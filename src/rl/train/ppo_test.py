# src/rl/train/ppo_test.py
import os
# 建議：MPS 上 Dirichlet 會缺 op，先開 fallback，避免漏掉時直接崩潰
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import datetime
import torch
import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from tqdm import trange

# === 設定專案路徑 ===
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]          # .../rl-trading/src
ROOT = HERE.parents[3]             # .../rl-trading
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# === 匯入模組（保留 PPO 與 Env、Logger） ===
from rl.models.ppo_agent import PPOAgent
from rl.env.StockTradingEnv import StockTradingEnv
from rl.train.logger import RunLogger


def split_infos(infos):
    """把 dict of arrays 轉成 list of dicts（給 vector env 用）"""
    if isinstance(infos, dict) and isinstance(list(infos.values())[0], (np.ndarray, list)):
        num_envs = len(next(iter(infos.values())))
        return [
            {k: (v[i] if isinstance(v, (np.ndarray, list)) else v) for k, v in infos.items()}
            for i in range(num_envs)
        ]
    elif isinstance(infos, dict):
        return [infos]
    else:
        return infos

def normalize_mask_batch(mask_any):
    """
    把 infos["action_mask_3d"] 規整成 ndarray(bool) 形狀：(num_envs, 3, N, QMAX+1)
    - Async/SubprocVectorEnv 會把每個 env 的 ndarray 裝成 list 或 object-dtype ndarray
    - 這裡把它 stack 起來並轉成 bool，避免 torch.from_numpy() 因 dtype=object 爆掉
    """
    import numpy as _np
    if mask_any is None:
        return None
    # list[ndarray] -> ndarray
    if isinstance(mask_any, list):
        return _np.stack(mask_any, axis=0).astype(bool, copy=False)
    if isinstance(mask_any, _np.ndarray):
        if mask_any.dtype == object:
            # 例如 array([ndarray, ndarray, ...], dtype=object)
            return _np.stack(list(mask_any), axis=0).astype(bool, copy=False)
        # 其他普通情況，補一個保險轉型
        return mask_any.astype(bool, copy=False)
    # 其他少見型別，嘗試轉 list 再 stack
    try:
        return _np.stack(list(mask_any), axis=0).astype(bool, copy=False)
    except Exception:
        raise TypeError(f"Unsupported mask container type: {type(mask_any)}")


if __name__ == "__main__":
    episode_entropy = []

    # === 讀取 config.yaml ===
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---- 訓練與 PPO 相關設定 ----
    n_episodes = config["training"]["n_episodes"]
    save_freq  = config["training"]["save_freq"]

    ppo_cfg    = config.get("ppo", {})
    num_envs   = ppo_cfg.get("num_envs", 2)
    use_subproc= ppo_cfg.get("use_subproc", True)

    # ---- 環境設定 ----
    init_cash    = config["environment"]["initial_cash"]
    lookback     = config["environment"]["lookback"]
    reward_mode  = str(config["environment"]["reward_mode"]).lower().strip()
    action_mode  = str(config["environment"]["action_mode"]).lower().strip()
    max_holdings = config["environment"].get("max_holdings", None)
    qmax_per_trade = int(config["environment"].get("qmax_per_trade", 1))  # 新增：若沒有就用 1

    # ---- 輸出目錄 ----
    outdir = ROOT / config["logging"]["outdir"]
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = outdir / f"run_{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    # 存一份 config 到 run 資料夾
    with open(outdir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # === 載入資料 ===
    data_file = config["data"].get("file", "training_data_20.parquet")
    data_path = ROOT / "data" / "processed" / data_file
    if data_file.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["date"])

    ids = sorted(df["stock_id"].unique())  # 全部股票
    num_stocks = len(ids)                  # 直接以股票清單長度為檔數（提供給 PPOAgent）

    # === 建立環境（只跑 PPO，多環境並行） ===
    def make_env():
        return StockTradingEnv(
            df=df,
            stock_ids=ids,
            lookback=lookback,
            initial_cash=init_cash,
            reward_mode=reward_mode,
            action_mode=action_mode,
            max_holdings=max_holdings,
            qmax_per_trade=qmax_per_trade,
        )

    if use_subproc:
        env = AsyncVectorEnv([make_env for _ in range(num_envs)])
        print(f"[INFO] PPO 使用 SubprocVectorEnv 並行 {num_envs} 個環境 (多核 CPU)")
    else:
        env = SyncVectorEnv([make_env for _ in range(num_envs)])
        print(f"[INFO] PPO 使用 SyncVectorEnv 並行 {num_envs} 個環境 (單核)")

    # === 初始化 agent ===
    # 使用 single_*（重點：VectorEnv 不能用 env.action_space 直接取維度）
    if hasattr(env, "single_observation_space"):
        single_os = env.single_observation_space
        single_as = env.single_action_space
    else:
        single_os = env.observation_space
        single_as = env.action_space

    obs_dim = int(np.prod(single_os.shape))

    # 這個 action_dim 只拿來做 shape 檢查（agent 內部不需要）
    if isinstance(single_as, gym.spaces.Box):
        action_dim = int(np.prod(single_as.shape))  # 例如 portfolio: N 或 N+1
    elif isinstance(single_as, gym.spaces.MultiDiscrete):
        action_dim = int(np.prod(single_as.nvec))
    elif isinstance(single_as, gym.spaces.Discrete):
        action_dim = single_as.n
    else:
        raise ValueError(f"Unsupported action_space type: {type(single_as)}")

    # ✅ 對齊你的 PPOAgent 簽名
    agent = PPOAgent(
        obs_dim=obs_dim,
        num_stocks=num_stocks,
        qmax_per_trade=qmax_per_trade,
        config=ppo_cfg,
    )

    # === Logging ===
    all_rewards, summary = [], []
    trade_sample_freq = config["logging"].get("trade_sample_freq", 10)
    logger = RunLogger(outdir, trade_sample_freq)

    progress_bar = trange(1, n_episodes + 1, desc="Training", unit="episode")

    # === PPO 訓練 (多環境並行) ===
    try:
        for ep in progress_bar:
            obs, infos = env.reset()
            daily_returns = []

            # ★ 取出並規整 batched 遮罩
            action_mask_batch = normalize_mask_batch(infos.get("action_mask_3d", None))

            for t in range(agent.n_steps):
                # === 逐環境選動作：每次只傳「單一環境」的遮罩 (3, N, QMAX+1) ===
                batch_actions, batch_actions_flat, batch_logps, batch_values, batch_masks_flat = [], [], [], [], []
                for i in range(int(getattr(obs, "shape", [env.num_envs])[0])):
                    obs_i = obs[i]
                    mask_i = None
                    if action_mask_batch is not None:
                        # 取出第 i 個環境的單一遮罩 (3, N, QMAX+1)
                        mask_i = action_mask_batch[i]

                    # ★ 先把當步要用的遮罩扁平化成 (A,)
                    if mask_i is not None:
                        mask_flat_i = agent.flatten_mask(mask_i)  # torch.bool tensor, shape (A,)
                        # 存 CPU numpy.bool_ 以防內部 buffer 需要 numpy
                        if hasattr(mask_flat_i, "detach"):
                            mask_flat_i = mask_flat_i.detach().to("cpu").numpy()
                        mask_flat_i = mask_flat_i.astype(bool, copy=False)
                    else:
                        mask_flat_i = None  # 若 agent 內部能容忍 None，也可傳 None；否則可做全 True
                    
                    action_tuple_i, action_flat_i, logp_i, value_i = agent.select_action(obs_i, action_mask_3d=mask_i)

                    batch_actions.append(np.asarray(action_tuple_i, dtype=np.int64))

                    # 累積資料
                    batch_actions.append(np.asarray(action_tuple_i, dtype=np.int64))  # 給 env.step()
                    batch_actions_flat.append(int(action_flat_i))                      # 給 buffer
                    batch_logps.append(float(logp_i))
                    batch_values.append(float(value_i))
                    batch_masks_flat.append(mask_flat_i)                               # ★ 給 buffer：當步扁平遮罩
                
                # 把每個環境的結果堆回批次
                actions = np.stack(batch_actions, axis=0).astype(np.int64)          # (B, 3)
                log_probs_arr = np.asarray(batch_logps, dtype=np.float32)           # (B,)
                values_arr    = np.asarray(batch_values, dtype=np.float32)          # (B,)
                actions_flat_arr = np.asarray(batch_actions_flat, dtype=np.int64)   # (B,)
                masks_flat_list = list(batch_masks_flat)                             # 長度 B，每個是 (A,) 或 None

                # ★ 統一修剪到 env.num_envs（避免 B > env.num_envs 時炸裂）
                B = actions.shape[0]
                if B != env.num_envs:
                    if B < env.num_envs:
                        raise ValueError(f"[batch undersize] got {B}, expected {env.num_envs}")
                    # 修剪多出來的（例如 B=4, env.num_envs=2）
                    actions           = actions[:env.num_envs]
                    log_probs_arr     = log_probs_arr[:env.num_envs]
                    values_arr        = values_arr[:env.num_envs]
                    actions_flat_arr  = actions_flat_arr[:env.num_envs]
                    masks_flat_list   = masks_flat_list[:env.num_envs]

                # 進環境
                next_obs, rewards, dones, truncs, infos = env.step(actions)

                # ★ 更新下一步的遮罩（規整為 ndarray(bool)）
                action_mask_batch = normalize_mask_batch(infos.get("action_mask_3d", None))

                # === 存資料與記錄（逐環境）===
                infos_list = split_infos(infos)

                # 把逐環境的結果堆成陣列，方便 index 對齊
                actions_flat = np.asarray(batch_actions_flat, dtype=np.int64)   # (num_envs,)
                log_probs    = np.asarray(batch_logps, dtype=np.float32)        # (num_envs,)
                values       = np.asarray(batch_values, dtype=np.float32)       # (num_envs,)

                for i in range(len(infos_list)):
                    agent.store_transition(
                        obs[i],                     # 觀測
                        int(actions_flat[i]),       # ★ 展平後的 action id（對應 policy 的 logits 索引）
                        float(rewards[i]),
                        bool(dones[i]),
                        float(log_probs[i]),
                        float(values[i]),
                        masks_flat_list[i],         # ★ 新增：當步使用的扁平遮罩 (A,) / numpy.bool_
                    )
                    logger.log_step(ep, infos_list[i])

                obs = next_obs
                daily_returns.extend(rewards.tolist())

            # 更新
            agent.update()

            # 紀錄本回合 entropy
            if len(agent.entropy_log) > 0:
                episode_entropy.append(agent.entropy_log[-1])

            final_V = np.mean([info.get("V", init_cash) for info in infos_list])

            # episode 的總報酬率
            """
            total_return = (final_V - init_cash) / init_cash

            # 年化（假設一年 252 個交易日）
            days = len(daily_returns) if len(daily_returns) > 0 else 1
            annualized_return = (1 + total_return) ** (252 / days) - 1

            """

            R_total = np.sum(daily_returns)
            total_return = np.exp(R_total) -1
            days = len(daily_returns) if len(daily_returns) > 0 else 1
            annualized_return = (1 + total_return ) ** (252/days) - 1

            ep_return = annualized_return * 100
            all_rewards.append(ep_return)
            summary.append({"episode": ep, "annualized_return_pct": ep_return})

            if ep % save_freq == 0:
                fig, ax1 = plt.subplots(figsize=(8, 4))

                # 左軸：Annualized Return
                x1 = range(1, len(all_rewards) + 1)
                ln1, = ax1.plot(
                    x1, all_rewards, label="Annualized Return (%)",
                    color="#1f77b4", linewidth=2
                )
                ax1.set_xlabel("Episode")
                ax1.set_ylabel("Annualized Return (%)", color=ln1.get_color())
                ax1.tick_params(axis="y", colors=ln1.get_color())
                ax1.grid(True, axis="both", alpha=0.3)

                lines = [ln1]
                labels = [ln1.get_label()]

                # 右軸：Entropy
                if len(episode_entropy) > 0:
                    ax2 = ax1.twinx()
                    x2 = range(1, len(episode_entropy) + 1)
                    ln2, = ax2.plot(
                        x2, episode_entropy, label="Entropy",
                        color="#ff7f0e", linewidth=2, linestyle="--"
                    )
                    ax2.set_ylabel("Entropy", color=ln2.get_color())
                    ax2.tick_params(axis="y", colors=ln2.get_color())
                    lines.append(ln2)
                    labels.append(ln2.get_label())

                # 合併 legend
                ax1.legend(lines, labels, loc="best")

                fig.suptitle("Training Progress (PPO)")
                fig.tight_layout()
                fig.savefig(outdir / "reward_entropy_curve.png")
                plt.close(fig)

    finally:
        # 確保正常關閉 vector 環境，避免 __del__ 噴錯
        try:
            env.close()
        except Exception:
            pass

    # === 儲存模型 ===
    torch.save(agent.actor.state_dict(), outdir / "ppo_actor.pt")
    torch.save(agent.critic.state_dict(), outdir / "ppo_critic.pt")

    # === 儲存訓練紀錄 ===
    if config["logging"]["save_summary"]:
        pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False)

    print(f"✅ Training finished. Model=PPO. Results saved in: {outdir}")
