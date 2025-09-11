import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import yaml
import numpy as np
import pandas as pd
import datetime
import torch
import gymnasium as gym
import wandb
from pathlib import Path
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from tqdm import trange

# === 專案路徑 ===
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]
ROOT = HERE.parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# === 模組 ===
from rl.models.ppo_agent import PPOAgent
from rl.env.StockTradingEnv import StockTradingEnv
from rl.train.logger import RunLogger


def split_infos(infos):
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
    import numpy as _np
    if mask_any is None:
        return None
    if isinstance(mask_any, list):
        return _np.stack(mask_any, axis=0).astype(bool, copy=False)
    if isinstance(mask_any, _np.ndarray):
        if mask_any.dtype == object:
            return _np.stack(list(mask_any), axis=0).astype(bool, copy=False)
        return mask_any.astype(bool, copy=False)
    return _np.stack(list(mask_any), axis=0).astype(bool, copy=False)


if __name__ == "__main__":
    episode_entropy = []

    # === 讀取 config.yaml ===
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---- 訓練設定 ----
    n_episodes = config["training"]["n_episodes"]
    save_freq = config["training"]["save_freq"]

    ckpt_freq = 100   # 每多少 episodes 存一次 checkpoint
    max_ckpts = 10    # 最多保留 10 個

    ppo_cfg = config.get("ppo", {})
    num_envs = ppo_cfg.get("num_envs", 2)
    use_subproc = ppo_cfg.get("use_subproc", True)

    # ---- 環境設定 ----
    init_cash = config["environment"]["initial_cash"]
    lookback = config["environment"]["lookback"]
    reward_mode = str(config["environment"]["reward_mode"]).lower().strip()
    action_mode = str(config["environment"]["action_mode"]).lower().strip()
    max_holdings = config["environment"].get("max_holdings", None)
    qmax_per_trade = int(config["environment"].get("qmax_per_trade", 1))

    # ---- 初始化 W&B ----
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project="rl-trading",
        name=f"run_{run_id}",
        group="full-data",
        job_type="train",
        config=config
    )

    # ---- 輸出目錄 ----
    outdir = ROOT / config["logging"]["outdir"]
    run_dir = outdir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # === 載入資料 ===
    data_file = config["data"].get("file")
    data_path = ROOT / "data" / "processed" / data_file
    if data_file.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["date"])

    # ---- 根據 config 過濾欄位 ----
    selected_feats = config["data"].get("features", None)
    if selected_feats is not None:
        keep_cols = ["date", "stock_id"] + selected_feats
        df = df[[c for c in keep_cols if c in df.columns]]

    print("DataFrame 欄位：", df.columns.tolist())
    ids = sorted(df["stock_id"].unique())
    num_stocks = len(ids)

    # === 建立環境 ===
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
        print(f"[INFO] PPO 使用 SubprocVectorEnv 並行 {num_envs} 個環境")
    else:
        env = SyncVectorEnv([make_env for _ in range(num_envs)])
        print(f"[INFO] PPO 使用 SyncVectorEnv 並行 {num_envs} 個環境")

    # === 初始化 agent ===
    if hasattr(env, "single_observation_space"):
        single_os = env.single_observation_space
        single_as = env.single_action_space
    else:
        single_os = env.observation_space
        single_as = env.action_space

    obs_dim = int(np.prod(single_os.shape))

    if isinstance(single_as, gym.spaces.Box):
        action_dim = int(np.prod(single_as.shape))
    elif isinstance(single_as, gym.spaces.MultiDiscrete):
        action_dim = int(np.prod(single_as.nvec))
    elif isinstance(single_as, gym.spaces.Discrete):
        action_dim = single_as.n
    else:
        raise ValueError(f"Unsupported action_space type: {type(single_as)}")

    agent = PPOAgent(
        obs_dim=obs_dim,
        num_stocks=num_stocks,
        qmax_per_trade=qmax_per_trade,
        config=ppo_cfg,
    )

    # === Logging ===
    all_rewards, summary = [], []
    trade_sample_freq = config["logging"].get("trade_sample_freq", 10)
    logger = RunLogger(run_dir, trade_sample_freq)

    progress_bar = trange(1, n_episodes + 1, desc="Training", unit="episode")

    # === PPO 訓練 ===
    try:
        for ep in progress_bar:
            obs, infos = env.reset()
            infos_list = split_infos(infos)
            daily_returns = []
            ep_trade_counts = [0 for _ in range(num_envs)]  # 每個環境 episode 的交易次數

            action_mask_batch = normalize_mask_batch(infos.get("action_mask_3d", None))

            for t in range(agent.n_steps):
                batch_actions, batch_actions_flat, batch_logps, batch_values, batch_masks_flat = [], [], [], [], []
                for i in range(int(getattr(obs, "shape", [env.num_envs])[0])):
                    obs_i = obs[i]
                    mask_i = action_mask_batch[i] if action_mask_batch is not None else None

                    if mask_i is not None:
                        mask_flat_i = agent.flatten_mask(mask_i)
                        if hasattr(mask_flat_i, "detach"):
                            mask_flat_i = mask_flat_i.detach().to("cpu").numpy()
                        mask_flat_i = mask_flat_i.astype(bool, copy=False)
                    else:
                        mask_flat_i = None

                    action_tuple_i, action_flat_i, logp_i, value_i = agent.select_action(
                        obs_i, action_mask_3d=mask_i
                    )

                    batch_actions.append(np.asarray(action_tuple_i, dtype=np.int64))
                    batch_actions_flat.append(int(action_flat_i))
                    batch_logps.append(float(logp_i))
                    batch_values.append(float(value_i))
                    batch_masks_flat.append(mask_flat_i)

                actions = np.stack(batch_actions, axis=0).astype(np.int64)
                next_obs, rewards, dones, truncs, infos = env.step(actions)
                action_mask_batch = normalize_mask_batch(infos.get("action_mask_3d", None))
                infos_list = split_infos(infos)

                for i in range(len(infos_list)):
                    agent.store_transition(
                        obs[i],
                        int(batch_actions_flat[i]),
                        float(rewards[i]),
                        bool(dones[i]),
                        float(batch_logps[i]),
                        float(batch_values[i]),
                        batch_masks_flat[i],
                    )
                    logger.log_step(ep, infos_list[i])

                    # 讀取 trade_count
                    if "trade_count" in infos_list[i]:
                        ep_trade_counts[i] = infos_list[i]["trade_count"]

                obs = next_obs
                daily_returns.extend(rewards.tolist())

            agent.update()

            if len(agent.entropy_log) > 0:
                episode_entropy.append(agent.entropy_log[-1])

            # === 報酬統計 ===
            R_total = np.sum(daily_returns)
            total_return = np.exp(R_total) - 1
            days = len(daily_returns) if len(daily_returns) > 0 else 1
            annualized_return = (1 + total_return) ** (252 / days) - 1

            ep_return = annualized_return * 100
            all_rewards.append(ep_return)

            # 多環境 → 取平均交易數
            avg_trades = float(np.mean(ep_trade_counts))

            summary.append({
                "episode": ep,
                "annualized_return_pct": ep_return,
                "avg_trade_count": avg_trades,
            })

            # === W&B logging ===
            wandb.log({
                "episode": ep,
                "annualized_return_pct": ep_return,
                "avg_trade_count": avg_trades,
                "baseline": 0.0,
                "actor_loss": agent.actor_loss_log[-1] if agent.actor_loss_log else None,
                "critic_loss": agent.critic_loss_log[-1] if agent.critic_loss_log else None,
                "entropy": episode_entropy[-1] if episode_entropy else None,
            })

            # === 定期存 checkpoint ===
            if ep % ckpt_freq == 0:
                ckpt_path = run_dir / f"checkpoint_ep{ep}.pt"
                torch.save({
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "episode": ep
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

                # 保留最近 10 個
                ckpts = sorted(run_dir.glob("checkpoint_ep*.pt"))
                if len(ckpts) > max_ckpts:
                    for old_ckpt in ckpts[:-max_ckpts]:
                        old_ckpt.unlink()
                        print(f"Removed old checkpoint: {old_ckpt}")

    finally:
        try:
            env.close()
        except Exception:
            pass

    # === 儲存最後模型 ===
    torch.save(agent.actor.state_dict(), run_dir / "ppo_actor.pt")
    torch.save(agent.critic.state_dict(), run_dir / "ppo_critic.pt")

    wandb.save(str(run_dir / "ppo_actor.pt"))
    wandb.save(str(run_dir / "ppo_critic.pt"))

    # === 儲存紀錄 ===
    if config["logging"]["save_summary"]:
        pd.DataFrame(summary).to_csv(run_dir / "summary.csv", index=False)

    print(f"✅ Training finished. Model=PPO. Results saved in: {run_dir}")
