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
import matplotlib.pyplot as plt
from pathlib import Path
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from tqdm import trange
from collections import deque

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
from rl.test.ppo_test import run_test_once   # ✅ 引用測試 function

# region 小工具部分

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

def _infer_spaces_and_dims(env):
    os_ = getattr(env, "single_observation_space", env.observation_space)
    as_ = getattr(env, "single_action_space", env.action_space)

    obs_dim = int(np.prod(os_.shape))
    if isinstance(as_, gym.spaces.MultiDiscrete):
        action_dim = int(np.prod(as_.nvec))
    elif isinstance(as_, gym.spaces.Discrete):
        action_dim = int(as_.n)
    elif isinstance(as_, gym.spaces.Box):
        action_dim = int(np.prod(as_.shape))
    else:
        raise ValueError(f"Unsupported action_space: {type(as_)} | {as_!r}")
    return os_, as_, obs_dim, action_dim

def save_checkpoint(run_dir: Path, agent, ep: int) -> Path:
    ckpt_path = run_dir / f"checkpoint_ep{ep}.pt"
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "episode": ep
    }, ckpt_path)
    return ckpt_path

def prune_checkpoints(run_dir: Path, max_ckpts: int) -> None:
    ckpts = sorted(run_dir.glob("checkpoint_ep*.pt"))
    if len(ckpts) > max_ckpts:
        for old_ckpt in ckpts[:-max_ckpts]:
            try:
                old_ckpt.unlink()
            except Exception:
                pass

def compute_episode_metrics(daily_returns: list[float]) -> dict:
    R_total = float(np.sum(daily_returns)) if daily_returns else 0.0
    total_return = float(np.exp(R_total) - 1.0)
    days = len(daily_returns) if daily_returns else 1
    annualized_return = float((1.0 + total_return) ** (252.0 / days) - 1.0)
    return {
        "R_total": R_total,
        "total_return": total_return,
        "days": days,
        "annualized_pct": annualized_return * 100.0,
    }

def run_eval_and_plot(
    ckpt_path: Path,
    config_path: Path,
    data_path_test: Path,
    ep: int,
    recent_curves,
    upload_wandb: bool
):
    # === 呼叫測試 ===
    total_ret, max_dd, df_perf, df_baseline = run_test_once(
        ckpt_path, data_path_test, config_path,
        plot=False, save_trades=False, tag=f"ep{ep}", verbose=False
    )

    # 更新最近曲線
    recent_curves.append((ep, df_perf))

    # 畫比較圖（與原本相同）
    plt.figure(figsize=(10, 6))
    for ep_id, df_c in recent_curves:
        plt.plot(df_c.index, df_c["value"], label=f"ep{ep_id}")
    plt.plot(df_baseline.index, df_baseline["baseline"], label="Baseline (0050)", linestyle="--", color="black")
    plt.title("Portfolio Value Comparison (Recent 10 Checkpoints)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    if upload_wandb:
        wandb.log({
            "test/total_return": total_ret,
            "test/max_drawdown": max_dd,
            "test/portfolio_curves": wandb.Image(plt)
        }, step=ep)
    plt.close()

    return total_ret, max_dd, df_perf, df_baseline
# endregion 小工具部分

# region 主程式
if __name__ == "__main__":

    episode_entropy = []

    # === 讀取 config.yaml ===
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # === 取出各區塊的別名 ===
    train_cfg = config["training"]
    env_cfg   = config["environment"]
    log_cfg   = config["logging"]
    data_cfg  = config["data"]
    ppo_cfg   = config.get("ppo", {}) 

    # ---- 訓練設定 ----
    n_episodes   = train_cfg["n_episodes"]
    save_freq    = train_cfg["save_freq"]
    upload_wandb = train_cfg["upload_wandb"]

    ckpt_freq = train_cfg["ckpt_freq"]      
    max_ckpts = train_cfg["max_ckpts"]

    num_envs    = ppo_cfg.get("num_envs")
    use_subproc = ppo_cfg.get("use_subproc")

    wandb_every       = log_cfg.get("wandb_every")

    # ---- 環境設定 ----
    init_cash      = env_cfg["initial_cash"]
    lookback       = env_cfg["lookback"]
    reward_mode    = str(env_cfg["reward_mode"]).lower().strip()
    action_mode    = str(env_cfg["action_mode"]).lower().strip()
    max_holdings   = env_cfg.get("max_holdings")
    qmax_per_trade = int(env_cfg.get("qmax_per_trade"))

    # ---- 近期曲線容器 ----
    recent_curves = deque(maxlen=max_ckpts)

    # ---- 初始化 W&B ----
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if upload_wandb:
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

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # === 載入資料 ===
    data_file = config["data"].get("file")
    data_path = ROOT / "data" / "processed" / data_file
    if data_file.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["date"])

    selected_feats = config["data"].get("features", None)
    if selected_feats is not None:
        keep_cols = ["date", "stock_id"] + selected_feats
        df = df[[c for c in keep_cols if c in df.columns]]

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
    else:
        env = SyncVectorEnv([make_env for _ in range(num_envs)])

    n_envs = getattr(env, "num_envs", num_envs)  # 統一使用的環境數

    # === 初始化 agent ===
    _, _, obs_dim, _ = _infer_spaces_and_dims(env)

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
            daily_returns = []
            ep_trade_counts = [0 for _ in range(num_envs)]

            action_mask_batch = normalize_mask_batch(infos.get("action_mask_3d", None))
            t_env_total, t_train_total, update_count = 0.0, 0.0, 0

            for t in range(agent.n_steps):
                batch_actions, batch_actions_flat, batch_logps, batch_values, batch_masks_flat = [], [], [], [], []
                for i in range(n_envs):
                    obs_i = obs[i]
                    mask_i = action_mask_batch[i] if action_mask_batch is not None else None
                    if mask_i is not None:
                        mask_flat_i = agent.flatten_mask(mask_i)
                        if hasattr(mask_flat_i, "detach"):
                            mask_flat_i = mask_flat_i.detach().to("cpu").numpy()
                        mask_flat_i = mask_flat_i.astype(bool, copy=False)
                    else:
                        mask_flat_i = None
                    action_tuple_i, action_flat_i, logp_i, value_i = agent.select_action(obs_i, action_mask_3d=mask_i)
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
                    if "trade_count" in infos_list[i]:
                        ep_trade_counts[i] = infos_list[i]["trade_count"]

                obs = next_obs
                daily_returns.extend(rewards.tolist())

            agent.update()

            if len(agent.entropy_log) > 0:
                episode_entropy.append(agent.entropy_log[-1])

            m = compute_episode_metrics(daily_returns)
            ep_return = m["annualized_pct"]
            all_rewards.append(ep_return)
            avg_trades = float(np.mean(ep_trade_counts))

            summary.append({
                "episode": ep,
                "annualized_return_pct": ep_return,
                "avg_trade_count": avg_trades,
            })

            # === W&B logging (Train) ===
            if upload_wandb and (ep % wandb_every) == 0:
                wandb.log({
                    "train/episode": ep,
                    "train/annualized_return_pct": ep_return,
                    "train/avg_trade_count": avg_trades,
                    "train/actor_loss": agent.actor_loss_log[-1] if agent.actor_loss_log else None,
                    "train/critic_loss": agent.critic_loss_log[-1] if agent.critic_loss_log else None,
                    "train/entropy": episode_entropy[-1] if episode_entropy else None,
                }, step=ep)

            # === 定期存 checkpoint & 測試 ===
            if ep % ckpt_freq == 0:
                ckpt_path = save_checkpoint(run_dir, agent, ep)
                prune_checkpoints(run_dir, max_ckpts)

                data_path_test = ROOT / "data" / "processed" / "full" / "walk_forward" / "WF_test_2020_full.parquet"
                config_path    = ROOT / "config.yaml"

                total_ret, max_dd, df_perf, df_baseline = run_eval_and_plot(
                    ckpt_path, config_path, data_path_test, ep, recent_curves, upload_wandb
                )

    finally:
        try:
            env.close()
        except Exception:
            pass

    # === 儲存最後模型 ===
    torch.save(agent.actor.state_dict(), run_dir / "ppo_actor.pt")
    torch.save(agent.critic.state_dict(), run_dir / "ppo_critic.pt")

    if upload_wandb:
        wandb.save(str(run_dir / "ppo_actor.pt"))
        wandb.save(str(run_dir / "ppo_critic.pt"))

    if config["logging"]["save_summary"]:
        pd.DataFrame(summary).to_csv(run_dir / "summary.csv", index=False)

    print(f"✅ Training finished. Model=PPO. Results saved in: {run_dir}")

# endregion 主程式