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

def reset_envs(envs):
    obs_list, infos_list = [], []
    for e in envs:
        o, i = e.reset()
        obs_list.append(o)
        infos_list.append(i)
    return obs_list, infos_list

def step_envs(envs, actions):
    results = [e.step(a) for e, a in zip(envs, actions)]
    next_obs, rewards, dones, truncs, infos = zip(*results)

    # next_obs 已經是 tensor → stack 成 batch
    next_obs = torch.stack(next_obs)          # [n_envs, obs_dim]

    # rewards 是 tensor → stack 成 batch float
    rewards = torch.stack(rewards).float()    # [n_envs]

    # dones → float (0.0 或 1.0)，因為 GAE 要數值型
    dones = torch.as_tensor(dones, dtype=torch.float32)  # [n_envs]

    # truncs 基本上也是 bool，建議也統一轉 float
    truncs = torch.as_tensor(truncs, dtype=torch.float32)  # [n_envs]

    # infos 保留 list of dicts
    return next_obs, rewards, dones, truncs, list(infos)



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
    if mask_any is None:
        return None
    if isinstance(mask_any, list):
        return torch.stack([torch.as_tensor(m, dtype=torch.bool) for m in mask_any])
    if isinstance(mask_any, torch.Tensor):
        return mask_any.to(dtype=torch.bool)
    return torch.stack([torch.as_tensor(m, dtype=torch.bool) for m in mask_any])

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

def compute_episode_metrics(daily_returns: list[torch.Tensor]) -> dict:
    if len(daily_returns) == 0:
        return {"R_total": 0.0, "total_return": 0.0, "days": 1, "annualized_pct": 0.0}

    daily_returns = torch.cat([r.flatten() for r in daily_returns])
    R_total = daily_returns.sum()
    total_return = (torch.exp(R_total) - 1.0).item()

    days = daily_returns.numel()
    annualized_return = ((1.0 + total_return) ** (252.0 / days) - 1.0)
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
    total_ret, max_dd, df_perf, df_baseline = run_test_once(
        ckpt_path, data_path_test, config_path,
        plot=False, save_trades=False, tag=f"ep{ep}", verbose=False
    )
    recent_curves.append((ep, df_perf))
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
    
    train_cfg = config["training"]
    env_cfg   = config["environment"]
    log_cfg   = config["logging"]
    data_cfg  = config["data"]
    ppo_cfg   = config.get("ppo", {}) 

    n_episodes   = train_cfg["n_episodes"]
    save_freq    = train_cfg["save_freq"]
    upload_wandb = train_cfg["upload_wandb"]
    ckpt_freq    = train_cfg["ckpt_freq"]      
    max_ckpts    = train_cfg["max_ckpts"]

    num_envs    = ppo_cfg.get("num_envs")
    use_subproc = ppo_cfg.get("use_subproc")
    wandb_every = log_cfg.get("wandb_every")

    init_cash      = env_cfg["initial_cash"]
    lookback       = env_cfg["lookback"]
    reward_mode    = str(env_cfg["reward_mode"]).lower().strip()
    action_mode    = str(env_cfg["action_mode"]).lower().strip()
    max_holdings   = env_cfg.get("max_holdings")
    qmax_per_trade = int(env_cfg.get("qmax_per_trade"))

    recent_curves = deque(maxlen=max_ckpts)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if upload_wandb:
        wandb.init(
            project="rl-trading",
            name=f"run_{run_id}",
            group="full-data",
            job_type="train",
            config=ppo_cfg
        )

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
    # 這裡完全不用 SyncVectorEnv / AsyncVectorEnv
    envs = [make_env() for _ in range(num_envs)]
    n_envs = len(envs)

    # === 初始化 agent ===
    obs_dim = envs[0].obs_dim
    action_dim = envs[0].action_dim

    agent = PPOAgent(
        obs_dim=None,
        num_stocks=num_stocks,
        qmax_per_trade=qmax_per_trade,
        config=config,
    )

    # === 一次性的 Debug Print ===
    print("=== [DEBUG TRAIN LOOP INIT] ===")
    print(f"n_envs={n_envs}, n_steps={agent.n_steps}, batch_size={agent.batch_size}, epochs={agent.epochs}")
    print(f"Stocks={num_stocks}, Lookback={lookback}, Features={len(selected_feats)}")
    print(f"Max_holdings={max_holdings}, QMAX={qmax_per_trade}")

    all_rewards, summary = [], []
    trade_sample_freq = config["logging"].get("trade_sample_freq", 10)
    logger = RunLogger(run_dir, trade_sample_freq)
    progress_bar = trange(1, n_episodes + 1, desc="Training", unit="episode")

    try:
        for ep in progress_bar:

            # 等同之前的gym .reset()
            obs, infos = reset_envs(envs)

            for t in range(agent.n_steps):
                actions = []
                for i in range(len(envs)):
                    action_tuple, action_flat, logp, value, obs_flat, mask_flat = agent.select_action(
                        obs[i], infos[i].get("action_mask_3d", None)
                    )
                    actions.append(action_tuple)

                obs, rewards, dones, truncs, infos = step_envs(envs, actions)

            if ep == 1:
                print("=== [DEBUG EPISODE START] ===")
                print(f"obs (reset): type={type(obs)}, len={len(obs) if hasattr(obs,'__len__') else 'N/A'}")

            daily_returns = []
            ep_trade_counts = [0 for _ in range(num_envs)]

            # infos 是 list of dicts
            action_masks = [i.get("action_mask_3d", None) for i in infos]
            action_mask_batch = normalize_mask_batch(action_masks)

            for t in range(agent.n_steps):
                if ep == 1 and t == 0:
                    print("=== [DEBUG STEP] ===")
                    print(f"step={t}, action_mask_batch={action_mask_batch.shape if action_mask_batch is not None else None}")

                batch_actions, batch_actions_flat, batch_logps, batch_values, batch_masks_flat = [], [], [], [], []
                for i in range(n_envs):
                    obs_i = obs[i]
                    mask_i = action_mask_batch[i] if action_mask_batch is not None else None
                    action_tuple_i, action_flat_i, logp_i, value_i, obs_flat_i, mask_flat_i = agent.select_action(obs_i, action_mask_3d=mask_i)
                    batch_actions.append(torch.as_tensor(action_tuple_i, dtype=torch.long))
                    batch_actions_flat.append(torch.as_tensor(action_flat_i, dtype=torch.long))
                    batch_logps.append(torch.as_tensor(logp_i, dtype=torch.float32))
                    batch_values.append(torch.as_tensor(value_i, dtype=torch.float32))
                    batch_masks_flat.append(mask_flat_i)

                actions = torch.stack(batch_actions, dim=0)
                next_obs, rewards, dones, truncs, infos = step_envs(envs, actions)

                if ep == 1 and t == 0:
                    print("=== [DEBUG ENV STEP RESULT] ===")
                    print(f"actions={actions.shape}, rewards={rewards.shape}, dones={dones.shape}")

                # infos 是 list of dicts
                action_masks = [i.get("action_mask_3d", None) for i in infos]
                action_mask_batch = normalize_mask_batch(action_masks)

                infos_list = split_infos(infos)
                for i in range(len(infos_list)):
                    agent.store_transition(
                        obs_flat_i,
                        batch_actions_flat[i],
                        rewards[i],
                        dones[i],
                        batch_logps[i],
                        batch_values[i],
                        batch_masks_flat[i],
                    )
                    logger.log_step(ep, infos_list[i])
                    if "trade_count" in infos_list[i]:
                        ep_trade_counts[i] = infos_list[i]["trade_count"]
                obs = next_obs
                daily_returns.append(rewards)   # rewards 已經是 tensor [n_envs]

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
            if upload_wandb and (ep % wandb_every) == 0:
                wandb.log({
                    "train/episode": ep,
                    "train/annualized_return_pct": ep_return,
                    "train/avg_trade_count": avg_trades,
                    "train/actor_loss": agent.actor_loss_log[-1] if agent.actor_loss_log else None,
                    "train/critic_loss": agent.critic_loss_log[-1] if agent.critic_loss_log else None,
                    "train/entropy": episode_entropy[-1] if episode_entropy else None,
                }, step=ep)
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
            envs.close()
        except Exception:
            pass

    torch.save(agent.actor.state_dict(), run_dir / "ppo_actor.pt")
    torch.save(agent.critic.state_dict(), run_dir / "ppo_critic.pt")
    if upload_wandb:
        wandb.save(str(run_dir / "ppo_actor.pt"))
        wandb.save(str(run_dir / "ppo_critic.pt"))
    if config["logging"]["save_summary"]:
        pd.DataFrame(summary).to_csv(run_dir / "summary.csv", index=False)
    print(f"✅ Training finished. Model=PPO. Results saved in: {run_dir}")
# endregion 主程式
