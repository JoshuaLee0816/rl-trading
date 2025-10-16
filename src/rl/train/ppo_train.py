import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import datetime
import multiprocessing as mp
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from gymnasium.vector import AsyncVectorEnv
from tqdm import trange

import wandb

# ==== 建議：Mac 避免 fork + MPS 衝突，強制 spawn ====
try:
    if platform.system() == "Darwin":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# 降低每個子程序的CPU thread 競爭
torch.set_num_threads(1)

# === 專案路徑 ===
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]
ROOT = HERE.parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl.env.StockTradingEnv import StockTradingEnv
# === 模組 ===
from rl.models.ppo_agent import PPOAgent
from rl.test.ppo_test import _resolve_test_path, run_test_once, run_test_random_start

# RAM記憶體觀察用
proc = psutil.Process(os.getpid())

# region --------- 小工具 ----------
def split_infos(infos):
    if isinstance(infos, dict) and len(infos) > 0 and isinstance(next(iter(infos.values())), (np.ndarray, list, tuple)):
        n = len(next(iter(infos.values())))
        out = []
        for i in range(n):
            d = {k: (v[i] if isinstance(v, (np.ndarray, list, tuple)) else v) for k, v in infos.items()}
            out.append(d)
        return out
    elif isinstance(infos, dict):
        return [infos]
    else:
        return infos

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
    segment_years = days / 252.0
    return {"R_total": R_total, "total_return": total_return, "days": days, "segment_years": segment_years, "annualized_pct": annualized_return * 100.0}
# endregion 小工具

# region 主程式
if __name__ == "__main__":
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    env_cfg   = config["environment"]
    data_cfg  = config["data"]
    ppo_cfg   = config.get("ppo", {})
    log_cfg   = config.get("logging", {})

    n_episodes   = int(train_cfg["n_episodes"])
    ckpt_freq    = int(train_cfg.get("ckpt_freq", 50))
    max_ckpts    = int(train_cfg.get("max_ckpts", 5))
    upload_wandb = bool(train_cfg.get("upload_wandb", False))
    resume_from_best = bool(train_cfg.get("resume_from_best", False))

    num_envs     = int(ppo_cfg.get("num_envs", 2))
    wandb_every  = int(log_cfg.get("wandb_every", 10))
    test_every   = int(log_cfg.get("test_every", train_cfg.get("test_every", 10)))

    init_cash      = env_cfg["initial_cash"]
    lookback       = env_cfg["lookback"]
    reward_mode    = str(env_cfg.get("reward_mode", "daily_return")).lower().strip()
    action_mode    = str(env_cfg.get("action_mode", "discrete")).lower().strip()
    max_holdings   = int(env_cfg.get("max_holdings"))
    qmax_per_trade = int(env_cfg.get("qmax_per_trade"))

    # region W&B 初始化
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if upload_wandb:
        test_cfg = config.get("testing", {})
        env_cfg  = config.get("environment", {})
        model = train_cfg.get("model")

        wandb.init(
            project="rl-trading",
            name=f"{model}_{run_id}",
            job_type="train",
            config=config,
            settings=wandb.Settings(_disable_stats=True)
        )

        print(f"Policy = {test_cfg.get('policy', 'argmax')} | Reward = {env_cfg.get('reward_mode', 'daily_return')} | Encoder = {env_cfg.get('encoder', 'mlp')}")
        recent_test_logs = deque(maxlen=max_ckpts)

    outdir = ROOT / log_cfg.get("outdir", "logs/runs")
    run_dir = outdir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # 載資料
    data_file = data_cfg.get("file")
    data_path = ROOT / "data" / "processed" / data_file
    if data_file.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["date"])
    selected_feats = data_cfg.get("features", None)
    if selected_feats is not None:
        keep_cols = ["date", "stock_id"] + selected_feats
        df = df[[c for c in keep_cols if c in df.columns]]
    ids = sorted(df["stock_id"].unique())
    num_stocks = len(ids)

    def make_env():
        def _init():
            return StockTradingEnv(
                df=df,
                stock_ids=ids,
                lookback=lookback,
                initial_cash=init_cash,
                reward_mode=reward_mode,
                action_mode=action_mode,
                max_holdings=max_holdings,
                qmax_per_trade=qmax_per_trade,
                device="cpu",
            )
        return _init

    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    n_envs = num_envs

    tmp_env = StockTradingEnv(
        df=df,
        stock_ids=ids,
        lookback=lookback,
        initial_cash=init_cash,
        reward_mode=reward_mode,
        action_mode=action_mode,
        max_holdings=max_holdings,
        qmax_per_trade=qmax_per_trade,
        device="cpu",
    )
    obs_dim = tmp_env.obs_dim
    del tmp_env

    agent = PPOAgent(obs_dim=obs_dim, num_stocks=num_stocks, qmax_per_trade=qmax_per_trade, config=config)

    ckpt_dir = ROOT / "logs/runs"
    actor_best = ckpt_dir / "actor_best.pt"
    critic_best = ckpt_dir / "critic_best.pt"

    if resume_from_best:
        if actor_best.exists() and critic_best.exists():
            try:
                agent.actor.load_state_dict(torch.load(actor_best, map_location=agent.device))
                agent.critic.load_state_dict(torch.load(critic_best, map_location=agent.device))
                print(f"[INFO] 從 best.pt 續訓成功 ({actor_best.name}, {critic_best.name})")
            except Exception as e:
                print(f"[WARN] 續訓模型載入失敗，改為從頭訓練：{e}")
        else:
            print("[WARN] 找不到 best.pt，改為從頭訓練")
    else:
        print("[INFO] resume_from_best=False，從頭開始訓練")

    progress_bar = trange(1, n_episodes + 1, unit="EP")

    try:
        for ep in progress_bar:
            obs_nd, infos = envs.reset()
            obs = agent.obs_to_tensor(obs_nd)
            infos_list = split_infos(infos)
            prev_V = torch.tensor([i["V"] for i in infos_list], dtype=torch.float32)
            daily_returns = []
            ep_trade_counts = [0 for _ in range(n_envs)]
            ep_rewards = []

            for t in range(agent.n_steps):
                mask_batch = [i.get("action_mask_3d", None) for i in infos_list]
                actions_tuple, actions_flat, logps, values, obs_batch, mask_batch_t = agent.select_action(obs, mask_batch)
                actions_np = np.asarray(actions_tuple, dtype=np.int64)
                obs_nd, rewards, dones, truncs, infos = envs.step(actions_np)
                ep_rewards.extend(rewards.tolist())
                obs = agent.obs_to_tensor(obs_nd)
                infos_list = split_infos(infos)

                for i in range(n_envs):
                    if "trade_count" in infos_list[i]:
                        ep_trade_counts[i] += infos_list[i]["trade_count"]

                cur_V = torch.tensor([i["V"] for i in infos_list], dtype=torch.float32)
                log_ret = torch.log(cur_V / (prev_V + 1e-8))
                mask_alive = ~(torch.as_tensor(dones, dtype=torch.bool) | torch.as_tensor(truncs, dtype=torch.bool))
                if mask_alive.any():
                    daily_returns.append(log_ret[mask_alive].detach().cpu())
                prev_V = cur_V

                for i in range(n_envs):
                    agent.store_transition(obs_batch[i], actions_flat[i], rewards[i], dones[i], logps[i], values[i], mask_batch_t[i])

            logs = agent.update()

            total_ep = ep * n_envs

            # === 🟩 訓練結果上傳到 W&B ===
            if upload_wandb and (ep % max(1, wandb_every) == 0):
                metrics = compute_episode_metrics(daily_returns)
                reward_mean = float(np.mean(ep_rewards)) if len(ep_rewards) > 0 else 0.0
                ep_mdd = min([i.get("mdd", 0.0) for i in infos_list]) * 100
                wandb.log({
                    "train/actor_loss": agent.actor_loss_log[-1] if agent.actor_loss_log else None,
                    "train/critic_loss": agent.critic_loss_log[-1] if agent.critic_loss_log else None,
                    "train/entropy": agent.entropy_log[-1] if agent.entropy_log else None,
                    "train/avg_trade_count": float(np.mean(ep_trade_counts)),
                    "train/mdd%": ep_mdd,
                    "train/reward_mean": reward_mean,
                    "train/annualized_pct": float(metrics["annualized_pct"]),
                }, step=total_ep)
            # === 🟩 End ===

            if upload_wandb and (ep % max(1, test_every) == 0):
                tmp_ckpt = run_dir / f"eval_tmp_ep{ep}.pt"
                torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()}, tmp_ckpt)
                with open(ROOT / "config.yaml", "r", encoding="utf-8") as _f:
                    _cfg_for_test = yaml.safe_load(_f)

                years = (2020, 2021, 2022, 2023, 2024)
                test_cfg = _cfg_for_test.get("testing", {})
                test_policy = test_cfg.get("policy", "argmax")
                test_conf_threshold = float(test_cfg.get("conf_threshold", 0.75))
                test_n_runs = int(test_cfg.get("n_runs", 5))

                # === 🔧 修改區域開始 ===
                # 固定五年測試 (固定初始資金 100000)
                fixed_results = {}
                for y in years:
                    data_path = _resolve_test_path(ROOT, _cfg_for_test, y)
                    if not data_path.exists():
                        print(f"[WARN] 找不到 {y} 的測試檔案：{data_path}")
                        continue
                    try:
                        tr, mdd, _df_perf, df_base, fig, _actions, sell_count = run_test_once(
                            actor_path=str(tmp_ckpt),
                            data_path=str(data_path),
                            config_path=str(ROOT / "config.yaml"),
                            plot=True,
                            save_trades=True,
                            tag=f"{y}_Fixed_ep{ep}",
                            verbose=True,
                            return_fig=True,
                            policy=test_policy,
                            conf_threshold=test_conf_threshold,
                            initial_cash=100000,
                        )
                        fixed_results[y] = {"total_return": tr, "max_drawdown": mdd, "trade_count": sell_count, "fig": fig}
                    except Exception as e:
                        print(f"[WARN] 固定年度測試 {y} 失敗：{e}")

                # 隨機五段測試
                try:
                    random_result = run_test_random_start(
                        actor_path=str(tmp_ckpt),
                        config_path=str(ROOT / "config.yaml"),
                        n_runs=test_n_runs,
                        save_trades=True,
                        plot=True,
                        tag=f"Random_ep{ep}",
                        verbose=True,
                        policy=test_policy,
                        conf_threshold=test_conf_threshold,
                    )
                    avg_return = random_result["total_return"]
                    avg_mdd = random_result["max_drawdown"]
                    avg_trade_count = random_result["sell_count"]
                    figs = random_result["figs"]
                    random_results = {}
                    for i, fig in enumerate(figs, 1):
                        random_results[f"random_{i}"] = {"total_return": avg_return, "max_drawdown": avg_mdd, "trade_count": avg_trade_count, "fig": fig}
                except Exception as e:
                    print(f"[WARN] Random-start 測試失敗：{e}")

                # === 上傳固定五年測試結果 ===
                if upload_wandb and len(fixed_results) > 0:
                    avg_return_fixed = np.mean([v["total_return"] for v in fixed_results.values()])
                    avg_mdd_fixed = np.mean([v["max_drawdown"] for v in fixed_results.values()])
                    avg_trades_fixed = np.mean([v["trade_count"] for v in fixed_results.values()])
                    wandb.log({
                        "test_fixed/mean_return": avg_return_fixed,
                        "test_fixed/mean_max_drawdown": avg_mdd_fixed,
                        "test_fixed/mean_trade_count": avg_trades_fixed,
                    }, step=total_ep)
                    imgs_fixed = [wandb.Image(v["fig"], caption=f"Fixed Test {y}") for y, v in fixed_results.items() if v.get("fig")]
                    if imgs_fixed:
                        wandb.log({"test_fixed/panel": imgs_fixed}, step=total_ep)

                # === 上傳隨機五段測試結果 ===
                if upload_wandb and len(random_results) > 0:
                    avg_return_rand = np.mean([v["total_return"] for v in random_results.values()])
                    avg_mdd_rand = np.mean([v["max_drawdown"] for v in random_results.values()])
                    avg_trades_rand = np.mean([v["trade_count"] for v in random_results.values()])
                    wandb.log({
                        "test_random/mean_return": avg_return_rand,
                        "test_random/mean_max_drawdown": avg_mdd_rand,
                        "test_random/mean_trade_count": avg_trades_rand,
                    }, step=total_ep)
                    imgs_rand = [wandb.Image(v["fig"], caption=f"Random Test {k}") for k, v in random_results.items() if v.get("fig")]
                    if imgs_rand:
                        wandb.log({"test_random/panel": imgs_rand}, step=total_ep)
                # === 🔧 修改區域結束 ===

                try:
                    tmp_ckpt.unlink(missing_ok=True)
                except Exception:
                    pass

            rss = proc.memory_info().rss / 1024**3
            print(f"[MEM] ep={ep} | RSS={rss:.2f} GB ")

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
    print(f"Training finished. Results saved in: {run_dir}")

# endregion 主程式
