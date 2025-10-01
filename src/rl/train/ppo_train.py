import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import yaml
import numpy as np
import pandas as pd
import datetime
import torch
import platform
import wandb
import psutil, os
import matplotlib.pyplot as plt
from pathlib import Path
from gymnasium.vector import AsyncVectorEnv
from tqdm import trange
from collections import deque
import multiprocessing as mp

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

# === 模組 ===
from rl.models.ppo_agent import PPOAgent
from rl.env.StockTradingEnv import StockTradingEnv
from rl.test.ppo_test import run_test_suite, run_test_once, _resolve_test_path  # ← 只在這裡新增 import：run_test_once, _resolve_test_path

# RAM記憶體觀察用
proc = psutil.Process(os.getpid())

# --------- 小工具 ----------
def split_infos(infos):
    """
    將 AsyncVectorEnv 的 infos(dict of arrays) 攤平成 list[dict]（每個env一份）
    """
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
    return {"R_total": R_total, "total_return": total_return, "days": days, "annualized_pct": annualized_return * 100.0}


# ============== 主程式 ==============
if __name__ == "__main__":
    # 讀設定
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
    num_envs     = int(ppo_cfg.get("num_envs", 2))
    wandb_every  = int(log_cfg.get("wandb_every", 10))
    test_every   = int(log_cfg.get("test_every", train_cfg.get("test_every", 10)))

    init_cash      = env_cfg["initial_cash"]
    lookback       = env_cfg["lookback"]
    reward_mode    = str(env_cfg.get("reward_mode", "daily_return")).lower().strip()
    action_mode    = str(env_cfg.get("action_mode", "discrete")).lower().strip()
    max_holdings   = int(env_cfg.get("max_holdings"))
    qmax_per_trade = int(env_cfg.get("qmax_per_trade"))

    # W&B
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if upload_wandb:
        wandb.init(project="rl-trading", name=f"run_{run_id}", job_type="train", config=ppo_cfg)
        # 建立固定長度的存放圖表區
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

    # 建立 Vector 環境（每個子程序環境僅用 CPU 記憶體）
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
                device="cpu",  # ⚠️ 環境固定CPU
            )
        return _init

    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    n_envs = num_envs

    # 建立一次性 env 取 obs_dim
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
    action_dim = tmp_env.action_dim
    del tmp_env

    # Agent（自動選擇 MPS / CUDA / CPU）
    agent = PPOAgent(
        obs_dim=obs_dim,
        num_stocks=num_stocks,
        qmax_per_trade=qmax_per_trade,
        config=config,
    )

    print("=== [DEBUG TRAIN LOOP INIT] ===")
    print(f"n_envs={n_envs}, n_steps={agent.n_steps}, batch_size={agent.batch_size}, epochs={agent.epochs}")
    print(f"Stocks={num_stocks}, Lookback={lookback}, Features={len(selected_feats) if selected_feats else 0}")
    print(f"Max_holdings={max_holdings}, QMAX={qmax_per_trade}")

    # 進度條：總回合 = 外層集數 * 環境數
    progress_bar = trange(1, n_episodes * n_envs + 1, unit="episode", unit_scale=n_envs)

    try:
        for ep in progress_bar:
            # reset
            obs_nd, infos = envs.reset()
            obs = agent.obs_to_tensor(obs_nd)
            infos_list = split_infos(infos)

            prev_V = torch.tensor([i["V"] for i in infos_list], dtype=torch.float32)
            daily_returns = []
            ep_trade_counts = [0 for _ in range(n_envs)]

            start = time.perf_counter()
            for t in range(agent.n_steps):
                mask_batch = [i.get("action_mask_3d", None) for i in infos_list]
                actions_tuple, actions_flat, logps, values, obs_batch, mask_batch_t = agent.select_action(
                    obs, mask_batch
                )
                actions_np = np.asarray(actions_tuple, dtype=np.int64)
                obs_nd, rewards, dones, truncs, infos = envs.step(actions_np)
                obs = agent.obs_to_tensor(obs_nd)
                infos_list = split_infos(infos)

                for i in range(n_envs):
                    if "trade_count" in infos_list[i]:
                        ep_trade_counts[i] += infos_list[i]["trade_count"]

                cur_V = torch.tensor([i["V"] for i in infos_list], dtype=torch.float32)
                log_ret = torch.log(cur_V / (prev_V + 1e-8))
                daily_returns.append(log_ret.detach().cpu())
                prev_V = cur_V

                for i in range(n_envs):
                    agent.store_transition(
                        obs_batch[i],
                        actions_flat[i],
                        rewards[i],
                        dones[i],
                        logps[i],
                        values[i],
                        mask_batch_t[i],
                    )

            end = time.perf_counter()
            #print(f"[DEBUG] Rollout (env interaction) 花費 {end - start:.3f} 秒")

            agent.update()

            metrics = compute_episode_metrics(daily_returns)
            days = metrics["days"]
            ep_trade_counts = [i.get("trade_count", 0) for i in infos_list]
            avg_trades = float(np.mean(ep_trade_counts)) / days * 252

            # check MDD 有沒有學會
            mdd_list = [i.get("mdd", 0.0) for i in infos_list]
            ep_mdd = min(mdd_list) * 100  # ← episode 最大回撤 (%)

            #print(f"[EP {ep}] days={metrics['days']} total_return={metrics['total_return']:.4f} "f"annualized={metrics['annualized_pct']:.2f}%")

            total_ep = ep * n_envs
            if total_ep % ckpt_freq == 0:
                ckpt_path = save_checkpoint(run_dir, agent, ep)
                prune_checkpoints(run_dir, max_ckpts)

            if upload_wandb and (ep % max(1, wandb_every) == 0):
                wandb.log({
                    "train/episode_outer": ep,
                    "train/episode_total": total_ep,
                    "train/actor_loss": agent.actor_loss_log[-1] if agent.actor_loss_log else None,
                    "train/critic_loss": agent.critic_loss_log[-1] if agent.critic_loss_log else None,
                    "train/entropy": agent.entropy_log[-1] if agent.entropy_log else None,
                    "eval/days": int(metrics["days"]),
                    "eval/total_return": float(metrics["total_return"]),
                    "eval/annualized_pct": float(metrics["annualized_pct"]),
                    "train/avg_trade_count": avg_trades,
                    "train/mdd": ep_mdd,
                }, step=total_ep)

            # === 每 test_every 個 outer-episode 跑一次 5 年測試並上傳到 W&B ===
            if upload_wandb and (ep % max(1, test_every) == 0):
                tmp_ckpt = run_dir / f"eval_tmp_ep{ep}.pt"
                torch.save({"actor": agent.actor.state_dict(),
                            "critic": agent.critic.state_dict()}, tmp_ckpt)

                years = (2020, 2021, 2022, 2023, 2024)
                # 先跑 Argmax（沿用既有 run_test_suite）
                results_arg = run_test_suite(
                    actor_path=tmp_ckpt,
                    config_path=ROOT / "config.yaml",
                    years=years,
                    plot=True,
                    save_trades=True,
                    verbose=True,
                )

                # 再跑 EV-greedy（逐年呼叫 run_test_once(policy="ev_greedy")）
                with open(ROOT / "config.yaml", "r", encoding="utf-8") as _f:
                    _cfg_for_test = yaml.safe_load(_f)

                results_ev = {}
                for y in years:
                    data_path = _resolve_test_path(ROOT, _cfg_for_test, y)
                    if not data_path.exists():
                        print(f"[WARN] EV-greedy 測試找不到 {y} 的檔案：{data_path}")
                        continue
                    try:
                        tr, mdd, _df_perf, df_base, fig, _actions = run_test_once(  # <<< 多接一個 _actions
                            actor_path=str(tmp_ckpt),
                            data_path=str(data_path),
                            config_path=str(ROOT / "config.yaml"),
                            plot=True, save_trades=True,
                            tag=f"{y}_EV",
                            verbose=True,
                            return_fig=True,
                            policy="ev_greedy"
                        )
                        results_ev[y] = {
                            "total_return": tr,
                            "max_drawdown": mdd,
                            "fig": fig,
                            "trades": None,
                        }
                    except Exception as e:
                        print(f"[WARN] EV-greedy 測試 {y} 失敗：{e}")

                if len(results_arg) == 0 and len(results_ev) == 0:
                    print("[WARN] run_test_suite / EV-greedy 都沒有任何年份成功（多半是找不到測試檔）。不上傳圖。")
                else:
                    log_dict = {}
                    panel_imgs = []

                    # Argmax：數值 + 單年圖
                    for y in years:
                        if y not in results_arg:
                            continue
                        r = results_arg[y]
                        # 只留面板，不上傳單項數值與單張圖
                        if r["fig"] is not None:
                            img = wandb.Image(r["fig"], caption=f"Argmax {y}")
                            panel_imgs.append(img)
                            plt.close(r["fig"])

                    # EV-greedy：數值 + 單年圖
                    panel_imgs_ev = []
                    for y in years:
                        if y not in results_ev:
                            continue
                        r = results_ev[y]
                        # 只留面板，不上傳單項數值與單張圖
                        if r["fig"] is not None:
                            img = wandb.Image(r["fig"], caption=f"EV-greedy {y}")
                            panel_imgs_ev.append(img)
                            plt.close(r["fig"])

                    # 面板：分開各一組，也可合併總覽
                    if panel_imgs:
                        log_dict[f"test_argmax/panel/slot"] = panel_imgs
                    if panel_imgs_ev:
                        log_dict[f"test_ev/panel/slot"] = panel_imgs_ev
                    if panel_imgs and panel_imgs_ev:
                        log_dict["test/panel_both"] = panel_imgs + panel_imgs_ev

                    # 保留最近 max_ckpts 次 test 結果（本地用）
                    if upload_wandb:
                        recent = {}
                        recent.update({k:v for k,v in log_dict.items() if isinstance(v, (int, float))})
                        recent_test_logs.append(recent)

                    # W&B 一次 log
                    wandb.log(log_dict, step=total_ep)

            # RAM記憶體檢查用
            rss = proc.memory_info().rss / 1024**3  # 常駐記憶體 (GB)
            vms = proc.memory_info().vms / 1024**3  # 虛擬記憶體 (GB)
            print(f"[MEM] ep={ep} | RSS={rss:.2f} GB | VMS={vms:.2f} GB")

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
