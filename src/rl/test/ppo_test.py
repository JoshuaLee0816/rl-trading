import torch
import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === 專案路徑 ===
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]
ROOT = HERE.parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl.env.StockTradingEnv import StockTradingEnv
from rl.models.ppo_agent import PPOAgent


def run_test_once(actor_path, data_path, config_path,
                  plot=True, save_trades=False, tag="2020", verbose=True):

    # === 載入 config.yaml ===
    with open(config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    env_cfg = full_cfg["environment"]
    feature_cols = full_cfg["data"]["features"]

    # === 載入測試資料 ===
    df = pd.read_parquet(data_path) if str(data_path).endswith(".parquet") else pd.read_csv(data_path)
    keep_cols = ["date", "stock_id"] + feature_cols
    df = df[keep_cols]
    ids = sorted(df["stock_id"].unique())

    # === 初始化環境 ===
    env = StockTradingEnv(
        df=df,
        stock_ids=ids,
        lookback=env_cfg["lookback"],
        initial_cash=env_cfg["initial_cash"],
        max_holdings=env_cfg["max_holdings"],
        qmax_per_trade=env_cfg["qmax_per_trade"],
    )

    obs, info = env.reset()
    obs_dim = env.obs_dim

    # === 建立 PPO Agent ===
    agent = PPOAgent(obs_dim, len(ids), env.QMAX, full_cfg)

    # === 載入已訓練好的 actor 參數 ===
    if full_cfg["training"].get("load_checkpoint", True) and actor_path is not None and os.path.exists(actor_path):
        try:
            ckpt = torch.load(actor_path, map_location=agent.device)
            if "actor" in ckpt:
                agent.actor.load_state_dict(ckpt["actor"])
            else:
                agent.actor.load_state_dict(ckpt)
            agent.actor.eval()
            print(f"[INFO] Loaded checkpoint from {actor_path}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}. Using random init.")
    else:
        print("[INFO] No checkpoint found or load_checkpoint=False. Using random init.")

    # === 測試 loop ===
    dates, values, actions = [], [], []
    terminated = False

    while not terminated:
        with torch.no_grad():
            obs_t = agent.obs_to_tensor(obs).unsqueeze(0)  # [1,obs_dim]
            mask_flat = agent.flatten_mask(info["action_mask_3d"]).unsqueeze(0)
            logits = agent.actor(obs_t)
            masked_logits = logits.masked_fill(~mask_flat, -1e9)
            a_flat = torch.argmax(masked_logits, dim=-1).item()
            action_tuple = agent.flat_to_tuple(a_flat)

        obs, reward, terminated, _, info = env.step(action_tuple)

        dates.append(info["date"])
        values.append(info["V"])
        actions.append((info["date"], info["side"], info["stock_id"], info["lots"], info["cash"], info["V"]))

    # === 績效分析 ===
    df_perf = pd.DataFrame({"date": dates, "value": values})
    df_perf["date"] = pd.to_datetime(df_perf["date"])
    df_perf.set_index("date", inplace=True)

    total_return = df_perf["value"].iloc[-1] / df_perf["value"].iloc[0] - 1
    roll_max = df_perf["value"].cummax()
    drawdown = df_perf["value"] / roll_max - 1
    max_drawdown = drawdown.min()

    if verbose:
        print(f"[TEST-{tag}] Total Return: {total_return:.2%}, Max Drawdown: {max_drawdown:.2%}")

    # === baseline 計算 ===
    baseline_value = (env.baseline_close / env.baseline_close[env.K]) * env.initial_cash
    df_baseline = pd.DataFrame({"date": env.dates[env.K:], "baseline": baseline_value[env.K:].cpu().numpy()})
    df_baseline["date"] = pd.to_datetime(df_baseline["date"])
    df_baseline.set_index("date", inplace=True)

    # === 繪圖 ===
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df_perf.index, df_perf["value"], label="Agent Portfolio")
        plt.plot(df_baseline.index, df_baseline["baseline"], label="Baseline (0050)", linestyle="--")
        plt.title(f"Portfolio Value Over Time ({tag})")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    # === 輸出交易紀錄 ===
    if save_trades:
        out_path = Path("src/rl/test/testing_output") / f"trades_{tag}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_trades = pd.DataFrame(actions, columns=["date", "side", "stock_id", "lots", "cash", "value"])
        df_trades.to_csv(out_path, index=False)

    return total_return, max_drawdown, df_perf, df_baseline


# === 主程式（獨立跑測試用） ===
if __name__ == "__main__":
    run_dirs = sorted((ROOT / "logs" / "runs").glob("run_*"))
    latest_run = run_dirs[-1]
    ACTOR_PATH = latest_run / "ppo_actor.pt"

    CONFIG_PATH = ROOT / "config.yaml"

    # 從 config.yaml 讀取測試資料路徑
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    DATA_PATH = ROOT / "data" / "processed" / cfg["data"]["test_file"]

    print(f"[INFO] Using actor checkpoint: {ACTOR_PATH}")
    print(f"[INFO] Using test dataset: {DATA_PATH}")

    run_test_once(
        actor_path=ACTOR_PATH,
        data_path=DATA_PATH,
        config_path=CONFIG_PATH,
        plot=True,
        save_trades=True,
        tag="2020",
        verbose=True,
    )
