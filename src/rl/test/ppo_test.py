import torch
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
    """
    單次測試 agent
    - actor_path: checkpoint 路徑
    - data_path: 測試資料 parquet/csv
    - config_path: config.yaml
    - plot: 是否畫資產曲線
    - save_trades: 是否存交易紀錄
    - tag: 測試標籤（例如年份）
    - verbose: 是否在 console 印結果
    """

    # === 載入 config.yaml ===
    with open(config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    env_cfg = full_cfg["environment"]
    ppo_cfg = full_cfg["ppo"]
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
    obs_dim = env.observation_space.shape[0]

    # === 建立 PPO Agent ===
    agent = PPOAgent(obs_dim, len(ids), env.QMAX, ppo_cfg)

    # === 載入已訓練好的 actor 參數 ===
    ckpt = torch.load(actor_path, map_location=agent.device)
    if "actor" in ckpt:
        agent.actor.load_state_dict(ckpt["actor"])
    else:
        agent.actor.load_state_dict(ckpt)
    agent.actor.eval()

    # === 測試 loop ===
    dates, values, actions = [], [], []
    terminated = False

    while not terminated:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
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
    df_baseline = pd.DataFrame({"date": env.dates[env.K:], "baseline": baseline_value[env.K:]})
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
        if verbose:
            print(f"交易紀錄已存 {out_path}")

    return total_return, max_drawdown, df_perf, df_baseline


# === 主程式（獨立跑測試用） ===
if __name__ == "__main__":
    run_dirs = sorted((ROOT / "logs" / "runs").glob("run_*"))
    latest_run = run_dirs[-1]
    ACTOR_PATH = latest_run / "ppo_actor.pt"

    print(ACTOR_PATH)

    DATA_PATH = "data/processed/full_300/walk_forward/WF_test_2020_full_300.parquet"
    CONFIG_PATH = ROOT / "config.yaml"

    run_test_once(ACTOR_PATH, DATA_PATH, CONFIG_PATH,
                  plot=True, save_trades=True, tag="2020", verbose=True)
