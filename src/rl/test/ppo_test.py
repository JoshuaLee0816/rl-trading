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

# === 路徑設定 ===
ACTOR_PATH = "logs/runs/run_20250911_213315/checkpoint_ep5900.pt"         # 訓練好的 PPO checkpoint
DATA_PATH = "data/processed/full/walk_forward/WF_test_2021_full.parquet"  # 測試資料
CONFIG_PATH = ROOT / "config.yaml"

# === 載入 config.yaml ===
with open(CONFIG_PATH, "r") as f:
    full_cfg = yaml.safe_load(f)

env_cfg = full_cfg["environment"]
ppo_cfg = full_cfg["ppo"]
feature_cols = full_cfg["data"]["features"]

# === 載入測試資料，過濾 feature ===
df = pd.read_parquet(DATA_PATH)
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
print(f"[INFO] Obs dim = {obs_dim}, N={len(ids)}, F={len(feature_cols)}, K={env_cfg['lookback']}")

# === 建立 PPO Agent ===
agent = PPOAgent(obs_dim, len(ids), env.QMAX, ppo_cfg)

# === 載入已訓練好的 actor 參數 ===
ckpt = torch.load(ACTOR_PATH, map_location=agent.device)
if "actor" in ckpt:  # checkpoint 存成 dict
    agent.actor.load_state_dict(ckpt["actor"])
else:                # checkpoint 是純 actor state_dict
    agent.actor.load_state_dict(ckpt)
agent.actor.eval()

# === 測試 loop ===
dates, values, actions = [], [], []
terminated = False

while not terminated:
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        mask_flat = agent.flatten_mask(info["action_mask_3d"]).unsqueeze(0)

        logits = agent.actor(obs_t)  # (1, A)
        masked_logits = logits.masked_fill(~mask_flat, -1e9)

        # deterministic：直接取最大機率的動作
        a_flat = torch.argmax(masked_logits, dim=-1).item()
        action_tuple = agent.flat_to_tuple(a_flat)

    obs, reward, terminated, _, info = env.step(action_tuple)

    dates.append(info["date"])
    values.append(info["V"])
    actions.append((
        info["date"], info["side"], info["stock_id"],
        info["lots"], info["cash"], info["V"]
    ))

# === 績效分析 ===
df_perf = pd.DataFrame({"date": dates, "value": values})
df_perf["date"] = pd.to_datetime(df_perf["date"])
df_perf.set_index("date", inplace=True)

# 總報酬率
total_return = df_perf["value"].iloc[-1] / df_perf["value"].iloc[0] - 1

# 最大回撤
roll_max = df_perf["value"].cummax()
drawdown = df_perf["value"] / roll_max - 1
max_drawdown = drawdown.min()

print(f"Total Return: {total_return:.2%}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# === baseline 計算 ===
baseline_value = (env.baseline_close / env.baseline_close[env.K]) * env.initial_cash
df_baseline = pd.DataFrame({"date": env.dates[env.K:], "baseline": baseline_value[env.K:]})
df_baseline["date"] = pd.to_datetime(df_baseline["date"])
df_baseline.set_index("date", inplace=True)

# === 繪圖 ===
plt.figure(figsize=(10, 6))
plt.plot(df_perf.index, df_perf["value"], label="Agent Portfolio")
plt.plot(df_baseline.index, df_baseline["baseline"], label="Baseline (0050)", linestyle="--")
plt.title("Portfolio Value Over Time (2020)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# === 輸出交易紀錄 ===
df_trades = pd.DataFrame(actions, columns=["date", "side", "stock_id", "lots", "cash", "value"])
df_trades.to_csv("src/rl/test/testing_output/trades_2020.csv", index=False)
print("交易紀錄已存 trades_2020.csv")
