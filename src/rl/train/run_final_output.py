import torch
import pandas as pd
import numpy as np
from src.rl.env.StockTradingEnv import StockTradingEnv
from src.rl.models.ppo_agent import Actor

# ========== Helper ==========
def flatten_mask(mask3, device="cpu"):
    """
    把 3D mask (3, N, QMAX+1) 攤平成 1D (A,)
    """
    if isinstance(mask3, np.ndarray):
        mask3 = torch.from_numpy(mask3)
    mask3 = mask3.to(device)

    buy  = mask3[0, :, 1:]                # (N, QMAX)
    sell = mask3[1, :, :1]                # (N, 1)
    hold = mask3[2:3, :1, :1].reshape(1)  # (1,)
    flat = torch.cat([buy.reshape(-1), sell.reshape(-1), hold], dim=0).bool()
    return flat

def flat_to_tuple(a_flat: int, N: int, QMAX: int):
    """
    將攤平類別還原為 (op, idx, q)
      op: 0=BUY, 1=SELL_ALL, 2=HOLD
    """
    A_buy = N * QMAX
    if a_flat < A_buy:
        rel = int(a_flat)
        idx = rel // QMAX
        q   = (rel % QMAX) + 1
        return (0, idx, q)
    elif a_flat < A_buy + N:
        idx = int(a_flat - A_buy)
        return (1, idx, 0)
    else:
        return (2, 0, 0)


# ========== 1. 載入測試資料 ==========
df = pd.read_parquet("data/processed/training_data_20.parquet")   # ⚠️ 這裡要準備好新的測試資料
stock_ids = sorted(df["stock_id"].unique())

env = StockTradingEnv(
    df=df,
    stock_ids=stock_ids,
    lookback=20,
    initial_cash=100000,
    max_holdings=5,
    qmax_per_trade=10,
    reward_mode="daily_return",
    action_mode="discrete"
)

# ========== 2. 載入 Actor ==========
obs_dim = env.observation_space.shape[0]
N = len(stock_ids)
QMAX = env.action_space.nvec[2] - 1   # 取 q 的上限

# ⚠️ hidden_dim 要跟訓練時一致
actor = Actor(obs_dim, N, QMAX, hidden_dim=64)
actor.load_state_dict(torch.load("logs/runs/run_20250909_122808/ppo_actor.pt", map_location="cpu"))
actor.eval()

# ========== 3. 跑一次完整 episode ==========
obs, info = env.reset()
done = False
total_reward = 0
trades = []

while not done:
    mask = info["action_mask_3d"]

    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = flatten_mask(mask, device="cpu").unsqueeze(0)
        logits = actor(obs_t)
        masked_logits = logits.masked_fill(~mask_t, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample().item()

    action_tuple =  flat_to_tuple(action, N, QMAX)
    obs, reward, done, _, info = env.step(action_tuple)
    total_reward += reward

    trades.append({
        "date": info["date"],
        "side": info["side"],
        "stock": info["stock_id"],
        "lots": info["lots"],
        "cash": info["cash"],
        "V": info["V"]
    })

# ========== 4. 輸出結果 ==========
trades_df = pd.DataFrame(trades)
trades_df.to_csv("logs/final_trades.csv", index=False)

# 年化報酬計算（用 log-return sum 換算）
days = len(trades) if len(trades) > 0 else 1
total_return = np.exp(total_reward) - 1
annualized_return = (1 + total_return) ** (252 / days) - 1

print(f"Final Portfolio Value: {info['V']}")
print(f"Total log-return sum: {total_reward:.6f}")
print(f"Annualized Return: {annualized_return*100:.2f}%")
print("交易紀錄已存到 logs/final_trades.csv")
