import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import gymnasium as gym

# === 專案路徑 ===
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]
ROOT = HERE.parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# === 模組 ===
from rl.models.ppo_agent import PPOAgent
from rl.env.StockTradingEnv import StockTradingEnv


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


def run_evaluation(config, actor_path, critic_path, outdir):
    # === 載入資料 ===
    data_file = config["data"].get("file", "stocks_20_with_market_index_2015-2020_long.parquet")
    data_path = ROOT / "data" / "processed" / data_file
    if data_file.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["date"])

    ids = sorted(df["stock_id"].unique())
    num_stocks = len(ids)

    # === 建立環境 ===
    env = StockTradingEnv(
        df=df,
        stock_ids=ids,
        lookback=config["environment"]["lookback"],
        initial_cash=config["environment"]["initial_cash"],
        reward_mode=config["environment"]["reward_mode"],
        action_mode=config["environment"]["action_mode"],
        max_holdings=config["environment"].get("max_holdings", None),
        qmax_per_trade=int(config["environment"].get("qmax_per_trade", 1)),
    )

    # === 初始化 agent ===
    single_os = env.observation_space
    obs_dim = int(np.prod(single_os.shape))

    agent = PPOAgent(
        obs_dim=obs_dim,
        num_stocks=num_stocks,
        qmax_per_trade=int(config["environment"].get("qmax_per_trade", 1)),
        config=config.get("ppo", {}),
    )

    # 載入訓練好的權重
    agent.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    agent.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
    agent.actor.eval()
    agent.critic.eval()

    # === Evaluation Loop ===
    obs, infos = env.reset()
    portfolio_values = [infos.get("V", config["environment"]["initial_cash"])]
    daily_returns = []

    action_mask_batch = normalize_mask_batch(infos.get("action_mask_3d", None))

    done, truncated = False, False
    while not (done or truncated):
        # 保證 mask 維度正確
        mask_i = None
        if action_mask_batch is not None:
            mask_i = action_mask_batch
            if mask_i.ndim == 2:  # [N, M] → [1, N, M]
                mask_i = np.expand_dims(mask_i, axis=0)

        action_tuple, action_flat, logp, value = agent.select_action(
            obs, action_mask_3d=mask_i
        )

        obs, reward, done, truncated, infos = env.step(action_tuple)
        action_mask_batch = normalize_mask_batch(infos.get("action_mask_3d", None))

        portfolio_values.append(infos.get("V", portfolio_values[-1]))
        daily_returns.append(reward)

    env.close()

    # === 統計 ===
    portfolio_values = np.array(portfolio_values)
    daily_returns = np.array(daily_returns)

    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1

    print("==== Evaluation Result ====")
    print(f"Final Portfolio Value: {portfolio_values[-1]:,.2f}")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Annualized Return: {annualized_return*100:.2f}%")

    # === 畫圖 ===
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(portfolio_values, color="blue", linewidth=2)
    ax[0].axhline(portfolio_values[0], color="gray", linestyle="--", linewidth=1, label="Initial Capital")
    ax[0].set_ylabel("Portfolio Value")
    ax[0].set_title("Portfolio Value Curve")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(daily_returns * 100, color="green", linewidth=1)
    ax[1].set_ylabel("Daily Return (%)")
    ax[1].set_xlabel("Time Step")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "eval_result.png")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python eval_final.py <ppo_actor.pt 路徑>")
        sys.exit(1)

    actor_path = Path(sys.argv[1]).resolve()
    run_dir = actor_path.parent
    critic_path = run_dir / "ppo_critic.pt"

    # === 載入 config.yaml ===
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"[INFO] 使用模型: {actor_path}")
    print(f"[INFO] 輸出將存到: {run_dir}")

    run_evaluation(config, actor_path, critic_path, run_dir)
