# src/rl/train/test.py
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]          # .../RL_Trading/src
ROOT = HERE.parents[3]             # .../RL_Trading
sys.path.append(str(SRC_DIR))

from rl.env.StockTradingEnv import StockTradingEnv
from rl.train.logger import RunLogger   # 若你放同檔，改成: from logger import RunLogger

if __name__ == "__main__":
    csv_path = ROOT / "data" / "processed" / "training_data.csv"
    outdir = ROOT / "logs" / "runs"     # 統一輸出資料夾

    df = pd.read_csv(csv_path, parse_dates=["date"])
    ids = sorted(df["stock_id"].unique())[:20]

    env = StockTradingEnv(
        df=df, stock_ids=ids, lookback=20,
        initial_cash=1_000_000, reward_mode="daily_return"
    )

    n_episodes = 3
    for ep in range(1, n_episodes + 1):
        logger = RunLogger(outdir)
        obs, info = env.reset()
        steps, ep_reward = 0, 0.0

        while True:
            action = env.action_space.sample()
            obs, r, done, trunc, info = env.step(action)
            ep_reward += r
            steps += 1

            # 記錄每步
            logger.log_step(ep, steps, r, info)

            if done or trunc:
                break

        final_V = info["V"]
        init_V = env.initial_cash
        ret_pct = (final_V - init_V) / init_V * 100

        # 存檔（tag 可以用 ep 編號或日期時間）
        eq_path, tr_path = logger.save_csv(tag=f"ep{ep:03d}")

        print(f"[Episode {ep}] steps={steps}, final_V={final_V}, return={ret_pct:+.2f}%")
        print(f"  ↳ equity curve: {eq_path}")
        print(f"  ↳ trades     : {tr_path}")
