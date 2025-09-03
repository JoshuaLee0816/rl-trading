# src/rl/train/test.py
import sys
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import datetime

HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]          # .../RL_Trading/src
ROOT = HERE.parents[3]             # .../RL_Trading
sys.path.append(str(SRC_DIR))

from rl.env.StockTradingEnv import StockTradingEnv
from rl.train.logger import RunLogger   # 若你放同檔，改成: from logger import RunLogger

if __name__ == "__main__":
    # 讀取 config.yaml
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    n_episodes = config["training"]["n_episodes"]
    save_freq  = config["training"]["save_freq"]

    init_cash  = config["environment"]["initial_cash"]
    lookback   = config["environment"]["lookback"]
    reward_mode = config["environment"]["reward_mode"]
    action_mode = config["environment"]["action_mode"]

    outdir = ROOT / config["logging"]["outdir"]
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = outdir / f"run_{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    # 存一份 config 到 run 資料夾，方便追蹤
    with open(outdir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # 載入資料
    csv_path = ROOT / "data" / "processed" / "training_data.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"])
    ids = sorted(df["stock_id"].unique())[:20]

    # 建立環境
    env = StockTradingEnv(
        df=df, stock_ids=ids, lookback=lookback,
        initial_cash=init_cash, reward_mode=reward_mode,
        action_mode=action_mode
    )

    all_rewards = []
    summary = []
    logger = RunLogger(outdir)

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        ep_reward = 0.0

        while True:
            action = env.action_space.sample()  # baseline: 隨機動作
            obs, r, done, trunc, info = env.step(action)
            ep_reward += r

            logger.log_step(ep, info)

            if done or trunc:
                break

        final_V = info["V"]
        ret_pct = (final_V - init_cash) / init_cash * 100
        all_rewards.append(ep_reward)
        summary.append({"episode": ep, "reward": ep_reward, "return_pct": ret_pct})

        if ep % save_freq == 0:
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(all_rewards)+1), all_rewards, marker="o")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Training Progress")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(outdir / "reward_curve.png")
            plt.close()

    if config["logging"]["save_summary"]:
        pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False)

    print(f"✅ Training finished. Results saved in: {outdir}")
