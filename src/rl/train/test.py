# src/rl/train/test.py
import sys, importlib
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import datetime
import time
from tqdm import trange

# === 設定專案路徑 ===
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]          # .../RL_Trading/src
ROOT = HERE.parents[3]             # .../RL_Trading
# 確保 src 是 sys.path[0]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# === 匯入模組 ===
from rl.models.random_agent import RandomAgent
from rl.models.dqn_agent import DQNAgent
from rl.models.ppo_agent import PPOAgent
from rl.models.a2c_agent import A2CAgent

from rl.env.StockTradingEnv import StockTradingEnv
from rl.train.logger import RunLogger



if __name__ == "__main__":
    # 讀取 config.yaml
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    n_episodes = config["training"]["n_episodes"]
    save_freq  = config["training"]["save_freq"]
    model_name = config["training"].get("model", "random").lower()

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

    #從config載入訓練資料
    data_file = config["data"].get("file", "training_data.csv")
    csv_path = ROOT / "data" / "processed" / data_file
    df = pd.read_csv(csv_path, parse_dates=["date"])

    ids = sorted(df["stock_id"].unique())[:20]

    # 建立環境
    max_holdings = config["environment"].get("max_holdings", None)

    env = StockTradingEnv(
        df=df, stock_ids=ids, lookback=lookback,
        initial_cash=init_cash, reward_mode=reward_mode,
        action_mode=action_mode,max_holdings=max_holdings
    )

    # 初始化 agent
    obs_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, "n"):
        action_dim = env.action_space.n
    elif hasattr(env.action_space, "nvec"):
        action_dim = int(env.action_space.nvec.prod())
    else:
        raise ValueError("Unsupported action_space type")

    if model_name == "random":
        agent = RandomAgent(env.action_space)
    elif model_name == "dqn":
        agent = DQNAgent(obs_dim, action_dim, config["dqn"])
    elif model_name == "ppo":
        agent = PPOAgent(obs_dim, action_dim, config.get("ppo", {}))
    elif model_name == "a2c":
        agent = A2CAgent(obs_dim, action_dim, config.get("a2c", {}))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 訓練迴圈
    all_rewards = []
    summary = []
    trade_sample_freq = config["logging"].get("trade_sample_freq", 10)
    logger = RunLogger(outdir, trade_sample_freq)

    progress_bar = trange(1, n_episodes + 1, desc="Training", unit="episode")

    for ep in progress_bar:
        obs, info = env.reset()
        #ep_reward = 0.0
        daily_returns = []   #先存每天的 return

        while True:
            # 用 agent 選擇動作
            action = agent.select_action(obs)
            next_obs, r, done, trunc, info = env.step(action)

            # 只在非 random 模型時存經驗 & 更新
            if model_name != "random":
                agent.store_transition(obs, action, r, next_obs, done)
                agent.update()

            obs = next_obs
            #ep_reward += r
            daily_returns.append(r)
            logger.log_step(ep, info)

            if done or trunc:
                break
        
        # episode 結束後，算累積報酬率
        ep_return = 1.0
        for dr in daily_returns:
            ep_return *= (1.0 + dr)
        ep_return -= 1.0   # 轉成百分比形式

        final_V = info["V"]
        ret_pct = (final_V - init_cash) / init_cash * 100
        all_rewards.append(ep_return)
        summary.append({"episode": ep, "reward": ep_return, "return_pct": ret_pct})

        # 更新 tqdm 狀態
        progress_bar.set_postfix({
            #"Reward": f"{ep_return:.2f}",
            #"Return%": f"{ret_pct:.2f}"   
            #自己決定要不要顯示(還在斟酌)
        })

        if ep % save_freq == 0:
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(all_rewards)+1), all_rewards, marker="o")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title(f"Training Progress ({model_name})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(outdir / "reward_curve.png")
            plt.close()
        
        #  強制休息機制
        cool_every = config.get("safety", {}).get("cool_down_every", None)
        cool_secs = config.get("safety", {}).get("sleep_seconds", 0)
        if cool_every and ep % cool_every == 0:
            print("休息一下")
            time.sleep(cool_secs)

    if config["logging"]["save_summary"]:
        pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False)

    print(f"✅ Training finished. Model={model_name}. Results saved in: {outdir}")
