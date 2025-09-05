# src/rl/train/test.py
import sys, importlib
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import datetime
import time
import torch
import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from tqdm import trange

# === 設定專案路徑 ===
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]          # .../RL_Trading/src
ROOT = HERE.parents[3]             # .../RL_Trading
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# === 匯入模組 ===
from rl.models.random_agent import RandomAgent
from rl.models.dqn_agent import DQNAgent
from rl.models.ppo_agent import PPOAgent
from rl.models.a2c_agent import A2CAgent

from rl.env.StockTradingEnv import StockTradingEnv
from rl.train.logger import RunLogger


def split_infos(infos):
    """把 dict of arrays 轉成 list of dicts"""
    if isinstance(infos, dict) and isinstance(list(infos.values())[0], (np.ndarray, list)):
        num_envs = len(next(iter(infos.values())))
        return [
            {k: (v[i] if isinstance(v, (np.ndarray, list)) else v) for k, v in infos.items()}
            for i in range(num_envs)
        ]
    elif isinstance(infos, dict):
        return [infos]
    else:
        return infos


if __name__ == "__main__":

    episode_entropy = []

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

    # 存一份 config 到 run 資料夾
    with open(outdir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # === 載入資料 ===
    data_file = config["data"].get("file", "training_data_20.parquet")
    data_path = ROOT / "data" / "processed" / data_file
    if data_file.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["date"])

    ids = sorted(df["stock_id"].unique())  # 全部股票
    max_holdings = config["environment"].get("max_holdings", None)

    # === 建立環境 ===
    if model_name == "ppo":
        num_envs = config["ppo"].get("num_envs", 4)
        use_subproc = config["ppo"].get("use_subproc", True)

        def make_env():
            return StockTradingEnv(
                df=df, stock_ids=ids, lookback=lookback,
                initial_cash=init_cash, reward_mode=reward_mode,
                action_mode=action_mode, max_holdings=max_holdings
            )

        if use_subproc:
            env = AsyncVectorEnv([make_env for _ in range(num_envs)])
            print(f"[INFO] PPO 使用 SubprocVectorEnv 並行 {num_envs} 個環境 (多核 CPU)")
        else:
            env = AsyncVectorEnv([make_env for _ in range(num_envs)])
            print(f"[INFO] PPO 使用 SyncVectorEnv 並行 {num_envs} 個環境 (單核)")

    else:
        env = StockTradingEnv(
            df=df, stock_ids=ids, lookback=lookback,
            initial_cash=init_cash, reward_mode=reward_mode,
            action_mode=action_mode, max_holdings=max_holdings
        )

    # === 初始化 agent ===
    obs_dim = env.single_observation_space.shape[0]    #為甚麼 維度討論

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
        action_dim = int(np.prod(env.action_space.nvec))
    elif isinstance(env.action_space, gym.spaces.Box):
        # Box 代表連續動作空間，用於 PPO 連續控制
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(f"Unsupported action_space type: {type(env.action_space)}")



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

    # === Logging ===
    all_rewards, summary = [], []
    trade_sample_freq = config["logging"].get("trade_sample_freq", 10)
    logger = RunLogger(outdir, trade_sample_freq)

    update_every = config["training"].get("update_every", 1)
    warmup_steps = config["training"].get("warmup_steps", 0)
    grad_steps   = config["training"].get("grad_steps", 1)
    global_step  = 0

    progress_bar = trange(1, n_episodes + 1, desc="Training", unit="episode")

    # === PPO 訓練 (多環境並行) ===
    if model_name == "ppo":
        for ep in progress_bar:
            obs, infos = env.reset()
            daily_returns = []

            # 收集 n_steps rollout
            for t in range(agent.n_steps):
                actions, log_probs, values = agent.select_action(obs)
                next_obs, rewards, dones, truncs, infos = env.step(actions)

                # === 新增：統一轉成 list[dict] ===
                infos_list = split_infos(infos)

                for i in range(len(infos_list)):
                    agent.store_transition(
                        obs[i], actions[i], rewards[i], dones[i],
                        log_probs[i], values[i]
                    )
                    logger.log_step(ep, infos_list[i])

                obs = next_obs
                daily_returns.extend(rewards.tolist())

            # 更新
            agent.update()
            """
            # 在 episode 結束時，直接取 agent.entropy_log 的最新值
            if len(agent.entropy_log) > 0:
                episode_entropy.append(agent.entropy_log[-1])
            
            ep_return = np.mean(daily_returns)
            final_V = np.mean([info.get("V", init_cash) for info in infos_list])
            ret_pct = (final_V - init_cash) / init_cash * 100

            all_rewards.append(ep_return)
            summary.append({"episode": ep, "reward": ep_return, "return_pct": ret_pct})
            """
            # 在 episode 結束時，直接取 agent.entropy_log 的最新值
            if len(agent.entropy_log) > 0:
                episode_entropy.append(agent.entropy_log[-1])

            # final_V = episode 最後的平均資產
            final_V = np.mean([info.get("V", init_cash) for info in infos_list])

            # 用最後資產 - 初始資金 = 總損益
            ep_return = final_V - init_cash
            ret_pct = (final_V - init_cash) / init_cash * 100

            all_rewards.append(ep_return)
            summary.append({"episode": ep, "reward": ep_return, "return_pct": ret_pct})


            if ep % save_freq == 0:
                fig, ax1 = plt.subplots(figsize=(8, 4))

                # 左邊 y 軸：Reward
                ax1.plot(range(1, len(all_rewards)+1), all_rewards, color="tab:blue", label="Reward")
                ax1.set_xlabel("Episode")
                ax1.set_ylabel("Total Reward", color="tab:blue")
                ax1.tick_params(axis="y", labelcolor="tab:blue")

                # 右邊 y 軸：Entropy
                if len(episode_entropy) > 0:
                    ax2 = ax1.twinx()
                    ax2.plot(range(1, len(episode_entropy)+1), episode_entropy, color="tab:orange", label="Entropy")
                    ax2.set_ylabel("Entropy", color="tab:orange")
                    ax2.tick_params(axis="y", labelcolor="tab:orange")

                # 標題 & 格式
                fig.suptitle(f"Training Progress ({model_name})")
                fig.tight_layout()
                plt.grid(True)
                plt.savefig(outdir / "reward_entropy_curve.png")
                plt.close()


    # === DQN / Random 訓練 (單環境) ===
    else:
        for ep in progress_bar:
            obs, info = env.reset()
            daily_returns = []

            while True:
                action = agent.select_action(obs)
                next_obs, r, done, trunc, info = env.step(action)

                if model_name == "dqn":
                    agent.store_transition(obs, action, r, next_obs, done)
                    if global_step >= warmup_steps and (global_step % update_every == 0):
                        for _ in range(grad_steps):
                            agent.update()

                global_step += 1
                obs = next_obs
                daily_returns.append(r)
                logger.log_step(ep, info)

                if done or trunc:
                    break

            ep_return = 1.0
            for dr in daily_returns:
                ep_return *= (1.0 + dr)
            ep_return -= 1.0

            final_V = info["V"]
            ret_pct = (final_V - init_cash) / init_cash * 100

            all_rewards.append(ep_return)
            summary.append({"episode": ep, "reward": ep_return, "return_pct": ret_pct})

            if ep % save_freq == 0:
                plt.figure(figsize=(8, 4))
                plt.plot(range(1, len(all_rewards)+1), all_rewards)
                plt.xlabel("Episode")
                plt.ylabel("Total Reward")
                plt.title(f"Training Progress ({model_name})")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(outdir / "reward_curve.png")
                plt.close()

    # === 儲存模型 ===
    if model_name == "dqn":
        torch.save(agent.q_network.state_dict(), outdir / "dqn_model.pt")
        torch.save(agent.target_network.state_dict(), outdir / "dqn_target.pt")
    elif model_name == "ppo":
        torch.save(agent.actor.state_dict(), outdir / "ppo_actor.pt")
        torch.save(agent.critic.state_dict(), outdir / "ppo_critic.pt")

    # === 儲存訓練紀錄 ===
    if config["logging"]["save_summary"]:
        pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False)

    print(f"✅ Training finished. Model={model_name}. Results saved in: {outdir}")
