# src/rl/train/test.py
import os
# 建議：MPS 上 Dirichlet 會缺 op，先開 fallback，避免漏掉時直接崩潰
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import importlib
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import datetime
import torch
import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from tqdm import trange

# === 設定專案路徑 ===
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]          # .../rl-trading/src
ROOT = HERE.parents[3]             # .../rl-trading
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# === 匯入模組 ===
from rl.models.random_agent import RandomAgent
from rl.models.dqn_agent import DQNAgent
from rl.models.old_ppo_agent import PPOAgent
from rl.models.a2c_agent import A2CAgent

from rl.env.OLD_StockTradingEnv import StockTradingEnv
from rl.train.logger import RunLogger


def split_infos(infos):
    """把 dict of arrays 轉成 list of dicts（給 vector env 用）"""
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

    # === 讀取 config.yaml ===
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    n_episodes = config["training"]["n_episodes"]
    save_freq  = config["training"]["save_freq"]
    model_name = config["training"].get("model", "random").lower()

    init_cash   = config["environment"]["initial_cash"]
    lookback    = config["environment"]["lookback"]
    reward_mode = str(config["environment"]["reward_mode"]).lower().strip()
    action_mode = str(config["environment"]["action_mode"]).lower().strip()

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
                df=df,
                stock_ids=ids,
                lookback=lookback,
                initial_cash=init_cash,
                reward_mode=reward_mode,
                action_mode=action_mode,
                max_holdings=max_holdings,
            )

        if use_subproc:
            env = AsyncVectorEnv([make_env for _ in range(num_envs)])
            print(f"[INFO] PPO 使用 SubprocVectorEnv 並行 {num_envs} 個環境 (多核 CPU)")
        else:
            env = SyncVectorEnv([make_env for _ in range(num_envs)])
            print(f"[INFO] PPO 使用 SyncVectorEnv 並行 {num_envs} 個環境 (單核)")
    else:
        env = StockTradingEnv(
            df=df,
            stock_ids=ids,
            lookback=lookback,
            initial_cash=init_cash,
            reward_mode=reward_mode,
            action_mode=action_mode,
            max_holdings=max_holdings,
        )

    # === 初始化 agent ===
    # 使用 single_*（重點：VectorEnv 不能用 env.action_space 直接取維度）
    if hasattr(env, "single_observation_space"):
        single_os = env.single_observation_space
        single_as = env.single_action_space
    else:
        single_os = env.observation_space
        single_as = env.action_space

    obs_dim = int(np.prod(single_os.shape))

    if isinstance(single_as, gym.spaces.Box):
        action_dim = int(np.prod(single_as.shape))  # portfolio: N+1, weights: N
    elif isinstance(single_as, gym.spaces.MultiDiscrete):
        action_dim = int(np.prod(single_as.nvec))
    elif isinstance(single_as, gym.spaces.Discrete):
        action_dim = single_as.n
    else:
        raise ValueError(f"Unsupported action_space type: {type(single_as)}")

    if model_name == "random":
        agent = RandomAgent(single_as)
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

                # 保障：動作 shape 必為 (num_envs, action_dim)
                actions = np.asarray(actions, dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions[None, :]
                assert actions.shape[1] == action_dim, \
                    f"[shape mismatch] got {actions.shape}, expected (*, {action_dim})"
                # 也檢查 batch 維
                if actions.shape[0] != env.num_envs:
                    # 有些 agent 會回 (action_dim,)；這裡複製到所有 env
                    if actions.shape == (1, action_dim):
                        actions = np.repeat(actions, env.num_envs, axis=0)
                #print(f"[DEBUG] step actions.shape={actions.shape}")  # 期望 (num_envs, action_dim)

                next_obs, rewards, dones, truncs, infos = env.step(actions)

                # 統一 infos 結構
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

            # 紀錄本回合 entropy
            if len(agent.entropy_log) > 0:
                episode_entropy.append(agent.entropy_log[-1])

            final_V = np.mean([info.get("V", init_cash) for info in infos_list])

            # episode 的總報酬率
            total_return = (final_V - init_cash) / init_cash

            # 年化（假設一年 252 個交易日）
            days = len(daily_returns) if len(daily_returns) > 0 else 1
            annualized_return = (1 + total_return) ** (252 / days) - 1

            ep_return = annualized_return * 100
            all_rewards.append(ep_return)
            summary.append({"episode": ep, "annualized_return_pct": ep_return})

            if ep % save_freq == 0:
                fig, ax1 = plt.subplots(figsize=(8, 4))

                # 左軸：Annualized Return
                x1 = range(1, len(all_rewards) + 1)
                ln1, = ax1.plot(
                    x1, all_rewards, label="Annualized Return (%)",
                    color="#1f77b4", linewidth=2
                )
                ax1.set_xlabel("Episode")
                ax1.set_ylabel("Annualized Return (%)", color=ln1.get_color())
                ax1.tick_params(axis="y", colors=ln1.get_color())
                ax1.grid(True, axis="both", alpha=0.3)

                lines = [ln1]
                labels = [ln1.get_label()]

                # 右軸：Entropy
                if len(episode_entropy) > 0:
                    ax2 = ax1.twinx()
                    x2 = range(1, len(episode_entropy) + 1)
                    ln2, = ax2.plot(
                        x2, episode_entropy, label="Entropy",
                        color="#ff7f0e", linewidth=2, linestyle="--"
                    )
                    ax2.set_ylabel("Entropy", color=ln2.get_color())
                    ax2.tick_params(axis="y", colors=ln2.get_color())
                    lines.append(ln2)
                    labels.append(ln2.get_label())

                # 合併 legend
                ax1.legend(lines, labels, loc="best")

                fig.suptitle(f"Training Progress ({model_name})")
                fig.tight_layout()
                fig.savefig(outdir / "reward_entropy_curve.png")
                plt.close(fig)


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

            # 將日報酬累乘成總報酬
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
