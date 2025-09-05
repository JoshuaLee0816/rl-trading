# src/rl/models/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import platform

class Actor (nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLu(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        return self.net(x)

class Critic (nn.Module):
    def __init__(self, obs_dim, hidden_dim = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)
    
class RolloutBuffer:
    """存放 n_steps * n_envs 的資料"""
    #RolloutBuffer = 短期收集一批 episode 的資料 → 算 advantage → 拿去更新 → 清空
    def __init__(self):
        self.clear()

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get(self):
        return (
            np.array(self.obs),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.dones),
            np.array(self.log_probs),
            np.array(self.values),
        )

    def clear(self):
        #理論上update結束就要clear buffer
        self.obs, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []

class PPOAgent:
    def __init__(self, obs_dim, action_dim, config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

        # === 超參數從 config.yaml ===
        self.gamma = config.get("gamma")
        self.lam = config.get("gae_lambda")
        self.clip_epsilon = config.get("clip_epsilon")
        self.batch_size = config.get("batch_size")
        self.n_steps = config.get("n_steps")
        self.epochs = config.get("epochs")
        self.entropy_coef = config.get("entropy_coef")
        self.value_coef = config.get("value_coef")

        # === 裝置選擇 ===
        device_cfg = config.get("device", "auto")
        if device_cfg == "cpu":
            self.device = torch.device("cpu")
        elif device_cfg == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device_cfg == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device_cfg == "auto":
            if platform.system() == "Darwin" and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        print(f"[INFO] Using device: {self.device}")

        # === 網路 ===
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        
        # === Optimizer === 更新actor and critic 的Network要不同的速率比較合理
        self.actor_lr = config["actor_lr"]
        self.critic_lr = config["critic_lr"]

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # rollout buffer
        self.buffer = RolloutBuffer()

    def select_action(self, obs):
        """obs 可以是 (n_envs, obs_dim) 或 (obs_dim,) [並行或單獨跑 輸入資料筆數不同]"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.dim() == 1: #單環境為了維度正確
            obs_tensor = obs_tensor.unsqueeze(0)
        
        probs = self.actor(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.critic(obs_tensor).squeeze()

        return (
            actions.cpu().numpy(),
            log_probs.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
        )

    def store_transition(self, obs, action, reward, done, log_prob, value):
        self.buffer.add(obs, action, reward, done, log_prob, value)

    def update(self):
        obs, actions, rewards, dones, old_log_probs, values = self.buffer.get()
        self.buffer.clear()

        # torch tensor
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)

        # 計算 GAE advantage
        returns, advantages = self._compute_gae(rewards, dones, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多次 epoch 更新
        dataset_size = len(obs)
        for _ in range(self.epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_returns = returns[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]

                # 計算新的 log_probs
                probs = self.actor(batch_obs)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # clipped surrogate
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic loss
                values_pred = self.critic(batch_obs).squeeze()
                critic_loss = (batch_returns - values_pred).pow(2).mean()

                # 總 loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _compute_gae(self, rewards, dones, values):
        returns, advs = [], []
        gae = 0
        next_value = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advs.insert(0, gae)
            ret = gae + values[t]
            returns.insert(0, ret)
            next_value = values[t]
        return (
            torch.tensor(returns, dtype=torch.float32, device=self.device),
            torch.tensor(advs, dtype=torch.float32, device=self.device),
        )