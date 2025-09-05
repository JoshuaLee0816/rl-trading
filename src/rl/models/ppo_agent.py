# src/rl/models/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import platform
import torch.nn.functional as F #處理Actor forward過小問題 （機率）

class Actor (nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

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
        return self.net(x)             # 直接輸出 value，不要 softmax 這很重要 他是value?

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
        self.entropy_log = []

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
        """obs 可以是 (n_envs, obs_dim) 或 (obs_dim,)"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.dim() == 1:  # 單環境 → 保持 batch 維度
            obs_tensor = obs_tensor.unsqueeze(0)

        probs = self.actor(obs_tensor)  # [batch, action_dim]
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()         # flat action (0 ~ N*5-1)
        log_probs = dist.log_prob(actions)
        values = self.critic(obs_tensor).squeeze()

        # === 將 flat action 還原成 (idx, lvl) ===
        idx = (actions // 5).cpu().numpy()
        lvl = (actions % 5).cpu().numpy()
        combined_actions = np.stack([idx, lvl], axis=-1)  # shape [batch, 2]

        return (
            combined_actions,                   # 給 env 用 (idx, lvl)
            log_probs.detach().cpu().numpy(),   # 存 buffer
            values.detach().cpu().numpy(),
        )


    def store_transition(self, obs, action, reward, done, log_prob, value):
        """
        action: (idx, lvl) → 還原成 flat action 存起來
        """
        
        if isinstance(action, (list, np.ndarray)) and len(action) == 2:
            idx, lvl = action
            flat_action = idx * 5 + lvl
        else:
            flat_action = action

        self.buffer.add(obs, flat_action, reward, done, log_prob, value)


    def update(self):
        

        obs, actions, rewards, dones, old_log_probs, values = self.buffer.get()
        self.buffer.clear()
        """
        print("Buffer obs shape:", obs.shape)
        print("Buffer actions sample:", actions[:10])
        print("Buffer rewards sample:", rewards[:10])
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)

        returns, advantages = self._compute_gae(rewards, dones, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        #print("Advantage stats:", advantages.min().item(), advantages.max().item(), advantages.mean().item())

        dataset_size = len(obs)
        entropies = []   # 收集這次 update 所有 batch 的 entropy

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

                probs = self.actor(batch_obs)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # === Debug print ===
                #print("batch_actions unique:", batch_actions.unique())
                #print("probs[0]:", probs[0].detach().cpu().numpy())
                #print("new_log_probs sample:", new_log_probs[:5].detach().cpu().numpy())

                #print("Entropy batch mean:", entropy.item())
                #print("Actor probs sample:", probs[0].detach().cpu().numpy()[:10])

                entropies.append(entropy.item())

                # ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_adv

                # actor loss (加上 entropy)
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # critic loss
                values_pred = self.critic(batch_obs).squeeze()
                critic_loss = (batch_returns - values_pred).pow(2).mean()

                # --- 更新 Actor ---
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # --- 更新 Critic ---
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                




        if entropies:  # entropies 是這次 update 收集的 batch 熵
            avg_entropy = float(np.mean(entropies))
            self.entropy_log.append(avg_entropy)

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