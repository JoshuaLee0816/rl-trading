# src/rl/models/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import platform
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, top_k=20):
        super().__init__()
        self.top_k = top_k
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # action_dim = 股票數量
        )

    def forward(self, x):
        return self.net(x)  # raw scores (not probs)

# === Critic：輸出 state value ===
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)  # (B,1)


# === Rollout Buffer ===
class RolloutBuffer:
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
        self.obs, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []


# === PPO Agent ===
class PPOAgent:
    def __init__(self, obs_dim, action_dim, config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.entropy_log = []

        # === Hyperparams ===
        self.gamma = config.get("gamma")
        self.lam = config.get("gae_lambda")
        self.clip_epsilon = config.get("clip_epsilon")
        self.batch_size = config.get("batch_size")
        self.n_steps = config.get("n_steps")
        self.epochs = config.get("epochs")
        self.entropy_coef = config.get("entropy_coef")
        self.value_coef = config.get("value_coef")

        # === Device ===
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

        # === Networks ===
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # === Optimizers ===
        self.actor_lr = config["actor_lr"]
        self.critic_lr = config["critic_lr"]
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.buffer = RolloutBuffer()
        #("[DEBUG] env obs_dim:", obs_dim)
        #print("[DEBUG] actor first layer in_features:", self.actor.net[0].in_features)


    def select_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        single = False
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            single = True

        # Actor 輸出 raw scores
        scores = self.actor(obs_tensor)  # (B, action_dim)

        # Top-k 選股
        topk_vals, topk_idx = torch.topk(scores, k=self.actor.top_k, dim=-1)

        # 對 top-k 做 softmax
        topk_probs = torch.softmax(topk_vals, dim=-1)

        # 把 allocation 放回原來的 700 維空間
        allocations = torch.zeros_like(scores)
        allocations.scatter_(1, topk_idx, topk_probs)

        # log_prob（cross-entropy 形式）
        log_probs = (allocations * torch.log(allocations + 1e-8)).sum(dim=-1)

        # Critic value
        values = self.critic(obs_tensor).squeeze(-1)

        if single:
            return (
                allocations[0].detach().cpu().numpy(),
                log_probs[0].detach().cpu().numpy(),
                values[0].detach().cpu().numpy(),
            )
        else:
            return (
                allocations.detach().cpu().numpy(),
                log_probs.detach().cpu().numpy(),
                values.detach().cpu().numpy(),
            )


    # === 儲存 transition ===
    def store_transition(self, obs, action, reward, done, log_prob, value):
        self.buffer.add(obs, action, reward, done, log_prob, value)

    # === 更新 ===
    def update(self):
        # 取出 buffer 並清空
        obs, actions, rewards, dones, old_log_probs, values = self.buffer.get()
        self.buffer.clear()

        # 張量化
        device = self.device
        obs           = torch.tensor(obs, dtype=torch.float32, device=device)
        actions       = torch.tensor(actions, dtype=torch.float32, device=device)   # allocation 向量（full N 維，top-k 以外通常為 0）
        rewards       = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones         = torch.tensor(dones, dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        values        = torch.tensor(values, dtype=torch.float32, device=device)

        # GAE
        returns, advantages = self._compute_gae(rewards, dones, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(obs)
        entropies = []

        # 取得 top-k（與 select_action 保持一致）
        top_k = getattr(self.actor, "top_k", None)
        if top_k is None or top_k <= 0:
            top_k = min(20, self.action_dim)  # 預設一個合理的值，避免未設定

        for _ in range(self.epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                batch_obs         = obs[batch_idx]             # (B, obs_dim)
                batch_actions     = actions[batch_idx]         # (B, N)
                batch_returns     = returns[batch_idx]         # (B,)
                batch_adv         = advantages[batch_idx]      # (B,)
                batch_old_logprob = old_log_probs[batch_idx]   # (B,)

                # === 新 policy：scores -> top-k -> softmax -> probs ===
                logits = self.actor(batch_obs)                 # (B, N)
                # 取得每列 top-k 的 index 與值
                topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)          # (B, k), (B, k)
                topk_probs = torch.softmax(topk_vals, dim=-1)                       # (B, k)

                # 將 top-k 機率 scatter 回 N 維空間，其餘為 0
                probs = torch.zeros_like(logits)                                     # (B, N)
                probs.scatter_(1, topk_idx, topk_probs)
                probs = torch.clamp(probs, min=1e-8)                                 # 避免 log(0)

                # === 熵（只對有效機率計算即可；scatter 後計算全量也等價）===
                entropy = -(probs * probs.log()).sum(dim=-1).mean()
                entropies.append(float(entropy.item()))

                # === log_prob（與 select_action 保持一致：交叉熵形式）===
                batch_actions = torch.clamp(batch_actions, min=1e-12)
                batch_actions = batch_actions / batch_actions.sum(dim=-1, keepdim=True)
                new_log_probs = (batch_actions * probs.log()).sum(dim=-1)            # (B,)

                # === PPO ratio / 損失 ===
                ratio = torch.exp(new_log_probs - batch_old_logprob)
                ratio = torch.nan_to_num(ratio, nan=0.0, posinf=1e6, neginf=-1e6)

                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # === Critic ===
                values_pred = self.critic(batch_obs).squeeze(-1)
                critic_loss = (batch_returns - values_pred).pow(2).mean()

                # 保險：防 NaN/Inf
                actor_loss  = torch.nan_to_num(actor_loss,  nan=0.0, posinf=1e6, neginf=-1e6)
                critic_loss = torch.nan_to_num(critic_loss, nan=0.0, posinf=1e6, neginf=-1e6)

                # === 反傳與更新 ===
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss = actor_loss + self.value_coef * critic_loss
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),  max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

        if entropies:
            self.entropy_log.append(float(np.mean(entropies)))

    # === GAE ===
    def _compute_gae(self, rewards, dones, values):
        returns, advs = [], []
        gae, next_value = 0.0, 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advs.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]
        return (
            torch.tensor(returns, dtype=torch.float32, device=self.device),
            torch.tensor(advs, dtype=torch.float32, device=self.device),
        )
