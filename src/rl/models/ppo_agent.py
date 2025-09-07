# src/rl/models/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import platform
from torch.distributions import Categorical

LARGE_NEG = -1e9


# ===================== Actor / Critic =====================
class Actor(nn.Module):
    """
    輸出「攤平後」的動作 logits 向量：
      A = N*QMAX + N + 1
      [0 .. N*QMAX-1] -> BUY(i, q>=1)
      [N*QMAX .. N*QMAX+N-1] -> SELL_ALL(i)
      [最後一格] -> HOLD
    """
    def __init__(self, obs_dim, num_stocks, qmax, hidden_dim=256):
        super().__init__()
        self.N = int(num_stocks)
        self.QMAX = int(qmax)
        self.action_dim = self.N * self.QMAX + self.N + 1
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim)
        )

    def forward(self, x):  # x: (B, obs_dim)
        return self.net(x)  # (B, A) raw logits


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):  # x: (B, obs_dim)
        return self.net(x).squeeze(-1)  # (B,)


# ===================== Rollout Buffer =====================
class RolloutBuffer:
    """
    只存 PPO 需要的最小集合（含 action_mask_flat，更新時重建 masked dist）
    """
    def __init__(self):
        self.clear()

    def add(self, obs, action_flat, reward, done, log_prob, value, action_mask_flat):
        self.obs.append(obs)
        self.actions.append(action_flat)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(action_mask_flat)

    def get(self):
        return (
            np.array(self.obs, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.masks, dtype=np.bool_)  # shape: (T, A)
        )

    def clear(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []
        self.masks = []


# ===================== PPO Agent（攤平動作版） =====================
class PPOAgent:
    """
    使用「攤平動作空間」的 PPO：
      - 需要從 env.info["action_mask"] 取得 (3, N, QMAX+1) 遮罩，並在這裡攤平成 (A,)
      - select_action 回傳 (action_tuple, action_flat, log_prob, value)
      - store_transition 請把 action_flat 與 action_mask_flat 存入 buffer
    """
    def __init__(self, obs_dim, num_stocks, qmax_per_trade, config):
        self.obs_dim = int(obs_dim)
        self.N = int(num_stocks)
        self.QMAX = int(qmax_per_trade)
        self.A = self.N * self.QMAX + self.N + 1  # 攤平後動作數
        self.config = config
        self.entropy_log = []

        # === Hyperparams ===
        self.gamma         = float(config.get("gamma", 0.99))
        self.lam           = float(config.get("gae_lambda", 0.95))
        self.clip_epsilon  = float(config.get("clip_epsilon", 0.2))
        self.batch_size    = int(config.get("batch_size", 64))
        self.n_steps       = int(config.get("n_steps", 2048))
        self.epochs        = int(config.get("epochs", 10))
        self.entropy_coef  = float(config.get("entropy_coef", 0.0))
        self.value_coef    = float(config.get("value_coef", 0.5))

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

        # === Networks & Optimizers ===
        self.actor  = Actor(self.obs_dim, self.N, self.QMAX, hidden_dim=config.get("actor_hidden", 256)).to(self.device)
        self.critic = Critic(self.obs_dim, hidden_dim=config.get("critic_hidden", 256)).to(self.device)

        self.actor_lr  = float(config.get("actor_lr", 3e-4))
        self.critic_lr = float(config.get("critic_lr", 1e-3))
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.buffer = RolloutBuffer()

    # ---------- Action flatten/unflatten helpers ----------
    def flatten_mask(self, mask3):
        """
        (3, N, QMAX+1) -> (A,) 布林向量
          BUY(i, q>=1)  -> 前 N*QMAX 格
          SELL_ALL(i)   -> 接著 N 格（用 q=0 那一格）
          HOLD          -> 最後 1 格（mask[2,0,0]）
        """
        if isinstance(mask3, np.ndarray):
            mask3 = torch.from_numpy(mask3)
        mask3 = mask3.to(self.device)

        buy  = mask3[0, :, 1:]              # (N, QMAX)
        sell = mask3[1, :, :1]              # (N, 1)   只取 q=0
        hold = mask3[2:3, :1, :1].reshape(1)  # (1,)
        flat = torch.cat([buy.reshape(-1), sell.reshape(-1), hold], dim=0).bool()
        return flat  # (A,)

    def flat_to_tuple(self, a_flat: int):
        """
        將攤平類別還原為 (op, idx, q)
          op: 0=BUY, 1=SELL_ALL, 2=HOLD
        """
        A_buy = self.N * self.QMAX
        if a_flat < A_buy:
            rel = int(a_flat)
            idx = rel // self.QMAX
            q   = (rel % self.QMAX) + 1
            return (0, idx, q)
        elif a_flat < A_buy + self.N:
            idx = int(a_flat - A_buy)
            return (1, idx, 0)
        else:
            return (2, 0, 0)

    # ---------- Select Action ----------
    def select_action(self, obs, action_mask_3d):
        """
        取得一個動作（抽樣版）：
          - obs: (obs_dim,)
          - action_mask_3d: (3, N, QMAX+1)；來自 env.info["action_mask"]
        回傳：
          - action_tuple: (op, idx, q)  -> 給 env.step()
          - action_flat:  int           -> 給 buffer 儲存
          - log_prob:     float
          - value:        float
        """
        self.actor.eval(); self.critic.eval()

        # 1) to tensor
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, obs_dim)
        mask_flat = self.flatten_mask(action_mask_3d).unsqueeze(0)                          # (1, A)

        # 2) logits -> masked categorical
        logits = self.actor(obs_t)                   # (1, A)
        masked_logits = logits.masked_fill(~mask_flat, LARGE_NEG)
        dist = Categorical(logits=masked_logits)     # 內部會做 softmax

        # 3) sample + log_prob + value
        a_flat = dist.sample()                       # (1,)
        logp   = dist.log_prob(a_flat)               # (1,)
        value  = self.critic(obs_t)                  # (1,)

        # 4) 還原為 (op, idx, q)
        a_flat_int = int(a_flat.item())
        action_tuple = self.flat_to_tuple(a_flat_int)

        return (
            action_tuple,
            a_flat_int,
            float(logp.item()),
            float(value.item()),
        )

    # ---------- Store transition ----------
    def store_transition(self, obs, action_flat, reward, done, log_prob, value, action_mask_flat):
        """
        action_mask_flat: (A,) 布林向量（建議在互動時就把 3D mask 攤平存起來）
        """
        self.buffer.add(obs, action_flat, reward, done, log_prob, value, action_mask_flat)

    # ---------- Update (PPO) ----------
    def update(self):
        obs, actions, rewards, dones, old_log_probs, values, masks = self.buffer.get()
        self.buffer.clear()

        device = self.device
        obs    = torch.tensor(obs,    dtype=torch.float32, device=device)  # (T, obs_dim)
        acts   = torch.tensor(actions, dtype=torch.long,    device=device) # (T,)
        rews   = torch.tensor(rewards, dtype=torch.float32, device=device) # (T,)
        dns    = torch.tensor(dones,   dtype=torch.float32, device=device) # (T,)
        old_lp = torch.tensor(old_log_probs, dtype=torch.float32, device=device)  # (T,)
        vals   = torch.tensor(values,  dtype=torch.float32, device=device)        # (T,)
        masks  = torch.tensor(masks,   dtype=torch.bool,    device=device)        # (T, A)

        # ---- GAE ----
        returns, advantages = self._compute_gae(rews, dns, vals)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = obs.size(0)
        entropies = []

        for _ in range(self.epochs):
            idxs = np.arange(N)
            np.random.shuffle(idxs)
            for start in range(0, N, self.batch_size):
                b = idxs[start:start+self.batch_size]
                b_obs   = obs[b]        # (B, obs_dim)
                b_acts  = acts[b]       # (B,)
                b_rets  = returns[b]    # (B,)
                b_advs  = advantages[b] # (B,)
                b_oldlp = old_lp[b]     # (B,)
                b_mask  = masks[b]      # (B, A)

                # 新 policy：masked logits -> dist
                logits = self.actor(b_obs)                         # (B, A)
                masked_logits = logits.masked_fill(~b_mask, LARGE_NEG)
                dist = Categorical(logits=masked_logits)

                new_logp = dist.log_prob(b_acts)                   # (B,)
                entropy  = dist.entropy().mean()                   # scalar
                entropies.append(float(entropy.item()))

                # ratio / PPO clip
                ratio = torch.exp(new_logp - b_oldlp)              # (B,)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advs
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # Critic
                v_pred = self.critic(b_obs)                        # (B,)
                critic_loss = (b_rets - v_pred).pow(2).mean() * self.value_coef

                # 反傳
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss = actor_loss + critic_loss
                total_loss.backward()

                # 梯度裁切避免爆炸
                nn.utils.clip_grad_norm_(self.actor.parameters(),  max_norm=0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

        if entropies:
            self.entropy_log.append(float(np.mean(entropies)))

    # ---------- GAE ----------
    def _compute_gae(self, rewards, dones, values):
        returns, advs = [], []
        gae, next_value = 0.0, 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advs.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]
        device = self.device
        return (
            torch.tensor(returns, dtype=torch.float32, device=device),
            torch.tensor(advs,    dtype=torch.float32, device=device),
        )
