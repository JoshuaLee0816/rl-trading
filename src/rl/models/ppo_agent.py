import platform
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

import wandb

LARGE_NEG = -1e9


class Actor(nn.Module):
    def __init__(self, obs_dim, num_stocks, qmax, hidden_dim=256):
        super().__init__()
        self.N = int(num_stocks)
        self.QMAX = int(qmax)
        # 扁平動作空間：BUY(N*QMAX) + SELL(N) + HOLD(1)
        self.action_dim = self.N * self.QMAX + self.N + 1
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        out = self.net(x)  # (B, A)
        """
        if not hasattr(self, "_printed"):
            print(f"[DEBUG] Actor.forward | input={x.shape} → logits={out.shape}")
            self._printed = True
        """
        return out


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        out = self.net(x).squeeze(-1)  # (B,)
        """
        if not hasattr(self, "_printed"):
            print(f"[DEBUG] Critic.forward | input={x.shape} → value={out.shape}")
            self._printed = True
        """
        return out


class RolloutBuffer:
    def __init__(self, device):
        self.device = device
        self.clear()

    def add(self, obs, action_flat, reward, done, log_prob, value, action_mask_flat):
        # ✅ 統一轉 dtype，避免 numpy.bool_ / numpy.float64 造成問題
        obs_t   = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t   = torch.as_tensor(action_flat, dtype=torch.long, device=self.device)
        rew_t   = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done_t  = torch.as_tensor(done, dtype=torch.float32, device=self.device)    # ← 轉成 float32 (0/1)
        logp_t  = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device)
        val_t   = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        mask_t  = action_mask_flat.to(self.device).to(torch.bool)

        self.obs.append(obs_t)
        self.actions.append(act_t)
        self.rewards.append(rew_t)
        self.dones.append(done_t)
        self.log_probs.append(logp_t)
        self.values.append(val_t)
        self.masks.append(mask_t)

    def get(self):
        # 這裡會回傳 shape 皆為 [T] 或 [T, ...]
        return (
            torch.stack(self.obs),
            torch.stack(self.actions),
            torch.stack(self.rewards),
            torch.stack(self.dones),
            torch.stack(self.log_probs),
            torch.stack(self.values),
            torch.stack(self.masks),
        )

    def clear(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []
        self.masks = []



class PPOAgent:
    def __init__(self, obs_dim, num_stocks, qmax_per_trade, config):
        self.N = int(num_stocks)
        self.QMAX = int(qmax_per_trade)
        self.A = self.N * self.QMAX + self.N + 1
        self.config = config
        #self.entropy_log, self.actor_loss_log, self.critic_loss_log = [], [], []
        self.entropy_log = deque(maxlen=1000)
        self.actor_loss_log = deque(maxlen=1000)
        self.critic_loss_log = deque(maxlen=1000)

        # === Hyperparams ===
        ppo_cfg = config.get("ppo", {})
        self.gamma        = float(ppo_cfg.get("gamma", 0.99))
        self.lam          = float(ppo_cfg.get("gae_lambda", 0.95))
        self.clip_epsilon = float(ppo_cfg.get("clip_epsilon", 0.2))
        self.batch_size   = int(ppo_cfg.get("batch_size", 256))
        self.n_steps      = int(ppo_cfg.get("n_steps", 512))
        self.epochs       = int(ppo_cfg.get("epochs", 3))
        self.entropy_coef = float(ppo_cfg.get("entropy_coef", 0.0))
        self.value_coef   = float(ppo_cfg.get("value_coef", 0.5))
        actor_hidden      = int(ppo_cfg.get("actor_hidden", 64))
        critic_hidden     = int(ppo_cfg.get("critic_hidden", 64))

        # === 選擇 device（支援 MPS）===
        device_cfg = ppo_cfg.get("device", "auto")
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
        if not hasattr(PPOAgent, "_printed_device"):
            print(f"[INFO] Using device: {self.device}")
            PPOAgent._printed_device = True

        # === 關鍵修正：obs_dim 直接採用環境提供的值 ===
        self.obs_dim = int(obs_dim)

        # === Actor / Critic ===
        self.actor  = Actor(self.obs_dim, self.N, self.QMAX, hidden_dim=actor_hidden).to(self.device)
        self.critic = Critic(self.obs_dim, hidden_dim=critic_hidden).to(self.device)

        self.actor_lr  = float(ppo_cfg.get("actor_lr", 3e-4))
        self.critic_lr = float(ppo_cfg.get("critic_lr", 3e-4))
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.buffer = RolloutBuffer(self.device)

        if not hasattr(PPOAgent, "_printed_init"):
            print("=== [DEBUG INIT] ===")
            print(f"obs_dim={self.obs_dim}, action_dim={self.A}")
            print(f"Actor hidden={actor_hidden}, Critic hidden={critic_hidden}")
            PPOAgent._printed_init = True

    def obs_to_tensor(self, obs):
        # 支援 [obs_dim] 或 [B, obs_dim] 的 np.ndarray
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return obs

    # ---- mask flat/unflat ----
    def flatten_mask(self, mask3):
        if isinstance(mask3, np.ndarray):
            mask3 = torch.from_numpy(mask3)
        mask3 = mask3.to(self.device).to(torch.bool)  # (3, N, QMAX+1)
        buy  = mask3[0, :, 1:]           # (N, QMAX)  (q>=1)
        sell = mask3[1, :, :1]           # (N, 1)
        hold = mask3[2:3, :1, :1].reshape(1)  # (1,)
        flat = torch.cat([buy.reshape(-1), sell.reshape(-1), hold], dim=0).bool()  # (A,)
        return flat

    def flat_to_tuple(self, a_flat: int):
        A_buy = self.N * self.QMAX
        if a_flat < A_buy:
            idx = a_flat // self.QMAX
            q   = (a_flat % self.QMAX) + 1
            return (0, idx, q)
        elif a_flat < A_buy + self.N:
            idx = a_flat - A_buy
            return (1, idx, 0)
        else:
            return (2, 0, 0)

    # ---- action ----
    def select_action(self, obs, action_mask_3d_batch):
        """
        永遠回傳批次型態：
        - actions_tuple: list[(op, idx, q)] 長度 B
        - a_flat: (B,)
        - logp:   (B,)
        - value:  (B,)
        - obs_t:  (B, obs_dim)
        - mask_flat: (B, A)  bool
        """
        self.actor.eval(); self.critic.eval()
        with torch.no_grad():
            obs_t = self.obs_to_tensor(obs)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)  # [1, obs_dim]
            B = obs_t.size(0)

            # per-sample normalize（穩定分佈）
            obs_t = (obs_t - obs_t.mean(dim=1, keepdim=True)) / (obs_t.std(dim=1, keepdim=True) + 1e-8)

            # 批次 mask
            if action_mask_3d_batch is not None:
                if isinstance(action_mask_3d_batch, (list, tuple)):
                    mask_flat = torch.stack([self.flatten_mask(m) for m in action_mask_3d_batch], dim=0)
                else:
                    mask_flat = self.flatten_mask(action_mask_3d_batch).unsqueeze(0)
            else:
                mask_flat = torch.ones((B, self.A), dtype=torch.bool, device=self.device)

            logits = self.actor(obs_t)                 # (B, A)
            masked_logits = logits.masked_fill(~mask_flat, LARGE_NEG)
            dist = Categorical(logits=masked_logits)
            a_flat = dist.sample()                     # (B,)
            logp   = dist.log_prob(a_flat)             # (B,)
            value  = self.critic(obs_t)                # (B,)

            actions_tuple = [self.flat_to_tuple(int(a)) for a in a_flat.tolist()]

            if not hasattr(self, "_printed_select"):
                print("=== [DEBUG SELECT_ACTION] ===")
                print("obs_t:", obs_t.shape, "| logits:", logits.shape, "| masked:", masked_logits.shape)
                print("action(flat):", a_flat.shape, "example:", int(a_flat[0].item()))
                print("action(tuple)[0]:", actions_tuple[0])
                self._printed_select = True

        return actions_tuple, a_flat, logp, value, obs_t, mask_flat

    def store_transition(self, obs, action_flat, reward, done, log_prob, value, action_mask_flat):
        self.buffer.add(obs, action_flat, reward, done, log_prob, value, action_mask_flat)

    # ---- update ----
    def update(self):
        t0 = time.perf_counter()
        obs, actions, rewards, dones, old_log_probs, values, masks = self.buffer.get()
        self.buffer.clear()
        t1 = time.perf_counter()

        returns, advantages_raw = self._compute_gae(rewards, dones, values)
        t2 = time.perf_counter()

        advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)
        """
        if wandb.run is not None:
            wandb.log({"adv_mean_raw": advantages_raw.mean().item(),
                       "adv_std_raw": advantages_raw.std().item()})
        """

        N = obs.size(0)
        entropies = []
        for _ in range(self.epochs):
            idxs = np.arange(N)
            np.random.shuffle(idxs)
            for start in range(0, N, self.batch_size):
                b = idxs[start:start+self.batch_size]
                b_obs   = obs[b]
                b_acts  = actions[b]
                b_rets  = returns[b]
                b_advs  = advantages[b]
                b_oldlp = old_log_probs[b]
                b_mask  = masks[b]

                logits = self.actor(b_obs)
                masked_logits = logits.masked_fill(~b_mask, LARGE_NEG)
                dist = Categorical(logits=masked_logits)
                new_logp = dist.log_prob(b_acts)
                entropy  = dist.entropy().mean()
                entropies.append(float(entropy.item()))

                ratio = torch.exp(new_logp - b_oldlp)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advs
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                v_pred = self.critic(b_obs)
                critic_loss = (b_rets - v_pred).pow(2).mean() * self.value_coef

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss = actor_loss + critic_loss
                total_loss.backward()
                self.actor_loss_log.append(float(actor_loss.item()))
                self.critic_loss_log.append(float(critic_loss.item()))
                nn.utils.clip_grad_norm_(self.actor.parameters(),  max_norm=0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        if entropies:
            self.entropy_log.append(float(np.mean(entropies)))

        if not hasattr(self, "_printed_update"):
            print("=== [DEBUG UPDATE] ===")
            print("obs:", obs.shape, "| actions:", actions.shape,
                  "| advantages:", advantages.shape, "| returns:", returns.shape)
            self._printed_update = True

        t3 = time.perf_counter()
        # print(f"[PROFILE] buffer={t1-t0:.4f}s, gae={t2-t1:.4f}s, update loop={t3-t2:.4f}s")

        # RAM記憶體DEBUG用
        print(f"[DEBUG] len(actor_loss_log)={len(self.actor_loss_log)}, len(critic_loss_log)={len(self.critic_loss_log)}")

    def _compute_gae(self, rewards, dones, values):
        T = len(rewards)
        returns = torch.zeros(T, dtype=torch.float32, device=self.device)
        advs    = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae, next_value = 0.0, 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae   = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advs[t]    = gae
            returns[t] = gae + values[t]
            next_value = values[t]
        return returns, advs
