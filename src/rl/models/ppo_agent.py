import platform
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import math
import wandb

LARGE_NEG = -1e9


class Actor(nn.Module):
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
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        out = self.net(x)
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
        out = self.net(x).squeeze(-1)
        return out


class RolloutBuffer:
    def __init__(self, device):
        self.device = device
        self.clear()

    def add(self, obs, action_flat, reward, done, log_prob, value, action_mask_flat):
        obs_t   = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t   = torch.as_tensor(action_flat, dtype=torch.long, device=self.device)
        rew_t   = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done_t  = torch.as_tensor(done, dtype=torch.float32, device=self.device)
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

        # === 新增：Adaptive Entropy Targeting ===
        self.entropy_target = float(ppo_cfg.get("entropy_target", 1.8))
        self.entropy_alpha  = float(ppo_cfg.get("entropy_alpha", 0.05))
        self.entropy_coef_min = float(ppo_cfg.get("entropy_coef_min", 0.001))
        self.entropy_coef_max = float(ppo_cfg.get("entropy_coef_max", 0.05))

        # === KL-aware 相關設定 ===
        self.target_kl   = float(ppo_cfg.get("target_kl", 0.02))
        self.kl_stop_mult = float(ppo_cfg.get("kl_stop_mult", 1.5))
        self.kl_low_mult  = float(ppo_cfg.get("kl_low_mult", 0.5))
        self.adapt_clip   = bool(ppo_cfg.get("adapt_clip", True))
        self.clip_min     = float(ppo_cfg.get("clip_min", 0.05))
        self.clip_max     = float(ppo_cfg.get("clip_max", 0.30))
        self.clip_up      = float(ppo_cfg.get("clip_up", 1.25))
        self.clip_down    = float(ppo_cfg.get("clip_down", 0.80))

        actor_hidden      = int(ppo_cfg.get("actor_hidden", 64))
        critic_hidden     = int(ppo_cfg.get("critic_hidden", 64))

        # === Device ===
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

        self.obs_dim = int(obs_dim)
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

    # ---- 其他函式不動 ----
    def obs_to_tensor(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return obs

    def flatten_mask(self, mask3):
        if isinstance(mask3, np.ndarray):
            mask3 = torch.from_numpy(mask3)
        mask3 = mask3.to(self.device).to(torch.bool)
        buy  = mask3[0, :, 1:]
        sell = mask3[1, :, :1]
        hold = mask3[2:3, :1, :1].reshape(1)
        flat = torch.cat([buy.reshape(-1), sell.reshape(-1), hold], dim=0).bool()
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

    def select_action(self, obs, action_mask_3d_batch):
        self.actor.eval(); self.critic.eval()
        with torch.no_grad():
            obs_t = self.obs_to_tensor(obs)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            B = obs_t.size(0)
            obs_t = (obs_t - obs_t.mean(dim=1, keepdim=True)) / (obs_t.std(dim=1, keepdim=True) + 1e-8)

            if action_mask_3d_batch is not None:
                if isinstance(action_mask_3d_batch, (list, tuple)):
                    mask_flat = torch.stack([self.flatten_mask(m) for m in action_mask_3d_batch], dim=0)
                else:
                    mask_flat = self.flatten_mask(action_mask_3d_batch).unsqueeze(0)
            else:
                mask_flat = torch.ones((B, self.A), dtype=torch.bool, device=self.device)

            logits = self.actor(obs_t)
            masked_logits = logits.masked_fill(~mask_flat, LARGE_NEG)
            dist = Categorical(logits=masked_logits)
            a_flat = dist.sample()
            logp   = dist.log_prob(a_flat)
            value  = self.critic(obs_t)

            actions_tuple = [self.flat_to_tuple(int(a)) for a in a_flat.tolist()]
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

        N = obs.size(0)
        entropies = []
        clip_eps_now = float(self.clip_epsilon)
        kl_list, kl_stop = [], False

        for _ in range(self.epochs):
            idxs = np.arange(N)
            np.random.shuffle(idxs)
            for start in range(0, N, self.batch_size):
                b = idxs[start:start+self.batch_size]
                b_obs, b_acts, b_rets = obs[b], actions[b], returns[b]
                b_advs, b_oldlp, b_mask = advantages[b], old_log_probs[b], masks[b]

                logits = self.actor(b_obs)
                masked_logits = logits.masked_fill(~b_mask, LARGE_NEG)
                dist = Categorical(logits=masked_logits)
                new_logp = dist.log_prob(b_acts)
                entropy  = dist.entropy().mean()
                entropies.append(float(entropy.item()))

                ratio = torch.exp(new_logp - b_oldlp)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1 - clip_eps_now, 1 + clip_eps_now) * b_advs
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

                with torch.no_grad():
                    approx_kl = (b_oldlp - new_logp).mean().abs()
                    kl_list.append(float(approx_kl.item()))
                if approx_kl.item() > self.kl_stop_mult * self.target_kl:
                    kl_stop = True
                    break
                if self.adapt_clip:
                    if approx_kl.item() < self.kl_low_mult * self.target_kl:
                        clip_eps_now = min(self.clip_max, clip_eps_now * self.clip_up)
                    elif approx_kl.item() > self.kl_stop_mult * self.target_kl:
                        clip_eps_now = max(self.clip_min, clip_eps_now * self.clip_down)
            if kl_stop:
                break

        # === Adaptive Entropy Update ===
        if entropies:
            current_entropy = float(np.mean(entropies))
            self.entropy_log.append(current_entropy)
            # 🔁 自動調整 entropy_coef
            self.entropy_coef *= math.exp(-self.entropy_alpha * (current_entropy - self.entropy_target))
            self.entropy_coef = float(np.clip(self.entropy_coef, self.entropy_coef_min, self.entropy_coef_max))

        avg_kl = float(np.mean(kl_list)) if kl_list else 0.0
        logs = {
            "actor_loss": self.actor_loss_log[-1] if self.actor_loss_log else None,
            "critic_loss": self.critic_loss_log[-1] if self.critic_loss_log else None,
            "entropy": self.entropy_log[-1] if self.entropy_log else None,
            "policy_kl": avg_kl,
            "clip_eps_now": float(clip_eps_now),
            "kl_early_stop": int(kl_stop),
            "entropy_coef_now": self.entropy_coef,  # ✅ 新增追蹤項
        }
        print(f"[DEBUG] len(actor_loss_log)={len(self.actor_loss_log)}, len(critic_loss_log)={len(self.critic_loss_log)}")
        return logs

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
