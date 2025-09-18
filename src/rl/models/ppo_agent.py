import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import platform
import wandb
from torch.distributions import Categorical
from rl.models.encoders import build_encoder

LARGE_NEG = -1e9

# region Actor/Critic Network
# ===================== Actor / Critic =====================
class Actor(nn.Module):
    def __init__(self, obs_dim, num_stocks, qmax, hidden_dim=256):
        super().__init__()
        self.N = int(num_stocks)
        self.QMAX = int(qmax)
        self.action_dim = self.N * self.QMAX + self.N + 1
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, x):  # x: (B, obs_dim)
        out = self.net(x)  # (B, A)
        if not hasattr(self, "_printed"):
            print(f"[DEBUG] Actor.forward | input={x.shape} → logits={out.shape}")
            self._printed = True
        return out


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):  # x: (B, obs_dim)
        out = self.net(x).squeeze(-1)  # (B,)
        if not hasattr(self, "_printed"):
            print(f"[DEBUG] Critic.forward | input={x.shape} → value={out.shape}")
            self._printed = True
        return out

# endregion Actor/Critic Network


class RolloutBuffer:
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
            np.array(self.masks, dtype=np.bool_)
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
        self.entropy_log, self.actor_loss_log, self.critic_loss_log = [], [], []

        # === Hyperparams ===
        self.gamma        = float(config.get("gamma", 0.99))
        self.lam          = float(config.get("gae_lambda", 0.95))
        self.clip_epsilon = float(config.get("clip_epsilon", 0.2))
        self.batch_size   = int(config.get("batch_size", 64))
        self.n_steps      = int(config.get("n_steps", 2048))
        self.epochs       = int(config.get("epochs", 10))
        self.entropy_coef = float(config.get("entropy_coef", 0.0))
        self.value_coef   = float(config.get("value_coef", 0.5))

        # === 選擇 device ===
        ppo_cfg = config.get("ppo", {})
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

        # === Encoder 初始化 ===
        features = config["data"].get("features_clean", config["data"].get("features", []))
        self.F = len(features)
        enc_cfg = config["model"].get("encoder", {"type": "identity"})
        self.encoder = build_encoder(enc_cfg, F=self.F)

        if enc_cfg["type"] == "identity":
            per_stock_dim = self.F * enc_cfg["params"].get("k_window", 20)
        else:
            per_stock_dim = enc_cfg["params"].get("d_model", 48)

        anchors = int(config["model"].get("n_anchor", 0))
        self.obs_dim = self.N * per_stock_dim + (1 + self.N) + 2 * config["environment"]["max_holdings"] + anchors

        # === Actor / Critic ===
        ppo_cfg = config.get("ppo", {})
        self.actor  = Actor(self.obs_dim, self.N, self.QMAX, hidden_dim=ppo_cfg.get("actor_hidden", 64)).to(self.device)
        self.critic = Critic(self.obs_dim, hidden_dim=ppo_cfg.get("critic_hidden", 64)).to(self.device)

        self.actor_lr  = float(ppo_cfg.get("actor_lr"))
        self.critic_lr = float(ppo_cfg.get("critic_lr"))
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.buffer = RolloutBuffer()

        # === Debug print (only once) ===
        if not hasattr(PPOAgent, "_printed_init"):
            print("=== [DEBUG INIT] ===")
            print(f"obs_dim={self.obs_dim}, action_dim={self.A}")
            print(f"Actor hidden={ppo_cfg.get('actor_hidden', 64)}, Critic hidden={ppo_cfg.get('critic_hidden', 64)}")
            PPOAgent._printed_init = True

    # region flatten/unflatten
    def flatten_mask(self, mask3):
        if isinstance(mask3, np.ndarray):
            mask3 = torch.from_numpy(mask3)
        mask3 = mask3.to(self.device)
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
    # endregion

    # region Select action 
    def select_action(self, obs, action_mask_3d):
        self.actor.eval(); self.critic.eval()
        with torch.no_grad():
            if isinstance(obs, dict):
                features_raw = torch.as_tensor(obs["features"], dtype=torch.float32, device=self.device).unsqueeze(0)
                z = self.encoder(features_raw)
                z_flat = z.reshape(1, -1)
                portfolio = torch.as_tensor(obs["portfolio"], dtype=torch.float32, device=self.device).unsqueeze(0)
                anchors_raw = obs.get("anchors", None)
                if anchors_raw is not None and len(anchors_raw) > 0:
                    anchors = torch.as_tensor(anchors_raw, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    anchors = torch.zeros((1, 0), dtype=torch.float32, device=self.device)
                obs_t = torch.cat([z_flat, portfolio, anchors], dim=1)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_t = (obs_t - obs_t.mean()) / (obs_t.std() + 1e-8)
            mask_flat = self.flatten_mask(action_mask_3d).unsqueeze(0)
            logits = self.actor(obs_t)
            masked_logits = logits.masked_fill(~mask_flat, LARGE_NEG)
            dist = Categorical(logits=masked_logits)
            a_flat = dist.sample()
            logp   = dist.log_prob(a_flat)
            value  = self.critic(obs_t)
            a_flat_int = int(a_flat.item())
            action_tuple = self.flat_to_tuple(a_flat_int)
            obs_flat_np  = obs_t.squeeze(0).cpu().numpy()
            mask_flat_np = mask_flat.squeeze(0).cpu().numpy()

            if not hasattr(self, "_printed_select"):
                print("=== [DEBUG SELECT_ACTION] ===")
                print("obs_t:", obs_t.shape)
                print("logits:", logits.shape)
                print("masked_logits:", masked_logits.shape)
                print("action:", a_flat_int, "tuple:", action_tuple)
                self._printed_select = True

        return action_tuple, a_flat_int, float(logp.item()), float(value.item()), obs_flat_np, mask_flat_np
    # endregion

    def store_transition(self, obs, action_flat, reward, done, log_prob, value, action_mask_flat):
        self.buffer.add(obs, action_flat, reward, done, log_prob, value, action_mask_flat)

    def update(self):
        obs, actions, rewards, dones, old_log_probs, values, masks = self.buffer.get()
        self.buffer.clear()
        device = self.device
        obs    = torch.tensor(obs,    dtype=torch.float32, device=device)  
        acts   = torch.tensor(actions, dtype=torch.long,   device=device) 
        rews   = torch.tensor(rewards, dtype=torch.float32, device=device) 
        dns    = torch.tensor(dones,   dtype=torch.float32, device=device) 
        old_lp = torch.tensor(old_log_probs, dtype=torch.float32, device=device)  
        vals   = torch.tensor(values,  dtype=torch.float32, device=device)        
        masks  = torch.tensor(masks,   dtype=torch.bool,   device=device)        

        returns, advantages_raw = self._compute_gae(rews, dns, vals)
        advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)

        if wandb.run is not None:
            wandb.log({"adv_mean_raw": advantages_raw.mean().item(),
                       "adv_std_raw": advantages_raw.std().item()})

        N = obs.size(0)
        entropies = []
        for _ in range(self.epochs):
            idxs = np.arange(N)
            np.random.shuffle(idxs)
            for start in range(0, N, self.batch_size):
                b = idxs[start:start+self.batch_size]
                b_obs, b_acts, b_rets, b_advs, b_oldlp, b_mask = obs[b], acts[b], returns[b], advantages[b], old_lp[b], masks[b]
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
            print("obs:", obs.shape)
            print("actions:", acts.shape)
            print("advantages:", advantages.shape)
            print("returns:", returns.shape)
            self._printed_update = True

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
