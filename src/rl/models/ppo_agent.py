import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import platform
import wandb
import time
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
    def __init__(self, device):
        self.device = device
        self.clear()

    def add(self, obs, action_flat, reward, done, log_prob, value, action_mask_flat):               #存成tensor
        self.obs.append(obs.to(self.device))
        self.actions.append(torch.as_tensor(action_flat, device=self.device))
        self.rewards.append(torch.as_tensor(reward, device=self.device))
        self.dones.append(torch.as_tensor(done, device=self.device))
        self.log_probs.append(torch.as_tensor(log_prob, device=self.device))
        self.values.append(torch.as_tensor(value, device=self.device))
        self.masks.append(action_mask_flat.to(self.device))

    def get(self):
        return (        #return torch datatype
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
        self.entropy_log, self.actor_loss_log, self.critic_loss_log = [], [], []

        # === Hyperparams ===
        ppo_cfg = config.get("ppo", {})
        self.gamma        = float(ppo_cfg.get("gamma"))
        self.lam          = float(ppo_cfg.get("gae_lambda"))
        self.clip_epsilon = float(ppo_cfg.get("clip_epsilon"))
        self.batch_size   = int(ppo_cfg.get("batch_size"))
        self.n_steps      = int(ppo_cfg.get("n_steps"))
        self.epochs       = int(ppo_cfg.get("epochs"))
        self.entropy_coef = float(ppo_cfg.get("entropy_coef"))
        self.value_coef   = float(ppo_cfg.get("value_coef"))

        # === 選擇 device ===
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
        self.encoder = build_encoder(enc_cfg, F=self.F).to(self.device)

        if enc_cfg["type"] == "identity":
            per_stock_dim = self.F * enc_cfg["params"].get("k_window", 20)
        else:
            per_stock_dim = enc_cfg["params"].get("d_model", 48)

        anchors = int(config["model"].get("n_anchor", 0))
        self.obs_dim = self.N * per_stock_dim + (1 + self.N) + 2 * config["environment"]["max_holdings"] + anchors

        # === Actor / Critic ===
        self.actor  = Actor(self.obs_dim, self.N, self.QMAX, hidden_dim=ppo_cfg.get("actor_hidden", 64)).to(self.device)
        self.critic = Critic(self.obs_dim, hidden_dim=ppo_cfg.get("critic_hidden", 64)).to(self.device)

        self.actor_lr  = float(ppo_cfg.get("actor_lr"))
        self.critic_lr = float(ppo_cfg.get("critic_lr"))
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.buffer = RolloutBuffer(self.device)

        # === Debug print (only once) ===
        if not hasattr(PPOAgent, "_printed_init"):
            print("=== [DEBUG INIT] ===")
            print(f"obs_dim={self.obs_dim}, action_dim={self.A}")
            print(f"Actor hidden={ppo_cfg.get('actor_hidden', 64)}, Critic hidden={ppo_cfg.get('critic_hidden', 64)}")
            PPOAgent._printed_init = True

    def obs_to_tensor(self, obs_dict):
        """
        將 env 輸出的 dict 轉成 [obs_dim] tensor
        """
        # features: (N,F,K) → (1,N,F,K)
        features_raw = obs_dict["features"].unsqueeze(0).to(self.device)
        z = self.encoder(features_raw)                  # [1,N,D]
        z_flat = z.reshape(1, -1)                       # [1,N*D]

        # portfolio: (1+N+2*max_holdings) → (1,dim)
        portfolio = obs_dict["portfolio"].unsqueeze(0).to(self.device)

        obs_t = torch.cat([z_flat, portfolio], dim=1)   # [1,obs_dim]
        return obs_t.squeeze(0)                         # [obs_dim]

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
            # === 判斷單環境還是多環境 ===
            if isinstance(obs, dict):   # 單一環境（dict 格式）
                features_raw = torch.as_tensor(obs["features"], dtype=torch.float32, device=self.device).unsqueeze(0)
                z = self.encoder(features_raw)
                z_flat = z.reshape(1, -1)
                portfolio = torch.as_tensor(obs["portfolio"], dtype=torch.float32, device=self.device).unsqueeze(0)
                anchors_raw = obs.get("anchors", None)
                if anchors_raw is not None and len(anchors_raw) > 0:
                    anchors = torch.as_tensor(anchors_raw, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    anchors = torch.zeros((1, 0), dtype=torch.float32, device=self.device)
                obs_t = torch.cat([z_flat, portfolio, anchors], dim=1)   # [1, obs_dim]
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                if obs_t.dim() == 1:  # [obs_dim]
                    obs_t = obs_t.unsqueeze(0)  # [1, obs_dim]

            B = obs_t.size(0)

            # === 標準化 ===
            obs_t = (obs_t - obs_t.mean(dim=1, keepdim=True)) / (obs_t.std(dim=1, keepdim=True) + 1e-8)

            # === 處理 mask ===
            if action_mask_3d is not None:
                if isinstance(action_mask_3d, (list, tuple)):  # 多環境
                    mask_flat = torch.stack([self.flatten_mask(m) for m in action_mask_3d], dim=0)
                else:  # 單一環境
                    mask_flat = self.flatten_mask(action_mask_3d).unsqueeze(0)
            else:
                mask_flat = torch.ones((B, self.A), dtype=torch.bool, device=self.device)

            # === Actor 前向 ===
            logits = self.actor(obs_t)                  # [B, A]
            masked_logits = logits.masked_fill(~mask_flat, LARGE_NEG)
            dist = Categorical(logits=masked_logits)
            a_flat = dist.sample()                      # [B]
            logp   = dist.log_prob(a_flat)              # [B]
            value  = self.critic(obs_t)                 # [B]

            # === Decode ===
            actions_tuple = [self.flat_to_tuple(int(a)) for a in a_flat.tolist()]

            if not hasattr(self, "_printed_select"):
                print("=== [DEBUG SELECT_ACTION] ===")
                print("obs_t:", obs_t.shape)
                print("logits:", logits.shape)
                print("masked_logits:", masked_logits.shape)
                print("action:", a_flat if B > 1 else int(a_flat.item()), 
                      "tuple:", actions_tuple if B > 1 else actions_tuple[0])
                self._printed_select = True

        # === 保持回傳順序 ===
        if B == 1:  # 單一環境 → 保持舊版格式
            return actions_tuple[0], int(a_flat.item()), float(logp.item()), float(value.item()), obs_t.squeeze(0), mask_flat.squeeze(0)
        else:       # 多環境 → 批次格式
            return actions_tuple, a_flat, logp, value, obs_t, mask_flat
    # endregion

    def store_transition(self, obs, action_flat, reward, done, log_prob, value, action_mask_flat):
        self.buffer.add(obs, action_flat, reward, done, log_prob, value, action_mask_flat)

    # region Update
    def update(self):
        t0 = time.perf_counter()

        obs, actions, rewards, dones, old_log_probs, values, masks = self.buffer.get()
        self.buffer.clear()
        t1 = time.perf_counter()

        returns, advantages_raw = self._compute_gae(rewards, dones, values)
        t2 = time.perf_counter()

        advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)

        if wandb.run is not None:
            wandb.log({"adv_mean_raw": advantages_raw.mean().item(),
                       "adv_std_raw": advantages_raw.std().item()})

        N = obs.size(0)
        entropies = []
        for _ in range(self.epochs):                    # 外層loop, 每次update重複幾個epochs
            idxs = np.arange(N)
            np.random.shuffle(idxs)
            for start in range(0, N, self.batch_size):  # Mini-batches切分 抽出這一批對應的資料
                b = idxs[start:start+self.batch_size]
                b_obs   = obs[b]
                b_acts  = actions[b]
                b_rets  = returns[b]
                b_advs  = advantages[b]
                b_oldlp = old_log_probs[b]
                b_mask  = masks[b]
                
                # 前向傳播+動作分布
                f0 = time.perf_counter()

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
                f1 = time.perf_counter()

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

                f2 = time.perf_counter()
                #print(f"[PROFILE] batch forward={f1-f0:.4f}s, backward={f2-f1:.4f}s")

        if entropies:
            self.entropy_log.append(float(np.mean(entropies)))

        if not hasattr(self, "_printed_update"):
            print("=== [DEBUG UPDATE] ===")
            print("obs:", obs.shape)
            print("actions:", actions.shape)
            print("advantages:", advantages.shape)
            print("returns:", returns.shape)
            self._printed_update = True
        
        t3 = time.perf_counter()
        print(f"[PROFILE] buffer={t1-t0:.4f}s, gae={t2-t1:.4f}s, update loop={t3-t2:.4f}s")

    # endregion Update

    # region Compute_GAE
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
    
    # endregion Compute_GAE
