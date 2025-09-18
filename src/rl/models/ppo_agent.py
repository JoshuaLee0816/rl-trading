# src/rl/models/ppo_agent.py
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
    """
    輸出動作action _dim

    EX -> N = 2, Qmax = 3, 這樣會是6格 + Sell_All 一格 + HOLD 1格 ，所以action dim = 8格 (N*Qmax + 1 + 1)
    """
    def __init__(self, obs_dim, num_stocks, qmax, hidden_dim=256):
        super().__init__()
        self.N = int(num_stocks)
        self.QMAX = int(qmax)
        self.action_dim = self.N * self.QMAX + self.N + 1
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), 
            nn.ReLU(), #block negative signal (Shall I change to Tanh()??)
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim)
        )

        # 初始化的時候標準化network 才不會讓初始化太大
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)
        #print(self.net[0].weight.abs().mean().item())


    def forward(self, x):  # x: (B, obs_dim)
        return self.net(x)  # (B, A) raw logits 這些logits再轉乘categorical (logits = ...)動作分布


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)   #輸出1dim =>某個狀態的評估價值
        )
    def forward(self, x):  # x: (B, obs_dim) (B是Bacth size, 然後去掉最後一dim可以計算loss)
        return self.net(x).squeeze(-1)  # (B,)

# endregion Actor/Critic Network

# ===================== Rollout Buffer =====================
class RolloutBuffer:
    """
    因為PPO是 On-policy, 必須用當前的policy來產生動作得到資料, and RolloutBuffer is something like register in this model.
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
    # region PPO 初始化
    def __init__(self, obs_dim, num_stocks, qmax_per_trade, config):
        #self.obs_dim = int(obs_dim)

        self.N = int(num_stocks)
        self.QMAX = int(qmax_per_trade)           # 現實中買幾張其實跟金額有關，但這裡限制大概在10張只是為了要縮小維度 
        self.A = self.N * self.QMAX + self.N + 1  # 動作空間攤平成一維後的大小 (用於policy network 輸出成Logits)
        self.config = config
        self.entropy_log = []
        self.actor_loss_log = []
        self.critic_loss_log = []


        # === Hyperparams ===
        self.gamma         = float(config.get("gamma", 0.99))
        self.lam           = float(config.get("gae_lambda", 0.95))
        self.clip_epsilon  = float(config.get("clip_epsilon", 0.2))
        self.batch_size    = int(config.get("batch_size", 64))
        self.n_steps       = int(config.get("n_steps", 2048))
        self.epochs        = int(config.get("epochs", 10))
        self.entropy_coef  = float(config.get("entropy_coef", 0.0))
        self.value_coef    = float(config.get("value_coef", 0.5))

        # region 選擇device

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
            
        if not hasattr(PPOAgent, "_printed_device"):
            print(f"[INFO] Using device: {self.device}")
            PPOAgent._printed_device = True

        # endregion 選擇device

        # --- Encoder 初始化 ---
        features = config["data"].get("features_clean", config["data"].get("features", []))
        self.F = len(features)
        enc_cfg = config["model"].get("encoder", {"type":"identity"})
        self.encoder = build_encoder(enc_cfg, F=self.F)

        # 每檔股票輸出的維度
        if enc_cfg["type"] == "identity":
            per_stock_dim = self.F * enc_cfg["params"].get("k_window", 20)
        else:
            per_stock_dim = enc_cfg["params"].get("d_model", 48)

        # 總 obs 維度 = 股票編碼 + portfolio 狀態 + anchor
        anchors = int(config["model"].get("n_anchor", 0))
        self.obs_dim = self.N * per_stock_dim + (1 + self.N) + 2 * config["environment"]["max_holdings"] + anchors          #定義了維度

        # Actor Critic Optimizer初始化
        ppo_cfg = config.get("ppo", {})  # 先抓出 ppo 區塊
        self.actor  = Actor(self.obs_dim, self.N, self.QMAX, hidden_dim=ppo_cfg.get("actor_hidden", 64)).to(self.device)
        self.critic = Critic(self.obs_dim, hidden_dim=ppo_cfg.get("critic_hidden", 64)).to(self.device)


        self.actor_lr  = float(ppo_cfg.get("actor_lr"))
        self.critic_lr = float(ppo_cfg.get("critic_lr"))

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.buffer = RolloutBuffer()

    # endregion PPO 初始化

    # region 三維動作mask flat成一維的工具
    def flatten_mask(self, mask3):
        """
        (3, N, QMAX+1) -> (A,) 布林向量
          BUY(i, q>=1)  -> 前 N*QMAX 格
          SELL_ALL(i)   -> 接著 N 格（用 q=0 那一格）
          HOLD          -> 最後 1 格（mask[2,0,0]）
        """
        if isinstance(mask3, np.ndarray):
            mask3 = torch.from_numpy(mask3) #確保轉成Torch tensor放到同一個裝置(CPU/CUDA/MPS)
        mask3 = mask3.to(self.device)

        buy  = mask3[0, :, 1:]              # 取出所有buy的動作
        sell = mask3[1, :, :1]              # 取出所有sell的動作
        hold = mask3[2:3, :1, :1].reshape(1)  # 取出所有hold的動作
        flat = torch.cat([buy.reshape(-1), sell.reshape(-1), hold], dim=0).bool() #三段攤平

        """
        [ BUY(stock0, q=1), BUY(stock0, q=2), BUY(stock0, q=3), SELL_ALL(stock0), SELL_ALL(stock1), HOLD ]
        現在HOLD就會全部都HOLD 要謹慎思考 我怕HOLD太容易導致都不進行交易
        """
        return flat  # (A,)
    
    # endregion 三維動作mask flat成一維的工具

    # region 一維 unflatten 還原成 三維
    """
    這樣做的原因：
    Torch 的 policy gradient (比如 categorical 分布）最容易處理 一維分類問題
    但是StockTradingEnv 的 step() 期待的動作格式是三維
    """
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
        
    # endregion 一維 unflatten 還原成 三維

        # region Select action 
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
          - obs_flat_np:  np.ndarray    -> 展平 obs，交給 train loop
          - mask_flat_np: np.ndarray    -> 動作 mask (A,)
        """
        self.actor.eval(); self.critic.eval()

        # 在 rollout 階段關閉梯度，避免建立無用計算圖（省記憶體、加速）
        with torch.no_grad():
            if isinstance(obs, dict):
                # 假設 obs 是 dict: {"features": (N,F,K), "portfolio": (1+N,), "anchors": (n_anchor,)}
                features_raw = torch.as_tensor(obs["features"], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,N,F,K)
                z = self.encoder(features_raw)      # (1,N,D) or (1,N,F*K)
                z_flat = z.reshape(1, -1)           # (1, N*D)

                portfolio = torch.as_tensor(obs["portfolio"], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,1+N)
                anchors_raw = obs.get("anchors", None)
                if anchors_raw is not None and len(anchors_raw) > 0:
                    anchors = torch.as_tensor(anchors_raw, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    anchors = torch.zeros((1,0), dtype=torch.float32, device=self.device)

                obs_t = torch.cat([z_flat, portfolio, anchors], dim=1)  # (1, obs_dim)
            else:
                # 假設 obs 已經是展平的一維向量
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, obs_dim)

            obs_t = (obs_t - obs_t.mean()) / (obs_t.std() + 1e-8)

            mask_flat = self.flatten_mask(action_mask_3d).unsqueeze(0)                          # 只允許True in Mask

            # 2) logits -> masked categorical
            logits = self.actor(obs_t)                   # (1, A)
            masked_logits = logits.masked_fill(~mask_flat, LARGE_NEG)
            dist = Categorical(logits=masked_logits)

            # 3) sample + log_prob + value
            a_flat = dist.sample()
            logp   = dist.log_prob(a_flat)
            value  = self.critic(obs_t)

            # 4) 還原為 (op, idx, q)
            a_flat_int = int(a_flat.item())
            action_tuple = self.flat_to_tuple(a_flat_int)

            # === 展平 obs 交給 train loop ===
            obs_flat_np  = obs_t.squeeze(0).cpu().numpy()
            mask_flat_np = mask_flat.squeeze(0).cpu().numpy()

        return (
            action_tuple,      # 給 env.step()
            a_flat_int,        # 給 buffer 儲存
            float(logp.item()),
            float(value.item()), # 算advantages
            obs_flat_np,       # 展平 obs
            mask_flat_np       # 動作 mask
        )
    # endregion Select action 


    # region Store transition into Rollout Buffer 
    """
    EX
    {
        "obs": [0.3, -0.1, 1.2],
        "action": 17,
        "reward": 0.05,
        "done": False,
        "log_prob": -2.35,
        "value": 0.12,
        "mask": [False, True, ..., False]
        }
    """
    def store_transition(self, obs, action_flat, reward, done, log_prob, value, action_mask_flat):
        self.buffer.add(obs, action_flat, reward, done, log_prob, value, action_mask_flat)

    # endregion Store transition into Rollout Buffer 

    # region Update (PPO)
    def update(self):
        obs, actions, rewards, dones, old_log_probs, values, masks = self.buffer.get()  #get rollout buffer 裡面全部的資料
        self.buffer.clear()                                                             #清空準備下一次蒐集

        device = self.device
        obs    = torch.tensor(obs,    dtype=torch.float32, device=device)  
        acts   = torch.tensor(actions, dtype=torch.long,    device=device) 
        rews   = torch.tensor(rewards, dtype=torch.float32, device=device) 
        dns    = torch.tensor(dones,   dtype=torch.float32, device=device) 
        old_lp = torch.tensor(old_log_probs, dtype=torch.float32, device=device)  
        vals   = torch.tensor(values,  dtype=torch.float32, device=device)        
        masks  = torch.tensor(masks,   dtype=torch.bool,    device=device)        

        # ---- GAE ----
        returns, advantages_raw = self._compute_gae(rews, dns, vals)
        adv_mean_raw = advantages_raw.mean().item()
        adv_std_raw  = advantages_raw.std().item()

        # normalize
        advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)

        # === W&B logging of advantages ===
        if wandb.run is not None:
            wandb.log({
                "adv_mean_raw": adv_mean_raw,
                "adv_std_raw": adv_std_raw,
            })

        N = obs.size(0)
        entropies = []

        # region 小批次更新(mini-batch SGD)
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

                self.actor_loss_log.append(float(actor_loss.item()))
                self.critic_loss_log.append(float(critic_loss.item()))


                # 梯度裁切避免爆炸
                nn.utils.clip_grad_norm_(self.actor.parameters(),  max_norm=0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # endregion 小批次更新(mini-batch SGD)

        if entropies:
            self.entropy_log.append(float(np.mean(entropies)))
    # endregion Update (PPO)

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
