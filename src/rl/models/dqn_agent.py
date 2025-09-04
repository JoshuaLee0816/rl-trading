# src/rl/models/dqn_agent.py

# === DQN Agent Steps ===
# 1. Initialize Agent
#    - Define obs_dim, action_dim, and hyperparameters (lr, gamma, epsilon, buffer size, etc.)
#    - Build Q-Network (estimate Q(s,a)) and Target Network (stabilize training)
#    - Create Replay Buffer to store (s, a, r, s’, done)

# 2. Action Selection (ε-greedy)
#    - With probability ε: choose random action (exploration)
#    - With probability 1-ε: choose argmax Q(s,a) (exploitation)
#    - Decay ε over time to reduce exploration

# 3. Store Transition
#    - After each step, push (obs, action, reward, next_obs, done) into Replay Buffer

# 4. Update Q-Network
#    - Sample mini-batch from Replay Buffer
#    - Compute target: y = r + γ * max_a’ Q_target(s’, a’)
#    - Compute loss: L = MSE(Q(s,a), y)
#    - Backpropagate and update Q-Network

# 5. Update Target Network
#    - Every fixed steps, copy weights from Q-Network to Target Network
#    - Or use soft update for smoother convergence

# 6. Training Loop
#    - For each episode:
#        * Observe state
#        * Select action via ε-greedy
#        * Step environment -> (next_state, reward, done)
#        * Store transition
#        * Update Q-Network
#        * Periodically update Target Network
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import numpy as np
import platform

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim = 64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
class ReplayBuffer:
    def __init__(self, capacity = 10000):
        self.capacity = capacity #buffer 容量設定
        self.buffer = []
        self.position = 0
    
    def push(self, obs, action, reward, next_obs, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, reward, next_obs, done)
        self.position = (self.position + 1) % self.capacity #????

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = map(np.array, zip(*batch))
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, obs_dim, action_dim, config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.lr = config.get("lr", 1e-3)
        self.batch_size = config.get("batch_size", 64)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)

        # 新增：探索衰減排程
        self.epsilon_schedule = config.get("epsilon_schedule", "per_step")  # per_step(舊行為) / per_episode(新)
        if self.epsilon_schedule == "per_episode":
            # 依照「N 集後到達 epsilon_min」自動推導每集衰減率
            N = max(1, int(config.get("epsilon_episodes_to_min", 5000)))
            self.epsilon_decay_episode = (self.epsilon_min / max(1e-9, self.epsilon)) ** (1.0 / N)
            self.epsilon_decay = None  # 不用 per-step
        else:
            # 保留舊的每步衰減機制
            self.epsilon_decay = config.get("epsilon_decay", 0.995)
            self.epsilon_decay_episode = None

        # --- 裝置選擇 ---
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

        #Networks
        self.q_network = QNetwork(obs_dim, action_dim).to(self.device)                              #用來學習狀態到動作價值（Q(s,a)）。
        self.target_network = copy.deepcopy(self.q_network).to(self.device)                             #主要用途是「計算 Q-learning 更新的目標值」。
        self.target_network.eval() #target net does not update directly

        self.optimizer = optim.Adam(self.q_network.parameters(), lr = self.lr)

        # Replay Buffer
        buffer_size = config.get("buffer_size", 10000)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    def select_action(self, obs):
        # Convert obs to tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)               #PyTorch QNetwork 期望輸入torch.Tensor

        # ε-greedy strategy
        if random.random() < self.epsilon:
            # Exploration: random action
            action = random.randrange(self.action_dim)
        else:
            # Exploitation: choose best action from Q-Network
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
                action = q_values.argmax().item()

        # Decay epsilon (but keep >= epsilon_min)
        # 衰減策略：每步 or 每集
        if self.epsilon_schedule == "per_step" and self.epsilon_decay is not None:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


        # === 修正：把 int 動作轉成 [idx, lvl] ===
        if isinstance(action, int):
            idx = action // 5
            lvl = action % 5
            action = [idx, lvl]                                 #一次只能調整一檔股票倉位，多檔的調法之後再寫

        return action


    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Store one transition into the replay buffer.
        Each transition is: (state, action, reward, next_state, done)
        """
        # 將 [idx, lvl] 轉換成單一 int
        if isinstance(action, (list, np.ndarray)) and len(action) == 2:
            idx, lvl = action
            action = idx * 5 + lvl   # flat index

        obs = np.array(obs, copy=False)
        next_obs = np.array(next_obs, copy=False)
        self.replay_buffer.push(obs, action, reward, next_obs, done)


    def update(self):
        # Only update if enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from replay buffer
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        obs      = torch.tensor(obs, dtype=torch.float32,device=self.device)
        actions  = torch.tensor(actions, dtype=torch.long,device=self.device).unsqueeze(1)  # shape: [batch, 1]
        rewards  = torch.tensor(rewards, dtype=torch.float32,device=self.device).unsqueeze(1)
        next_obs = torch.tensor(next_obs, dtype=torch.float32,device=self.device)
        dones    = torch.tensor(dones, dtype=torch.float32,device=self.device).unsqueeze(1)

        # Current Q values for chosen actions
        q_values = self.q_network(obs).gather(1, actions)

        # Target Q values
        with torch.no_grad():
            max_next_q = self.target_network(next_obs).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * (1 - dones) * max_next_q

        # Compute loss (MSE)
        loss = nn.MSELoss()(q_values, target_q)

        # Optimize Q-Network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Hard update target network every fixed steps ---
        if not hasattr(self, "update_count"):
            self.update_count = 0
        self.update_count += 1
        if self.update_count % self.config.get("target_update_freq", 100) == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    def on_episode_end(self):
        if self.epsilon_schedule == "per_episode" and self.epsilon_decay_episode is not None:
            self.epsilon = max(self.epsilon * self.epsilon_decay_episode, self.epsilon_min)


