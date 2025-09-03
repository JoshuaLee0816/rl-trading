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


class DQNAgent:
    def __init__(self, obs_dim, action_dim, config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        # TODO: 初始化 Q-network, target network, memory buffer

    def select_action(self, obs):
        # TODO: ε-greedy 策略
        raise NotImplementedError("DQNAgent.select_action 尚未實作")

    def store_transition(self, obs, action, reward, next_obs, done):
        # TODO: 存到 replay buffer
        pass

    def update(self):
        # TODO: 從 replay buffer 抽樣並更新 Q-network
        pass
