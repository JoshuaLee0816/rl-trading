# src/rl/models/dqn_agent.py
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
