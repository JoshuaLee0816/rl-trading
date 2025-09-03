# src/rl/models/ppo_agent.py
class PPOAgent:
    def __init__(self, obs_dim, action_dim, config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        # TODO: 初始化 actor-critic 網路, buffer

    def select_action(self, obs):
        # TODO: 用 actor 輸出 action distribution
        raise NotImplementedError("PPOAgent.select_action 尚未實作")

    def store_transition(self, obs, action, reward, next_obs, done):
        # TODO: 存到 buffer
        pass

    def update(self):
        # TODO: 執行多次 PPO 更新
        pass
