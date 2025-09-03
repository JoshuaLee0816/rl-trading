# src/rl/models/a2c_agent.py
class A2CAgent:
    def __init__(self, obs_dim, action_dim, config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        # TODO: 初始化 actor-critic 網路

    def select_action(self, obs):
        # TODO: 用 actor 決策
        raise NotImplementedError("A2CAgent.select_action 尚未實作")

    def store_transition(self, obs, action, reward, next_obs, done):
        # TODO: 存到 buffer（或直接用 on-policy）
        pass

    def update(self):
        # TODO: 執行一次 A2C 更新
        pass
