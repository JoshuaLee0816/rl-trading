# src/rl/models/random_agent.py
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs):
        # baseline: 隨機選擇動作
        return self.action_space.sample()

    def store_transition(self, *args, **kwargs):
        pass

    def update(self):
        pass


