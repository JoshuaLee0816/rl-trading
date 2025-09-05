# src/rl/models/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import platform

class Actor (nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLu(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        return self.net(x)

class Critic (nn.Module):
    def __init__(self, obs_dim, hidden_dim = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)
    
class RolloutBuffer:
    """存放 n_steps * n_envs 的資料"""
    #RolloutBuffer = 短期收集一批 episode 的資料 → 算 advantage → 拿去更新 → 清空
    def __init__(self):
        self.clear()

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get(self):
        return (
            np.array(self.obs),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.dones),
            np.array(self.log_probs),
            np.array(self.values),
        )

    def clear(self):
        #理論上update結束就要clear buffer
        self.obs, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []

class PPOAgent:
    def __init__(self, obs_dim, action_dim, config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

        # === 超參數從 config.yaml ===
        self.gamma = config.get("gamma")
        self.lam = config.get("gae_lambda")
        self.clip_epsilon = config.get("clip_epsilon")
        self.lr = config.get("lr")
        self.batch_size = config.get("batch_size")
        self.n_steps = config.get("n_steps")
        self.epochs = config.get("epochs")
        self.entropy_coef = config.get("entropy_coef")
        self.value_coef = config.get("value_coef")