import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.__init__()

    def compute_returns(self, gamma):
        returns, discounted = [], 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted = 0
            discounted = reward + gamma * discounted
            returns.insert(0, discounted)
        return torch.tensor(returns, dtype=torch.float32)
