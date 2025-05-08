import torch
from torch.optim import Adam
from .actor_critic import ActorCritic
from .buffer import RolloutBuffer

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            action, logprob, _ = self.policy.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        return action.numpy()

    def update(self):
        returns = self.buffer.compute_returns(self.gamma)
        states = torch.stack(self.buffer.states)
        actions = torch.stack(self.buffer.actions)
        old_logprobs = torch.stack(self.buffer.logprobs).detach()

        for _ in range(self.k_epochs):
            logprobs, values, entropy = self.policy.evaluate(states, actions)
            advantages = returns - values.detach()
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean()                    + 0.5 * (values - returns).pow(2).mean()                    - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.buffer.clear()
