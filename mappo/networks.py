import torch
from torch import nn
from torch.distributions import Categorical


def _mlp(input_dim, hidden_sizes, output_dim):
    layers = []
    previous = input_dim
    for hidden in hidden_sizes:
        layers.extend([nn.Linear(previous, hidden), nn.Tanh()])
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class SharedActor(nn.Module):
    def __init__(self, obs_dim, action_dim=2, hidden_sizes=(64, 64)):
        super().__init__()
        self.network = _mlp(obs_dim, hidden_sizes, action_dim)

    def distribution(self, observations, action_masks):
        logits = self.network(observations)
        logits = logits.masked_fill(~action_masks.bool(), -1e9)
        return Categorical(logits=logits)

    def act(self, observations, action_masks, deterministic=False):
        distribution = self.distribution(observations, action_masks)
        actions = (
            torch.argmax(distribution.logits, dim=-1)
            if deterministic
            else distribution.sample()
        )
        return actions, distribution.log_prob(actions)

    def evaluate_actions(self, observations, action_masks, actions):
        distribution = self.distribution(observations, action_masks)
        return distribution.log_prob(actions), distribution.entropy()


class CentralCritic(nn.Module):
    def __init__(self, state_dim, num_agents, hidden_sizes=(128, 128)):
        super().__init__()
        self.num_agents = num_agents
        self.network = _mlp(state_dim + num_agents, hidden_sizes, 1)

    def forward(self, states, agent_ids):
        one_hot = torch.nn.functional.one_hot(
            agent_ids.long(),
            num_classes=self.num_agents,
        ).float()
        return self.network(torch.cat([states, one_hot], dim=-1)).squeeze(-1)

