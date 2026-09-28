from pathlib import Path

import numpy as np
import torch
from torch import nn

from mappo.networks import CentralCritic, SharedActor


class MAPPOTrainer:
    def __init__(
        self,
        obs_dim,
        state_dim,
        num_agents,
        learning_rate=5e-4,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=10.0,
        update_epochs=5,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.num_agents = num_agents
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.actor = SharedActor(obs_dim).to(self.device)
        self.critic = CentralCritic(state_dim, num_agents).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
        )

    def _critic_inputs(self, states):
        if states.ndim == 1:
            states = np.repeat(states[None, :], self.num_agents, axis=0)
        elif states.ndim == 2:
            states = np.repeat(states[:, None, :], self.num_agents, axis=1)
        return states

    @torch.no_grad()
    def act(self, observations, state, action_masks, deterministic=False):
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
        actions, log_probs = self.actor.act(obs_tensor, mask_tensor, deterministic)
        state_batch = torch.as_tensor(
            self._critic_inputs(np.asarray(state)),
            dtype=torch.float32,
            device=self.device,
        )
        agent_ids = torch.arange(self.num_agents, device=self.device)
        values = self.critic(state_batch, agent_ids)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def values(self, state):
        state_batch = torch.as_tensor(
            self._critic_inputs(np.asarray(state)),
            dtype=torch.float32,
            device=self.device,
        )
        agent_ids = torch.arange(self.num_agents, device=self.device)
        return self.critic(state_batch, agent_ids).cpu().numpy()

    def update(self, buffer, last_values, gamma=0.99, gae_lambda=0.95):
        data = buffer.arrays()
        advantages, returns = buffer.advantages_and_returns(
            last_values,
            gamma,
            gae_lambda,
        )
        time_steps, num_agents = data["actions"].shape
        batch_size = time_steps * num_agents

        observations = torch.as_tensor(
            data["observations"].reshape(batch_size, -1),
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.as_tensor(
            data["action_masks"].reshape(batch_size, -1),
            dtype=torch.bool,
            device=self.device,
        )
        actions = torch.as_tensor(
            data["actions"].reshape(-1),
            dtype=torch.long,
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            data["log_probs"].reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        state_batch = np.repeat(data["states"][:, None, :], num_agents, axis=1)
        states = torch.as_tensor(
            state_batch.reshape(batch_size, -1),
            dtype=torch.float32,
            device=self.device,
        )
        agent_ids = torch.arange(num_agents, device=self.device).repeat(time_steps)
        advantages = torch.as_tensor(
            advantages.reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.as_tensor(
            returns.reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )

        latest = {}
        for _ in range(self.update_epochs):
            log_probs, entropy = self.actor.evaluate_actions(
                observations,
                masks,
                actions,
            )
            ratios = torch.exp(log_probs - old_log_probs)
            surrogate_1 = ratios * advantages
            surrogate_2 = torch.clamp(
                ratios,
                1.0 - self.clip_ratio,
                1.0 + self.clip_ratio,
            ) * advantages
            policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()
            values = self.critic(states, agent_ids)
            value_loss = nn.functional.mse_loss(values, returns)
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy_mean
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()
            latest = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy_mean.item()),
            }
        return latest

    def save(self, path, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "extra": extra or {},
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("extra", {})

