import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.states = []
        self.action_masks = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(
        self,
        observations,
        state,
        action_masks,
        actions,
        log_probs,
        rewards,
        done,
        values,
    ):
        self.observations.append(np.asarray(observations, dtype=np.float32))
        self.states.append(np.asarray(state, dtype=np.float32))
        self.action_masks.append(np.asarray(action_masks, dtype=np.bool_))
        self.actions.append(np.asarray(actions, dtype=np.int64))
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.dones.append(float(done))
        self.values.append(np.asarray(values, dtype=np.float32))

    def __len__(self):
        return len(self.rewards)

    def arrays(self):
        return {
            "observations": np.asarray(self.observations, dtype=np.float32),
            "states": np.asarray(self.states, dtype=np.float32),
            "action_masks": np.asarray(self.action_masks, dtype=np.bool_),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
        }

    def advantages_and_returns(self, last_values, gamma=0.99, gae_lambda=0.95):
        data = self.arrays()
        rewards = data["rewards"]
        dones = data["dones"]
        values = data["values"]
        advantages = np.zeros_like(rewards)
        gae = np.zeros(rewards.shape[1], dtype=np.float32)
        next_values = np.asarray(last_values, dtype=np.float32)

        for step in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + gamma * next_values * nonterminal - values[step]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[step] = gae
            next_values = values[step]

        return advantages, advantages + values

