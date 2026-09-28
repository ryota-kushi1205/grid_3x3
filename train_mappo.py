import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from mappo.rollout_buffer import RolloutBuffer
from mappo.trainer import MAPPOTrainer
from sumo_mappo_env import (
    DEMAND_END,
    ENVIRONMENT_VERSION,
    MAX_SIMULATION_END,
    WARMUP_END,
    SumoMAPPOEnv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAPPO on the 3x3 SUMO grid")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/mappo_warmup300_latest.pt",
    )
    parser.add_argument("--log", default="mappo_training_warmup300.csv")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--warmup", type=float, default=WARMUP_END)
    parser.add_argument("--demand-end", type=float, default=DEMAND_END)
    parser.add_argument("--max-end", type=float, default=MAX_SIMULATION_END)
    return parser.parse_args()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    seed_everything(args.seed)
    env = SumoMAPPOEnv(
        use_gui=args.gui,
        seed=args.seed,
        warmup_end=args.warmup,
        demand_end=args.demand_end,
        max_simulation_end=args.max_end,
    )
    trainer = MAPPOTrainer(
        env.obs_dim,
        env.state_dim,
        env.num_agents,
        device=args.device,
    )
    buffer = RolloutBuffer()
    observations, state, masks = env.reset()
    episode = 0
    episode_return = 0.0
    episode_steps = 0
    log_path = Path(args.log)
    write_header = not log_path.exists()

    try:
        with log_path.open("a", newline="", encoding="utf-8-sig") as log_file:
            writer = csv.DictWriter(
                log_file,
                fieldnames=[
                    "total_steps",
                    "episode",
                    "episode_steps",
                    "episode_return",
                    "end_time_s",
                    "cleared",
                    "truncated",
                ],
            )
            if write_header:
                writer.writeheader()

            total_steps = 0
            while total_steps < args.total_steps:
                actions, log_probs, values = trainer.act(observations, state, masks)
                (
                    next_observations,
                    next_state,
                    rewards,
                    terminated,
                    truncated,
                    next_masks,
                    info,
                ) = env.step(actions)
                done = terminated or truncated
                buffer.add(
                    observations,
                    state,
                    masks,
                    actions,
                    log_probs,
                    rewards,
                    done,
                    values,
                )
                observations, state, masks = (
                    next_observations,
                    next_state,
                    next_masks,
                )
                total_steps += 1
                episode_steps += 1
                episode_return += float(rewards[0])

                if done:
                    writer.writerow({
                        "total_steps": total_steps,
                        "episode": episode,
                        "episode_steps": episode_steps,
                        "episode_return": f"{episode_return:.6f}",
                        "end_time_s": f"{info['time_s']:.0f}",
                        "cleared": terminated,
                        "truncated": truncated,
                    })
                    log_file.flush()
                    episode += 1
                    episode_steps = 0
                    episode_return = 0.0
                    observations, state, masks = env.reset(seed=args.seed)

                if len(buffer) >= args.rollout_steps or total_steps >= args.total_steps:
                    last_values = (
                        np.zeros(env.num_agents, dtype=np.float32)
                        if done
                        else trainer.values(state)
                    )
                    losses = trainer.update(buffer, last_values)
                    buffer.clear()
                    trainer.save(
                        args.checkpoint,
                        extra={
                            "total_steps": total_steps,
                            "episode": episode,
                            "environment_version": ENVIRONMENT_VERSION,
                            "warmup_end": args.warmup,
                            "demand_end": args.demand_end,
                            "max_simulation_end": args.max_end,
                            "seed": args.seed,
                        },
                    )
                    print(
                        f"steps={total_steps} episode={episode} "
                        f"loss={losses['loss']:.4f} entropy={losses['entropy']:.4f}"
                    )
    finally:
        env.close()


if __name__ == "__main__":
    main()
