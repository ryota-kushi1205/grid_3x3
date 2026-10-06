import argparse
import csv
import os
import time
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


DEFAULT_CHECKPOINT = "checkpoints/mappo_warmup300_maxgreen60_latest.pt"
LOG_FIELDS = [
    "total_steps",
    "episode",
    "episode_steps",
    "episode_return",
    "end_time_s",
    "cleared",
    "truncated",
    "scenario_id",
    "scenario_profile",
    "scenario_demand_level",
]
UPDATE_LOG_FIELDS = [
    "total_steps",
    "episode",
    "rollout_steps",
    "agent_samples",
    "wall_time_s",
    "mean_team_reward",
    "min_team_reward",
    "max_team_reward",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "requested_switch_rate",
    "executed_switch_rate",
    "forced_switch_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAPPO on the 3x3 SUMO grid")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint",
        help="Output checkpoint path; defaults to --resume or the standard path.",
    )
    parser.add_argument(
        "--resume",
        metavar="CHECKPOINT",
        help="Resume actor, critic, optimizer, counters, and RNG state from a checkpoint.",
    )
    parser.add_argument("--log", default="mappo_training_warmup300_maxgreen60.csv")
    parser.add_argument(
        "--update-log",
        help="Per-update CSV path; defaults to <log stem>_updates.csv.",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--warmup", type=float, default=WARMUP_END)
    parser.add_argument("--demand-end", type=float, default=DEMAND_END)
    parser.add_argument("--max-end", type=float, default=MAX_SIMULATION_END)
    parser.add_argument(
        "--max-wall-time-seconds",
        type=float,
        help="Stop after this many real seconds, saving the latest completed rollout.",
    )
    parser.add_argument(
        "--scenario-manifest",
        help="CSV made by generate_training_scenarios.py; samples one train scenario per episode.",
    )
    return parser.parse_args()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_training_scenarios(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scenario manifest not found: {manifest_path}")
    required = {
        "scenario_id",
        "split",
        "profile",
        "demand_level",
        "seed",
        "vehicle_route_file",
        "pedestrian_route_file",
    }
    scenarios = []
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"scenario manifest missing columns: {sorted(missing)}")
        for row in reader:
            if row["split"] != "train":
                continue
            vehicle_route = (manifest_path.parent / row["vehicle_route_file"]).resolve()
            pedestrian_route = (
                manifest_path.parent / row["pedestrian_route_file"]
            ).resolve()
            if not vehicle_route.is_file() or not pedestrian_route.is_file():
                raise FileNotFoundError(
                    f"scenario {row['scenario_id']} references a missing route file"
                )
            row["vehicle_route"] = vehicle_route
            row["pedestrian_route"] = pedestrian_route
            row["seed"] = int(row["seed"])
            scenarios.append(row)
    if not scenarios:
        raise ValueError("scenario manifest has no rows with split=train")
    return scenarios


def reset_episode(env, scenarios, scenario_rng, default_seed):
    if not scenarios:
        env.set_route_files(None)
        observations, state, masks = env.reset(seed=default_seed)
        return observations, state, masks, None

    scenario = scenarios[int(scenario_rng.integers(len(scenarios)))]
    env.set_route_files(
        [scenario["vehicle_route"], scenario["pedestrian_route"]]
    )
    observations, state, masks = env.reset(seed=scenario["seed"])
    return observations, state, masks, scenario


def normalized_manifest_path(manifest_path):
    return str(Path(manifest_path).resolve()) if manifest_path else ""


def validate_resume_settings(extra, args):
    expected = {
        "environment_version": ENVIRONMENT_VERSION,
        "warmup_end": args.warmup,
        "demand_end": args.demand_end,
        "max_simulation_end": args.max_end,
        "seed": args.seed,
        "scenario_manifest": normalized_manifest_path(args.scenario_manifest),
        "rollout_steps": args.rollout_steps,
    }
    for key, expected_value in expected.items():
        actual = extra.get(key)
        if key in {"warmup_end", "demand_end", "max_simulation_end"}:
            matches = actual is not None and float(actual) == float(expected_value)
        else:
            matches = actual == expected_value
        if not matches:
            raise ValueError(
                f"checkpoint setting {key}={actual!r} does not match "
                f"current setting {expected_value!r}"
            )


def restore_rng_state(extra, scenario_rng):
    if "numpy_random_state" in extra:
        np.random.set_state(extra["numpy_random_state"])
    if "torch_rng_state" in extra:
        torch.set_rng_state(extra["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "torch_cuda_rng_state_all" in extra:
        torch.cuda.set_rng_state_all(extra["torch_cuda_rng_state_all"])
    if "scenario_rng_state" in extra:
        scenario_rng.bit_generator.state = extra["scenario_rng_state"]


def checkpoint_extra(args, total_steps, episode, scenario_rng):
    extra = {
        "total_steps": total_steps,
        "episode": episode,
        "environment_version": ENVIRONMENT_VERSION,
        "warmup_end": args.warmup,
        "demand_end": args.demand_end,
        "max_simulation_end": args.max_end,
        "seed": args.seed,
        "scenario_manifest": normalized_manifest_path(args.scenario_manifest),
        "rollout_steps": args.rollout_steps,
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "scenario_rng_state": scenario_rng.bit_generator.state,
    }
    if torch.cuda.is_available():
        extra["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return extra


def validate_log_header(log_path, fields, option_name):
    if not log_path.exists() or log_path.stat().st_size == 0:
        return True
    with log_path.open(newline="", encoding="utf-8-sig") as log_file:
        header = next(csv.reader(log_file), [])
    if header != fields:
        raise ValueError(
            f"log header in {log_path} is incompatible; "
            f"use a new {option_name} path"
        )
    return False


def default_update_log_path(log_path):
    suffix = log_path.suffix or ".csv"
    return log_path.with_name(f"{log_path.stem}_updates{suffix}")


def main():
    args = parse_args()
    if args.total_steps <= 0:
        raise ValueError("--total-steps must be positive")
    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive")
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
    scenarios = (
        load_training_scenarios(args.scenario_manifest)
        if args.scenario_manifest
        else []
    )
    scenario_rng = np.random.default_rng(args.seed)
    checkpoint_path = Path(
        args.checkpoint or args.resume or DEFAULT_CHECKPOINT
    )
    total_steps = 0
    episode = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resume_extra = trainer.load(resume_path, load_optimizer=True)
        validate_resume_settings(resume_extra, args)
        total_steps = int(resume_extra.get("total_steps", 0))
        episode = int(resume_extra.get("episode", 0))
        restore_rng_state(resume_extra, scenario_rng)
        if total_steps >= args.total_steps:
            raise ValueError(
                f"checkpoint already has {total_steps} steps; "
                f"set --total-steps above that value"
            )
        print(
            f"resumed from {resume_path}: steps={total_steps} episode={episode}"
        )

    observations, state, masks, active_scenario = reset_episode(
        env,
        scenarios,
        scenario_rng,
        args.seed,
    )
    episode_return = 0.0
    episode_steps = 0
    started_at = time.monotonic()
    log_path = Path(args.log)
    update_log_path = (
        Path(args.update_log) if args.update_log else default_update_log_path(log_path)
    )
    if log_path.resolve() == update_log_path.resolve():
        raise ValueError("--log and --update-log must be different files")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    update_log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = validate_log_header(log_path, LOG_FIELDS, "--log")
    write_update_header = validate_log_header(
        update_log_path,
        UPDATE_LOG_FIELDS,
        "--update-log",
    )
    rollout_requested_switches = 0
    rollout_executed_switches = 0
    rollout_forced_switches = 0
    rollout_agent_decisions = 0
    print(f"episode log: {log_path}")
    print(f"update log: {update_log_path}")

    try:
        with (
            log_path.open("a", newline="", encoding="utf-8-sig") as log_file,
            update_log_path.open(
                "a", newline="", encoding="utf-8-sig"
            ) as update_log_file,
        ):
            episode_writer = csv.DictWriter(
                log_file,
                fieldnames=LOG_FIELDS,
            )
            update_writer = csv.DictWriter(
                update_log_file,
                fieldnames=UPDATE_LOG_FIELDS,
            )
            if write_header:
                episode_writer.writeheader()
            if write_update_header:
                update_writer.writeheader()

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
                control_results = info["control_results"].values()
                rollout_requested_switches += sum(
                    result.requested_action == 1 for result in control_results
                )
                rollout_executed_switches += sum(
                    result.executed_action == 1
                    for result in info["control_results"].values()
                )
                rollout_forced_switches += sum(
                    result.forced for result in info["control_results"].values()
                )
                rollout_agent_decisions += env.num_agents

                if done:
                    episode_writer.writerow({
                        "total_steps": total_steps,
                        "episode": episode,
                        "episode_steps": episode_steps,
                        "episode_return": f"{episode_return:.6f}",
                        "end_time_s": f"{info['time_s']:.0f}",
                        "cleared": terminated,
                        "truncated": truncated,
                        "scenario_id": (
                            active_scenario["scenario_id"] if active_scenario else "fixed"
                        ),
                        "scenario_profile": (
                            active_scenario["profile"] if active_scenario else "fixed"
                        ),
                        "scenario_demand_level": (
                            active_scenario["demand_level"]
                            if active_scenario
                            else "fixed"
                        ),
                    })
                    log_file.flush()
                    episode += 1
                    episode_steps = 0
                    episode_return = 0.0
                    observations, state, masks, active_scenario = reset_episode(
                        env,
                        scenarios,
                        scenario_rng,
                        args.seed,
                    )

                time_limit_reached = (
                    args.max_wall_time_seconds is not None
                    and time.monotonic() - started_at >= args.max_wall_time_seconds
                )
                if (
                    len(buffer) >= args.rollout_steps
                    or total_steps >= args.total_steps
                    or time_limit_reached
                ):
                    rollout_length = len(buffer)
                    team_rewards = np.asarray(buffer.rewards, dtype=np.float32)[:, 0]
                    last_values = (
                        np.zeros(env.num_agents, dtype=np.float32)
                        if done
                        else trainer.values(state)
                    )
                    losses = trainer.update(buffer, last_values)
                    buffer.clear()
                    trainer.save(
                        checkpoint_path,
                        extra=checkpoint_extra(
                            args,
                            total_steps,
                            episode,
                            scenario_rng,
                        ),
                    )
                    elapsed = time.monotonic() - started_at
                    update_writer.writerow({
                        "total_steps": total_steps,
                        "episode": episode,
                        "rollout_steps": rollout_length,
                        "agent_samples": rollout_length * env.num_agents,
                        "wall_time_s": f"{elapsed:.3f}",
                        "mean_team_reward": f"{team_rewards.mean():.6f}",
                        "min_team_reward": f"{team_rewards.min():.6f}",
                        "max_team_reward": f"{team_rewards.max():.6f}",
                        "loss": f"{losses['loss']:.6f}",
                        "policy_loss": f"{losses['policy_loss']:.6f}",
                        "value_loss": f"{losses['value_loss']:.6f}",
                        "entropy": f"{losses['entropy']:.6f}",
                        "requested_switch_rate": (
                            f"{rollout_requested_switches / rollout_agent_decisions:.6f}"
                        ),
                        "executed_switch_rate": (
                            f"{rollout_executed_switches / rollout_agent_decisions:.6f}"
                        ),
                        "forced_switch_rate": (
                            f"{rollout_forced_switches / rollout_agent_decisions:.6f}"
                        ),
                    })
                    update_log_file.flush()
                    rollout_requested_switches = 0
                    rollout_executed_switches = 0
                    rollout_forced_switches = 0
                    rollout_agent_decisions = 0
                    print(
                        f"steps={total_steps} episode={episode} "
                        f"loss={losses['loss']:.4f} entropy={losses['entropy']:.4f}"
                    )
                if time_limit_reached:
                    elapsed = time.monotonic() - started_at
                    print(
                        f"wall-time limit reached after {elapsed:.1f} s; "
                        f"checkpoint saved at steps={total_steps}"
                    )
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
