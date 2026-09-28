"""Evaluate a trained MAPPO checkpoint on the fixed 3x3 SUMO demand."""

import argparse
import csv
from pathlib import Path

from mappo.trainer import MAPPOTrainer
from sumo_mappo_env import (
    DEMAND_END,
    ENVIRONMENT_VERSION,
    MAX_SIMULATION_END,
    WARMUP_END,
    SumoMAPPOEnv,
)


DETAIL_FIELDS = [
    "time_s",
    "period",
    "junction",
    "phase",
    "requested_action",
    "executed_action",
    "forced",
    "control_reason",
    "stopped_red_vehicles",
    "idle_energy_mj",
    "wait_c0",
    "wait_c1",
    "wait_c2",
    "wait_c3",
    "wait_east_west",
    "wait_north_south",
    "unresolved_pedestrians",
    "affected_vehicles_for_east_west_ped",
    "affected_vehicles_for_north_south_ped",
    "clear_time_east_west_s",
    "clear_time_north_south_s",
    "predicted_energy_east_west_mj",
    "predicted_energy_north_south_mj",
    "predicted_pedestrian_energy_mj",
    "total_congestion_mj",
]

SUMMARY_FIELDS = [
    "checkpoint",
    "checkpoint_total_steps",
    "checkpoint_episode",
    "seed",
    "warmup_end_s",
    "demand_end_s",
    "simulation_end_s",
    "cleared",
    "truncated",
    "remaining_vehicles",
    "remaining_persons",
    "main_decisions",
    "cooldown_decisions",
    "total_decisions",
    "main_idle_score_mj",
    "main_prediction_score_mj",
    "main_congestion_score_mj",
    "cooldown_idle_score_mj",
    "cooldown_prediction_score_mj",
    "cooldown_congestion_score_mj",
    "total_idle_score_mj",
    "total_prediction_score_mj",
    "total_congestion_score_mj",
    "episode_return",
    "requested_switches",
    "executed_switches",
    "forced_switches",
    "max_waiting_pedestrians_at_junction",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MAPPO checkpoint on SUMO"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/mappo_warmup300_latest.pt",
    )
    parser.add_argument("--output", default="mappo_evaluation.csv")
    parser.add_argument("--summary", default="mappo_evaluation_summary.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using the highest-probability action",
    )
    parser.add_argument("--warmup", type=float, default=WARMUP_END)
    parser.add_argument("--demand-end", type=float, default=DEMAND_END)
    parser.add_argument("--max-end", type=float, default=MAX_SIMULATION_END)
    return parser.parse_args()


def empty_totals():
    return {"decisions": 0, "idle": 0.0, "prediction": 0.0, "cost": 0.0}


def update_totals(totals, info):
    totals["decisions"] += 1
    totals["idle"] += info["network_idle_energy_mj"]
    totals["prediction"] += info["network_prediction_energy_mj"]
    totals["cost"] += info["network_cost_mj"]


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

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
    checkpoint_extra = trainer.load(checkpoint_path, load_optimizer=False)
    checkpoint_version = checkpoint_extra.get("environment_version")
    if checkpoint_version != ENVIRONMENT_VERSION:
        raise ValueError(
            "checkpoint was trained with an incompatible environment: "
            f"expected {ENVIRONMENT_VERSION!r}, got {checkpoint_version!r}. "
            "Retrain MAPPO after the warm-up and observation changes."
        )
    expected_settings = {
        "warmup_end": args.warmup,
        "demand_end": args.demand_end,
        "max_simulation_end": args.max_end,
    }
    for key, expected in expected_settings.items():
        actual = checkpoint_extra.get(key)
        if actual is None or float(actual) != float(expected):
            raise ValueError(
                f"checkpoint setting {key}={actual!r} does not match "
                f"evaluation setting {expected!r}"
            )

    detail_path = Path(args.output)
    summary_path = Path(args.summary)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    main_totals = empty_totals()
    cooldown_totals = empty_totals()
    requested_switches = 0
    executed_switches = 0
    forced_switches = 0
    maximum_waiting_pedestrians = 0
    episode_return = 0.0
    final_info = None

    try:
        observations, state, masks = env.reset(seed=args.seed)
        with detail_path.open("w", newline="", encoding="utf-8-sig") as detail_file:
            writer = csv.DictWriter(detail_file, fieldnames=DETAIL_FIELDS)
            writer.writeheader()

            while True:
                actions, _, _ = trainer.act(
                    observations,
                    state,
                    masks,
                    deterministic=not args.stochastic,
                )
                (
                    observations,
                    state,
                    rewards,
                    terminated,
                    truncated,
                    masks,
                    info,
                ) = env.step(actions)
                final_info = info
                episode_return += float(rewards[0])
                period = "main" if info["time_s"] <= args.demand_end else "cooldown"
                update_totals(
                    main_totals if period == "main" else cooldown_totals,
                    info,
                )

                metrics_by_junction = {
                    row["junction"]: row for row in info["junction_metrics"]
                }
                for junction in env.agent_ids:
                    result = info["control_results"][junction]
                    requested_switches += int(result.requested_action == 1)
                    executed_switches += int(result.executed_action == 1)
                    forced_switches += int(result.forced)
                    row = metrics_by_junction[junction]
                    maximum_waiting_pedestrians = max(
                        maximum_waiting_pedestrians,
                        row["wait_east_west"],
                        row["wait_north_south"],
                    )
                    writer.writerow({
                        "time_s": f"{info['time_s']:.0f}",
                        "period": period,
                        "junction": junction,
                        "phase": row["phase"],
                        "requested_action": result.requested_action,
                        "executed_action": result.executed_action,
                        "forced": result.forced,
                        "control_reason": result.reason,
                        "stopped_red_vehicles": row["stopped_red_vehicles"],
                        "idle_energy_mj": f"{row['idle_energy_mj']:.6f}",
                        "wait_c0": row["wait_c0"],
                        "wait_c1": row["wait_c1"],
                        "wait_c2": row["wait_c2"],
                        "wait_c3": row["wait_c3"],
                        "wait_east_west": row["wait_east_west"],
                        "wait_north_south": row["wait_north_south"],
                        "unresolved_pedestrians": row["unresolved_pedestrians"],
                        "affected_vehicles_for_east_west_ped": (
                            row["affected_vehicles_for_east_west_ped"]
                        ),
                        "affected_vehicles_for_north_south_ped": (
                            row["affected_vehicles_for_north_south_ped"]
                        ),
                        "clear_time_east_west_s": (
                            f"{row['clear_time_east_west_s']:.3f}"
                        ),
                        "clear_time_north_south_s": (
                            f"{row['clear_time_north_south_s']:.3f}"
                        ),
                        "predicted_energy_east_west_mj": (
                            f"{row['predicted_energy_east_west_mj']:.6f}"
                        ),
                        "predicted_energy_north_south_mj": (
                            f"{row['predicted_energy_north_south_mj']:.6f}"
                        ),
                        "predicted_pedestrian_energy_mj": (
                            f"{row['predicted_pedestrian_energy_mj']:.6f}"
                        ),
                        "total_congestion_mj": f"{row['total_congestion_mj']:.6f}",
                    })

                if terminated or truncated:
                    break
    finally:
        env.close()

    if final_info is None:
        raise RuntimeError("evaluation ended before the first controlled step")

    total_decisions = main_totals["decisions"] + cooldown_totals["decisions"]
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_total_steps": checkpoint_extra.get("total_steps", ""),
        "checkpoint_episode": checkpoint_extra.get("episode", ""),
        "seed": args.seed,
        "warmup_end_s": f"{args.warmup:.0f}",
        "demand_end_s": f"{args.demand_end:.0f}",
        "simulation_end_s": f"{final_info['time_s']:.0f}",
        "cleared": final_info["cleared"],
        "truncated": final_info["truncated"],
        "remaining_vehicles": final_info["remaining_vehicles"],
        "remaining_persons": final_info["remaining_persons"],
        "main_decisions": main_totals["decisions"],
        "cooldown_decisions": cooldown_totals["decisions"],
        "total_decisions": total_decisions,
        "main_idle_score_mj": f"{main_totals['idle']:.6f}",
        "main_prediction_score_mj": f"{main_totals['prediction']:.6f}",
        "main_congestion_score_mj": f"{main_totals['cost']:.6f}",
        "cooldown_idle_score_mj": f"{cooldown_totals['idle']:.6f}",
        "cooldown_prediction_score_mj": f"{cooldown_totals['prediction']:.6f}",
        "cooldown_congestion_score_mj": f"{cooldown_totals['cost']:.6f}",
        "total_idle_score_mj": (
            f"{main_totals['idle'] + cooldown_totals['idle']:.6f}"
        ),
        "total_prediction_score_mj": (
            f"{main_totals['prediction'] + cooldown_totals['prediction']:.6f}"
        ),
        "total_congestion_score_mj": (
            f"{main_totals['cost'] + cooldown_totals['cost']:.6f}"
        ),
        "episode_return": f"{episode_return:.6f}",
        "requested_switches": requested_switches,
        "executed_switches": executed_switches,
        "forced_switches": forced_switches,
        "max_waiting_pedestrians_at_junction": maximum_waiting_pedestrians,
    }
    with summary_path.open("w", newline="", encoding="utf-8-sig") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(summary)

    print(
        f"evaluation complete: end={summary['simulation_end_s']} s, "
        f"cleared={summary['cleared']}, "
        f"congestion_score={summary['total_congestion_score_mj']} MJ"
    )
    print(f"detail: {detail_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
