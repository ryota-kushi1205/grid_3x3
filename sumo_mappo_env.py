"""Multi-agent SUMO environment used by MAPPO.

No Gym dependency is required. Arrays have a fixed leading agent dimension:
observations [9, obs_dim], state [state_dim], and masks [9, 2].
"""

import os
import sys
from pathlib import Path

import numpy as np


if "SUMO_HOME" not in os.environ:
    raise RuntimeError("SUMO_HOME is not set")
sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci

import run_v1 as metrics
from signal_controller import SignalController


ENVIRONMENT_VERSION = "warmup300-demand-aware-v1"
WARMUP_END = 300.0
DEMAND_END = 3600.0
MAX_SIMULATION_END = 5400.0
VEHICLE_COUNT_SCALE = 30.0
PEDESTRIAN_COUNT_SCALE = 10.0
ROAD_SPEED = 13.89


class PedestrianWaitTracker:
    def __init__(self, junction_ids):
        self.junction_ids = tuple(junction_ids)
        self.wait_seconds = {}

    def reset(self):
        self.wait_seconds.clear()

    def update(self, connection, seconds=1.0):
        current = {}
        by_junction = {
            junction: {"east_west": 0.0, "north_south": 0.0}
            for junction in self.junction_ids
        }

        for junction in self.junction_ids:
            signals = metrics.get_ped_signal_states(junction)
            for walking_area in metrics.WALKING_AREAS:
                edge_id = f":{junction}_{walking_area}"
                red_crossings = [
                    crossing
                    for crossing in metrics.WALKING_AREA_CROSSINGS[walking_area]
                    if metrics.is_red(signals[crossing])
                ]
                if len(red_crossings) != 1:
                    continue
                crossing = red_crossings[0]
                group = (
                    "east_west"
                    if crossing in metrics.PEDESTRIAN_GROUPS["east_west"]
                    else "north_south"
                )
                for person_id in connection.edge.getLastStepPersonIDs(edge_id):
                    if connection.person.getSpeed(person_id) >= metrics.STOP_SPEED:
                        continue
                    key = (junction, person_id)
                    wait = self.wait_seconds.get(key, 0.0) + seconds
                    current[key] = wait
                    by_junction[junction][group] = max(
                        by_junction[junction][group],
                        wait,
                    )

        self.wait_seconds = current
        return by_junction


class SumoMAPPOEnv:
    def __init__(
        self,
        sumo_cfg=metrics.SUMO_CFG,
        use_gui=False,
        seed=42,
        warmup_end=WARMUP_END,
        demand_end=DEMAND_END,
        max_simulation_end=MAX_SIMULATION_END,
    ):
        self.sumo_cfg = str(Path(sumo_cfg).resolve())
        self.use_gui = bool(use_gui)
        self.seed = int(seed)
        self.warmup_end = float(warmup_end)
        self.demand_end = float(demand_end)
        self.max_simulation_end = float(max_simulation_end)
        self.agent_ids = tuple(metrics.INTERSECTIONS)
        self.num_agents = len(self.agent_ids)
        self.obs_dim = 31
        self.state_dim = self.num_agents * self.obs_dim
        self.action_dim = 2
        self.connection = None
        self.controller = None
        self.wait_tracker = PedestrianWaitTracker(self.agent_ids)
        self.pedestrian_waits = {
            junction: {"east_west": 0.0, "north_south": 0.0}
            for junction in self.agent_ids
        }
        self.last_metrics = {}

    def _sumo_command(self):
        binary = "sumo-gui" if self.use_gui else "sumo"
        return [
            binary,
            "-c",
            self.sumo_cfg,
            "--step-length",
            str(metrics.SIMULATION_STEP),
            "--end",
            str(self.max_simulation_end),
            "--seed",
            str(self.seed),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ]

    def reset(self, seed=None):
        self.close()
        if seed is not None:
            self.seed = int(seed)
        traci.start(self._sumo_command())
        self.connection = traci
        self.wait_tracker.reset()
        self.pedestrian_waits = {
            junction: {"east_west": 0.0, "north_south": 0.0}
            for junction in self.agent_ids
        }
        while self.connection.simulation.getTime() < self.warmup_end:
            self.connection.simulationStep()
            self.pedestrian_waits = self.wait_tracker.update(
                self.connection,
                metrics.SIMULATION_STEP,
            )

        self.controller = SignalController(
            self.connection,
            self.agent_ids,
            decision_interval=metrics.DECISION_INTERVAL,
        )
        self.controller.synchronize_from_sumo()
        self.last_metrics = self._collect_all_metrics()
        observations = self._observations()
        return observations, self._global_state(observations), self._action_masks()

    def _collect_junction_metrics(self, junction):
        vehicles = metrics.get_detector_vehicle_ids_by_direction(junction)
        direction_data = {}
        for direction, vehicle_ids in vehicles.items():
            speeds = [
                self.connection.vehicle.getSpeed(vehicle_id)
                for vehicle_id in vehicle_ids
            ]
            direction_data[direction] = {
                "count": len(vehicle_ids),
                "stopped": sum(speed < metrics.STOP_SPEED for speed in speeds),
                "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            }

        walking_counts = {
            walking_area: metrics.get_stopped_peds(junction, walking_area)
            for walking_area in metrics.WALKING_AREAS
        }
        signals = metrics.get_ped_signal_states(junction)
        crossing_counts, unresolved = metrics.infer_waiting_crossings(
            walking_counts,
            signals,
        )
        stopped_red = metrics.get_stopped_red_vehicles(junction, vehicles)
        affected = metrics.get_affected_vehicles_by_pedestrian_group(
            crossing_counts,
            vehicles,
        )
        clear_times, predicted_energy = metrics.calculate_pedestrian_prediction(
            crossing_counts,
            affected,
        )
        idle_energy_mj = (
            len(stopped_red)
            * metrics.IDLE_POWER
            * metrics.SIMULATION_STEP
            / 1_000_000.0
        )
        prediction_energy_mj = sum(predicted_energy.values()) / 1_000_000.0
        return {
            "directions": direction_data,
            "crossing_counts": crossing_counts,
            "unresolved": unresolved,
            "stopped_red_vehicles": len(stopped_red),
            "idle_energy_mj": idle_energy_mj,
            "prediction_energy_mj": prediction_energy_mj,
            "clear_times": clear_times,
            "affected_counts": {
                group: len(vehicle_ids)
                for group, vehicle_ids in affected.items()
            },
        }

    def _collect_all_metrics(self):
        return {
            junction: self._collect_junction_metrics(junction)
            for junction in self.agent_ids
        }

    def _observation_for(self, junction):
        data = self.last_metrics[junction]
        features = []
        for direction in metrics.DETECTOR_DIRECTIONS:
            features.append(
                data["directions"][direction]["count"] / VEHICLE_COUNT_SCALE
            )
        for direction in metrics.DETECTOR_DIRECTIONS:
            features.append(
                data["directions"][direction]["stopped"] / VEHICLE_COUNT_SCALE
            )
        for direction in metrics.DETECTOR_DIRECTIONS:
            features.append(
                min(data["directions"][direction]["mean_speed"] / ROAD_SPEED, 1.0)
            )
        for crossing in metrics.CROSSINGS:
            features.append(
                min(data["crossing_counts"][crossing] / PEDESTRIAN_COUNT_SCALE, 1.0)
            )
        features.append(min(data["unresolved"] / PEDESTRIAN_COUNT_SCALE, 1.0))
        waits = self.pedestrian_waits[junction]
        features.extend([
            min(waits["east_west"] / 45.0, 1.0),
            min(waits["north_south"] / 45.0, 1.0),
        ])

        phase = self.controller.phase(junction)
        features.extend([1.0 if phase == index else 0.0 for index in range(6)])
        features.append(min(self.controller.stable_elapsed[junction] / 45.0, 1.0))
        previous_action = self.controller.previous_action[junction]
        features.extend([1.0 if previous_action == action else 0.0 for action in range(2)])
        row = ord(junction[0]) - ord("A")
        column = int(junction[1])
        features.extend([row / 2.0, column / 2.0])
        features.append(
            float(
                self.controller.action_mask(
                    junction,
                    self._phase_demands(junction),
                )[1]
            )
        )

        observation = np.asarray(features, dtype=np.float32)
        if observation.shape != (self.obs_dim,):
            raise RuntimeError(
                f"observation shape mismatch: {observation.shape}, expected {(self.obs_dim,)}"
            )
        return observation

    def _observations(self):
        return np.stack(
            [self._observation_for(junction) for junction in self.agent_ids]
        )

    def _global_state(self, observations):
        return observations.reshape(-1).astype(np.float32, copy=False)

    def _phase_demands(self, junction):
        data = self.last_metrics[junction]
        directions = data["directions"]
        crossings = data["crossing_counts"]
        waits = self.pedestrian_waits[junction]
        phase_0_demand = (
            directions["north"]["count"] > 0
            or directions["south"]["count"] > 0
            or crossings["c1"] > 0
            or crossings["c3"] > 0
            or waits["north_south"] > 0.0
        )
        phase_3_demand = (
            directions["east"]["count"] > 0
            or directions["west"]["count"] > 0
            or crossings["c0"] > 0
            or crossings["c2"] > 0
            or waits["east_west"] > 0.0
        )
        return {0: phase_0_demand, 3: phase_3_demand}

    def _action_masks(self):
        return np.asarray(
            [
                self.controller.action_mask(
                    junction,
                    self._phase_demands(junction),
                )
                for junction in self.agent_ids
            ],
            dtype=np.bool_,
        )

    def _is_cleared(self):
        if self.connection.simulation.getTime() < self.demand_end:
            return False
        return (
            self.connection.simulation.getMinExpectedNumber() == 0
            and len(self.connection.vehicle.getIDList()) == 0
            and len(self.connection.person.getIDList()) == 0
        )

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_agents,):
            raise ValueError(
                f"actions shape must be {(self.num_agents,)}, got {actions.shape}"
            )

        control_results = {}
        for index, junction in enumerate(self.agent_ids):
            control_results[junction] = self.controller.apply_action(
                junction,
                int(actions[index]),
                self._phase_demands(junction),
            )

        terminated = False
        truncated = False
        executed_seconds = 0
        for _ in range(metrics.DECISION_INTERVAL):
            self.connection.simulationStep()
            executed_seconds += 1
            self.controller.update_elapsed(metrics.SIMULATION_STEP)
            self.pedestrian_waits = self.wait_tracker.update(
                self.connection,
                metrics.SIMULATION_STEP,
            )
            self.last_metrics = self._collect_all_metrics()

            terminated = self._is_cleared()
            truncated = (
                self.connection.simulation.getTime() >= self.max_simulation_end
                and not terminated
            )
            if terminated or truncated:
                break

        endpoint_idle_mj = sum(
            data["idle_energy_mj"] for data in self.last_metrics.values()
        )
        endpoint_prediction_mj = sum(
            data["prediction_energy_mj"] for data in self.last_metrics.values()
        )
        network_cost_mj = endpoint_idle_mj + endpoint_prediction_mj
        team_reward = -network_cost_mj
        rewards = np.full(self.num_agents, team_reward, dtype=np.float32)
        observations = self._observations()
        info = {
            "time_s": self.connection.simulation.getTime(),
            "executed_seconds": executed_seconds,
            "network_idle_energy_mj": endpoint_idle_mj,
            "network_prediction_energy_mj": endpoint_prediction_mj,
            "network_cost_mj": network_cost_mj,
            "cleared": terminated,
            "truncated": truncated,
            "remaining_vehicles": len(self.connection.vehicle.getIDList()),
            "remaining_persons": len(self.connection.person.getIDList()),
            "control_results": control_results,
            "junction_metrics": self.evaluation_rows(),
        }
        return (
            observations,
            self._global_state(observations),
            rewards,
            terminated,
            truncated,
            self._action_masks(),
            info,
        )

    def evaluation_rows(self):
        rows = []
        for junction in self.agent_ids:
            data = self.last_metrics[junction]
            crossings = data["crossing_counts"]
            rows.append({
                "junction": junction,
                "phase": self.controller.phase(junction),
                "stopped_red_vehicles": data["stopped_red_vehicles"],
                "idle_energy_mj": data["idle_energy_mj"],
                "wait_c0": crossings["c0"],
                "wait_c1": crossings["c1"],
                "wait_c2": crossings["c2"],
                "wait_c3": crossings["c3"],
                "wait_east_west": crossings["c0"] + crossings["c2"],
                "wait_north_south": crossings["c1"] + crossings["c3"],
                "unresolved_pedestrians": data["unresolved"],
                "affected_vehicles_for_east_west_ped": (
                    data["affected_counts"]["east_west"]
                ),
                "affected_vehicles_for_north_south_ped": (
                    data["affected_counts"]["north_south"]
                ),
                "clear_time_east_west_s": data["clear_times"]["east_west"],
                "clear_time_north_south_s": data["clear_times"]["north_south"],
                "predicted_energy_east_west_mj": (
                    data["affected_counts"]["east_west"]
                    * (
                        metrics.IDLE_POWER * data["clear_times"]["east_west"]
                        + metrics.RESTART_ENERGY
                    )
                    / 1_000_000.0
                ),
                "predicted_energy_north_south_mj": (
                    data["affected_counts"]["north_south"]
                    * (
                        metrics.IDLE_POWER * data["clear_times"]["north_south"]
                        + metrics.RESTART_ENERGY
                    )
                    / 1_000_000.0
                ),
                "predicted_pedestrian_energy_mj": data["prediction_energy_mj"],
                "total_congestion_mj": (
                    data["idle_energy_mj"] + data["prediction_energy_mj"]
                ),
            })
        return rows

    def close(self):
        if self.connection is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
