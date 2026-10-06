"""Safe two-action traffic-light controller for the 3x3 network."""

from dataclasses import dataclass


STABLE_PHASES = (0, 3)
TRANSITION_PHASES = (1, 2, 4, 5)
DEFAULT_MAX_GREEN = 60.0


@dataclass
class ControllerResult:
    requested_action: int
    executed_action: int
    forced: bool
    reason: str


class SignalController:
    """Convert hold/switch actions into the network's six safe SUMO phases.

    Phase 0 serves north-south vehicles and north-south pedestrians (c1/c3).
    Phase 3 serves east-west vehicles and east-west pedestrians (c0/c2).
    SUMO advances 1 -> 2 -> 3 and 4 -> 5 -> 0 using the durations in the net.
    """

    def __init__(
        self,
        traci_connection,
        junction_ids,
        decision_interval=5.0,
        min_green=15.0,
        max_green=DEFAULT_MAX_GREEN,
    ):
        self.traci = traci_connection
        self.junction_ids = tuple(junction_ids)
        self.decision_interval = float(decision_interval)
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.stable_elapsed = {junction: 0.0 for junction in self.junction_ids}
        self.previous_phase = {junction: None for junction in self.junction_ids}
        self.previous_action = {junction: 0 for junction in self.junction_ids}

    def reset(self, initial_phase=0):
        if initial_phase not in STABLE_PHASES:
            raise ValueError(f"initial_phase must be 0 or 3, got {initial_phase}")
        for junction in self.junction_ids:
            self.traci.trafficlight.setPhase(junction, initial_phase)
            self.traci.trafficlight.setPhaseDuration(junction, self.min_green)
            self.stable_elapsed[junction] = 0.0
            self.previous_phase[junction] = initial_phase
            self.previous_action[junction] = 0

    def synchronize_from_sumo(self):
        """Take control without changing the phase reached during warm-up."""
        for junction in self.junction_ids:
            phase = self.phase(junction)
            self.previous_phase[junction] = phase
            self.previous_action[junction] = 0
            self.stable_elapsed[junction] = (
                float(self.traci.trafficlight.getSpentDuration(junction))
                if phase in STABLE_PHASES
                else 0.0
            )

    def update_elapsed(self, seconds=1.0):
        for junction in self.junction_ids:
            phase = self.phase(junction)
            previous = self.previous_phase[junction]
            if phase in STABLE_PHASES:
                if phase == previous:
                    self.stable_elapsed[junction] += seconds
                else:
                    self.stable_elapsed[junction] = seconds
            else:
                self.stable_elapsed[junction] = 0.0
            self.previous_phase[junction] = phase

    def phase(self, junction):
        return int(self.traci.trafficlight.getPhase(junction))

    def action_mask(self, junction, phase_demands):
        """Return [hold_allowed, switch_allowed]."""
        phase = self.phase(junction)
        if phase not in STABLE_PHASES:
            return (True, False)

        elapsed = self.stable_elapsed[junction]
        opposite_phase = 3 if phase == 0 else 0
        forced = elapsed >= self.max_green and bool(
            phase_demands.get(opposite_phase, False)
        )
        if forced:
            return (False, True)
        return (True, elapsed >= self.min_green)

    def apply_action(self, junction, action, phase_demands):
        if action not in (0, 1):
            raise ValueError(f"action must be 0 (hold) or 1 (switch), got {action}")

        mask = self.action_mask(junction, phase_demands)
        forced = not mask[0] and mask[1]
        if forced:
            executed = 1
            reason = "maximum green with opposing demand"
        elif not mask[action]:
            executed = 0
            reason = "switch blocked during transition or minimum green"
        else:
            executed = action
            reason = "policy action"

        phase = self.phase(junction)
        if executed == 1 and phase in STABLE_PHASES:
            transition_phase = 1 if phase == 0 else 4
            self.traci.trafficlight.setPhase(junction, transition_phase)
        elif executed == 0 and phase in STABLE_PHASES:
            # Keep the stable phase alive until the next decision.
            self.traci.trafficlight.setPhaseDuration(
                junction,
                self.decision_interval + 0.1,
            )

        self.previous_action[junction] = executed
        return ControllerResult(action, executed, forced, reason)
