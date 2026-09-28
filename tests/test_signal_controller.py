import unittest

from signal_controller import SignalController


class FakeTrafficLight:
    def __init__(self):
        self.phases = {"A0": 0}
        self.durations = {}
        self.spent_durations = {"A0": 0.0}

    def getPhase(self, junction):
        return self.phases[junction]

    def setPhase(self, junction, phase):
        self.phases[junction] = phase

    def setPhaseDuration(self, junction, duration):
        self.durations[junction] = duration

    def getSpentDuration(self, junction):
        return self.spent_durations[junction]


class FakeTraci:
    def __init__(self):
        self.trafficlight = FakeTrafficLight()


class SignalControllerTest(unittest.TestCase):
    def setUp(self):
        self.traci = FakeTraci()
        self.controller = SignalController(self.traci, ["A0"])
        self.controller.reset()

    def test_switch_is_blocked_before_minimum_green(self):
        result = self.controller.apply_action(
            "A0",
            1,
            {0: True, 3: True},
        )
        self.assertEqual(result.executed_action, 0)
        self.assertEqual(self.traci.trafficlight.getPhase("A0"), 0)

    def test_switch_uses_safe_transition_phase(self):
        self.controller.stable_elapsed["A0"] = 15.0
        result = self.controller.apply_action(
            "A0",
            1,
            {0: True, 3: True},
        )
        self.assertEqual(result.executed_action, 1)
        self.assertEqual(self.traci.trafficlight.getPhase("A0"), 1)

    def test_maximum_green_forces_switch_with_opposing_demand(self):
        self.controller.stable_elapsed["A0"] = 45.0
        result = self.controller.apply_action(
            "A0",
            0,
            {0: True, 3: True},
        )
        self.assertTrue(result.forced)
        self.assertEqual(result.executed_action, 1)

    def test_maximum_green_does_not_force_without_opposing_demand(self):
        self.controller.stable_elapsed["A0"] = 45.0
        result = self.controller.apply_action(
            "A0",
            0,
            {0: True, 3: False},
        )
        self.assertFalse(result.forced)
        self.assertEqual(result.executed_action, 0)

    def test_transition_masks_switch(self):
        self.traci.trafficlight.phases["A0"] = 2
        self.assertEqual(
            self.controller.action_mask(
                "A0",
                {0: True, 3: True},
            ),
            (True, False),
        )

    def test_synchronize_keeps_warmup_phase_and_elapsed_time(self):
        self.traci.trafficlight.phases["A0"] = 3
        self.traci.trafficlight.spent_durations["A0"] = 22.0
        self.controller.synchronize_from_sumo()
        self.assertEqual(self.controller.phase("A0"), 3)
        self.assertEqual(self.controller.stable_elapsed["A0"], 22.0)


if __name__ == "__main__":
    unittest.main()
