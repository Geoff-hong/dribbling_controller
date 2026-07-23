import math
import unittest

from sim2sim_benchmark import engine
from sim2sim_benchmark.conditions import condition_row
from sim2sim_benchmark.html_report import condition_stats, nested_strict_success
from sim2sim_benchmark.topup import PROTOCOL_VERSION


class LatencyProtocolTest(unittest.TestCase):
    def test_nominal_matches_cpp_sim2sim_without_synthetic_latency(self):
        nominal = condition_row("nominal", "baseline", 0.0)
        self.assertEqual(nominal["ball_obs_delay_steps"], 0)
        self.assertEqual(nominal["action_delay_ms"], 0)

    def test_latency_axes_only_change_their_own_channel(self):
        obs = condition_row("obslag_2", "obs_latency", 2,
                            ball_obs_delay_steps=2)
        act = condition_row("actlag_10", "act_latency", 10,
                            action_delay_ms=10)
        self.assertEqual((obs["ball_obs_delay_steps"], obs["action_delay_ms"]),
                         (2, 0))
        self.assertEqual((act["ball_obs_delay_steps"], act["action_delay_ms"]),
                         (0, 10))

    def test_protocol_was_bumped_for_incomparable_nominal(self):
        self.assertGreaterEqual(PROTOCOL_VERSION, 5)

    def test_every_condition_carries_the_cpp_bridge_staleness(self):
        # the 100 Hz topic hop is structural to the C++ stack, so it must be on
        # for the baseline AND under every latency axis (synthetic lag stacks on
        # top of it, it does not replace it)
        for cond in (condition_row("nominal", "baseline", 0.0),
                     condition_row("obslag_2", "obs_latency", 2, ball_obs_delay_steps=2),
                     condition_row("actlag_10", "act_latency", 10, action_delay_ms=10)):
            self.assertEqual(cond["bridge_delay_ms"], 10.0)

    def test_bridge_delay_resolves_to_two_substeps(self):
        # 10 ms at the MJCF's physics step = 2 sub-steps; a timestep change
        # would silently redefine what "one publish stale" means
        import re
        with open(engine.SINGLE_MJCF) as f:
            dt = float(re.search(r'timestep="([\d.]+)"', f.read()).group(1))
        self.assertEqual(int(round(10.0 / (dt * 1000.0))), 2)


class RouteParityTest(unittest.TestCase):
    def test_route_rng_matches_libstdcpp_mt19937_uniform_distribution(self):
        # Produced by GCC/libstdc++ std::mt19937(42) followed by
        # std::uniform_real_distribution<double>(0, 1).
        expected = [
            0.79654298428784598,
            0.18343478789336848,
            0.77969099761266125,
            0.59685016158005655,
        ]
        rng = engine.CppMt19937Uniform(42)
        self.assertEqual([rng.uniform(0.0, 1.0) for _ in expected], expected)

    def test_human_route_uses_cpp_lazy_initial_fill(self):
        cfg = dict(engine.ROUTE_CFG, routeLength=10.0)
        route = engine.Route(cfg, seed=0)
        route.reset([0.0, 0.0], [1.0, 0.0], cmd_mode=4)
        self.assertEqual(route.filled, cfg["routeInitSegments"])

    def test_python_only_route_governors_are_not_in_cpp_parity_config(self):
        self.assertNotIn("routeHumanCumCapDeg", engine.ROUTE_CFG)
        self.assertNotIn("routeMinClearanceM", engine.ROUTE_CFG)


class CapabilityVerdictTest(unittest.TestCase):
    def test_verdicts_are_always_nested(self):
        cases = [
            (False, True, "", math.nan),
            (False, True, "", 0.0),
            (False, True, "", 1.0),
            (False, True, "off_route", math.nan),
            (False, False, "", math.nan),
            (True, True, "fell", math.nan),
        ]
        for args in cases:
            possession, route, strict = engine.capability_success_verdicts(*args)
            self.assertGreaterEqual(possession, route, args)
            self.assertGreaterEqual(route, strict, args)

    def test_lost_ball_cannot_be_strict_success(self):
        self.assertEqual(
            engine.capability_success_verdicts(False, False, "", math.nan),
            (0.0, 0.0, 0.0),
        )

    def test_unfinished_arc_is_not_strict_success(self):
        self.assertEqual(
            engine.capability_success_verdicts(False, True, "", 0.0),
            (1.0, 1.0, 0.0),
        )

    def test_report_repairs_protocol_two_strict_success(self):
        self.assertEqual(nested_strict_success(1.0, 0.0, None), 0.0)
        self.assertEqual(nested_strict_success(1.0, 1.0, 0.0), 0.0)
        self.assertEqual(nested_strict_success(0.0, 1.0, None), 1.0)


def _report_row(*, success, ct):
    return dict(
        fell=0.0,
        ball_lost=0.0,
        ball_lost_05=0.0,
        foot_ball_dist=0.1,
        success=success,
        success_possession=1.0,
        success_route=success,
        ach=1.0,
        cmd=1.0,
        ct=ct,
        duration=15.0,
        progress=10.0,
        ball_dist=0.5,
        min_z=0.7,
        reason="" if success else "off_route",
        completed=None,
    )


class CrossTrackAggregationTest(unittest.TestCase):
    def test_capability_reports_success_conditioned_and_censored_ct_separately(self):
        point = condition_stats([
            _report_row(success=1.0, ct=0.1),
            _report_row(success=0.0, ct=0.9),
        ], fail_fast=True)

        self.assertEqual(point["success"], 50.0)
        self.assertEqual(point["cross_track_success"], 0.1)
        self.assertEqual(point["cross_track_success_n"], 1)
        self.assertEqual(point["cross_track"], 0.5)
        self.assertEqual(point["cross_track_n"], 2)


if __name__ == "__main__":
    unittest.main()
