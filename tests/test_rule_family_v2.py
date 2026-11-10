"""Unit tests for the hysteresis rule: pure logic, no simulator needed."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "rule_family_v2", Path(__file__).resolve().parents[1] / "scripts" / "rule_family_v2.py")
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)


def run(rule, avail):
    """Commanded fraction of AVAILABLE power per step."""
    return np.array([0.5 * (float(rule.act(np.array([p]))[0]) + 1.0) for p in avail])


def test_stays_off_until_the_resource_has_been_high_enough_for_k_up_steps():
    rule = rf.HysteresisRule(p_on=0.1, k_up=5, k_down=5, slew=1.0, cap=1.0)
    u = run(rule, [0.2] * 4 + [0.2])           # 4 steps: still off, 5th step switches on
    assert (u[:4] == 0).all() and u[4] > 0


def test_a_single_dip_does_not_switch_it_off_but_a_long_dip_does():
    rule = rf.HysteresisRule(p_on=0.1, k_up=1, k_down=5, slew=1.0, cap=1.0)
    run(rule, [0.3] * 3)
    assert rule.on
    run(rule, [0.01] * 4)                       # 4 low steps < k_down
    assert rule.on
    run(rule, [0.01])                           # 5th low step
    assert not rule.on


def test_cap_and_availability_bound_the_command():
    rule = rf.HysteresisRule(p_on=0.1, k_up=1, k_down=5, slew=1.0, cap=0.5)
    avail = [0.9] * 5
    u = run(rule, avail)
    assert np.allclose(u * np.array(avail), 0.5)     # command = cap, expressed as a fraction of 0.9
    u2 = run(rf.HysteresisRule(p_on=0.1, k_up=1, k_down=5, slew=1.0, cap=0.5), [0.3] * 5)
    assert np.allclose(u2, 1.0)                      # resource below the cap: take everything


def test_slew_limit_ramps_the_command_up_gradually():
    rule = rf.HysteresisRule(p_on=0.1, k_up=1, k_down=5, slew=0.01, cap=1.0)
    avail = [0.5] * 6
    cmd = run(rule, avail) * np.array(avail)         # commanded power as a fraction of rated
    assert np.allclose(np.diff(cmd), 0.01, atol=1e-6)


def test_the_command_never_exceeds_the_resource():
    rng = np.random.default_rng(0)
    rule = rf.HysteresisRule(p_on=0.08, k_up=5, k_down=5, slew=0.003, cap=0.7)
    avail = np.clip(rng.normal(0.3, 0.2, 2000), 0, 1)
    u = run(rule, avail)
    assert (u >= 0).all() and (u <= 1 + 1e-6).all()
