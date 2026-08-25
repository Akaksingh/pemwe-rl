"""Plant-model tests. OWNER: Person A.

The four the brief names, plus the invariants that would silently corrupt every number
downstream if they broke.
"""

import numpy as np
import pytest


# --- the brief's item 1: j_from_power must invert power_w --------------------------

@pytest.mark.parametrize("frac", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_j_from_power_inverts_power_w(stack, frac):
    p = frac * stack.p_rated
    if p < stack.p_min_on:
        pytest.skip("below the ON threshold")
    j = stack.j_from_power(p)
    assert stack.j_min <= j <= stack.j_rated
    assert stack.power_w(j) == pytest.approx(p, rel=1e-6)


def test_full_power_command_reaches_exactly_rated(stack):
    """P_rated must include BoP, or an action of +1 can never reach j_rated.

    The Day-0 stub defined P_rated as stack-only power, so power_w(j_rated) was
    1.02 x P_rated and rated current density was unreachable.
    """
    assert stack.power_w(stack.j_rated) == pytest.approx(stack.p_rated, rel=1e-9)
    assert stack.j_from_power(stack.p_rated) == pytest.approx(stack.j_rated, rel=1e-9)


def test_power_never_exceeds_the_commanded_setpoint(stack):
    """DECISIONS.md 3 is a HARD cap: the plant may never draw more than is available."""
    for p in np.linspace(0.0, stack.p_rated, 400):
        j = stack.j_from_power(float(p))
        drawn = float(stack.power_w(j)) if j > 0 else 0.0
        assert drawn <= p + 1e-6, f"drew {drawn:.1f} W from an available {p:.1f} W"


def test_power_is_monotonic_in_j(stack):
    """The bisection in j_from_power is only valid if this holds."""
    j = np.linspace(stack.j_min, stack.j_rated, 500)
    assert np.all(np.diff(stack.power_w(j)) > 0)


# --- the brief's item 2: efficiency must be non-monotonic (Gate G1.1) --------------

def test_efficiency_is_non_monotonic_and_peaks_mid_range(stack):
    j, v, eta, pf = stack.efficiency_curve(600)
    k = int(np.argmax(eta))
    assert eta[k] > eta[0], "efficiency must rise from the low-j end"
    assert eta[k] > eta[-1], "efficiency must fall again toward rated"
    assert 0.5 <= j[k] <= 0.8, f"peak at j={j[k]:.3f}, DECISIONS.md 7 wants 0.5-0.8"
    assert 0.25 <= j[k] / stack.j_rated <= 0.40


def test_max_h2_rate_and_max_h2_per_joule_are_different_setpoints(stack):
    """This IS the control problem. If these coincide there is nothing to learn."""
    j, v, eta, pf = stack.efficiency_curve(600)
    rate = stack.h2_rate_kg_s(j)
    assert int(np.argmax(rate)) == len(j) - 1, "max H2 rate should be at rated"
    assert int(np.argmax(eta)) < len(j) - 1, "max H2 per joule must NOT be at rated"


def test_crossover_collapses_faradaic_efficiency_at_low_j(stack):
    assert stack.faradaic_efficiency(stack.j_rated) > 0.98
    assert stack.faradaic_efficiency(0.1) < 0.9
    assert stack.faradaic_efficiency(stack.j_min) < 0.75


def test_cell_voltage_in_published_range_and_increasing(stack):
    j = np.linspace(stack.j_min, stack.j_rated, 500)
    v = stack.v_cell(j)
    assert np.all(np.diff(v) > 0)
    assert 1.75 <= float(stack.v_cell(stack.j_rated)) <= 1.85
    assert float(stack.v_cell(stack.j_min)) > stack.e_rev()


def test_tafel_slopes_match_reported_electrocatalysis(stack):
    """alpha is only defensible through the Tafel slope it implies at 60 C."""
    b_a, b_c = stack.tafel_slopes_mv_per_dec()
    assert 40.0 <= b_a <= 60.0, f"anode {b_a:.1f} mV/dec outside IrO2 OER range"
    assert 25.0 <= b_c <= 40.0, f"cathode {b_c:.1f} mV/dec outside Pt HER range"


def test_membrane_thickness_drives_both_asr_and_crossover(cfg):
    """The two ends of the efficiency curve must not be independently tunable."""
    from pemwe import StackModel
    thin = dict(cfg); thin["stack"] = dict(cfg["stack"], membrane_thickness_cm=0.0035)
    a, b = StackModel(cfg), StackModel(thin)
    assert b.asr < a.asr, "a thinner membrane must lower ohmic loss"
    assert b.j_cross > a.j_cross, "...and must raise crossover. Both, or the trade is fake"


def test_degradation_raises_voltage_and_costs_yield(stack):
    """dv_deg has to reach the polarization curve, or there is no long-term tradeoff."""
    j = 1.0
    v0 = float(stack.v_cell(j, 0.0))
    v1 = float(stack.v_cell(j, 0.177))          # end of life, 10% rise
    assert v1 == pytest.approx(v0 + 0.177, abs=1e-9)
    # at fixed POWER, a degraded stack reaches a lower current and makes less H2
    p = 0.5 * stack.p_rated
    h2_new = stack.h2_rate_kg_s(stack.j_from_power(p, 0.0))
    h2_old = stack.h2_rate_kg_s(stack.j_from_power(p, 0.177), 0.177)
    assert h2_old < h2_new
