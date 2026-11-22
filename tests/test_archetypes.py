"""Property and hand-worked tests for the payoff functions."""
import numpy as np
import pytest

from railca import archetypes as arc

RNG = np.random.default_rng(7)
N = 2000
FARE = RNG.uniform(20, 300, N)
NONFARE = RNG.uniform(0, 40, N)
OPEX = RNG.uniform(50, 300, N)
CHARGE = RNG.uniform(-50, 80, N)
SUBSIDY = RNG.uniform(0, 30, N)
BANDS = dict(deadband=0.05, shortfall_share=0.5, band1=0.05, rate1=0.5, band2=0.08, rate2=0.95)


def hybrid(fare, nonfare, opex, charge, proj=150.0, subsidy=0.0, **kw):
    b = dict(BANDS, **kw)
    return arc.hybrid(fare, nonfare, opex, charge, proj, b["deadband"], b["shortfall_share"], b["band1"], b["rate1"],
                      b["band2"], b["rate2"], subsidy)


# --- accounting identity: operator + authority == system surplus --------------------------------
def test_identity_net_cost():
    op, au = arc.net_cost(FARE, NONFARE, OPEX, CHARGE, SUBSIDY)
    assert np.allclose(op + au, arc.system_surplus(FARE, NONFARE, OPEX), atol=1e-9)


def test_identity_gross_cost():
    fee = RNG.uniform(50, 300, N)
    inc = arc.incentive_payment(fee, RNG.normal(0, 2, N), 0.01, 0.02, 0.03)
    op, au = arc.gross_cost(FARE, NONFARE, OPEX, fee, inc)
    assert np.allclose(op + au, arc.system_surplus(FARE, NONFARE, OPEX), atol=1e-9)


def test_identity_hybrid():
    op, au = hybrid(FARE, NONFARE, OPEX, CHARGE, proj=RNG.uniform(20, 300, N), subsidy=SUBSIDY)
    assert np.allclose(op + au, arc.system_surplus(FARE, NONFARE, OPEX), atol=1e-9)


def test_identity_concession():
    op, au = arc.concession(FARE, NONFARE, OPEX, 8.0, CHARGE, 0.03)
    assert np.allclose(op + au, arc.system_surplus(FARE, NONFARE, OPEX, 8.0), atol=1e-9)


def test_system_surplus_by_hand():
    assert arc.system_surplus(100.0, 10.0, 80.0) == 30.0
    assert arc.system_surplus(100.0, 10.0, 80.0, 5.0) == 25.0


# --- hand-worked values ---------------------------------------------------------------------------
def test_net_cost_by_hand():
    assert arc.net_cost(100.0, 10.0, 80.0, 20.0, 3.0) == (13.0, 17.0)


def test_gross_cost_by_hand():
    assert arc.gross_cost(100.0, 10.0, 80.0, 90.0, 2.0) == (12.0, 18.0)


def test_concession_by_hand():
    op, au = arc.concession(100.0, 10.0, 80.0, 8.0, 5.0, 0.1)
    assert (op, au) == (pytest.approx(7.0), pytest.approx(15.0))


def test_hybrid_by_hand_shortfall():
    # projection 150, deadband 5% -> threshold 142.5; fare 100 -> relief 0.5 * 42.5 = 21.25
    op, au = hybrid(100.0, 10.0, 100.0, 10.0)
    assert arc.shortfall_relief(100.0, 150.0, 0.05, 0.5) == pytest.approx(21.25)
    # pre = 110 - 100 - 10 + 21.25 = 21.25; bands 5.5 and 8.8 of revenue 110
    # share = 0.5 * (8.8 - 5.5) + 0.95 * (21.25 - 8.8) = 1.65 + 11.8275
    assert op == pytest.approx(21.25 - 13.4775)
    assert au == pytest.approx(10.0 - 21.25 + 13.4775)


def test_hybrid_by_hand_with_subsidy():
    op, au = hybrid(150.0, 0.0, 150.0, 0.0, subsidy=4.0)   # pre = 4 < band1 * 150 = 7.5 -> no share
    assert (op, au) == (pytest.approx(4.0), pytest.approx(-4.0))


# --- band boundaries --------------------------------------------------------------------------------
def test_relief_zero_at_and_above_threshold():
    assert arc.shortfall_relief(142.5, 150.0, 0.05, 0.5) == 0.0
    assert arc.shortfall_relief(150.0, 150.0, 0.05, 0.5) == 0.0
    assert arc.shortfall_relief(142.4, 150.0, 0.05, 0.5) == pytest.approx(0.05)


def test_profit_share_at_boundaries():
    rev = 200.0   # bands at 10 and 16
    assert arc.profit_share(10.0, rev, 0.05, 0.5, 0.08, 0.95) == 0.0
    assert arc.profit_share(9.0, rev, 0.05, 0.5, 0.08, 0.95) == 0.0
    assert arc.profit_share(-30.0, rev, 0.05, 0.5, 0.08, 0.95) == 0.0
    assert arc.profit_share(13.0, rev, 0.05, 0.5, 0.08, 0.95) == pytest.approx(1.5)
    assert arc.profit_share(16.0, rev, 0.05, 0.5, 0.08, 0.95) == pytest.approx(3.0)
    assert arc.profit_share(20.0, rev, 0.05, 0.5, 0.08, 0.95) == pytest.approx(3.0 + 3.8)


def test_profit_share_is_continuous_at_the_bands():
    rev, eps = 200.0, 1e-7
    for edge in (10.0, 16.0):
        lo = arc.profit_share(edge - eps, rev, 0.05, 0.5, 0.08, 0.95)
        hi = arc.profit_share(edge + eps, rev, 0.05, 0.5, 0.08, 0.95)
        assert abs(hi - lo) < 1e-6


def test_incentive_clipped_at_caps():
    assert arc.incentive_payment(100.0, 0.0, 0.01, 0.02, 0.03) == 0.0
    assert arc.incentive_payment(100.0, 1.0, 0.01, 0.02, 0.03) == pytest.approx(1.0)
    assert arc.incentive_payment(100.0, 2.0, 0.01, 0.02, 0.03) == pytest.approx(2.0)
    assert arc.incentive_payment(100.0, 5.0, 0.01, 0.02, 0.03) == pytest.approx(2.0)
    assert arc.incentive_payment(100.0, -3.0, 0.01, 0.02, 0.03) == pytest.approx(-3.0)
    assert arc.incentive_payment(100.0, -9.0, 0.01, 0.02, 0.03) == pytest.approx(-3.0)


# --- monotonicity -------------------------------------------------------------------------------------
def test_operator_payoff_never_falls_with_revenue():
    fare = np.linspace(20, 400, 800)
    for op in (arc.net_cost(fare, 10.0, 120.0, 20.0)[0], hybrid(fare, 10.0, 120.0, 20.0)[0],
               arc.concession(fare, 10.0, 120.0, 8.0, 5.0, 0.05)[0]):
        assert np.all(np.diff(op) > 0)
    assert np.ndim(arc.gross_cost(fare, 10.0, 120.0, 130.0)[0]) == 0      # does not depend on revenue at all


def test_authority_payoff_never_falls_with_revenue():
    fare = np.linspace(20, 400, 800)
    assert np.all(np.diff(hybrid(fare, 10.0, 120.0, 20.0)[1]) >= -1e-12)
    assert np.all(np.diff(arc.gross_cost(fare, 10.0, 120.0, 130.0)[1]) > 0)
    assert np.ndim(arc.net_cost(fare, 10.0, 120.0, 20.0)[1]) == 0


def test_operator_payoff_falls_with_cost_everywhere():
    opex = np.linspace(50, 300, 600)
    for op in (arc.net_cost(150.0, 10.0, opex, 20.0)[0], arc.gross_cost(150.0, 10.0, opex, 130.0)[0],
               hybrid(150.0, 10.0, opex, 20.0)[0], arc.concession(150.0, 10.0, opex, 8.0, 5.0, 0.0)[0]):
        assert np.all(np.diff(op) < 0)


def test_hybrid_narrows_the_operator_range_relative_to_net_cost():
    fare = np.array([90.0, 150.0, 210.0])
    a = arc.net_cost(fare, 10.0, 130.0, 20.0)[0]
    c = hybrid(fare, 10.0, 130.0, 20.0)[0]
    assert c[0] > a[0] and c[2] < a[2]
    assert (c[2] - c[0]) < (a[2] - a[0])


def test_higher_charge_moves_money_one_for_one():
    a0, g0 = arc.net_cost(150.0, 10.0, 130.0, 20.0)
    a1, g1 = arc.net_cost(150.0, 10.0, 130.0, 21.0)
    assert (a1 - a0, g1 - g0) == (-1.0, 1.0)
    b0, h0 = arc.gross_cost(150.0, 10.0, 130.0, 140.0)
    b1, h1 = arc.gross_cost(150.0, 10.0, 130.0, 141.0)
    assert (b1 - b0, h1 - h0) == (1.0, -1.0)
