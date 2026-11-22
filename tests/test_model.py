"""Tests for driver paths, calibration, simulation metrics, risk allocation, RTS arithmetic and procurement."""
import numpy as np
import pytest

from railca import analysis, params, paths, procurement, rts

P = params.base_params()
TERMS = analysis.calibrate(P)


# --- paths ----------------------------------------------------------------------------------------------
def test_regulated_fares_by_hand():
    g = np.array([[0.0, 0.08, 0.01, 0.00, 0.02]])
    idx, deferred = paths.regulated_fares(g, 0.05)
    assert deferred[0] == pytest.approx([0.0, 0.03, 0.0, 0.0, 0.0])
    assert idx[0] == pytest.approx([1.0, 1.05, 1.05 * 1.04, 1.05 * 1.04, 1.05 * 1.04 * 1.02])


def test_regulated_fares_exactly_at_cap_defers_nothing():
    idx, deferred = paths.regulated_fares(np.array([[0.0, 0.05, 0.05]]), 0.05)
    assert deferred[0] == pytest.approx([0.0, 0.0, 0.0])
    assert idx[0, 2] == pytest.approx(1.05 ** 2)


def test_regulated_fares_pass_decreases_through():
    idx, deferred = paths.regulated_fares(np.array([[0.0, -0.02]]), 0.05)
    assert idx[0, 1] == pytest.approx(0.98) and deferred[0, 1] == 0.0


def test_pandemic_multiplier_by_hand():
    starts = np.zeros((2, 8), dtype=bool)
    starts[0, 2] = True
    starts[0, 3] = True      # ignored: an event is already running
    starts[1, 6] = True
    m = paths.pandemic_multiplier(starts, [0.6, 0.7, 0.9])
    assert m[0] == pytest.approx([1, 1, 0.6, 0.7, 0.9, 1, 1, 1])
    assert m[1] == pytest.approx([1, 1, 1, 1, 1, 1, 0.6, 0.7])


def test_pandemic_can_restart_after_recovery():
    starts = np.zeros((1, 6), dtype=bool)
    starts[0, 0] = starts[0, 2] = True
    assert paths.pandemic_multiplier(starts, [0.5, 0.8])[0] == pytest.approx([0.5, 0.8, 0.5, 0.8, 1, 1])


def test_deterministic_path_by_hand():
    d = paths.deterministic(P)
    cost_g = 0.5 * 0.02 + 0.4 * 0.035 + 0.1 * 0.02
    formula = cost_g - 0.001 + 0.011
    assert d["cpi_idx"][0, :3] == pytest.approx([1.0, 1.02, 1.02 ** 2])
    assert d["wage_idx"][0, 2] == pytest.approx(1.035 ** 2)
    assert d["cost_idx"][0, 3] == pytest.approx((1 + cost_g) ** 3)
    assert d["fare_regulated"][0, 3] == pytest.approx(P["fare"] * (1 + formula) ** 3)
    assert d["fare_free"][0, 3] == pytest.approx(P["fare"] * (1 + cost_g) ** 3)
    assert d["trips"][0, 4] == pytest.approx(P["trips_m"] * 1.01 ** 4)
    assert d["deferred"].max() == 0.0
    assert d["opex"][0, 0] == pytest.approx(P["opex_m"])
    assert d["maintenance"][0, 2] == pytest.approx(P["maintenance_m"] * 1.02 ** 2 * 1.01 ** 2)
    assert d["staff"][0, 2] == pytest.approx(P["staff_m"] * 1.035 ** 2)
    assert d["energy"][0, 2] == pytest.approx(P["energy_m"] * 1.02 ** 2)
    assert d["other"][0, 2] == pytest.approx(P["other_m"] * 1.02 ** 2)


def test_deterministic_shocks():
    base = paths.deterministic(P)
    assert paths.deterministic(P, shock={"ridership_mult": 0.9})["trips"] == pytest.approx(base["trips"] * 0.9)
    e = paths.deterministic(P, shock={"energy_mult": 1.3})
    assert e["energy"][0, 0] == pytest.approx(base["energy"][0, 0])
    assert e["energy"][0, 1:] == pytest.approx(base["energy"][0, 1:] * 1.3)
    assert e["cost_idx"][0, 1] > base["cost_idx"][0, 1]
    capped = paths.deterministic(P, shock={"fare_cap": 0.02})
    assert capped["fare_regulated"][0, 1] == pytest.approx(P["fare"] * 1.02)
    assert capped["deferred"][0, 1] == pytest.approx(0.5 * 0.02 + 0.4 * 0.035 + 0.1 * 0.02 + 0.01 - 0.02)
    pan = paths.deterministic(P, shock={"pandemic_year": 3})
    assert (pan["trips"] / base["trips"])[0, 2:8] == pytest.approx([1.0] + P["pandemic_path"] + [1.0])
    assert paths.deterministic(P, shock={"wage_add": 0.01})["wage_idx"][0, 1] == pytest.approx(1.045)
    assert paths.deterministic(P, shock={"cpi_add": 0.01})["cpi_idx"][0, 1] == pytest.approx(1.03)
    assert np.all(paths.deterministic(P, shock={"reliability": -1.0})["reliability"] == -1.0)


def test_simulation_is_reproducible_and_has_the_right_moments():
    a, b = paths.simulate(P, 4000, 1), paths.simulate(P, 4000, 1)
    assert np.array_equal(a["trips"], b["trips"])
    assert not np.array_equal(a["trips"], paths.simulate(P, 4000, 2)["trips"])
    cpi_g = a["cpi_idx"][:, 1:] / a["cpi_idx"][:, :-1] - 1
    assert cpi_g.mean() == pytest.approx(P["cpi_mean"], abs=0.001)
    assert cpi_g.std() == pytest.approx(P["cpi_sd"], abs=0.001)
    wage_g = a["wage_idx"][:, 1:] / a["wage_idx"][:, :-1] - 1
    assert wage_g.mean() == pytest.approx(P["wage_mean"], abs=0.001)
    energy_g = a["energy_idx"][:, 1:] / a["energy_idx"][:, :-1] - 1
    assert energy_g.mean() == pytest.approx(P["energy_mean"], abs=0.01)
    assert np.log1p(energy_g).std() == pytest.approx(P["energy_sd"], abs=0.01)
    assert np.all(a["cpi_idx"][:, 0] == 1.0) and np.all(a["trips"][:, 0] == P["trips_m"])
    assert a["reliability"].std() == pytest.approx(1.0, abs=0.02)


def test_simulated_pandemic_frequency_and_ridership_noise():
    q = dict(P, pandemic_prob=0.0)
    a = paths.simulate(q, 6000, 3)
    noise = np.log(a["trips"] / (P["trips_m"] * 1.01 ** np.arange(P["years"])))
    assert noise[:, 1].std() == pytest.approx(P["ridership_sd"], abs=0.002)
    stationary = P["ridership_sd"] / np.sqrt(1 - P["ridership_rho"] ** 2)
    assert noise[:, -1].std() == pytest.approx(stationary, abs=0.003)
    assert np.corrcoef(noise[:, -1], noise[:, -2])[0, 1] == pytest.approx(P["ridership_rho"], abs=0.04)
    b = paths.simulate(dict(P, ridership_sd=0.0), 20000, 4)
    hit = (b["trips"] / (P["trips_m"] * 1.01 ** np.arange(P["years"])) < 0.99)
    assert hit[:, 0].mean() == 0.0
    assert hit[:, 1].mean() == pytest.approx(P["pandemic_prob"], abs=0.004)
    assert hit.any(axis=1).mean() == pytest.approx(1 - (1 - P["pandemic_prob"]) ** 14, abs=0.01)


# --- payoffs on paths --------------------------------------------------------------------------------------
def test_calibration_hits_the_target_margin_for_every_archetype():
    res = analysis.payoffs(P, paths.deterministic(P), TERMS)
    for k in analysis.ARCHETYPES:
        m = analysis.cumulative_margin(res[k]["operator"], res[k]["revenue"])[0]
        assert m == pytest.approx(P["target_margin"], abs=1e-9)
    long = analysis.calibrate(P, 30)
    r = analysis.payoffs(P, paths.deterministic(P, 30), long)["D"]
    assert analysis.cumulative_margin(r["operator"], r["revenue"])[0] == pytest.approx(P["target_margin"], abs=1e-9)


def test_payoffs_base_year_by_hand():
    res = analysis.payoffs(P, paths.deterministic(P), TERMS)
    fare, nonfare = P["fare"] * P["trips_m"], P["nonfare_share"] * P["fare"] * P["trips_m"]
    assert res["A"]["operator"][0, 0] == pytest.approx(fare + nonfare - P["opex_m"] - TERMS["A"])
    assert res["A"]["authority"][0, 0] == pytest.approx(TERMS["A"])
    assert res["B"]["operator"][0, 0] == pytest.approx(TERMS["B"] - P["opex_m"])
    assert res["B"]["revenue"][0, 0] == pytest.approx(TERMS["B"])
    assert res["D"]["authority"][0, 0] == pytest.approx(TERMS["D"])
    assert res["D"]["surplus"][0, 0] == pytest.approx(fare + nonfare - P["opex_m"] - P["capital_charge_m"])
    assert res["A"]["revenue"][0, 0] == pytest.approx(fare + nonfare)


def test_payoffs_later_years_use_the_right_indices():
    d = paths.deterministic(P)
    res = analysis.payoffs(P, d, TERMS)
    t = 6
    fee = TERMS["B"] * (1 + P["fee_indexation"] * (d["cost_idx"][0, t] - 1))
    assert res["B"]["revenue"][0, t] == pytest.approx(fee)
    assert res["A"]["authority"][0, t] == pytest.approx(TERMS["A"] * d["cpi_idx"][0, t])
    assert res["D"]["authority"][0, t] == pytest.approx(TERMS["D"] * d["cpi_idx"][0, t])
    nonfare = P["nonfare_share"] * P["fare"] * d["trips"][0, t] * d["cpi_idx"][0, t]
    assert res["D"]["revenue"][0, t] == pytest.approx(d["trips"][0, t] * d["fare_free"][0, t] + nonfare)
    assert res["A"]["revenue"][0, t] == pytest.approx(d["trips"][0, t] * d["fare_regulated"][0, t] + nonfare)


def test_deferral_subsidy_goes_to_the_fare_collecting_operator_only():
    d = paths.deterministic(P, shock={"fare_cap": 0.02})
    covered = analysis.payoffs(P, d, TERMS)
    bare = analysis.payoffs(dict(P, deferral_cover=0.0), d, TERMS)
    sub = d["deferred"] * d["trips"] * d["fare_regulated"]
    assert sub[0, 5] > 0
    assert (covered["A"]["operator"] - bare["A"]["operator"])[0] == pytest.approx(sub[0])
    assert (covered["A"]["authority"] - bare["A"]["authority"])[0] == pytest.approx(-sub[0])
    assert np.array_equal(covered["B"]["operator"], bare["B"]["operator"])
    assert np.array_equal(covered["D"]["operator"], bare["D"]["operator"])
    half = analysis.payoffs(dict(P, deferral_cover=0.5), d, TERMS)
    assert (half["A"]["operator"] - bare["A"]["operator"])[0] == pytest.approx(0.5 * sub[0])


def test_identity_holds_on_simulated_paths():
    res = analysis.payoffs(P, paths.simulate(P, 3000, 5), TERMS)
    for k in analysis.ARCHETYPES:
        assert np.allclose(res[k]["operator"] + res[k]["authority"], res[k]["surplus"], atol=1e-8)


def test_npv_and_cvar_and_margin_by_hand():
    assert analysis.npv(np.array([[1.0, 1.1, 1.21]]), 0.1)[0] == pytest.approx(3.0)
    x = np.arange(1.0, 101.0)
    assert analysis.cvar(x, 0.05) == pytest.approx(3.0)
    assert analysis.cvar(np.array([5.0, 1.0, 9.0]), 0.05) == 1.0
    m = analysis.cumulative_margin(np.array([[1.0, 3.0], [0.0, -2.0]]), np.array([[10.0, 30.0], [10.0, 10.0]]))
    assert m == pytest.approx([0.1, -0.1])


def test_metrics_by_hand():
    op = np.array([[10.0, 10.0], [-5.0, -5.0], [4.0, -1.0], [2.0, 2.0]])
    res = {"A": {"operator": op, "authority": 3.0 - op, "revenue": np.full((4, 2), 100.0), "surplus": np.full((4, 2), 3.0)}}
    m = analysis.metrics(dict(P, discount=0.0), res)["A"]
    assert m["p_loss_year"] == pytest.approx(3 / 8)
    assert m["p_loss_cumulative"] == pytest.approx(1 / 4)
    assert m["margin_mean"] == pytest.approx((0.1 - 0.05 + 0.015 + 0.02) / 4)
    assert m["operator_npv_mean"] == pytest.approx((20 - 10 + 3 + 4) / 4)
    assert m["operator_npv_cvar5"] == -10.0
    assert m["authority_npv_mean"] == pytest.approx(6 - 17 / 4)
    assert m["authority_npv_sd"] == pytest.approx(np.std([20.0, -10.0, 3.0, 4.0]))
    assert m["identity_max_abs_error"] == 0.0
    assert m["margin_p50"] == pytest.approx(np.median([0.1, -0.05, 0.015, 0.02]))


def test_risk_allocation_follows_the_fare_box():
    rows = {r["shock"]: r for r in analysis.risk_allocation(P, TERMS)}
    rid = rows["ridership"]
    assert rid["A"]["operator_share"] == pytest.approx(1.0) and rid["A"]["authority"] == pytest.approx(0.0)
    assert rid["B"]["operator"] == pytest.approx(0.0) and rid["B"]["authority"] < 0
    assert 0.3 < rid["C"]["operator_share"] < 0.7
    assert rid["A"]["operator"] == pytest.approx(rid["B"]["authority"])
    assert rows["reliability"]["B"]["operator"] == pytest.approx(-rows["reliability"]["B"]["authority"])
    assert rows["reliability"]["B"]["operator"] < 0 and rows["reliability"]["B"]["operator_share"] is None
    assert rows["reliability"]["A"]["operator"] == 0.0
    cap, bare = rows["fare_cap"], rows["fare_cap_uncovered"]
    assert abs(cap["A"]["authority"]) > 10 * abs(cap["A"]["operator"])
    assert bare["A"]["authority"] == pytest.approx(0.0) and bare["A"]["operator"] < -100
    assert cap["D"]["operator"] == 0.0 and cap["D"]["operator_share"] is None
    assert rows["pandemic"]["A"]["operator"] < 0 and rows["pandemic"]["C"]["authority"] < 0
    assert rows["energy"]["D"]["operator"] < 0 and rows["wages"]["B"]["operator"] < 0


def test_marginal_incentives():
    inc = analysis.marginal_incentives(P, TERMS)
    e = inc["expected"]
    assert e["A"]["revenue"] == pytest.approx(1.0) and e["A"]["cost"] == pytest.approx(1.0)
    assert e["B"]["revenue"] == pytest.approx(0.0) and e["B"]["cost"] == pytest.approx(1.0)
    assert e["D"]["revenue"] == pytest.approx(1.0)
    assert e["B"]["reliability_m"] == pytest.approx(0.01 * analysis.payoffs(P, paths.deterministic(P), TERMS)["B"]["revenue"][0, 5])
    assert e["A"]["reliability_m"] == 0.0
    low, high = inc["ridership 20% below projection"], inc["ridership 15% above projection"]
    assert low["C"]["cost"] == pytest.approx(1.0)
    assert low["C"]["revenue"] == pytest.approx(1 - P["shortfall_share"] / (1 + P["nonfare_share"] * 1.02 ** 5 * (P["fare"] / paths.deterministic(P)["fare_regulated"][0, 5])), abs=0.01)
    assert high["C"]["cost"] == pytest.approx(1 - P["rate2"])
    assert high["C"]["revenue"] < e["C"]["revenue"] < 1.0


def test_tornado_signs():
    rows = {r["driver"]: r for r in analysis.tornado(P, TERMS)}
    rid = rows["Ridership -10% / +10%"]
    assert rid["A"]["adverse"] < 0 < rid["A"]["favourable"]
    assert rid["B"]["adverse"] == pytest.approx(0.0)
    assert abs(rid["C"]["adverse"]) < abs(rid["A"]["adverse"])
    assert rid["C"]["favourable"] < rid["A"]["favourable"]
    assert rows["Reliability -1 / +1 sd"]["B"]["favourable"] > 0
    assert rows["Fare cap 2% / 8%"]["A"]["favourable"] == pytest.approx(0.0)


def test_run_small():
    out = analysis.run(P, n=1500, seed=11)
    assert out["n_sims"] == 1500 and out["years"] == 15
    for k in analysis.ARCHETYPES:
        assert out["deterministic_margin"][k] == pytest.approx(0.05, abs=1e-9)
        assert out["metrics"][k]["identity_max_abs_error"] < 1e-8
    m = out["metrics"]
    assert m["A"]["operator_npv_sd"] > m["C"]["operator_npv_sd"]
    assert m["B"]["authority_npv_sd"] > m["C"]["authority_npv_sd"] > m["A"]["authority_npv_sd"]
    assert m["A"]["p_loss_cumulative"] > m["C"]["p_loss_cumulative"]
    assert out["concession_30_years"]["p_at_least_one_pandemic"] == pytest.approx(1 - 0.98 ** 29)
    assert out["concession_30_years"]["years"] == 30
    assert out["base_year"]["fare_to_opex"] == pytest.approx(P["fare"] * P["trips_m"] / 135.0)


# --- parameters ------------------------------------------------------------------------------------------
def test_derived_parameters_by_hand():
    d = params.derived()
    assert d["implied_rail_fare_revenue_m"] == pytest.approx(23.94 / 0.30 / 0.07)
    assert d["rail_boardings_m"] == pytest.approx((3490 + 209) * 365 / 1000)
    assert d["fare_per_boarding"] == pytest.approx(1140.0 / 1350.135)
    assert d["line_trips_m"] == pytest.approx(171.55)
    assert sum(d["cost_shares"].values()) == pytest.approx(1.0)
    assert d["cost_shares"]["staff"] == pytest.approx(765.596 / 1449.230)
    assert d["pandemic_path"] == pytest.approx([2023 / 3384, 2100 / 3384, 2745 / 3384, 3243 / 3384])
    assert P["staff_m"] + P["energy_m"] + P["maintenance_m"] + P["other_m"] == pytest.approx(P["opex_m"])


# --- RTS Link ----------------------------------------------------------------------------------------------
def test_rts_arithmetic_by_hand():
    assert rts.annual_revenue_m(40_000, 5.0) == pytest.approx(73.0)
    assert rts.break_even_daily_trips(73.0, 5.0) == pytest.approx(40_000)
    assert rts.break_even_daily_trips(100.0, 4.0) == pytest.approx(100e6 / 1460)
    c = rts.daily_ceiling(10_000, 18, 4, 0.25)
    assert c == {"all_day_full": 360_000.0, "peaked_profile": 150_000.0}
    assert rts.time_savings_m(40_000, 30, 15.0) == pytest.approx(40_000 * 365 * 0.5 * 15 / 1e6)


def test_rts_exchange_rate_functions():
    assert rts.fx_revenue_factor(3.2, 3.2) == 1.0
    assert rts.fx_revenue_factor(4.0, 3.2, 0.5) == pytest.approx(0.9)
    assert rts.fx_revenue_factor(4.0, 3.2, 0.0) == 1.0
    assert rts.fx_revenue_factor(1.6, 3.2, 1.0) == pytest.approx(2.0)
    assert rts.fx_margin_m(100.0, 80.0, 4.0, 3.2, 0.5, 0.5) == pytest.approx(18.0)
    assert rts.fx_margin_m(100.0, 80.0, 4.0, 3.2, 0.5, 0.0) == pytest.approx(10.0)


def test_rts_fx_statistics():
    series = [("2020-01-30", 3.0), ("2020-01-31", 3.0), ("2020-02-28", 3.3), ("2020-03-31", 3.0), ("2020-04-30", 3.3)]
    s = rts.fx_stats(series)
    assert (s["latest"], s["min"], s["max"], s["n_days"]) == (3.3, 3.0, 3.3, 5)
    assert s["first_date"] == "2020-01-30" and s["last_date"] == "2020-04-30"
    steps = np.diff(np.log([3.0, 3.3, 3.0, 3.3]))
    assert s["annual_volatility"] == pytest.approx(np.std(steps, ddof=1) * np.sqrt(12))
    real = rts.fx_series()
    assert real[0][0] >= "2015-01-01" and 2.5 < min(v for _, v in real) and max(v for _, v in real) < 3.7


def test_rts_run_is_consistent():
    r = rts.run()
    assert r["opening_share_of_woodlands"] == pytest.approx(40_000 / 300_000)
    assert r["reference_revenue_m"] == pytest.approx(73.0)
    assert r["revenue"][0]["by_fare"][1]["revenue_myr_m"] == pytest.approx(73.0 * r["fx"]["latest"])
    assert r["ridership"][1]["daily"] == pytest.approx(30_000) and r["ridership"][0]["provenance"] == "REAL"
    be = r["break_even"][1]["by_fare"][1]
    assert be["daily_trips"] == pytest.approx(100e6 / 1825) and be["share_of_opening_estimate"] == pytest.approx(100e6 / 1825 / 40_000)
    assert be["share_of_peaked_ceiling"] == pytest.approx(100e6 / 1825 / 150_000)
    strong, latest, weak = r["fx_sensitivity"]
    assert strong["revenue_factor"] > 1.0 == latest["revenue_factor"] > weak["revenue_factor"]
    assert latest["margin_matched_m"] == pytest.approx(73.0 - 100.0)
    assert abs(weak["margin_matched_m"] - latest["margin_matched_m"]) < abs(weak["margin_sgd_costs_m"] - latest["margin_sgd_costs_m"])
    assert r["shuttle_fare_myr_in_sgd"] == pytest.approx(5.0 / r["fx"]["latest"])
    assert [t["value_m"] for t in r["time_savings"]] == pytest.approx([73.0, 146.0, 219.0])
    assert r["opening_share_of_peaked_ceiling"] == pytest.approx(40_000 / 150_000)


# --- procurement ---------------------------------------------------------------------------------------------
def test_procurement_scores_and_switching_weight():
    assert procurement.score(80.0, 100.0, 0.4, 100.0, 80.0) == pytest.approx(0.4 * 0.8 + 0.6 * 0.8)
    bidders = [{"name": "cheap", "quality": 80.0, "price": 92.0}, {"name": "good", "quality": 100.0, "price": 100.0}]
    w = procurement.switching_weight(bidders)
    assert w == pytest.approx(0.08 / (0.08 + 0.2))
    assert procurement.winner(bidders, w - 0.01) == "cheap" and procurement.winner(bidders, w + 0.01) == "good"
    assert procurement.winner(bidders, 0.0) == "cheap" and procurement.winner(bidders, 1.0) == "good"
    assert procurement.BIDDERS[0]["price"] / procurement.BIDDERS[1]["price"] == pytest.approx(0.92)


def test_aggressive_bids_get_riskier():
    rows = procurement.aggressive_bids(P, n=3000, seed=9)
    assert [r["discount"] for r in rows] == [0.0, 0.04, 0.08, 0.12]
    assert rows[0]["margin_mean"] == pytest.approx(0.05, abs=0.01)
    assert rows[1]["fee_m"] == pytest.approx(rows[0]["fee_m"] * 0.96)
    for a, b in zip(rows, rows[1:]):
        assert b["p_loss_cumulative"] >= a["p_loss_cumulative"] and b["margin_mean"] < a["margin_mean"]
        assert b["margin_p5"] < a["margin_p5"] and b["p_loss_year"] >= a["p_loss_year"]
    assert rows[0]["p_loss_cumulative"] < 0.1 < rows[2]["p_loss_cumulative"]
    assert rows[0]["margin_p5"] < rows[0]["margin_mean"]
