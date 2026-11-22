"""Calibration, Monte Carlo metrics, risk allocation, marginal incentives and sensitivities."""
from __future__ import annotations

import numpy as np

from railca import archetypes as arc
from railca import paths

ARCHETYPES = ["A", "B", "C", "D"]
NAMES = {"A": "Net-cost concession", "B": "Gross-cost service fee", "C": "Hybrid with sharing bands",
         "D": "Long concession, operator-set fares"}


def payoffs(p: dict, drv: dict, terms: dict) -> dict:
    """Operator and authority payoff, operator revenue and system surplus per archetype, shape (n, T)."""
    fare_rev = drv["trips"] * drv["fare_regulated"]
    nonfare = p["nonfare_share"] * p["fare"] * drv["trips"] * drv["cpi_idx"]
    subsidy = p["deferral_cover"] * drv["deferred"] * fare_rev
    opex = drv["opex"]
    out = {}
    op, au = arc.net_cost(fare_rev, nonfare, opex, terms["A"] * drv["cpi_idx"], subsidy)
    out["A"] = {"operator": op, "authority": au, "revenue": fare_rev + nonfare + subsidy,
                "surplus": arc.system_surplus(fare_rev, nonfare, opex)}
    fee = terms["B"] * (1.0 + p["fee_indexation"] * (drv["cost_idx"] - 1.0))
    inc = arc.incentive_payment(fee, drv["reliability"], p["incentive_rate"], p["bonus_cap"], p["penalty_cap"])
    op, au = arc.gross_cost(fare_rev, nonfare, opex, fee, inc)
    out["B"] = {"operator": op, "authority": au, "revenue": fee + inc,
                "surplus": arc.system_surplus(fare_rev, nonfare, opex)}
    op, au = arc.hybrid(fare_rev, nonfare, opex, terms["C"] * drv["cpi_idx"], terms["projected_fare_rev"],
                        p["deadband"], p["shortfall_share"], p["band1"], p["rate1"], p["band2"], p["rate2"], subsidy)
    out["C"] = {"operator": op, "authority": au, "revenue": fare_rev + nonfare + subsidy,
                "surplus": arc.system_surplus(fare_rev, nonfare, opex)}
    fare_rev_d = drv["trips"] * drv["fare_free"]
    op, au = arc.concession(fare_rev_d, nonfare, opex, p["capital_charge_m"], terms["D"] * drv["cpi_idx"],
                            p["revenue_share"])
    out["D"] = {"operator": op, "authority": au, "revenue": fare_rev_d + nonfare,
                "surplus": arc.system_surplus(fare_rev_d, nonfare, opex, p["capital_charge_m"])}
    return out


def cumulative_margin(operator: np.ndarray, revenue: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return operator.sum(axis=1) / revenue.sum(axis=1)


def npv(x: np.ndarray, rate: float) -> np.ndarray:
    t = np.arange(x.shape[1])
    return (x / (1.0 + rate) ** t).sum(axis=1)


def _solve(f, lo: float, hi: float, increasing: bool) -> float:
    """Bisection for f(x) = 0 on [lo, hi] where f is monotone."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == increasing:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def calibrate(p: dict, years: int | None = None) -> dict:
    """Set each archetype's charge or fee so that the expected-path cumulative operator margin hits the target.

    This puts the four contracts on the same expected footing, so that differences in the results
    come from how each one passes risk around and not from one being priced more generously.
    """
    drv = paths.deterministic(p, years)
    terms = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0,
             "projected_fare_rev": (drv["trips"] * drv["fare_regulated"])[0]}
    for k in ARCHETYPES:
        def gap(x, k=k):
            trial = dict(terms)
            trial[k] = x
            r = payoffs(p, drv, trial)[k]
            return float(cumulative_margin(r["operator"], r["revenue"])[0]) - p["target_margin"]
        terms[k] = _solve(gap, -2000.0, 2000.0, increasing=(k == "B"))
    return terms


def cvar(x: np.ndarray, alpha: float = 0.05) -> float:
    """Mean of the worst alpha share of outcomes."""
    k = max(1, int(np.floor(alpha * len(x))))
    return float(np.sort(x)[:k].mean())


def metrics(p: dict, res: dict) -> dict:
    out = {}
    for k, r in res.items():
        m = cumulative_margin(r["operator"], r["revenue"])
        op_npv, au_npv = npv(r["operator"], p["discount"]), npv(r["authority"], p["discount"])
        out[k] = {
            "name": NAMES[k],
            "margin_mean": float(m.mean()), "margin_p5": float(np.percentile(m, 5)),
            "margin_p50": float(np.percentile(m, 50)), "margin_p95": float(np.percentile(m, 95)),
            "p_loss_year": float((r["operator"] < 0).mean()),
            "p_loss_cumulative": float((op_npv < 0).mean()),
            "operator_npv_mean": float(op_npv.mean()), "operator_npv_sd": float(op_npv.std()),
            "operator_npv_cvar5": cvar(op_npv),
            "authority_npv_mean": float(au_npv.mean()), "authority_npv_sd": float(au_npv.std()),
            "authority_npv_p5": float(np.percentile(au_npv, 5)), "authority_npv_cvar5": cvar(au_npv),
            "identity_max_abs_error": float(np.abs(r["operator"] + r["authority"] - r["surplus"]).max()),
        }
    return out


SHOCKS = [
    ("ridership", "Ridership 10% lower in every year", {"ridership_mult": 0.9}),
    ("pandemic", "Pandemic-like fall starting in year 3", {"pandemic_year": 3}),
    ("energy", "Energy prices 30% higher from year 1", {"energy_mult": 1.3}),
    ("wages", "Wage growth 1 point higher each year", {"wage_add": 0.01}),
    ("fare_cap", "Yearly fare increases capped at 2% (deferral covered)", {"fare_cap": 0.02}),
    ("reliability", "Reliability 1 standard deviation below target every year", {"reliability": -1.0}),
]


def _npv_pair(p, terms, shock, years=None):
    res = payoffs(p, paths.deterministic(p, years, shock), terms)
    return {k: (float(npv(r["operator"], p["discount"])[0]), float(npv(r["authority"], p["discount"])[0]))
            for k, r in res.items()}


def risk_allocation(p: dict, terms: dict) -> list[dict]:
    """Change in present value for each party under one-at-a-time shocks, and the operator's share of it."""
    base = _npv_pair(p, terms, None)
    rows = []
    for key, label, shock in SHOCKS:
        hit = _npv_pair(p, terms, shock)
        row = {"shock": key, "label": label}
        for k in ARCHETYPES:
            d_op, d_au = hit[k][0] - base[k][0], hit[k][1] - base[k][1]
            total = d_op + d_au
            row[k] = {"operator": d_op, "authority": d_au,
                      "operator_share": (d_op / total) if abs(total) > 1e-6 else None}
        rows.append(row)
    uncovered = dict(p, deferral_cover=0.0)
    base_u, hit_u = _npv_pair(uncovered, terms, None), _npv_pair(uncovered, terms, {"fare_cap": 0.02})
    row = {"shock": "fare_cap_uncovered", "label": "Yearly fare increases capped at 2% (deferral not covered)"}
    for k in ARCHETYPES:
        d_op, d_au = hit_u[k][0] - base_u[k][0], hit_u[k][1] - base_u[k][1]
        total = d_op + d_au
        row[k] = {"operator": d_op, "authority": d_au, "operator_share": (d_op / total) if abs(total) > 1e-6 else None}
    rows.append(row)
    return rows


def marginal_incentives(p: dict, terms: dict, year: int = 5) -> dict:
    """Operator's gain from S$1 of extra revenue, S$1 of cost saved, and one standard deviation of reliability.

    Measured by finite differences in one year, in three states of the world.
    """
    states = {"expected": {}, "ridership 20% below projection": {"ridership_mult": 0.8},
              "ridership 15% above projection": {"ridership_mult": 1.15}}
    out = {}
    for name, shock in states.items():
        drv = paths.deterministic(p, None, shock)
        base = payoffs(p, drv, terms)
        up = dict(drv, trips=drv["trips"] * 1.001)
        more = payoffs(p, up, terms)
        cut = dict(drv, opex=drv["opex"] - 0.1)
        less = payoffs(p, cut, terms)
        rel = dict(drv, reliability=drv["reliability"] + 1.0)
        better = payoffs(p, rel, terms)
        row = {}
        for k in ARCHETYPES:
            d_w = (more[k]["surplus"] - base[k]["surplus"])[0, year]
            row[k] = {
                "revenue": float((more[k]["operator"] - base[k]["operator"])[0, year] / d_w),
                "cost": float((less[k]["operator"] - base[k]["operator"])[0, year] / 0.1),
                "reliability_m": float((better[k]["operator"] - base[k]["operator"])[0, year]),
            }
        out[name] = row
    return out


TORNADO = [
    ("Ridership -10% / +10%", {"ridership_mult": 0.9}, {"ridership_mult": 1.1}),
    ("Energy price +30% / -30%", {"energy_mult": 1.3}, {"energy_mult": 0.7}),
    ("Wage growth +1 / -1 point", {"wage_add": 0.01}, {"wage_add": -0.01}),
    ("Inflation +1 / -1 point", {"cpi_add": 0.01}, {"cpi_add": -0.01}),
    ("Fare cap 2% / 8%", {"fare_cap": 0.02}, {"fare_cap": 0.08}),
    ("Reliability -1 / +1 sd", {"reliability": -1.0}, {"reliability": 1.0}),
]


def tornado(p: dict, terms: dict) -> list[dict]:
    """Operator present value under a low and a high case of each driver, as a change from the base case."""
    base = _npv_pair(p, terms, None)
    rows = []
    for label, lo, hi in TORNADO:
        a, b = _npv_pair(p, terms, lo), _npv_pair(p, terms, hi)
        rows.append({"driver": label,
                     **{k: {"adverse": a[k][0] - base[k][0], "favourable": b[k][0] - base[k][0]} for k in ARCHETYPES}})
    return rows


def run(p: dict, n: int = 20000, seed: int = 20261123) -> dict:
    terms = calibrate(p)
    res = payoffs(p, paths.simulate(p, n, seed), terms)
    m = metrics(p, res)
    long_terms = calibrate(p, 30)
    long_res = payoffs(p, paths.simulate(p, n, seed + 1, 30), long_terms)
    long_m = metrics(p, long_res)["D"]
    det = payoffs(p, paths.deterministic(p), terms)
    return {
        "n_sims": n, "seed": seed, "years": p["years"],
        "terms": {k: float(terms[k]) for k in ARCHETYPES},
        "base_year": {
            "fare_revenue_m": float(terms["projected_fare_rev"][0]),
            "nonfare_revenue_m": float(p["nonfare_share"] * p["fare"] * p["trips_m"]),
            "opex_m": float(p["opex_m"]),
            "fare_to_opex": float(terms["projected_fare_rev"][0] / p["opex_m"]),
        },
        "deterministic_margin": {k: float(cumulative_margin(det[k]["operator"], det[k]["revenue"])[0]) for k in ARCHETYPES},
        "metrics": m,
        "concession_30_years": {**long_m, "years": 30, "fixed_payment_m": float(long_terms["D"]),
                                "p_at_least_one_pandemic": float(1.0 - (1.0 - p["pandemic_prob"]) ** 29)},
        "risk_allocation": risk_allocation(p, terms),
        "marginal_incentives": marginal_incentives(p, terms),
        "tornado": tornado(p, terms),
    }
