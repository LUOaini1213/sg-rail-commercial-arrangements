"""Scenario arithmetic for the Johor Bahru-Singapore Rapid Transit System (RTS) Link.

Published inputs (capacity, opening-ridership estimate, checkpoint volume, benchmark fares) are REAL.
Fares had not been announced when this was built, and the operator's costs are not public, so fare
levels and cost bands are ILLUSTRATIVE. Nothing here is a forecast or a view on what fares should be.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "data" / "ecb_eur_sgd_myr.csv"

CAPACITY_PPHPD = 10_000            # REAL  F62
OPENING_ESTIMATE = 40_000          # REAL  F69 (MRT Corp, passengers a day upon opening)
WOODLANDS_DAILY = 300_000          # REAL  F77 (travellers a day through Woodlands Checkpoint)
CONCESSION_YEARS = 30              # REAL  F64
SHUTTLE_FARE_SGD = 5.0             # REAL  F80 (Woodlands to JB Sentral, non-Malaysian)
SHUTTLE_FARE_MYR = 5.0             # REAL  F80 (JB Sentral to Woodlands)
PRESS_RANGE_SGD = (5.0, 7.0)       # UNVERIFIED F75 (press report; scenario range only)
HOURS = 18                         # ASSUMPTION
PEAK_HOURS = 4                     # ASSUMPTION
OFFPEAK_LOAD = 0.25                # ASSUMPTION
SHARES = [0.10, 0.20, 0.35]        # ILLUSTRATIVE shares of Woodlands Checkpoint volume
FARES_SGD = [4.0, 5.0, 6.0, 7.0]   # ILLUSTRATIVE
COST_BANDS_M = [60.0, 100.0, 140.0]  # ILLUSTRATIVE yearly cost incl. financing of operating assets
MYR_REVENUE_SHARE = 0.5            # ILLUSTRATIVE share of tickets priced in ringgit
MYR_COST_SHARE = 0.5               # ILLUSTRATIVE share of costs in ringgit (the depot is in Johor Bahru)
VALUE_OF_TIME = 15.0               # ASSUMPTION S$ per hour
MINUTES_SAVED = [20, 40, 60]       # ASSUMPTION


def fx_series() -> list[tuple[str, float]]:
    """Daily ringgit per Singapore dollar, derived as EUR/MYR divided by EUR/SGD."""
    with FX.open(encoding="utf-8", newline="") as fh:
        return [(r["date"], float(r["eur_myr"]) / float(r["eur_sgd"])) for r in csv.DictReader(fh)]


def fx_stats(series: list[tuple[str, float]]) -> dict:
    """Level statistics on daily rates and annualised volatility of month-end log changes."""
    rates = np.array([v for _, v in series])
    month_end = {}
    for d, v in series:
        month_end[d[:7]] = v
    m = np.array(list(month_end.values()))
    vol = float(np.std(np.diff(np.log(m)), ddof=1) * math.sqrt(12.0))
    return {"first_date": series[0][0], "last_date": series[-1][0], "n_days": len(series),
            "latest": float(rates[-1]), "min": float(rates.min()), "max": float(rates.max()),
            "p5": float(np.percentile(rates, 5)), "p50": float(np.percentile(rates, 50)),
            "p95": float(np.percentile(rates, 95)), "annual_volatility": vol}


def daily_ceiling(capacity=CAPACITY_PPHPD, hours=HOURS, peak_hours=PEAK_HOURS, offpeak_load=OFFPEAK_LOAD) -> dict:
    """Trips a day if both directions run full all day, and under a peaked load profile."""
    full = capacity * 2 * hours
    peaked = capacity * 2 * (peak_hours + (hours - peak_hours) * offpeak_load)
    return {"all_day_full": float(full), "peaked_profile": float(peaked)}


def annual_revenue_m(daily_trips: float, fare_sgd: float) -> float:
    return daily_trips * 365.0 * fare_sgd / 1e6


def break_even_daily_trips(annual_cost_m: float, fare_sgd: float) -> float:
    return annual_cost_m * 1e6 / (365.0 * fare_sgd)


def fx_revenue_factor(rate: float, rate0: float, myr_share: float = MYR_REVENUE_SHARE) -> float:
    """Revenue in S$ relative to launch if a share of tickets is fixed in ringgit at the launch rate.

    rate is ringgit per Singapore dollar; a higher rate (weaker ringgit) lowers the S$ value.
    """
    return (1.0 - myr_share) + myr_share * rate0 / rate


def fx_margin_m(annual_revenue: float, annual_cost: float, rate: float, rate0: float,
                rev_share: float = MYR_REVENUE_SHARE, cost_share: float = MYR_COST_SHARE) -> float:
    """S$ margin when shares of revenue and cost are fixed in ringgit."""
    return annual_revenue * fx_revenue_factor(rate, rate0, rev_share) - annual_cost * fx_revenue_factor(rate, rate0, cost_share)


def time_savings_m(daily_trips: float, minutes: float, value_per_hour: float = VALUE_OF_TIME) -> float:
    """Yearly value of time saved, S$ million. An economic benefit to travellers, not operator revenue."""
    return daily_trips * 365.0 * minutes / 60.0 * value_per_hour / 1e6


def run() -> dict:
    series = fx_series()
    fx = fx_stats(series)
    rate0 = fx["latest"]
    ridership = [{"label": "Published opening estimate", "daily": float(OPENING_ESTIMATE), "provenance": "REAL"}]
    ridership += [{"label": f"{int(s * 100)}% of Woodlands Checkpoint volume", "daily": s * WOODLANDS_DAILY,
                   "provenance": "ILLUSTRATIVE"} for s in SHARES]
    ceiling = daily_ceiling()
    revenue = [{"label": r["label"], "daily": r["daily"],
                "by_fare": [{"fare_sgd": f, "revenue_sgd_m": annual_revenue_m(r["daily"], f),
                             "revenue_myr_m": annual_revenue_m(r["daily"], f) * rate0} for f in FARES_SGD]}
               for r in ridership]
    breakeven = [{"cost_m": c, "by_fare": [{"fare_sgd": f, "daily_trips": break_even_daily_trips(c, f),
                                             "share_of_opening_estimate": break_even_daily_trips(c, f) / OPENING_ESTIMATE,
                                             "share_of_peaked_ceiling": break_even_daily_trips(c, f) / ceiling["peaked_profile"]}
                                            for f in FARES_SGD]} for c in COST_BANDS_M]
    ref_rev = annual_revenue_m(OPENING_ESTIMATE, 5.0)
    ref_cost = COST_BANDS_M[1]
    fx_sens = []
    for label, rate in (("Ringgit strong (5th percentile since 2015)", fx["p5"]), ("Latest", rate0),
                        ("Ringgit weak (95th percentile since 2015)", fx["p95"])):
        fx_sens.append({"label": label, "myr_per_sgd": rate,
                        "revenue_factor": fx_revenue_factor(rate, rate0),
                        "revenue_sgd_m": ref_rev * fx_revenue_factor(rate, rate0),
                        "margin_matched_m": fx_margin_m(ref_rev, ref_cost, rate, rate0),
                        "margin_sgd_costs_m": fx_margin_m(ref_rev, ref_cost, rate, rate0, cost_share=0.0)})
    time_rows = [{"minutes": m, "value_m": time_savings_m(OPENING_ESTIMATE, m)} for m in MINUTES_SAVED]
    return {
        "inputs": {"capacity_pphpd": CAPACITY_PPHPD, "opening_estimate": OPENING_ESTIMATE,
                   "woodlands_daily": WOODLANDS_DAILY, "concession_years": CONCESSION_YEARS,
                   "shuttle_fare_sgd": SHUTTLE_FARE_SGD, "shuttle_fare_myr": SHUTTLE_FARE_MYR,
                   "press_range_low": PRESS_RANGE_SGD[0], "press_range_high": PRESS_RANGE_SGD[1],
                   "hours": HOURS, "peak_hours": PEAK_HOURS, "offpeak_load": OFFPEAK_LOAD,
                   "value_of_time": VALUE_OF_TIME, "reference_fare": 5.0, "reference_cost_m": ref_cost,
                   "myr_revenue_share": MYR_REVENUE_SHARE, "myr_cost_share": MYR_COST_SHARE},
        "opening_share_of_woodlands": OPENING_ESTIMATE / WOODLANDS_DAILY,
        "opening_share_of_peaked_ceiling": OPENING_ESTIMATE / ceiling["peaked_profile"],
        "ceiling": ceiling, "fx": fx, "ridership": ridership, "revenue": revenue, "break_even": breakeven,
        "reference_revenue_m": ref_rev,
        "shuttle_fare_myr_in_sgd": SHUTTLE_FARE_MYR / rate0,
        "fx_sensitivity": fx_sens, "time_savings": time_rows,
    }
