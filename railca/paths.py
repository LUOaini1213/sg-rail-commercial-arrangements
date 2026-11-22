"""Driver paths (ridership, prices, fares, costs) for one stylised line: deterministic and random."""
from __future__ import annotations

import numpy as np


def regulated_fares(formula_g: np.ndarray, cap: float):
    """Apply a yearly cap to formula-allowed fare increases and carry the rest forward.

    formula_g has shape (n, T); column 0 is the base year (no change). Returns (index, deferred)
    where index is the fare index and deferred is the quantum still owed after each year.
    allowable_t = formula_t + deferred_{t-1}; granted_t = min(allowable_t, cap).
    """
    n, T = formula_g.shape
    index = np.ones((n, T))
    deferred = np.zeros((n, T))
    for t in range(1, T):
        allowable = formula_g[:, t] + deferred[:, t - 1]
        granted = np.minimum(allowable, cap)
        deferred[:, t] = allowable - granted
        index[:, t] = index[:, t - 1] * (1.0 + granted)
    return index, deferred


def pandemic_multiplier(starts: np.ndarray, path) -> np.ndarray:
    """Ridership multiplier given a boolean (n, T) array of event starts and a recovery path.

    An event occupies len(path) consecutive years; a start drawn while an event is running is ignored.
    """
    n, T = starts.shape
    mult = np.ones((n, T))
    stage = np.full(n, -1)
    for t in range(T):
        new = (stage < 0) & starts[:, t]
        stage = np.where(new, 0, stage)
        active = stage >= 0
        idx = np.clip(stage, 0, len(path) - 1)
        mult[:, t] = np.where(active, np.asarray(path)[idx], 1.0)
        stage = np.where(active, stage + 1, stage)
        stage = np.where(stage >= len(path), -1, stage)
    return mult


def build(p: dict, cpi_g, wage_g, energy_g, ridership_noise, pandemic, reliability, shock: dict | None = None) -> dict:
    """Turn growth rates and shocks into yearly drivers. All arrays are (n, T); column 0 is the base year."""
    s = {"ridership_mult": 1.0, "energy_mult": 1.0, "opex_add": 0.0}
    s.update(shock or {})
    n, T = cpi_g.shape
    t = np.arange(T)
    energy_g = energy_g.copy()
    if T > 1:  # a level shock to energy prices arrives in year 1 and feeds the cost and fare indices
        energy_g[:, 1] = (1.0 + energy_g[:, 1]) * s["energy_mult"] - 1.0
    cpi_idx = np.cumprod(1.0 + cpi_g, axis=1)
    wage_idx = np.cumprod(1.0 + wage_g, axis=1)
    energy_idx = np.cumprod(1.0 + energy_g, axis=1)
    cost_g = p["w_cpi"] * cpi_g + p["w_wage"] * wage_g + p["w_energy"] * energy_g
    cost_idx = np.cumprod(1.0 + cost_g, axis=1)
    formula_g = cost_g - p["productivity"] + p["capacity"]
    formula_g[:, 0] = 0.0
    fare_idx, deferred = regulated_fares(formula_g, p["fare_cap"])
    trips = p["trips_m"] * (1.0 + p["ridership_growth"]) ** t * np.exp(ridership_noise) * pandemic * s["ridership_mult"]
    staff = p["staff_m"] * wage_idx
    energy = p["energy_m"] * energy_idx
    maintenance = p["maintenance_m"] * cpi_idx * (1.0 + p["ageing"]) ** t
    other = p["other_m"] * cpi_idx + s["opex_add"]
    return {
        "cpi_idx": cpi_idx, "wage_idx": wage_idx, "energy_idx": energy_idx, "cost_idx": cost_idx,
        "fare_regulated": p["fare"] * fare_idx, "deferred": deferred, "fare_free": p["fare"] * cost_idx,
        "trips": trips, "reliability": reliability,
        "staff": staff, "energy": energy, "maintenance": maintenance, "other": other,
        "opex": staff + energy + maintenance + other,
    }


def deterministic(p: dict, years: int | None = None, shock: dict | None = None) -> dict:
    """Expected-growth path with no noise. shock may also set wage_add, cpi_add, reliability, pandemic_year."""
    T = years or p["years"]
    s = shock or {}
    one = np.ones((1, T))
    cpi_g = one * (p["cpi_mean"] + s.get("cpi_add", 0.0))
    wage_g = one * (p["wage_mean"] + s.get("wage_add", 0.0))
    energy_g = one * p["energy_mean"]
    for g in (cpi_g, wage_g, energy_g):
        g[:, 0] = 0.0
    starts = np.zeros((1, T), dtype=bool)
    if "pandemic_year" in s:
        starts[0, s["pandemic_year"]] = True
    q = dict(p)
    if "fare_cap" in s:
        q["fare_cap"] = s["fare_cap"]
    return build(q, cpi_g, wage_g, energy_g, np.zeros((1, T)), pandemic_multiplier(starts, p["pandemic_path"]),
                 one * s.get("reliability", 0.0), s)


def simulate(p: dict, n: int, seed: int, years: int | None = None) -> dict:
    """Random paths. Ridership noise is AR(1); energy growth is lognormal; pandemics start with a yearly probability."""
    T = years or p["years"]
    rng = np.random.default_rng(seed)
    cpi_g = rng.normal(p["cpi_mean"], p["cpi_sd"], (n, T))
    wage_g = rng.normal(p["wage_mean"], p["wage_sd"], (n, T))
    energy_g = np.exp(rng.normal(np.log1p(p["energy_mean"]) - 0.5 * p["energy_sd"] ** 2, p["energy_sd"], (n, T))) - 1.0
    for g in (cpi_g, wage_g, energy_g):
        g[:, 0] = 0.0
    z = rng.normal(0.0, p["ridership_sd"], (n, T))
    noise = np.zeros((n, T))
    for t in range(1, T):
        noise[:, t] = p["ridership_rho"] * noise[:, t - 1] + z[:, t]
    starts = rng.random((n, T)) < p["pandemic_prob"]
    starts[:, 0] = False
    reliability = rng.normal(0.0, 1.0, (n, T))
    return build(p, cpi_g, wage_g, energy_g, noise, pandemic_multiplier(starts, p["pandemic_path"]), reliability)
