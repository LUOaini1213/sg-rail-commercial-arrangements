"""Plant bugs in the model one at a time and check that the test suite notices each one.

Usage: python scripts/mutate.py --record first|final|ci [--require-all]
The golden test (committed results against a fresh run) is excluded, so a kill has to come from a
behavioural test with hand-worked expectations. Every file is restored after its mutant has run
(and restored again on exit). Results go to results/mutation.json under the given key. Mutants
judged equivalent (no input can tell them apart) are listed in EQUIVALENT with a reason.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "mutation.json"
PYTEST = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-m", "not golden",
          "--deselect", "tests/test_repo.py::test_mutation_record_is_complete"]
ARC, PATHS, ANA, RTS, PROC, WB = ("railca/archetypes.py", "railca/paths.py", "railca/analysis.py", "railca/rts.py",
                                  "railca/procurement.py", "railca/workbook.py")

MUTANTS = [
    # payoff functions
    (ARC, "return fare_rev + nonfare_rev - opex - capital_charge", "return fare_rev + nonfare_rev - opex"),
    (ARC, "operator = fare_rev + nonfare_rev - opex - licence_charge + deferral_subsidy",
     "operator = fare_rev + nonfare_rev - opex - licence_charge - deferral_subsidy"),
    (ARC, "authority = licence_charge - deferral_subsidy", "authority = licence_charge"),
    (ARC, "np.clip(rate * reliability_score, -penalty_cap, bonus_cap)", "np.clip(rate * reliability_score, -bonus_cap, penalty_cap)"),
    (ARC, "operator = service_fee + incentive - opex", "operator = service_fee - opex"),
    (ARC, "authority = fare_rev + nonfare_rev - service_fee - incentive", "authority = fare_rev - service_fee - incentive"),
    (ARC, "threshold = projected_fare_rev * (1.0 - deadband)", "threshold = projected_fare_rev"),
    (ARC, "return share * np.maximum(0.0, threshold - fare_rev)", "return share * (threshold - fare_rev)"),
    (ARC, "tier1 = rate1 * np.maximum(0.0, np.minimum(pre_share_profit, upper) - lower)",
     "tier1 = rate1 * np.maximum(0.0, pre_share_profit - lower)"),
    (ARC, "tier2 = rate2 * np.maximum(0.0, pre_share_profit - upper)", "tier2 = rate2 * np.maximum(0.0, pre_share_profit - lower)"),
    (ARC, "return tier1 + tier2", "return tier1"),
    (ARC, "pre = revenue - opex - base_charge + relief + deferral_subsidy", "pre = revenue - opex - base_charge + deferral_subsidy"),
    (ARC, "operator = pre - share", "operator = pre"),
    (ARC, "authority = base_charge - relief - deferral_subsidy + share", "authority = base_charge - relief - deferral_subsidy"),
    (ARC, "payment = fixed_payment + revenue_share * fare_rev", "payment = fixed_payment"),
    (ARC, "operator = fare_rev + nonfare_rev - opex - capital_charge - payment", "operator = fare_rev + nonfare_rev - opex - payment"),
    # driver paths and simulation
    (PATHS, "allowable = formula_g[:, t] + deferred[:, t - 1]", "allowable = formula_g[:, t]"),
    (PATHS, "granted = np.minimum(allowable, cap)", "granted = allowable"),
    (PATHS, "index[:, t] = index[:, t - 1] * (1.0 + granted)", "index[:, t] = 1.0 + granted"),
    (PATHS, "new = (stage < 0) & starts[:, t]", "new = starts[:, t]"),
    (PATHS, "stage = np.where(stage >= len(path), -1, stage)", "stage = np.where(stage > len(path), -1, stage)"),
    (PATHS, 'energy_g[:, 1] = (1.0 + energy_g[:, 1]) * s["energy_mult"] - 1.0', 'energy_g[:, 1] = energy_g[:, 1] * s["energy_mult"]'),
    (PATHS, 'cost_g = p["w_cpi"] * cpi_g + p["w_wage"] * wage_g + p["w_energy"] * energy_g', 'cost_g = p["w_cpi"] * cpi_g + p["w_wage"] * wage_g'),
    (PATHS, 'formula_g = cost_g - p["productivity"] + p["capacity"]', 'formula_g = cost_g + p["productivity"] + p["capacity"]'),
    (PATHS, 'staff = p["staff_m"] * wage_idx', 'staff = p["staff_m"] * cpi_idx'),
    (PATHS, 'maintenance = p["maintenance_m"] * cpi_idx * (1.0 + p["ageing"]) ** t', 'maintenance = p["maintenance_m"] * cpi_idx'),
    (PATHS, '"fare_free": p["fare"] * cost_idx', '"fare_free": p["fare"] * cpi_idx'),
    (PATHS, 'np.exp(rng.normal(np.log1p(p["energy_mean"]) - 0.5 * p["energy_sd"] ** 2, p["energy_sd"], (n, T))) - 1.0',
     'np.exp(rng.normal(np.log1p(p["energy_mean"]), p["energy_sd"], (n, T))) - 1.0'),
    (PATHS, 'noise[:, t] = p["ridership_rho"] * noise[:, t - 1] + z[:, t]', "noise[:, t] = z[:, t]"),
    (PATHS, "starts[:, 0] = False", "starts[:, 0] = starts[:, 0]"),
    (PATHS, 'starts = rng.random((n, T)) < p["pandemic_prob"]', 'starts = rng.random((n, T)) < 2 * p["pandemic_prob"]'),
    # analysis
    (ANA, 'nonfare = p["nonfare_share"] * p["fare"] * drv["trips"] * drv["cpi_idx"]', 'nonfare = p["nonfare_share"] * p["fare"] * drv["trips"]'),
    (ANA, 'subsidy = p["deferral_cover"] * drv["deferred"] * fare_rev', 'subsidy = drv["deferred"] * fare_rev'),
    (ANA, 'fee = terms["B"] * (1.0 + p["fee_indexation"] * (drv["cost_idx"] - 1.0))', 'fee = terms["B"] * drv["cost_idx"]'),
    (ANA, 'fare_rev_d = drv["trips"] * drv["fare_free"]', 'fare_rev_d = drv["trips"] * drv["fare_regulated"]'),
    (ANA, "return operator.sum(axis=1) / revenue.sum(axis=1)", "return (operator / revenue).mean(axis=1)"),
    (ANA, "return (x / (1.0 + rate) ** t).sum(axis=1)", "return (x / (1.0 + rate) ** (t + 1)).sum(axis=1)"),
    (ANA, 'terms[k] = _solve(gap, -2000.0, 2000.0, increasing=(k == "B"))', 'terms[k] = _solve(gap, -2000.0, 2000.0, increasing=(k == "A"))'),
    (ANA, "return float(np.sort(x)[:k].mean())", "return float(np.sort(x)[-k:].mean())"),
    (ANA, '"p_loss_cumulative": float((op_npv < 0).mean()),', '"p_loss_cumulative": float((op_npv > 0).mean()),'),
    (ANA, '"p_loss_year": float((r["operator"] < 0).mean()),', '"p_loss_year": float((r["operator"] < 1).mean()),'),
    (ANA, '"operator_share": (d_op / total) if abs(total) > 1e-6 else None}', '"operator_share": (d_au / total) if abs(total) > 1e-6 else None}'),
    (ANA, "uncovered = dict(p, deferral_cover=0.0)", "uncovered = dict(p, deferral_cover=1.0)"),
    (ANA, 'cut = dict(drv, opex=drv["opex"] - 0.1)', 'cut = dict(drv, opex=drv["opex"] + 0.1)'),
    (ANA, '"p_at_least_one_pandemic": float(1.0 - (1.0 - p["pandemic_prob"]) ** 29)', '"p_at_least_one_pandemic": float(1.0 - (1.0 - p["pandemic_prob"]) ** 30)'),
    # RTS Link arithmetic
    (RTS, "peaked = capacity * 2 * (peak_hours + (hours - peak_hours) * offpeak_load)", "peaked = capacity * (peak_hours + (hours - peak_hours) * offpeak_load)"),
    (RTS, "return annual_cost_m * 1e6 / (365.0 * fare_sgd)", "return annual_cost_m * 1e6 / (360.0 * fare_sgd)"),
    (RTS, "return (1.0 - myr_share) + myr_share * rate0 / rate", "return (1.0 - myr_share) + myr_share * rate / rate0"),
    (RTS, "month_end[d[:7]] = v", "month_end[d[:4]] = v"),
    (RTS, "vol = float(np.std(np.diff(np.log(m)), ddof=1) * math.sqrt(12.0))", "vol = float(np.std(np.diff(np.log(m)), ddof=1) * math.sqrt(252.0))"),
    (RTS, "return daily_trips * 365.0 * minutes / 60.0 * value_per_hour / 1e6", "return daily_trips * 365.0 * minutes * value_per_hour / 1e6"),
    # procurement
    (PROC, "return w_quality * quality / best_quality + (1.0 - w_quality) * lowest_price / price",
     "return w_quality * quality / best_quality + (1.0 - w_quality) * price / lowest_price"),
    (PROC, "return dp / (dp + dq)", "return dq / (dp + dq)"),
    (PROC, 'trial = dict(terms, B=terms["B"] * (1.0 - d))', 'trial = dict(terms, B=terms["B"] * (1.0 + d))'),
    # workbook formulas and their Python counterpart (the comparison test has to notice either side moving)
    (WB, '("granted", "0", "MIN({c}8,fare_cap)"),', '("granted", "0", "MAX({c}8,fare_cap)"),'),
    (WB, '("deferred", "0", "{c}8-{c}9"),', '("deferred", "0", "{c}8"),'),
    (WB, '("B_operator", "{c}25-{c}21", "{c}25-{c}21"),', '("B_operator", "{c}25-{c}21", "{c}25-{c}20"),'),
    (WB, '"F": f"=C{r}+D{r}-E{r}-term_A", "G": "=term_A",', '"F": f"=C{r}+D{r}-E{r}", "G": "=term_A",'),
    (WB, '"J": f"=shortfall_share*MAX(0,fare*trips_m*(1-deadband)-C{r})",', '"J": f"=shortfall_share*MAX(0,fare*trips_m*(1+deadband)-C{r})",'),
    (WB, '+rate2*MAX(0,K{r}-band2*(C{r}+D{r}))",', '+rate2*MAX(0,K{r}-band1*(C{r}+D{r}))",'),
    (WB, 'c = arc.hybrid(fr, nf, ox, terms["C"], base_fare, p["deadband"], p["shortfall_share"],',
     'c = arc.hybrid(fr, nf, ox, terms["C"], base_fare, p["shortfall_share"], p["deadband"],'),
    (WB, 'out[f"{k}_authority"] = res[k]["authority"]', 'out[f"{k}_authority"] = res[k]["operator"]'),
    (WB, '("C_relief", "shortfall_share*MAX(0,{c}36*(1-deadband)-{c}14)", "shortfall_share*MAX(0,{c}36*(1-deadband)-{c}14)"),',
     '("C_relief", "shortfall_share*MAX(0,{c}36*(1-deadband)-{c}14)", "shortfall_share*MAX(0,{c}14*(1-deadband)-{c}14)"),'),
]

# index in MUTANTS -> reason no test can distinguish it from the original
EQUIVALENT: dict[int, str] = {}


def run_tests() -> bool:
    return subprocess.run(PYTEST, cwd=ROOT, capture_output=True, text=True).returncode == 0


def main() -> int:
    args = sys.argv[1:]
    key = args[args.index("--record") + 1] if "--record" in args else "ci"
    if not run_tests():
        print("the unmutated suite fails; fix that first")
        return 2
    originals = {f: (ROOT / f).read_bytes() for f in {m[0] for m in MUTANTS}}
    survivors = []
    try:
        for i, (f, old, new) in enumerate(MUTANTS):
            src = originals[f].decode("utf-8")
            if src.count(old) < 1:
                raise SystemExit(f"mutant {i}: pattern not found in {f}: {old}")
            (ROOT / f).write_bytes(src.replace(old, new, 1).encode("utf-8"))
            try:
                alive = run_tests()
            finally:
                (ROOT / f).write_bytes(originals[f])
            print(f"{i:02d} {'SURVIVED' if alive else 'killed  '} {f}: {old[:70]}", flush=True)
            if alive:
                survivors.append({"index": i, "file": f, "original": old, "mutant": new})
    finally:
        for f, data in originals.items():
            (ROOT / f).write_bytes(data)
    real = [s for s in survivors if s["index"] not in EQUIVALENT]
    record = {"total": len(MUTANTS), "killed": len(MUTANTS) - len(survivors), "survivors": survivors,
              "equivalent": [{"index": i, "reason": r} for i, r in sorted(EQUIVALENT.items()) if any(s["index"] == i for s in survivors)]}
    if key != "ci":
        data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        data[key] = record
        if key == "first":
            data["final"] = record
        OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"killed {record['killed']} of {record['total']}; unexplained survivors: {len(real)}")
    return 1 if ("--require-all" in args and real) else 0


if __name__ == "__main__":
    raise SystemExit(main())
