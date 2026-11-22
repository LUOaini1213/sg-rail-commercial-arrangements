"""Run every analysis offline and write results/*.json, the Excel workbook and the figures.

Usage: python scripts/run_all.py [--no-figures]
Deterministic: fixed seeds, no network. tests/test_repo.py re-runs the numbers and compares.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from railca import analysis, factbase, params, procurement, rts, workbook  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical order (A, B, C, D)
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def dump(name: str, obj) -> None:
    (RESULTS / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")


def sanity(arche: dict) -> dict:
    d = params.derived()
    b = arche["base_year"]
    return {
        "derived": {k: v for k, v in d.items() if k not in ("pandemic_path", "cost_shares")},
        "cost_shares": d["cost_shares"], "pandemic_path": d["pandemic_path"],
        "model_fare_to_opex": b["fare_to_opex"],
        "smrt_trains_fy2018_fare_to_opex": params.SMRT_TRAINS_FY2018_M["fare"] / params.SMRT_TRAINS_FY2018_M["opex"],
        "dtl_fy2017_fare_to_opex": params.DTL_FY2017_M["fare"] / params.DTL_FY2017_M["opex"],
        "model_opex_m": b["opex_m"], "dtl_fy2017_opex_m": params.DTL_FY2017_M["opex"],
        "model_opex_over_dtl_fy2017": b["opex_m"] / params.DTL_FY2017_M["opex"],
        "model_fare_revenue_m": b["fare_revenue_m"], "dtl_fy2017_fare_m": params.DTL_FY2017_M["fare"],
        "model_fare_over_dtl_fy2017": b["fare_revenue_m"] / params.DTL_FY2017_M["fare"],
        "target_margin": params.base_params()["target_margin"], "sbs_group_margin_2025": d["sbs_group_margin"],
    }


def mapping() -> list[dict]:
    with (ROOT / "data" / "arrangement_mapping.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def workbook_check(p: dict, terms: dict) -> dict:
    from openpyxl.utils import get_column_letter as col

    path = workbook.build(p, terms)
    ev = workbook.evaluate(path)
    worst, n = 0.0, 0
    for name, arr in workbook.python_projection(p, terms).items():
        for t, v in enumerate(arr):
            worst = max(worst, abs(float(ev[("PROJECTION", f"{col(t + 2)}{workbook.PROJECTION_ROW[name]}")]) - float(v)))
            n += 1
    for i, row in enumerate(workbook.python_grid(p, terms)):
        for k, v in row.items():
            worst = max(worst, abs(float(ev[("SCENARIOGRID", f"{col(workbook.GRID_COLS.index(k) + 1)}{i + 2}")]) - v))
            n += 1
    n_formulas = sum(1 for (s, _), _v in ev.items() if s in ("PROJECTION", "SCENARIOGRID"))
    return {"file": "results/rail_contract_model.xlsx", "cells_compared": n, "cells_evaluated": n_formulas,
            "max_abs_difference": worst, "grid_rows": len(workbook.GRID_RIDERSHIP) * len(workbook.GRID_OPEX)}


def figures(arche: dict, r: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
    A = analysis.ARCHETYPES
    note = "Stylised model, illustrative contract terms. Not any real licence or concession."

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for i, k in enumerate(A):
        m = arche["metrics"][k]
        y = len(A) - 1 - i
        ax.plot([m["margin_p5"] * 100, m["margin_p95"] * 100], [y, y], color=SERIES[i], lw=2, solid_capstyle="round")
        ax.plot([m["margin_p50"] * 100], [y], "o", color=SERIES[i], ms=8, mec=SURFACE, mew=2)
        ax.text(m["margin_p95"] * 100 + 0.3, y, f"median {m['margin_p50'] * 100:.1f}%", va="center", color=MUTED, fontsize=9)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(A)))
    ax.set_yticklabels([f"{k}. {analysis.NAMES[k]}" for k in reversed(A)])
    ax.set_xlabel("Operator margin over the term, % of revenue (5th to 95th percentile)", fontsize=9)
    ax.set_title("Same expected margin, different spread", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=ax.get_xlim()[1] + 2.2)
    fig.text(0.01, 0.01, note, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "margin_spread.png", dpi=160)
    plt.close(fig)

    SHORT = {"ridership": "Ridership 10% lower every year", "energy": "Energy prices 30% higher",
             "fare_cap": "Fare increases capped at 2% a year"}
    shocks = [x for x in arche["risk_allocation"] if x["shock"] in ("ridership", "energy", "fare_cap")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, row in zip(axes, shocks):
        x = range(len(A))
        ax.bar([i - 0.2 for i in x], [row[k]["operator"] for k in A], width=0.38, color=SERIES[0], label="Operator")
        ax.bar([i + 0.2 for i in x], [row[k]["authority"] for k in A], width=0.38, color=SERIES[1], label="Authority")
        ax.set_xticks(list(x))
        ax.set_xticklabels(A)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(SHORT[row["shock"]], loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Change in present value, S$ million")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(0.01, 0.01, note, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "risk_allocation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True, sharex=True)
    labels = [t["driver"] for t in arche["tornado"]]
    for i, (ax, k) in enumerate(zip(axes, A)):
        for j, t in enumerate(arche["tornado"]):
            y = len(labels) - 1 - j
            ax.barh(y, t[k]["adverse"], color=SERIES[i], height=0.6)
            ax.barh(y, t[k]["favourable"], color=SERIES[i], height=0.6, alpha=0.45)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(f"{k}. {analysis.NAMES[k].split(',')[0]}", loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(list(reversed(labels)))
    fig.supxlabel("Change in operator present value, S$ million (solid: first-named case; pale: second)", fontsize=9, color=MUTED, y=0.075)
    fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGURES / "tornado.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, band in enumerate(r["break_even"]):
        xs = [b["fare_sgd"] for b in band["by_fare"]]
        ys = [b["daily_trips"] / 1000 for b in band["by_fare"]]
        ax.plot(xs, ys, color=SERIES[i], lw=2, marker="o", ms=8, mec=SURFACE, mew=2)
        ax.text(xs[-1] + 0.08, ys[-1] + (2.5 if i == 1 else 0), f"S${band['cost_m']:.0f}m a year", va="center", fontsize=9, color=MUTED)
    ax.axhline(r["inputs"]["opening_estimate"] / 1000, color=MUTED, lw=1, ls="--")
    ax.text(5.25, r["inputs"]["opening_estimate"] / 1000 + 1.5, "Published opening estimate: 40,000 a day", fontsize=9, color=MUTED)
    ax.set_xlabel("Average fare per trip, S$ (illustrative; no fare had been announced)")
    ax.set_ylabel("Break-even trips a day, thousand")
    ax.set_xlim(3.9, 8.3)
    ax.set_xticks([4, 5, 6, 7])
    ax.set_title("RTS Link: break-even ridership against illustrative cost bands", loc="left", fontsize=11)
    fig.text(0.01, 0.01, "Cost bands are illustrative; the operator's costs are not public.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "rts_break_even.png", dpi=160)
    plt.close(fig)

    series = rts.fx_series()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(range(len(series)), [v for _, v in series], color=SERIES[0], lw=1.5)
    ticks = [i for i, (d, _) in enumerate(series) if d[5:7] == "01" and (i == 0 or series[i - 1][0][5:7] != "01")][::2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([series[i][0][:4] for i in ticks])
    ax.set_ylabel("Ringgit per Singapore dollar")
    ax.set_title("Ringgit per Singapore dollar, daily", loc="left", fontsize=11)
    fig.text(0.01, 0.01, "Source: ECB statistics (euro reference rates); cross rate derived.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "myr_per_sgd.png", dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    params.check_anchor_ids()
    p = params.base_params()
    arche = analysis.run(p)
    arche["sanity"] = sanity(arche)
    arche["params"] = {k: v for k, v in p.items() if isinstance(v, (int, float))}
    arche["params"]["shock_fare_cap"] = dict(analysis.SHOCKS[4][2])["fare_cap"]
    ra = {x["shock"]: x for x in arche["risk_allocation"]}
    arche["headline"] = {"fare_cap_cost_authority_A": -ra["fare_cap"]["A"]["authority"],
                         "fare_cap_cost_operator_A_uncovered": -ra["fare_cap_uncovered"]["A"]["operator"],
                         "ridership_share_C": ra["ridership"]["C"]["operator_share"],
                         "pandemic_share_C": ra["pandemic"]["C"]["operator_share"],
                         "reliability_swing_B": ra["reliability"]["B"]["authority"]}
    arche["mapping"] = mapping()
    dump("archetypes.json", arche)
    r = rts.run()
    dump("rts.json", r)
    proc = procurement.run(p)
    proc["params"] = {"gap": params.JRL_BID_GAP}
    dump("procurement.json", proc)
    labels = [row[1] for row in params.PROVENANCE]
    dump("provenance.json", {"n_inputs": len(labels), **{k: labels.count(k) for k in
                             ("REAL", "DERIVED", "ASSUMPTION", "ILLUSTRATIVE", "SYNTHETIC")},
                             "facts": factbase.counts()})
    dump("workbook.json", workbook_check(p, analysis.calibrate(p)))
    if "--no-figures" not in sys.argv:
        figures(arche, r)
    print("done")


if __name__ == "__main__":
    main()
