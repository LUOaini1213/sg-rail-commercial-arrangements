"""Render README.md and the documents in docs/ from their templates, results/*.json and data/facts.csv.

Placeholders:
  {{file.json:dotted.path|fmt}}   fmt: .Nf  ,.Nf  pctN  int  str
  {{fact:F22}}                    the value column of a fact
  {{table:name}}                  tables built from results files
Usage: python scripts/build_docs.py [--check]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from railca import analysis, factbase, params  # noqa: E402

RESULTS = ROOT / "results"
TEMPLATES = ROOT / "docs" / "templates"
TARGETS = {
    ROOT / "README.md": TEMPLATES / "README.md",
    ROOT / "docs" / "policy_brief.md": TEMPLATES / "policy_brief.md",
    ROOT / "docs" / "one_page_summary.md": TEMPLATES / "one_page_summary.md",
}
PLACEHOLDER = re.compile(r"\{\{([^}]+)\}\}")
A = analysis.ARCHETYPES


def load_results() -> dict:
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("*.json"))}


def lookup(data, path: str):
    for part in path.split("."):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def fmt_value(v, fmt: str) -> str:
    if fmt == "str":
        return str(v)
    if fmt == "int":
        return f"{int(round(v)):,}"
    if fmt.startswith("pct"):
        return f"{v * 100:.{int(fmt[3:])}f}%"
    if fmt[:1] in ".,":
        return format(v, fmt)
    raise ValueError(f"unknown format {fmt}")


def _m(v: float) -> str:
    return f"{v:+,.0f}".replace("+0", "0").replace("-0", "0") if abs(v) < 0.5 else f"{v:+,.0f}"


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def table(name: str, res: dict) -> str:
    ar, rts, pr = res["archetypes.json"], res["rts.json"], res["procurement.json"]
    if name == "metrics":
        out = [_row(["Archetype", "Margin: 5th percentile", "Median", "95th percentile", "Loss in a given year",
                     "Loss over the term", "Operator worst-tail average", "Authority: standard deviation"]),
               _row(["---"] + ["---:"] * 7)]
        for k in A:
            m = ar["metrics"][k]
            out.append(_row([f"{k}. {m['name']}", f"{m['margin_p5'] * 100:.1f}%", f"{m['margin_p50'] * 100:.1f}%",
                             f"{m['margin_p95'] * 100:.1f}%", f"{m['p_loss_year'] * 100:.0f}%",
                             f"{m['p_loss_cumulative'] * 100:.0f}%", f"{m['operator_npv_cvar5']:,.0f}",
                             f"{m['authority_npv_sd']:,.0f}"]))
        return "\n".join(out)
    if name == "risk":
        out = [_row(["Shock (one at a time, expected path)"] + [f"{k}: operator / authority" for k in A]),
               _row(["---"] + ["---:"] * 4)]
        for row in ar["risk_allocation"]:
            out.append(_row([row["label"]] + [f"{_m(row[k]['operator'])} / {_m(row[k]['authority'])}" for k in A]))
        return "\n".join(out)
    if name == "incentives":
        out = [_row(["State of the world", "Archetype", "Kept from S$1 of extra surplus from ridership",
                     "Kept from S$1 of cost saved", "Gain from one standard deviation of reliability (S$m a year)"]),
               _row(["---", "---", "---:", "---:", "---:"])]
        for state, row in ar["marginal_incentives"].items():
            for k in A:
                out.append(_row([state.capitalize(), k, f"{row[k]['revenue']:.2f}", f"{row[k]['cost']:.2f}",
                                 f"{row[k]['reliability_m']:.1f}"]))
        return "\n".join(out)
    if name == "mapping":
        out = [_row(["Real arrangement", "Closest archetype", "Confidence", "Facts relied on", "What public sources do not say"]),
               _row(["---"] * 5)]
        for r in ar["mapping"]:
            out.append(_row([r["arrangement"], r["closest_archetype"], r["confidence"], r["basis"],
                             r["what_public_sources_do_not_say"]]))
        return "\n".join(out)
    if name == "rts_revenue":
        fares = [b["fare_sgd"] for b in rts["revenue"][0]["by_fare"]]
        out = [_row(["Ridership scenario", "Trips a day"] + [f"S${f:.0f} fare: S$m / RMm" for f in fares]),
               _row(["---", "---:"] + ["---:"] * len(fares))]
        for r in rts["revenue"]:
            out.append(_row([r["label"], f"{r['daily']:,.0f}"] +
                            [f"{b['revenue_sgd_m']:,.0f} / {b['revenue_myr_m']:,.0f}" for b in r["by_fare"]]))
        return "\n".join(out)
    if name == "rts_break_even":
        fares = [b["fare_sgd"] for b in rts["break_even"][0]["by_fare"]]
        out = [_row(["Illustrative yearly cost"] + [f"S${f:.0f} fare" for f in fares]), _row(["---"] + ["---:"] * len(fares))]
        for band in rts["break_even"]:
            out.append(_row([f"S${band['cost_m']:.0f}m"] +
                            [f"{b['daily_trips']:,.0f} ({b['share_of_opening_estimate'] * 100:.0f}%)" for b in band["by_fare"]]))
        return "\n".join(out)
    if name == "rts_fx":
        out = [_row(["Exchange rate case", "Ringgit per S$", "Revenue in S$m", "Margin, costs half in ringgit (S$m)",
                     "Margin, costs all in S$ (S$m)"]), _row(["---", "---:", "---:", "---:", "---:"])]
        for r in rts["fx_sensitivity"]:
            out.append(_row([r["label"], f"{r['myr_per_sgd']:.2f}", f"{r['revenue_sgd_m']:.1f}",
                             f"{r['margin_matched_m']:.1f}", f"{r['margin_sgd_costs_m']:.1f}"]))
        return "\n".join(out)
    if name == "rts_time":
        out = [_row(["Minutes saved per trip", "Value a year (S$m)"]), _row(["---:", "---:"])]
        out += [_row([r["minutes"], f"{r['value_m']:,.0f}"]) for r in rts["time_savings"]]
        return "\n".join(out)
    if name == "bids":
        out = [_row(["Fee relative to the calibrated level", "Mean margin", "5th percentile margin", "Loss over the term"]),
               _row(["---:", "---:", "---:", "---:"])]
        for r in pr["aggressive_bids"]:
            out.append(_row([f"-{r['discount'] * 100:.0f}%" if r["discount"] else "calibrated", f"{r['margin_mean'] * 100:.1f}%",
                             f"{r['margin_p5'] * 100:.1f}%", f"{r['p_loss_cumulative'] * 100:.0f}%"]))
        return "\n".join(out)
    if name == "sanity":
        s = ar["sanity"]
        out = [_row(["Check", "Stylised line", "Published figure", "Comment"]), _row(["---", "---:", "---:", "---"]),
               _row(["Operating cost, S$m a year", f"{s['model_opex_m']:.0f}", f"{s['dtl_fy2017_opex_m']:.0f}",
                     "Downtown Line, 2017 (F54). Same order of magnitude; the published figure may include charges the model treats separately"]),
               _row(["Fare revenue, S$m a year", f"{s['model_fare_revenue_m']:.0f}", f"{s['dtl_fy2017_fare_m']:.0f}",
                     "Downtown Line, 2017 (F54). The model is about twice as high: it uses 2025 ridership and a 2026 network-average fare"]),
               _row(["Fare revenue / operating cost", f"{s['model_fare_to_opex']:.2f}",
                     f"{s['dtl_fy2017_fare_to_opex']:.2f} and {s['smrt_trains_fy2018_fare_to_opex']:.2f}",
                     "Downtown Line 2017 and SMRT Trains to March 2018 (F53, F54). The stylised line is healthier than either published loss-making year"]),
               _row(["Operating margin", f"{s['target_margin'] * 100:.1f}%", f"{s['sbs_group_margin_2025'] * 100:.1f}%",
                     "SBS Transit group 2025, bus and rail together (F55). Close, but not a rail-only figure"])]
        return "\n".join(out)
    if name == "provenance":
        out = [_row(["Input", "Label", "Value or range", "Source or basis", "Used for"]), _row(["---"] * 5)]
        out += [_row([r[0], f"**{r[1]}**", r[2], r[3], r[4]]) for r in params.PROVENANCE]
        return "\n".join(out)
    raise KeyError(name)


def render(text: str, res: dict) -> str:
    facts = factbase.by_id()

    def sub(m: re.Match) -> str:
        token = m.group(1).strip()
        if token.startswith("table:"):
            return table(token[6:], res)
        if token.startswith("fact:"):
            return facts[token[5:]]["value"]
        src, fmt = token.rsplit("|", 1)
        fname, path = src.split(":", 1)
        return fmt_value(lookup(res[fname], path), fmt)

    return PLACEHOLDER.sub(sub, text)


def build() -> dict:
    res = load_results()
    out = {target: render(tpl.read_text(encoding="utf-8"), res) for target, tpl in TARGETS.items()}
    out[ROOT / "docs" / "fact_base.md"] = factbase.render()
    return out


def main() -> int:
    stale = []
    for target, text in build().items():
        if "--check" in sys.argv:
            if not target.exists() or target.read_text(encoding="utf-8") != text:
                stale.append(target.name)
        else:
            target.write_text(text, encoding="utf-8", newline="\n")
    if stale:
        print("out of date:", ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
