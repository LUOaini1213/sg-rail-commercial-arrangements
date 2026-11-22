"""Excel workbook holding the deterministic model as live formulas, and its Python counterpart.

Sheets: Inputs (every parameter with its provenance label), Projection (expected path, four
archetypes, one column per year) and ScenarioGrid (single-year payoffs over a ridership x cost grid).
tests/test_workbook.py evaluates the formulas and compares them with the Python model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from railca import analysis, paths
from railca import archetypes as arc

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "results" / "rail_contract_model.xlsx"

INPUT_KEYS = [
    ("trips_m", "DERIVED"), ("fare", "DERIVED"), ("nonfare_share", "ASSUMPTION"),
    ("staff_m", "ASSUMPTION"), ("energy_m", "ASSUMPTION"), ("maintenance_m", "ASSUMPTION"), ("other_m", "ASSUMPTION"),
    ("ridership_growth", "ASSUMPTION"), ("cpi_mean", "ASSUMPTION"), ("wage_mean", "ASSUMPTION"),
    ("energy_mean", "ASSUMPTION"), ("ageing", "ASSUMPTION"),
    ("w_cpi", "REAL"), ("w_wage", "REAL"), ("w_energy", "REAL"), ("productivity", "REAL"), ("capacity", "REAL"),
    ("fare_cap", "ASSUMPTION"), ("deferral_cover", "ASSUMPTION"),
    ("fee_indexation", "ILLUSTRATIVE"), ("deadband", "ILLUSTRATIVE"), ("shortfall_share", "ILLUSTRATIVE"),
    ("band1", "ILLUSTRATIVE"), ("rate1", "ILLUSTRATIVE"), ("band2", "ILLUSTRATIVE"), ("rate2", "ILLUSTRATIVE"),
    ("capital_charge_m", "ILLUSTRATIVE"), ("revenue_share", "ILLUSTRATIVE"),
    ("term_A", "DERIVED"), ("term_B", "DERIVED"), ("term_C", "DERIVED"), ("term_D", "DERIVED"),
]
REF = {k: f"Inputs!$B${i + 2}" for i, (k, _) in enumerate(INPUT_KEYS)}
GRID_RIDERSHIP = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
GRID_OPEX = [0.9, 1.0, 1.1]

# Projection rows: name -> (row number, formula for the base year, formula for later years).
# {c} is the column letter, {p} the previous column, {y} the year cell, and input names are replaced by cell references.
PROJECTION = [
    ("cpi_idx", "1", "{p}2*(1+cpi_mean)"),
    ("wage_idx", "1", "{p}3*(1+wage_mean)"),
    ("energy_idx", "1", "{p}4*(1+energy_mean)"),
    ("cost_g", "0", "w_cpi*cpi_mean+w_wage*wage_mean+w_energy*energy_mean"),
    ("cost_idx", "1", "{p}6*(1+{c}5)"),
    ("formula_g", "0", "{c}5-productivity+capacity"),
    ("allowable", "0", "{c}7+{p}10"),
    ("granted", "0", "MIN({c}8,fare_cap)"),
    ("deferred", "0", "{c}8-{c}9"),
    ("fare_idx", "1", "{p}11*(1+{c}9)"),
    ("fare", "fare*{c}11", "fare*{c}11"),
    ("trips", "trips_m*(1+ridership_growth)^{y}", "trips_m*(1+ridership_growth)^{y}"),
    ("fare_rev", "{c}13*{c}12", "{c}13*{c}12"),
    ("nonfare", "nonfare_share*fare*{c}13*{c}2", "nonfare_share*fare*{c}13*{c}2"),
    ("subsidy", "deferral_cover*{c}10*{c}14", "deferral_cover*{c}10*{c}14"),
    ("staff", "staff_m*{c}3", "staff_m*{c}3"),
    ("energy", "energy_m*{c}4", "energy_m*{c}4"),
    ("maintenance", "maintenance_m*{c}2*(1+ageing)^{y}", "maintenance_m*{c}2*(1+ageing)^{y}"),
    ("other", "other_m*{c}2", "other_m*{c}2"),
    ("opex", "SUM({c}17:{c}20)", "SUM({c}17:{c}20)"),
    ("surplus", "{c}14+{c}15-{c}21", "{c}14+{c}15-{c}21"),
    ("A_operator", "{c}14+{c}15-{c}21-term_A*{c}2+{c}16", "{c}14+{c}15-{c}21-term_A*{c}2+{c}16"),
    ("A_authority", "term_A*{c}2-{c}16", "term_A*{c}2-{c}16"),
    ("B_fee", "term_B*(1+fee_indexation*({c}6-1))", "term_B*(1+fee_indexation*({c}6-1))"),
    ("B_operator", "{c}25-{c}21", "{c}25-{c}21"),
    ("B_authority", "{c}14+{c}15-{c}25", "{c}14+{c}15-{c}25"),
    ("C_relief", "shortfall_share*MAX(0,{c}36*(1-deadband)-{c}14)", "shortfall_share*MAX(0,{c}36*(1-deadband)-{c}14)"),
    ("C_pre", "{c}14+{c}15-{c}21-term_C*{c}2+{c}28+{c}16", "{c}14+{c}15-{c}21-term_C*{c}2+{c}28+{c}16"),
    ("C_share", "rate1*MAX(0,MIN({c}29,band2*({c}14+{c}15))-band1*({c}14+{c}15))+rate2*MAX(0,{c}29-band2*({c}14+{c}15))",
     "rate1*MAX(0,MIN({c}29,band2*({c}14+{c}15))-band1*({c}14+{c}15))+rate2*MAX(0,{c}29-band2*({c}14+{c}15))"),
    ("C_operator", "{c}29-{c}30", "{c}29-{c}30"),
    ("C_authority", "term_C*{c}2-{c}28-{c}16+{c}30", "term_C*{c}2-{c}28-{c}16+{c}30"),
    ("D_fare_rev", "{c}13*fare*{c}6", "{c}13*fare*{c}6"),
    ("D_operator", "{c}33+{c}15-{c}21-capital_charge_m-(term_D*{c}2+revenue_share*{c}33)",
     "{c}33+{c}15-{c}21-capital_charge_m-(term_D*{c}2+revenue_share*{c}33)"),
    ("D_authority", "term_D*{c}2+revenue_share*{c}33", "term_D*{c}2+revenue_share*{c}33"),
]
PROJECTION_ROW = {name: i + 2 for i, (name, _, _) in enumerate(PROJECTION)}
PROJECTED_ROW = len(PROJECTION) + 2   # projected fare revenue agreed at the start: values, not formulas
assert PROJECTED_ROW == 36
GRID_COLS = ["ridership_mult", "opex_mult", "fare_rev", "nonfare", "opex", "A_operator", "A_authority",
             "B_operator", "B_authority", "C_relief", "C_pre", "C_share", "C_operator", "C_authority",
             "D_operator", "D_authority", "surplus"]


def _sub(expr: str) -> str:
    """Replace input names by absolute cell references (longest names first)."""
    for k in sorted(REF, key=len, reverse=True):
        expr = expr.replace(k, REF[k])
    return expr


def input_values(p: dict, terms: dict) -> dict:
    v = {k: p[k] for k, _ in INPUT_KEYS if not k.startswith("term_")}
    v.update({f"term_{k}": float(terms[k]) for k in analysis.ARCHETYPES})
    return v


def build(p: dict, terms: dict, path: Path = PATH) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Inputs"
    ws.append(["Input", "Value", "Provenance"])
    vals = input_values(p, terms)
    for k, label in INPUT_KEYS:
        ws.append([k, vals[k], label])
    ws.append([])
    ws.append(["Stylised and illustrative. Not the terms of any real licence or concession. term_A..D are solved in Python so that each archetype earns the target margin on the expected path."])
    pr = wb.create_sheet("Projection")
    T = p["years"]
    pr.cell(row=1, column=1, value="year")
    for t in range(T):
        pr.cell(row=1, column=t + 2, value=t)
    for name, first, later in PROJECTION:
        r = PROJECTION_ROW[name]
        pr.cell(row=r, column=1, value=name)
        for t in range(T):
            c, pc = get_column_letter(t + 2), get_column_letter(t + 1)
            expr = (first if t == 0 else later).format(c=c, p=pc, y=f"{c}$1")
            pr.cell(row=r, column=t + 2, value="=" + _sub(expr))
    pr.cell(row=PROJECTED_ROW, column=1, value="projected_fare_rev (fixed at award)")
    for t in range(T):
        pr.cell(row=PROJECTED_ROW, column=t + 2, value=float(terms["projected_fare_rev"][t]))
    gr = wb.create_sheet("ScenarioGrid")
    gr.append(GRID_COLS)
    r = 2
    for rm in GRID_RIDERSHIP:
        for om in GRID_OPEX:
            f = {
                "C": f"=fare*trips_m*A{r}", "D": f"=nonfare_share*fare*trips_m*A{r}",
                "E": f"=(staff_m+energy_m+maintenance_m+other_m)*B{r}",
                "F": f"=C{r}+D{r}-E{r}-term_A", "G": "=term_A",
                "H": f"=term_B-E{r}", "I": f"=C{r}+D{r}-term_B",
                "J": f"=shortfall_share*MAX(0,fare*trips_m*(1-deadband)-C{r})",
                "K": f"=C{r}+D{r}-E{r}-term_C+J{r}",
                "L": f"=rate1*MAX(0,MIN(K{r},band2*(C{r}+D{r}))-band1*(C{r}+D{r}))+rate2*MAX(0,K{r}-band2*(C{r}+D{r}))",
                "M": f"=K{r}-L{r}", "N": f"=term_C-J{r}+L{r}",
                "O": f"=C{r}+D{r}-E{r}-capital_charge_m-(term_D+revenue_share*C{r})",
                "P": f"=term_D+revenue_share*C{r}", "Q": f"=C{r}+D{r}-E{r}",
            }
            gr.cell(row=r, column=1, value=rm)
            gr.cell(row=r, column=2, value=om)
            for col, expr in f.items():
                gr[f"{col}{r}"] = _sub(expr)
            r += 1
    for sheet in (ws, pr, gr):
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        sheet.column_dimensions["A"].width = 22
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def python_projection(p: dict, terms: dict) -> dict:
    """The same quantities as the Projection sheet, from the Python model. Arrays of length years."""
    drv = paths.deterministic(p)
    res = analysis.payoffs(p, drv, terms)
    fare_rev = drv["trips"] * drv["fare_regulated"]
    out = {"cpi_idx": drv["cpi_idx"], "cost_idx": drv["cost_idx"], "deferred": drv["deferred"],
           "fare": drv["fare_regulated"], "trips": drv["trips"], "fare_rev": fare_rev, "opex": drv["opex"],
           "subsidy": p["deferral_cover"] * drv["deferred"] * fare_rev, "surplus": res["A"]["surplus"]}
    for k in analysis.ARCHETYPES:
        out[f"{k}_operator"] = res[k]["operator"]
        out[f"{k}_authority"] = res[k]["authority"]
    return {k: np.asarray(v)[0] for k, v in out.items()}


def python_grid(p: dict, terms: dict) -> list[dict]:
    """The same quantities as the ScenarioGrid sheet, from the payoff functions."""
    rows = []
    base_fare = p["fare"] * p["trips_m"]
    opex0 = p["staff_m"] + p["energy_m"] + p["maintenance_m"] + p["other_m"]
    for rm in GRID_RIDERSHIP:
        for om in GRID_OPEX:
            fr, nf, ox = base_fare * rm, p["nonfare_share"] * base_fare * rm, opex0 * om
            a = arc.net_cost(fr, nf, ox, terms["A"])
            b = arc.gross_cost(fr, nf, ox, terms["B"])
            c = arc.hybrid(fr, nf, ox, terms["C"], base_fare, p["deadband"], p["shortfall_share"],
                           p["band1"], p["rate1"], p["band2"], p["rate2"])
            d = arc.concession(fr, nf, ox, p["capital_charge_m"], terms["D"], p["revenue_share"])
            rows.append({"ridership_mult": rm, "opex_mult": om, "fare_rev": fr, "nonfare": nf, "opex": ox,
                         "A_operator": float(a[0]), "A_authority": float(a[1]),
                         "B_operator": float(b[0]), "B_authority": float(b[1]),
                         "C_operator": float(c[0]), "C_authority": float(c[1]),
                         "D_operator": float(d[0]), "D_authority": float(d[1]),
                         "surplus": float(arc.system_surplus(fr, nf, ox))})
    return rows


def evaluate(path: Path = PATH) -> dict:
    """Evaluate every formula with the `formulas` package. Returns {(sheet, cell): value}."""
    import formulas

    model = formulas.ExcelModel().loads(str(path)).finish()
    sol = model.calculate()
    out = {}
    for key, rng in sol.items():
        try:
            sheet, cell = str(key).rsplit("!", 1)
        except ValueError:
            continue
        if ":" in cell:
            continue
        sheet = sheet.strip("'").split("]")[-1].upper()
        val = rng.value[0, 0] if hasattr(rng, "value") else rng
        out[(sheet, cell.replace("$", ""))] = val
    return out
