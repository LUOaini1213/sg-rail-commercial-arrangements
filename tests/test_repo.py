"""Workbook-versus-Python comparison, fact-base integrity, and repository hygiene."""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter as col

from railca import analysis, factbase, params, workbook

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
P = params.base_params()
TERMS = analysis.calibrate(P)


# --- workbook ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    path = tmp_path_factory.mktemp("wb") / "model.xlsx"
    workbook.build(P, TERMS, path)
    return path, workbook.evaluate(path)


def test_workbook_projection_matches_python(evaluated):
    _, ev = evaluated
    proj = workbook.python_projection(P, TERMS)
    assert set(proj) >= {"A_operator", "B_operator", "C_operator", "D_operator", "A_authority", "B_authority",
                         "C_authority", "D_authority", "surplus", "fare", "trips", "opex", "deferred", "subsidy"}
    for name, arr in proj.items():
        assert len(arr) == P["years"]
        for t, v in enumerate(arr):
            got = float(ev[("PROJECTION", f"{col(t + 2)}{workbook.PROJECTION_ROW[name]}")])
            assert got == pytest.approx(float(v), abs=1e-9), (name, t)


def test_workbook_grid_matches_python(evaluated):
    _, ev = evaluated
    grid = workbook.python_grid(P, TERMS)
    assert len(grid) == 27
    for i, row in enumerate(grid):
        for k, v in row.items():
            got = float(ev[("SCENARIOGRID", f"{col(workbook.GRID_COLS.index(k) + 1)}{i + 2}")])
            assert got == pytest.approx(v, abs=1e-9), (k, i)


def test_workbook_grid_exercises_both_bands(evaluated):
    _, ev = evaluated
    relief = [float(ev[("SCENARIOGRID", f"J{r}")]) for r in range(2, 29)]
    share = [float(ev[("SCENARIOGRID", f"L{r}")]) for r in range(2, 29)]
    assert max(relief) > 0 and min(relief) == 0 and max(share) > 0 and min(share) == 0


def test_workbook_identity_inside_the_sheet(evaluated):
    _, ev = evaluated
    for r in range(2, 29):
        surplus = float(ev[("SCENARIOGRID", f"Q{r}")])
        for a, b in (("F", "G"), ("H", "I"), ("M", "N")):
            assert float(ev[("SCENARIOGRID", f"{a}{r}")]) + float(ev[("SCENARIOGRID", f"{b}{r}")]) == pytest.approx(surplus)
        d = float(ev[("SCENARIOGRID", f"O{r}")]) + float(ev[("SCENARIOGRID", f"P{r}")])
        assert d == pytest.approx(surplus - P["capital_charge_m"])


def test_workbook_cells_are_formulas_and_inputs_are_labelled(evaluated):
    path, _ = evaluated
    wb = load_workbook(path)
    assert wb.sheetnames == ["Inputs", "Projection", "ScenarioGrid"]
    assert all(str(wb["Projection"].cell(row=r, column=c).value).startswith("=")
               for r in range(2, len(workbook.PROJECTION) + 2) for c in range(2, P["years"] + 2))
    assert all(str(wb["ScenarioGrid"].cell(row=r, column=c).value).startswith("=") for r in range(2, 29) for c in range(3, 18))
    labels = {wb["Inputs"].cell(row=r, column=3).value for r in range(2, len(workbook.INPUT_KEYS) + 2)}
    assert labels <= {"REAL", "DERIVED", "ASSUMPTION", "ILLUSTRATIVE"}
    vals = workbook.input_values(P, TERMS)
    for i, (k, _) in enumerate(workbook.INPUT_KEYS):
        assert wb["Inputs"].cell(row=i + 2, column=1).value == k
        assert wb["Inputs"].cell(row=i + 2, column=2).value == pytest.approx(vals[k])


def test_workbook_responds_to_an_input_change(tmp_path):
    q = dict(P, fare_cap=0.02, shortfall_share=0.8)
    path = tmp_path / "alt.xlsx"
    workbook.build(q, TERMS, path)
    ev = workbook.evaluate(path)
    proj = workbook.python_projection(q, TERMS)
    assert proj["deferred"].max() > 0 and proj["subsidy"].max() > 0
    for name in ("deferred", "subsidy", "fare", "A_operator", "C_operator", "C_authority"):
        for t, v in enumerate(proj[name]):
            assert float(ev[("PROJECTION", f"{col(t + 2)}{workbook.PROJECTION_ROW[name]}")]) == pytest.approx(float(v), abs=1e-9)
    for i, row in enumerate(workbook.python_grid(q, TERMS)):
        assert float(ev[("SCENARIOGRID", f"M{i + 2}")]) == pytest.approx(row["C_operator"], abs=1e-9)


# --- fact base -----------------------------------------------------------------------------------------------
FACTS = factbase.load()


def test_fact_schema():
    assert len(FACTS) >= 40
    assert list(FACTS[0].keys()) == factbase.FIELDS
    assert [r["id"] for r in FACTS] == [f"F{i:02d}" for i in range(1, len(FACTS) + 1)]
    for r in FACTS:
        assert r["status"] in ("VERIFIED", "UNVERIFIED")
        assert r["claim"] and r["value"] and r["as_of"] and r["publisher"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["accessed"])
        assert r["source_url"].startswith("https://")
        if r["status"] == "VERIFIED":
            assert len(r["quote"]) >= 15 and r["source_type"] == "primary"
            assert "(press)" not in r["publisher"]
        else:
            assert r["note"], r["id"]


def test_fact_counts_and_conflicts():
    c = factbase.counts()
    assert c["n_verified"] + c["n_unverified"] == c["n_facts"]
    assert c["n_conflicts"] == sum("CONFLICT" in r["note"] for r in FACTS) >= 2
    check = json.loads((ROOT / "results" / "fact_check.json").read_text(encoding="utf-8"))
    assert check["n_facts"] == c["n_facts"] and check["n_verified"] == c["n_verified"]
    assert check["verified_quotes_missing"] == [] and check["n_verified_quotes_found"] == c["n_verified"]


def test_manifest_covers_every_source():
    with (ROOT / "data" / "sources_manifest.csv").open(encoding="utf-8", newline="") as fh:
        man = {r["url"]: r for r in csv.DictReader(fh)}
    assert set(man) == {r["source_url"] for r in FACTS}
    assert all(len(m["sha256"]) == 64 and int(m["bytes"]) > 0 for m in man.values())


def test_model_anchors_agree_with_the_fact_base():
    f = factbase.by_id()
    params.check_anchor_ids()
    for year, k in params.MRT_DAILY_K.items():
        assert f"{year} " in (f["F50"]["quote"] + f["F50"]["note"] + f["F51"]["quote"])
        assert f"{k:,}" in (f["F50"]["quote"] + f["F50"]["note"] + f["F51"]["quote"])
    assert f"{params.LRT_DAILY_K_2025}" in f["F51"]["quote"]
    assert f"{params.DTL_DAILY_2025:,}" in f["F52"]["quote"]
    assert f"${params.PT_FUND_CONTRIBUTION_M} million" in f["F45"]["quote"] and "30%" in f["F45"]["claim"]
    assert "7.0%" in f["F42"]["quote"] and params.FARE_INCREASE_2026 == 0.07
    assert f["F38"]["value"].startswith("0.5 cCPI + 0.4 WI + 0.1 EI")
    assert "0.1%" in f["F39"]["quote"] and "1.1%" in f["F40"]["quote"]
    for v in ("765,596", "196,643", "204,342", "1,449,230"):
        assert v in f["F55"]["quote"] + f["F55"]["note"]
    assert "$86 million" in f["F53"]["quote"] and "$838 million" in f["F53"]["quote"] and "$608 million" in f["F53"]["quote"]
    assert "$132 million" in f["F54"]["quote"] and "$69 million" in f["F54"]["quote"]
    assert "8%" in f["F23"]["quote"] and "11 years" in f["F22"]["quote"]
    from railca import rts
    assert "10,000" in f["F62"]["quote"] and rts.CAPACITY_PPHPD == 10_000
    assert "40,000" in f["F69"]["quote"] and "300,000" in f["F77"]["quote"] and "30-year" in f["F64"]["quote"]
    assert "RM5" in f["F80"]["quote"] and "SGD5" in f["F80"]["quote"]
    assert f["F75"]["status"] == "UNVERIFIED" and "S$5 to S$7" in f["F75"]["quote"]


def test_mapping_cites_existing_facts_and_claims_no_certainty():
    ids = set(factbase.by_id())
    with (ROOT / "data" / "arrangement_mapping.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 7
    for r in rows:
        assert set(re.findall(r"F\d\d", r["basis"])) <= ids and re.findall(r"F\d\d", r["basis"])
        assert r["what_public_sources_do_not_say"]


def test_provenance_labels():
    labels = {r[1] for r in params.PROVENANCE}
    assert labels == {"REAL", "DERIVED", "ASSUMPTION", "ILLUSTRATIVE", "SYNTHETIC"}
    ids = set(factbase.by_id())
    for r in params.PROVENANCE:
        assert set(re.findall(r"F\d\d", r[3])) <= ids
        if r[1] == "REAL":
            assert re.search(r"F\d\d", r[3]), r[0]


# --- documents ------------------------------------------------------------------------------------------------
def test_generated_documents_are_up_to_date():
    from scripts import build_docs
    for target, text in build_docs.build().items():
        assert target.read_text(encoding="utf-8") == text, f"{target.name} is stale: run scripts/build_docs.py"


ALLOWED = re.compile(r"\{\{[^}]+\}\}|`[^`]*`|\]\([^)]*\)|\bF\d\d\b|\b(19|20)\d\d\b|^\d\. |^#+ .*$|<!--.*?-->|\.png|S\$m|LUOaini1213", re.M)


@pytest.mark.parametrize("name", ["README.md", "policy_brief.md", "one_page_summary.md"])
def test_templates_contain_no_hand_typed_numbers(name):
    text = (ROOT / "docs" / "templates" / name).read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    left = ALLOWED.sub("", text)
    assert re.findall(r"\d[\d,.%]*", left) == []


def test_policy_brief_length_and_tone():
    text = (ROOT / "docs" / "policy_brief.md").read_text(encoding="utf-8")
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    words = [w for w in re.split(r"\s+", body) if re.search(r"[A-Za-z0-9]", w)]
    assert len(words) <= 1200, len(words)
    for section in ("## Issue", "## Options", "## Analysis", "## Trade-offs", "## What would change the picture"):
        assert section in text
    for word in ("should adopt", "must adopt", "we recommend", "unfair", "failed to"):
        assert word not in text.lower()
    summary = (ROOT / "docs" / "one_page_summary.md").read_text(encoding="utf-8")
    assert len(re.split(r"\s+", summary)) <= 520


def test_readme_opening_and_licence():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    head = readme[:1200]
    for phrase in ("Read this first", "built with AI assistance", "not affiliated", "no\n> non-public information",
                   "Wenjie Luo (GitHub LUOaini1213)", "stylised and illustrative"):
        assert phrase in head, phrase
    assert "Copyright (c) 2026 Wenjie Luo (LUOaini1213)" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "{{" not in readme


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return [ROOT / f for f in out]


def test_no_personal_data_or_local_paths():
    bad = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|[A-Za-z]:\\\\?Users|/Users/|/home/[a-z]+/|\+65[ \d]{8}|(api[_-]?key|token|secret)\s*=\s*[\"'][A-Za-z0-9_\-]{12,}", re.I)
    files = tracked_files()
    assert files
    for f in files:
        if f.suffix in (".png", ".xlsx") or not f.exists() or f.name == "test_repo.py":
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        hits = [m.group(0) for m in bad.finditer(text)]
        assert hits == [], (f.name, hits[:3])
    assert sum(f.stat().st_size for f in files if f.exists()) < 30e6


def test_ci_workflow_runs_offline_steps():
    wf = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for step in ("run_all.py", "build_docs.py --check", "pytest", "mutate.py"):
        assert step in wf
    assert "verify_facts" not in wf and "fetch_fx" not in wf


def test_test_count_matches_record(request):
    rec = json.loads((ROOT / "results" / "tests.json").read_text(encoding="utf-8"))
    opt = request.config.option
    if not opt.markexpr and not opt.deselect and not opt.keyword and len(request.session.items) > 60:   # full run only
        assert rec["n_tests"] == len(request.session.items)


def test_mutation_record_is_complete():
    rec = json.loads((ROOT / "results" / "mutation.json").read_text(encoding="utf-8"))
    for key in ("first", "final"):
        assert rec[key]["total"] >= 40 and 0 < rec[key]["killed"] <= rec[key]["total"]
        assert len(rec[key]["survivors"]) == rec[key]["total"] - rec[key]["killed"]
    assert rec["final"]["killed"] + len(rec["final"]["equivalent"]) == rec["final"]["total"]
    assert rec["final"]["killed"] >= rec["first"]["killed"]


# --- golden: committed results equal a fresh run -----------------------------------------------------------------
@pytest.mark.golden
def test_committed_results_match_a_fresh_run():
    from railca import procurement, rts

    def close(a, b, path=""):
        if isinstance(a, dict):
            assert set(a) == set(b), path
            for k in a:
                close(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            assert len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                close(x, y, f"{path}.{i}")
        elif isinstance(a, float) or isinstance(b, float):
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9), path
        else:
            assert a == b, path

    committed = json.loads((ROOT / "results" / "archetypes.json").read_text(encoding="utf-8"))
    fresh = analysis.run(P)
    for k in fresh:
        close(json.loads(json.dumps(fresh[k])), committed[k], k)
    close(json.loads(json.dumps(rts.run())), json.loads((ROOT / "results" / "rts.json").read_text(encoding="utf-8")), "rts")
    proc = json.loads((ROOT / "results" / "procurement.json").read_text(encoding="utf-8"))
    close(json.loads(json.dumps(procurement.run(P)["aggressive_bids"])), proc["aggressive_bids"], "bids")
    wb = json.loads((ROOT / "results" / "workbook.json").read_text(encoding="utf-8"))
    assert wb["max_abs_difference"] < 1e-9 and wb["cells_compared"] > 500
