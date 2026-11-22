"""Download the ECB euro reference rates and keep the SGD and MYR columns from 2015 onwards.

Usage: python scripts/fetch_fx.py     (needs network; the committed CSV is what the analysis uses)
Source: ECB statistics. Values are copied unmodified; reuse is permitted with attribution (fact F81).
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
OUT = ROOT / "data" / "ecb_eur_sgd_myr.csv"
META = ROOT / "data" / "ecb_eur_sgd_myr.meta.json"


def main() -> None:
    raw = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=90).read()
    rows = list(csv.reader(io.TextIOWrapper(zipfile.ZipFile(io.BytesIO(raw)).open("eurofxref-hist.csv"))))
    head = rows[0]
    i_sgd, i_myr = head.index("SGD"), head.index("MYR")
    keep = [(r[0], r[i_sgd], r[i_myr]) for r in rows[1:] if r[0] >= "2015-01-01" and r[i_sgd] != "N/A" and r[i_myr] != "N/A"]
    keep.sort()
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["date", "eur_sgd", "eur_myr"])
        w.writerows(keep)
    META.write_text(json.dumps({"source": "ECB statistics", "url": URL, "zip_bytes": len(raw),
                                "zip_sha256": hashlib.sha256(raw).hexdigest(), "rows": len(keep),
                                "first": keep[0][0], "last": keep[-1][0], "accessed": datetime.date.today().isoformat()}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(len(keep), keep[0], keep[-1])


if __name__ == "__main__":
    main()
