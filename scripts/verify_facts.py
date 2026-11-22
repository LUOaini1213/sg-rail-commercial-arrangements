"""Re-open every source in data/facts.csv and check that each quoted passage is really there.

Usage: python scripts/verify_facts.py [--cache DIR] [--refresh]
Needs network access unless every source is already in the cache directory, so it is not part of
the offline test suite. Source pages are copyrighted and are NOT committed; only their URL, size and
SHA-256 go into data/sources_manifest.csv. Results go to results/fact_check.json.
A quote "matches" when it appears in the page text after whitespace and typographic normalisation.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACTS = ROOT / "data" / "facts.csv"
MANIFEST = ROOT / "data" / "sources_manifest.csv"
OUT = ROOT / "results" / "fact_check.json"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
INLINE = r"(?i)</?(b|strong|em|i|span|a|u|sup|sub|font|small|mark)(\s[^>]*)?>"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for a, b in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'), ("–", "-"),
                 ("—", "-"), ("−", "-"), ("­", ""), ("​", "")):
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:)])", r"\1", s)
    return s.strip()


def page_text(raw: Path) -> str:
    data = raw.read_bytes()
    if data[:5] == b"%PDF-":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(raw)).pages)
    t = data.decode("utf-8", errors="replace")
    t = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", "", t)
    t = re.sub(r"(?s)<!--.*?-->", "", t)
    t = re.sub(INLINE, "", t)
    return html.unescape(re.sub(r"<[^>]+>", "\n", t))


def cache_name(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:20]


def fetch(url: str, cache: Path, refresh: bool) -> Path:
    raw = cache / cache_name(url)
    if refresh or not raw.exists() or raw.stat().st_size == 0:
        subprocess.run(["curl", "-sL", "-m", "90", "-A", UA, "-o", str(raw), url], check=False)
    return raw


def main() -> int:
    args = sys.argv[1:]
    cache = Path(args[args.index("--cache") + 1]) if "--cache" in args else Path.home() / ".cache" / "sg-rail-commercial-arrangements"
    cache.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(FACTS.open(encoding="utf-8", newline="")))
    texts, manifest, report = {}, {}, []
    for r in rows:
        url = r["source_url"]
        if not url:
            continue
        if url not in texts:
            raw = fetch(url, cache, "--refresh" in args)
            data = raw.read_bytes() if raw.exists() else b""
            try:
                texts[url] = norm(page_text(raw)) if data else ""
            except Exception as exc:  # unreadable source counts as a failed check
                texts[url] = ""
                print("unreadable", url, exc)
            manifest[url] = {"url": url, "publisher": r["publisher"], "bytes": len(data),
                             "sha256": hashlib.sha256(data).hexdigest(), "accessed": r["accessed"]}
        found = bool(r["quote"]) and norm(r["quote"]) in texts[url]
        report.append({"id": r["id"], "status": r["status"], "quote_found": found})
    bad = [x["id"] for x in report if x["status"] == "VERIFIED" and not x["quote_found"]]
    with MANIFEST.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["url", "publisher", "bytes", "sha256", "accessed"], lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(manifest.values(), key=lambda m: m["url"]))
    summary = {
        "n_facts": len(rows),
        "n_verified": sum(r["status"] == "VERIFIED" for r in rows),
        "n_unverified": sum(r["status"] == "UNVERIFIED" for r in rows),
        "n_sources": len(manifest),
        "n_verified_quotes_found": sum(x["quote_found"] for x in report if x["status"] == "VERIFIED"),
        "verified_quotes_missing": bad,
        "n_unverified_quotes_found": sum(x["quote_found"] for x in report if x["status"] == "UNVERIFIED"),
        "checked_on": max(r["accessed"] for r in rows),
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
