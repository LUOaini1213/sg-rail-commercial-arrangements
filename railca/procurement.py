"""Two small procurement illustrations. The bidders, scores and prices are ILLUSTRATIVE.

1. How the weight on quality changes the winner of a price-quality evaluation.
2. What a fixed fee bid below the cost-reflective level does to the chance of a loss over the term.
The only published anchors are the size of one real bid gap (8%) and one contract length (11 years);
no real bidder's quality score or cost is public.
"""
from __future__ import annotations

import numpy as np

from railca import analysis, paths
from railca.params import JRL_BID_GAP, JRL_TERM_YEARS

BIDDERS = [
    {"name": "Bidder X", "quality": 80.0, "price": 100.0},
    {"name": "Bidder Y", "quality": 88.0, "price": 100.0 / (1.0 - JRL_BID_GAP)},
]
DISCOUNTS = [0.0, 0.04, 0.08, 0.12]


def score(quality: float, price: float, w_quality: float, best_quality: float, lowest_price: float) -> float:
    """Weighted score: quality relative to the best, price relative to the lowest (both 0 to 1)."""
    return w_quality * quality / best_quality + (1.0 - w_quality) * lowest_price / price


def winner(bidders: list[dict], w_quality: float) -> str:
    best_q = max(b["quality"] for b in bidders)
    low_p = min(b["price"] for b in bidders)
    return max(bidders, key=lambda b: (score(b["quality"], b["price"], w_quality, best_q, low_p), -b["price"]))["name"]


def switching_weight(bidders: list[dict]) -> float:
    """Quality weight at which a cheaper, lower-quality bidder and a dearer, better one tie (two bidders)."""
    x, y = sorted(bidders, key=lambda b: b["price"])
    best_q = max(x["quality"], y["quality"])
    dq = (y["quality"] - x["quality"]) / best_q
    dp = 1.0 - x["price"] / y["price"]
    return dp / (dp + dq)


def aggressive_bids(p: dict, n: int = 20000, seed: int = 20261124, years: int = JRL_TERM_YEARS) -> list[dict]:
    """Operator outcomes under a gross-cost contract when the fee is bid below the calibrated level."""
    terms = analysis.calibrate(p, years)
    drv = paths.simulate(p, n, seed, years)
    rows = []
    for d in DISCOUNTS:
        trial = dict(terms, B=terms["B"] * (1.0 - d))
        r = analysis.payoffs(p, drv, trial)["B"]
        m = analysis.cumulative_margin(r["operator"], r["revenue"])
        rows.append({"discount": d, "fee_m": float(trial["B"]), "margin_mean": float(m.mean()),
                     "margin_p5": float(np.percentile(m, 5)),
                     "p_loss_cumulative": float((r["operator"].sum(axis=1) < 0).mean()),
                     "p_loss_year": float((r["operator"] < 0).mean())})
    return rows


def run(p: dict) -> dict:
    grid = [round(0.05 * i, 2) for i in range(21)]
    return {"bidders": BIDDERS, "switching_weight": switching_weight(BIDDERS),
            "winners": [{"w_quality": w, "winner": winner(BIDDERS, w)} for w in grid],
            "years": JRL_TERM_YEARS, "aggressive_bids": aggressive_bids(p)}
