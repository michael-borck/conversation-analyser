#!/usr/bin/env python3
"""Calibrate the 0-100 critical-thinking score against labelled transcripts.

The authoritative engagement signal is the ordinal `band` (One-Shot < Delegator <
Directed < Iterative < Critical). The composite score should be monotonic with it:
a higher-band conversation should score higher. This tool measures that — counting
*band-order inversions* (pairs where a lower-band transcript scores above a
higher-band one) — and can grid-search weights to minimise them.

    python scripts/calibrate_ct_score.py --signals /path/to/signals/individual
    python scripts/calibrate_ct_score.py --grid --signals ...

Input JSONs use the marking_pipeline shape: each has
`transcript.labels`, `transcript.pushback_regex_hits`, `transcript.suggested_band`.
With no --signals, a small built-in representative corpus is used (no real data),
which is also what the regression test exercises. Nothing is written; the
recommended weights are printed for you to paste into config.py.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from conversation_analyser import taxonomy  # noqa: E402
from conversation_analyser.config import (  # noqa: E402
    BAND_ORDER,
    CHAIN_CAP,
    CT_SCORE_WEIGHTS,
    PUSHBACK_CAP,
)

# Built-in representative corpus: (labels, pushback_hits). Bands are computed, not
# hard-coded. Spans the spectrum; contains no real student data.
CORPUS = [
    (["DG"], 0),
    (["DG", "DG"], 0),
    (["DG", "DG", "DG", "NQ"], 0),
    (["NQ", "DG", "AC", "DG", "NQ"], 0),
    (["NQ", "FU", "NQ", "AC"], 1),
    (["NQ", "NQ", "FU", "EX", "NQ", "NQ"], 1),
    (["NQ", "EX", "FU", "EX", "NQ"], 1),
    (["NQ", "EX", "FU", "EX", "CH", "FU"], 2),
    (["FU", "EX", "EX", "CH", "NQ"], 2),
    (["CH", "EX", "CH", "EX", "CH"], 4),
    (["CH", "EX", "FU", "CH", "EX", "CH"], 5),
    (["NQ", "CH", "EX", "CH", "EX", "FU", "CH"], 4),
]


def _norm(x: float, cap: int) -> float:
    return min(x / cap, 1.0) if cap else 0.0


def score(ratios: dict, chain: int, pushback: int, w: dict, cc: int, pc: int) -> float:
    raw = (
        w["ratio"] * ratios["critical_thinking"]
        + w["chain"] * _norm(chain, cc)
        + w["pushback"] * _norm(pushback, pc)
        - w["deleg"] * ratios["delegation"]
        - w["filler"] * ratios["filler"]
    )
    return round(100.0 * max(0.0, min(1.0, raw)), 1)


def item_from_labels(labels: list[str], pushback: int) -> dict:
    ratios = taxonomy.compute_ratios(labels)
    chain = taxonomy.longest_engaged_chain(labels)
    band = taxonomy.suggested_band(
        ratios=ratios, longest_chain=chain, pushback_hits=pushback,
        user_turn_count=len(labels), labels=labels,
    )
    return {"ratios": ratios, "chain": chain, "pushback": pushback, "band": band}


def load_signals(path: str) -> list[dict]:
    items = []
    for f in sorted(Path(path).glob("*.json")):
        t = json.load(open(f)).get("transcript", {})
        labels = t.get("labels")
        if not labels:
            continue
        items.append(item_from_labels(labels, t.get("pushback_regex_hits", 0)))
    return items


def inversions(items: list[dict], w: dict, cc: int, pc: int) -> tuple[int, int]:
    scored = [(BAND_ORDER.index(it["band"]), score(it["ratios"], it["chain"], it["pushback"], w, cc, pc)) for it in items]
    inv = tot = 0
    for (bi, si), (bj, sj) in itertools.combinations(scored, 2):
        if bi == bj:
            continue
        tot += 1
        lo, hi = (si, sj) if bi < bj else (sj, si)
        if lo > hi:  # lower band scored strictly higher
            inv += 1
    return inv, tot


def report(items: list[dict], w: dict, cc: int, pc: int, label: str) -> float:
    by = defaultdict(list)
    for it in items:
        by[it["band"]].append(score(it["ratios"], it["chain"], it["pushback"], w, cc, pc))
    print(f"\n[{label}] weights={w} caps=(chain={cc}, pushback={pc})")
    for b in BAND_ORDER:
        if by[b]:
            v = by[b]
            print(f"  {b:<10} n={len(v):<3} score min/median/max = {min(v):>5.1f} / {median(v):>5.1f} / {max(v):>5.1f}")
    inv, tot = inversions(items, w, cc, pc)
    rate = inv / tot if tot else 0.0
    print(f"  band-order inversions: {inv}/{tot} ({100 * rate:.1f}%)")
    return rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", help="dir of transcript signal JSONs (else built-in corpus)")
    ap.add_argument("--grid", action="store_true", help="grid-search weights for fewest inversions")
    a = ap.parse_args()

    items = load_signals(a.signals) if a.signals else [item_from_labels(l, p) for l, p in CORPUS]
    print(f"loaded {len(items)} labelled transcripts ({'real' if a.signals else 'built-in corpus'})")
    report(items, CT_SCORE_WEIGHTS, CHAIN_CAP, PUSHBACK_CAP, "current")

    if a.grid:
        best = None
        grid = itertools.product(
            [0.4, 0.5, 0.6, 0.7], [0.1, 0.2, 0.25], [0.1, 0.15, 0.2],
            [0.2, 0.3, 0.4], [0.1, 0.2],
        )
        for r, c, p, d, f in grid:
            w = {"ratio": r, "chain": c, "pushback": p, "deleg": d, "filler": f}
            inv, tot = inversions(items, w, CHAIN_CAP, PUSHBACK_CAP)
            rate = inv / tot if tot else 0.0
            # tie-break toward higher ratio weight (more interpretable)
            key = (rate, -r)
            if best is None or key < best[0]:
                best = (key, w)
        print("\n=== recommended (fewest inversions) ===")
        report(items, best[1], CHAIN_CAP, PUSHBACK_CAP, "best")
        print(f"\nCT_SCORE_WEIGHTS = {best[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
