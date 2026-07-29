import json
from pathlib import Path
from typing import Dict, List, Optional

import config


def _load(model_key: str, cfg: str) -> Optional[List[Dict]]:
    path = config.RESULTS_DIR / model_key / f"detail_{cfg}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _summary(model_key: str, cfg: str) -> Optional[Dict]:
    path = config.RESULTS_DIR / model_key / f"summary_{cfg}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def auroc(scores: List[float], labels: List[int]) -> Optional[float]:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def ex_at_coverage(confidence: List[float], correct: List[int], coverage: float) -> Optional[float]:
    n = len(confidence)
    if n == 0:
        return None
    keep = max(1, int(round(coverage * n)))
    order = sorted(range(n), key=lambda i: confidence[i], reverse=True)[:keep]
    return sum(correct[i] for i in order) / keep


def table_one():
    print("\n" + "=" * 78)
    print("TABLE I — EXECUTION ACCURACY ON BIRD-SQL dev")
    print("=" * 78)
    header = f"{'Model':12s} {'Config':16s} {'EX%':>7s} {'Valid%':>8s} {'AvgSteps':>9s} {'Conv%':>7s} {'N':>5s}"
    print(header)
    print("-" * 78)
    for mk in config.MODELS:
        for cfg in config.CONFIGS:
            s = _summary(mk, cfg)
            if not s or s.get("n", 0) == 0:
                print(f"{mk:12s} {cfg:16s} {'--':>7s}")
                continue
            print(f"{mk:12s} {cfg:16s} "
                  f"{100*s['execution_accuracy']:7.1f} "
                  f"{100*s['valid_sql_rate']:8.1f} "
                  f"{s['avg_steps']:9.2f} "
                  f"{100*s['convergence_rate']:7.1f} "
                  f"{s['n']:5d}")
    print("-" * 78)


def table_two():
    print("\n" + "=" * 78)
    print("TABLE II — ENTROPY vs ENERGY AS ERROR DETECTORS (standard run)")
    print("=" * 78)
    print(f"{'Model':12s} {'Signal':10s} {'AUROC':>7s} {'EX@80%cov':>11s} {'EX@all':>8s} {'N':>5s}")
    print("-" * 78)
    for mk in config.MODELS:
        recs = _load(mk, "standard")
        if not recs:
            print(f"{mk:12s} {'--':>10s}")
            continue
        scored = [r for r in recs if r["pred_valid"] is not None]
        correct = [int(r["correct"]) for r in scored]
        incorrect = [1 - c for c in correct]
        base_ex = sum(correct) / len(correct) if correct else 0.0
        for sig in ("entropy", "energy"):
            vals = [float(r[sig]) for r in scored]
            a = auroc(vals, incorrect)
            ex80 = ex_at_coverage([-v for v in vals], correct, 0.80)
            a_s = f"{a:.3f}" if a is not None else "n/a"
            e_s = f"{100*ex80:.1f}" if ex80 is not None else "n/a"
            print(f"{mk:12s} {sig:10s} {a_s:>7s} {e_s:>11s} {100*base_ex:8.1f} {len(scored):5d}")
    print("-" * 78)
    print("AUROC > 0.5 means the signal ranks wrong queries as less confident.")
    print("Report whichever signal wins; do not assume energy.")


def risk_coverage_table(points=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    print("\n" + "=" * 78)
    print("RISK–COVERAGE (EX% vs coverage, standard run)")
    print("=" * 78)
    cov_hdr = " ".join(f"{int(100*p):>6d}%" for p in points)
    print(f"{'Model':12s} {'Signal':10s} {cov_hdr}")
    print("-" * 78)
    for mk in config.MODELS:
        recs = _load(mk, "standard")
        if not recs:
            continue
        scored = [r for r in recs if r["pred_valid"] is not None]
        correct = [int(r["correct"]) for r in scored]
        for sig in ("entropy", "energy"):
            conf = [-float(r[sig]) for r in scored]
            row = " ".join(
                f"{100*ex_at_coverage(conf, correct, p):6.1f}" if ex_at_coverage(conf, correct, p) is not None else "   n/a"
                for p in points
            )
            print(f"{mk:12s} {sig:10s} {row}")
    print("-" * 78)


if __name__ == "__main__":
    table_one()
    table_two()
    risk_coverage_table()
