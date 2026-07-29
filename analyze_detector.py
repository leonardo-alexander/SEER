import json
import os
import sys
from pathlib import Path

_SIGNALS = [
    "entropy", "energy",
    "entropy_first", "energy_first",
    "entropy_mean", "energy_mean",
    "entropy_max", "energy_max",
]


def _results_dir() -> Path:
    try:
        import config
        return Path(config.RESULTS_DIR)
    except Exception:
        return Path(os.environ.get("SEER_RESULTS", "results"))


def auroc(scores, labels):
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    n_pos = sum(1 for _, l in pairs if l == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    sum_ranks_pos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def risk_coverage(scores, correct, points=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    rows = [(s, c) for s, c in zip(scores, correct) if s is not None]
    if not rows:
        return [], None
    rows.sort(key=lambda x: x[0])
    n = len(rows)
    out = []
    for cov in points:
        k = max(1, int(round(cov * n)))
        acc = sum(1 for _, c in rows[:k] if c) / k
        out.append((cov, acc))
    aurc = sum(a for _, a in out) / len(out)
    return out, aurc


def analyze_file(detail_path: Path):
    recs = json.load(open(detail_path))
    recs = [r for r in recs if r.get("pred_valid") is not None]
    if not recs:
        print(f"\n{detail_path.parent.name}/{detail_path.stem}: no scored records")
        return

    correct = [bool(r["correct"]) for r in recs]
    labels = [0 if c else 1 for c in correct]
    n, n_err = len(recs), sum(labels)
    ex = (n - n_err) / n
    print(f"\n{detail_path.parent.name}/{detail_path.stem}: "
          f"n={n}  EX={ex:.3f}  (correct={n - n_err}, incorrect={n_err})")

    if n_err == 0 or n_err == n:
        print("  (all-correct or all-incorrect — AUROC undefined for this run)")
        return

    present = [k for k in _SIGNALS if any(r.get(k) is not None for r in recs)]
    print(f"  {'signal':16s} {'AUROC':>6s} {'AURC':>6s}   risk-coverage (EX @ coverage)")
    for key in present:
        scores = [r.get(key) for r in recs]
        a = auroc(scores, labels)
        rc, aurc = risk_coverage(scores, correct)
        a_s = f"{a:.3f}" if a is not None else "  n/a"
        aurc_s = f"{aurc:.3f}" if aurc is not None else "  n/a"
        rc_s = "  ".join(f"{int(c * 100)}%:{acc:.2f}" for c, acc in rc)
        print(f"  {key:16s} {a_s:>6s} {aurc_s:>6s}   {rc_s}")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    rd = _results_dir()
    details = sorted(rd.glob("*/detail_*.json"))
    if only:
        details = [d for d in details if only in d.stem]
    if not details:
        print(f"No matching detail_*.json under {rd}. Run run_experiment.py first.")
        return

    print(f"Detector analysis over {rd}")
    for d in details:
        analyze_file(d)

    print("\nReading the table:")
    print("  - AUROC > 0.5 means the signal ranks wrong queries as more uncertain.")
    print("  - Compare *_first (SQL-start token) vs *_mean / *_max (whole query) to")
    print("    see whether sequence-level uncertainty is the better detector, and")
    print("    entropy vs energy within each.")


if __name__ == "__main__":
    main()
