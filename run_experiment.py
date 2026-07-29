import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import config
import bird_loader
import seer_runner
from executor import execution_match


def _summary(records: List[Dict]) -> Dict:
    scored = [r for r in records if r["pred_valid"] is not None]
    n = len(scored)
    if n == 0:
        return {"n": 0}
    ex = sum(1 for r in scored if r["correct"]) / n
    valid = sum(1 for r in scored if r["pred_valid"]) / n
    steps = sum(r["steps"] for r in scored) / n
    converged = sum(1 for r in scored if r["converged"]) / n
    by_diff: Dict[str, Dict] = {}
    for r in scored:
        d = r.get("difficulty", "unknown")
        by_diff.setdefault(d, {"n": 0, "correct": 0})
        by_diff[d]["n"] += 1
        by_diff[d]["correct"] += int(r["correct"])
    for d in by_diff:
        by_diff[d]["ex"] = by_diff[d]["correct"] / by_diff[d]["n"]
    return {
        "n": n,
        "execution_accuracy": round(ex, 4),
        "valid_sql_rate": round(valid, 4),
        "avg_steps": round(steps, 4),
        "convergence_rate": round(converged, 4),
        "by_difficulty": by_diff,
    }


def run_config(model_key: str, model_id: str, cfg: str, questions: List[Dict], taus: Dict[str, float]) -> Dict:
    out_dir = config.RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {model_key} | {cfg} | {len(questions)} questions ===")

    records: List[Dict] = []
    t0 = time.time()
    for i, q in enumerate(questions, 1):
        db_id = q["db_id"]
        gold = q.get("SQL", "").strip()
        try:
            schema = bird_loader.get_schema(db_id, config.SPLIT)
        except FileNotFoundError as e:
            print(f"  [{i}] skip ({e})")
            continue
        evidence = q.get("evidence", "") or ""
        question = q["question"]
        db_path = bird_loader._db_path(db_id, config.SPLIT)

        if cfg == "standard":
            res = seer_runner.run_standard(model_id, schema, evidence, question)
        else:
            gate = cfg.replace("seer_", "")
            res = seer_runner.run_seer(
                model_id, gate, taus[gate], schema, evidence, question, db_path=db_path
            )

        correct, valid, err = execution_match(db_path, res["sql"], gold)

        records.append({
            "idx": i - 1,
            "db_id": db_id,
            "question": question,
            "difficulty": q.get("difficulty", "unknown"),
            "gold_sql": gold,
            "pred_sql": res["sql"],
            "correct": bool(correct),
            "pred_valid": valid,
            "error": err,
            "steps": res["steps"],
            "converged": res["converged"],
            "exec_guided": res.get("exec_guided"),
            "changed_from_first": res.get("changed_from_first"),
            "entropy": res["entropy"],
            "energy": res["energy"],
            "entropy_first": res.get("entropy_first"),
            "energy_first": res.get("energy_first"),
            "entropy_mean": res.get("entropy_mean"),
            "energy_mean": res.get("energy_mean"),
            "entropy_max": res.get("entropy_max"),
            "energy_max": res.get("energy_max"),
            "gen_tokens": res.get("gen_tokens"),
            "gate_trace": res.get("gate_trace"),
        })

        if i % 25 == 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if True:
            acc = sum(1 for r in records if r["correct"]) / len(records)
            print(f"  [{i}/{len(questions)}] running EX={acc:.3f}  elapsed={(time.time()-t0)/60:.1f}m", flush=True)

    with open(out_dir / f"detail_{cfg}.json", "w") as f:
        json.dump(records, f, indent=2)
    summ = _summary(records)
    summ["model"] = model_id
    summ["config"] = cfg
    with open(out_dir / f"summary_{cfg}.json", "w") as f:
        json.dump(summ, f, indent=2)

    print(f"--- {model_key}/{cfg}: EX={summ.get('execution_accuracy')} "
          f"valid={summ.get('valid_sql_rate')} steps={summ.get('avg_steps')}")
    return summ


def main():
    p = argparse.ArgumentParser(description="SEER experiments (re-prompting)")
    p.add_argument("--model", choices=list(config.MODELS.keys()) + ["all"], default="all")
    p.add_argument("--standard-only", action="store_true")
    p.add_argument("--seer-only", action="store_true")
    args = p.parse_args()

    if args.standard_only:
        cfgs = ["standard"]
    elif args.seer_only:
        cfgs = [c for c in config.CONFIGS if c != "standard"]
    else:
        cfgs = config.CONFIGS

    model_keys = list(config.MODELS.keys()) if args.model == "all" else [args.model]
    questions = bird_loader.load_questions(config.SPLIT, limit=config.MAX_SAMPLES)
    print(f"Loaded {len(questions)} {config.SPLIT} questions (seed={config.SHUFFLE_SEED})")

    all_summaries = {}
    for mk in model_keys:
        model_id = config.MODELS[mk]
        need_gates = [c.replace("seer_", "") for c in cfgs if c.startswith("seer_")]
        taus = seer_runner.calibrate(model_id, need_gates) if need_gates else {}
        for cfg in cfgs:
            all_summaries[f"{mk}/{cfg}"] = run_config(mk, model_id, cfg, questions, taus)

    print("\n================ SUMMARY ================")
    for k, s in all_summaries.items():
        print(f"{k:28s}  EX={s.get('execution_accuracy')}  "
              f"valid={s.get('valid_sql_rate')}  steps={s.get('avg_steps')}  "
              f"conv={s.get('convergence_rate')}")


if __name__ == "__main__":
    main()
