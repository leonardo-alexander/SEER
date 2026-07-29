import json
from statistics import mean, pstdev
from typing import Dict, List, Optional

import config
import hf_model
import prompt_builder
from executor import run_sql


def _reduction() -> str:
    return getattr(config, "GATE_REDUCTION", "mean")


def _exec_guided() -> bool:
    return getattr(config, "EXEC_GUIDED", True)


_CALIB_SCHEMA = """CREATE TABLE departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL,
    location TEXT
);

CREATE TABLE employees (
    emp_id INTEGER PRIMARY KEY,
    emp_name TEXT NOT NULL,
    dept_id INTEGER,
    salary INTEGER,
    hire_date TEXT,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);"""

_CALIB_QUESTIONS = [
    "List the names of all employees.",
    "How many employees are there?",
    "What is the average salary of all employees?",
    "List all department names.",
    "Find the names of employees in the department with dept_id 1.",
    "What is the highest salary among all employees?",
    "List employee names ordered by salary in descending order.",
    "How many departments are located in 'New York'?",
    "Find the total salary paid to employees in each department.",
    "List the names of employees hired after '2020-01-01'.",
]


def _model_key(model_id: str) -> str:
    for k, v in config.MODELS.items():
        if v == model_id:
            return k
    return model_id


def calibrate(model_id: str, gates: List[str]) -> Dict[str, float]:
    red = _reduction()
    mkey = _model_key(model_id)
    cache = {}
    if config.CALIB_PATH.exists():
        with open(config.CALIB_PATH, "r") as f:
            cache = json.load(f)

    def key(g):
        return f"{mkey}/{g}"

    def _stale(g):
        rec = cache.get(key(g))
        return not isinstance(rec, dict) or rec.get("reduction") != red

    need = [g for g in gates if _stale(g)]
    if not need:
        return {g: cache[key(g)]["tau"] for g in gates}

    print(f"[calib] Calibrating {need} (reduction={red}) for {model_id} "
          f"on {len(_CALIB_QUESTIONS)} queries")
    tokenizer, model, device = hf_model.load_model(model_id)

    per_gate_values: Dict[str, List[float]] = {g: [] for g in need}
    for q in _CALIB_QUESTIONS:
        user = prompt_builder.build_user(_CALIB_SCHEMA, "None", q)
        prompt = hf_model.build_prompt(tokenizer, prompt_builder.system_prompt(), user)
        _, metrics = hf_model.generate_with_uncertainty(model, tokenizer, prompt, device)
        for g in need:
            per_gate_values[g].append(hf_model.reduce_metric(metrics, g, red))

    for g in need:
        vals = per_gate_values[g]
        mu, sigma = mean(vals), pstdev(vals)
        tau = mu + config.CALIB_SIGMA * sigma
        cache[key(g)] = {
            "gate": g,
            "reduction": red,
            "n_calibration": len(_CALIB_QUESTIONS),
            "mu": mu,
            "sigma": sigma,
            "tau": tau,
        }
        print(f"[calib]   {g}/{red}: mu={mu:.4f} sigma={sigma:.4f} -> tau={tau:.4f}")

    config.CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CALIB_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    return {g: cache[key(g)]["tau"] for g in gates}


def calibration_detail(model_id: str, reductions: List[str] = ("first", "mean")) -> Dict:
    tokenizer, model, device = hf_model.load_model(model_id)

    rows = []
    for q in _CALIB_QUESTIONS:
        user = prompt_builder.build_user(_CALIB_SCHEMA, "None", q)
        prompt = hf_model.build_prompt(tokenizer, prompt_builder.system_prompt(), user)
        _, metrics = hf_model.generate_with_uncertainty(model, tokenizer, prompt, device)
        row = {"question": q}
        for gate in ("entropy", "energy"):
            for red in reductions:
                row[f"{gate}_{red}"] = hf_model.reduce_metric(metrics, gate, red)
        rows.append(row)

    stats = {}
    for gate in ("entropy", "energy"):
        for red in reductions:
            vals = [r[f"{gate}_{red}"] for r in rows]
            mu, sigma = mean(vals), pstdev(vals)
            stats[f"{gate}_{red}"] = {
                "mu": mu,
                "sigma": sigma,
                "tau": mu + config.CALIB_SIGMA * sigma,
            }

    return {"rows": rows, "stats": stats}


def _logged_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    red = _reduction()
    return {
        "entropy": metrics[f"entropy_{red}"],
        "energy": metrics[f"energy_{red}"],
        "entropy_first": metrics["entropy_first"],
        "energy_first": metrics["energy_first"],
        "entropy_mean": metrics["entropy_mean"],
        "energy_mean": metrics["energy_mean"],
        "entropy_max": metrics["entropy_max"],
        "energy_max": metrics["energy_max"],
        "gen_tokens": metrics["n_tokens"],
    }


def run_standard(model_id: str, schema: str, evidence: str, question: str) -> Dict:
    tokenizer, model, device = hf_model.load_model(model_id)
    user = prompt_builder.build_user(schema, evidence, question)
    prompt = hf_model.build_prompt(tokenizer, prompt_builder.system_prompt(), user)

    sql, metrics = hf_model.generate_with_uncertainty(model, tokenizer, prompt, device)
    res = {"sql": sql, "steps": 1, "converged": True}
    res.update(_logged_metrics(metrics))
    return res


def _generic_cue(idx: int) -> str:
    cues = config.REFLECTION_CUES
    return cues[min(idx, len(cues) - 1)]


def _repair_cue(prev_sql: str, err: Optional[str], idx: int) -> str:
    prev = prev_sql.strip() or "(your previous query was empty)"
    if err:
        return (
            f"\n\nYour previous attempt was:\n{prev}\n\n"
            f"Executing it on the database produced this SQLite error:\n{err}\n\n"
            "Rewrite the query to fix this error. Output only the corrected "
            "SQLite query, with no explanation."
        )
    return (
        f"\n\nYour previous attempt was:\n{prev}\n\n"
        f"It executes, but double-check it: {_generic_cue(idx)} "
        "Output only the corrected SQLite query, with no explanation."
    )


def run_seer(model_id: str, gate: str, tau: float, schema: str,
             evidence: str, question: str, db_path=None) -> Dict:
    tokenizer, model, device = hf_model.load_model(model_id)
    system = prompt_builder.system_prompt()
    base_user = prompt_builder.build_user(schema, evidence, question)
    red = _reduction()
    exec_guided = _exec_guided() and db_path is not None

    gate_trace: List[float] = []
    converged = False
    cue = ""
    cues_used = 0

    first_sql: Optional[str] = None
    last_sql, last_metrics = "", {}
    best = None

    for step in range(1, config.K_MAX + 1):
        if exec_guided:
            user = base_user + cue
        else:
            user = base_user
            for c in range(cues_used):
                user += "\n\nReflection: " + _generic_cue(c)
        prompt = hf_model.build_prompt(tokenizer, system, user)

        sql, metrics = hf_model.generate_with_uncertainty(model, tokenizer, prompt, device)
        val = hf_model.reduce_metric(metrics, gate, red)
        gate_trace.append(val)
        last_sql, last_metrics = sql, metrics
        if first_sql is None:
            first_sql = sql

        err = None
        if exec_guided:
            _, err = run_sql(db_path, sql)
        valid = err is None

        cand = (1 if valid else 0, -val, sql, metrics)
        if best is None or cand[:2] > best[:2]:
            best = cand

        if val <= tau and (valid or not exec_guided):
            converged = True
            break

        if step < config.K_MAX:
            if exec_guided:
                cue = _repair_cue(sql, err, cues_used)
            cues_used += 1

    if exec_guided and best is not None:
        _, _, sql_final, metrics_final = best
    else:
        sql_final, metrics_final = last_sql, last_metrics

    res = {
        "sql": sql_final,
        "steps": len(gate_trace),
        "converged": converged,
        "gate": gate,
        "tau": tau,
        "gate_trace": [round(v, 5) for v in gate_trace],
        "exec_guided": exec_guided,
        "changed_from_first": sql_final.strip() != (first_sql or "").strip(),
    }
    res.update(_logged_metrics(metrics_final))
    return res
