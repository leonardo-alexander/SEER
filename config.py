import os
from pathlib import Path

BIRD_ROOT = Path(os.environ.get("BIRD_ROOT", "./bird"))
RESULTS_DIR = Path(os.environ.get("SEER_RESULTS", "./results"))
CALIB_PATH = RESULTS_DIR / "calibration.json"

MODELS = {
    "qwen": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
}

SPLIT = "dev"
MAX_SAMPLES = 150
SHUFFLE_BEFORE_LIMIT = True
SHUFFLE_SEED = 42

K_MAX = 5
CALIB_SIGMA = 2.0

GATE_REDUCTION = "mean"

EXEC_GUIDED = True

GATES = ["entropy", "energy"]
CONFIGS = ["standard"] + [f"seer_{g}" for g in GATES]

MAX_SEQ_LEN = 4096
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.1
LOAD_IN_4BIT = True

REFLECTION_CUES = [
    "Re-examine the schema: verify that every column and table name you use "
    "exists exactly as written and that foreign-key relationships are correct.",
    "Re-check the query logic: verify JOIN predicates, GROUP BY, aggregation, "
    "and WHERE filters against the question and the provided evidence.",
    "Re-read the question once more and confirm the query returns exactly what "
    "is asked, with the correct columns, ordering, and any required DISTINCT.",
    "Carefully reconsider value formatting and literals (exact strings, casing, "
    "and units) so filters match the actual data in the database.",
]

SQL_TIMEOUT_SEC = 30

HF_TOKEN = os.environ.get("HF_TOKEN", "")
