import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import config


def run_sql(db_path: Path, sql: str) -> Tuple[Optional[List[tuple]], Optional[str]]:
    if not sql or not sql.strip():
        return None, "empty query"

    con = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, timeout=config.SQL_TIMEOUT_SEC
    )
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")

    timed_out = {"flag": False}

    def _interrupt():
        timed_out["flag"] = True
        con.interrupt()

    timer = threading.Timer(config.SQL_TIMEOUT_SEC, _interrupt)
    timer.start()
    try:
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return rows, None
    except sqlite3.Error as e:
        return None, "timeout" if timed_out["flag"] else str(e)
    finally:
        timer.cancel()
        con.close()


def _normalize(rows: List[tuple]) -> Counter:
    return Counter(tuple("" if v is None else str(v) for v in r) for r in rows)


def execution_match(
    db_path: Path, pred_sql: str, gold_sql: str
) -> Tuple[bool, Optional[bool], Optional[str]]:
    gold_rows, gold_err = run_sql(db_path, gold_sql)
    if gold_err is not None:
        return False, None, f"gold error: {gold_err}"

    pred_rows, pred_err = run_sql(db_path, pred_sql)
    if pred_err is not None:
        return False, False, pred_err

    correct = _normalize(pred_rows) == _normalize(gold_rows)
    return correct, True, None
