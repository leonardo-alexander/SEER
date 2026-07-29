import json
import random
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import config


def _excluded(p: Path) -> bool:
    return "__MACOSX" in p.parts


@lru_cache(maxsize=8)
def _questions_path(split: str) -> Path:
    candidates = [
        config.BIRD_ROOT / split / f"{split}.json",
        config.BIRD_ROOT / f"{split}.json",
        config.BIRD_ROOT / split / "dev_20240627" / f"{split}.json",
        config.BIRD_ROOT / split / split / f"{split}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = [p for p in config.BIRD_ROOT.rglob(f"{split}.json") if not _excluded(p)]
    if matches:
        matches.sort(key=lambda p: len(p.parts))
        print(f"[bird_loader] questions auto-detected: {matches[0]}")
        return matches[0]
    return candidates[0]


@lru_cache(maxsize=1024)
def _db_path(db_id: str, split: str) -> Path:
    candidates = [
        config.BIRD_ROOT / split / f"{split}_databases" / db_id / f"{db_id}.sqlite",
        config.BIRD_ROOT / split / "dev_20240627" / "dev_databases" / "dev_databases" / db_id / f"{db_id}.sqlite",
        config.BIRD_ROOT / split / split / f"{split}_databases" / f"{split}_databases" / db_id / f"{db_id}.sqlite",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = [p for p in config.BIRD_ROOT.rglob(f"{db_id}.sqlite") if not _excluded(p)]
    if matches:
        matches.sort(key=lambda p: len(p.parts))
        return matches[0]
    return candidates[0]


def load_questions(split: str = "dev", limit: Optional[int] = None) -> List[Dict]:
    path = _questions_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {split}.json under {config.BIRD_ROOT}. "
            f"Set BIRD_ROOT (currently {config.BIRD_ROOT}) to your unzipped BIRD release."
        )
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if limit and getattr(config, "SHUFFLE_BEFORE_LIMIT", False):
        random.Random(getattr(config, "SHUFFLE_SEED", 42)).shuffle(questions)

    return questions[:limit] if limit else questions


@lru_cache(maxsize=256)
def get_schema(db_id: str, split: str = "dev", sample_rows: int = 3) -> str:
    db = _db_path(db_id, split)
    if not db.exists():
        raise FileNotFoundError(f"Database not found for db_id='{db_id}' under {config.BIRD_ROOT}")

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    cur = con.cursor()

    cur.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables = cur.fetchall()

    parts: List[str] = []
    for name, create_sql in tables:
        if not create_sql:
            continue
        block = create_sql.strip().rstrip(";") + ";"
        if sample_rows > 0:
            try:
                cur.execute(f'SELECT * FROM "{name}" LIMIT {sample_rows}')
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                if rows:
                    block += f"\n/* {sample_rows} example rows from {name}:\n"
                    block += "\t".join(cols) + "\n"
                    for r in rows:
                        block += "\t".join("" if v is None else str(v) for v in r) + "\n"
                    block += "*/"
            except sqlite3.Error:
                pass
        parts.append(block)

    con.close()
    return "\n\n".join(parts)
