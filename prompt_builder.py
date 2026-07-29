_SYSTEM = (
    "You are an expert data analyst who writes correct SQLite queries. "
    "Given a database schema and a question, output exactly one valid SQLite "
    "query that answers it. Output only the SQL query, with no explanation, "
    "no comments, and no markdown fences."
)

_USER_TMPL = """Database schema:
{schema}

External knowledge / evidence: {evidence}

Question: {question}

Write the SQLite query that answers the question."""


def build_user(schema: str, evidence: str, question: str) -> str:
    return _USER_TMPL.format(
        schema=schema,
        evidence=evidence.strip() or "None",
        question=question.strip(),
    )


def system_prompt() -> str:
    return _SYSTEM
