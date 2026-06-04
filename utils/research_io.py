import csv
import io
import os
from datetime import datetime


RESEARCH_DIR = os.path.join("data", "research")
REFLECTION_PATH = os.path.join(RESEARCH_DIR, "reflection_entries.csv")
USER_TEST_PATH = os.path.join(RESEARCH_DIR, "user_test_feedback.csv")

USER_TEST_FIELDNAMES = [
    "created_at",
    "pre_emotion",
    "post_emotion",
    "understood",
    "naturalness",
    "supportiveness",
    "discomfort",
    "willingness",
    "pbq_grief",
    "pbq_guilt",
    "pbq_anger",
    "icg_yearning",
    "icg_impairment",
    "notes",
]


def _ensure_research_dir():
    os.makedirs(RESEARCH_DIR, exist_ok=True)


def _append_csv(path: str, row: dict, fieldnames: list[str]):
    _ensure_research_dir()
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def rows_to_csv_bytes(rows: list[dict], fieldnames: list[str]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    return output.getvalue().encode("utf-8-sig")


def csv_file_bytes(path: str, fieldnames: list[str] | None = None) -> bytes:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    if fieldnames:
        return rows_to_csv_bytes([], fieldnames)
    return b""


def save_reflection_entry(exercise: dict, response_text: str, meta: dict | None = None):
    meta = meta or {}
    risk = meta.get("risk", {}) or {}
    strategy = meta.get("strategy", {}) or {}
    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exercise_key": exercise.get("key", ""),
        "exercise_title": exercise.get("title", ""),
        "exercise_category": exercise.get("category", ""),
        "response_text": response_text,
        "risk_level": risk.get("level", ""),
        "strategy_name": strategy.get("name", ""),
        "grief_stage": meta.get("grief_stage", ""),
    }
    _append_csv(
        REFLECTION_PATH,
        row,
        [
            "created_at",
            "exercise_key",
            "exercise_title",
            "exercise_category",
            "response_text",
            "risk_level",
            "strategy_name",
            "grief_stage",
        ],
    )


def save_user_test_feedback(feedback: dict):
    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **feedback,
    }
    _append_csv(USER_TEST_PATH, row, USER_TEST_FIELDNAMES)
    return row
