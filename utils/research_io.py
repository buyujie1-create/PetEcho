import csv
import io
import os
from datetime import datetime


RESEARCH_DIR = os.path.join("data", "research")
REFLECTION_PATH = os.path.join(RESEARCH_DIR, "reflection_entries.csv")
USER_TEST_PATH = os.path.join(RESEARCH_DIR, "user_test_feedback.csv")
CHAT_TRANSCRIPT_PATH = os.path.join(RESEARCH_DIR, "chat_transcripts.csv")

USER_TEST_FIELDNAMES = [
    "created_at",
    "participant_id",
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

REFLECTION_FIELDNAMES = [
    "created_at",
    "participant_id",
    "exercise_key",
    "exercise_title",
    "exercise_category",
    "response_text",
    "risk_level",
    "strategy_name",
    "grief_stage",
]

CHAT_TRANSCRIPT_FIELDNAMES = [
    "created_at",
    "participant_id",
    "turn_id",
    "pet_name",
    "user_message",
    "assistant_message",
    "emotion_sadness",
    "emotion_loneliness",
    "emotion_yearning",
    "emotion_guilt",
    "emotion_anger",
    "emotion_numbness",
    "grief_stage",
    "risk_level",
    "risk_action",
    "risk_reasons",
    "strategy_name",
    "guidance_mode",
    "guidance_focus",
    "memory_retrieval_allowed",
    "memory_retrieval_count",
    "safety_template_used",
]


def _ensure_research_dir():
    os.makedirs(RESEARCH_DIR, exist_ok=True)


def _append_csv(path: str, row: dict, fieldnames: list[str]):
    _ensure_research_dir()
    file_exists = os.path.exists(path)
    if file_exists:
        try:
            with open(path, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
                existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames != fieldnames:
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for existing_row in existing_rows:
                        writer.writerow({name: existing_row.get(name, "") for name in fieldnames})
        except Exception:
            pass
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
        if fieldnames:
            try:
                with open(path, "r", newline="", encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                return rows_to_csv_bytes(rows, fieldnames)
            except Exception:
                return rows_to_csv_bytes([], fieldnames)
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
        "participant_id": meta.get("participant_id", ""),
        "exercise_key": exercise.get("key", ""),
        "exercise_title": exercise.get("title", ""),
        "exercise_category": exercise.get("category", ""),
        "response_text": response_text,
        "risk_level": risk.get("level", ""),
        "strategy_name": strategy.get("name", ""),
        "grief_stage": meta.get("grief_stage", ""),
    }
    _append_csv(REFLECTION_PATH, row, REFLECTION_FIELDNAMES)


def save_user_test_feedback(feedback: dict):
    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **feedback,
    }
    _append_csv(USER_TEST_PATH, row, USER_TEST_FIELDNAMES)
    return row


def save_chat_transcript_turn(
    participant_id: str,
    user_message: str,
    assistant_message: str,
    meta: dict | None = None,
):
    meta = meta or {}
    emotion = meta.get("emotion", {}) or {}
    risk = meta.get("risk", {}) or {}
    strategy = meta.get("strategy", {}) or {}
    reasons = risk.get("reasons", [])
    if isinstance(reasons, list):
        risk_reasons = " | ".join(str(item) for item in reasons)
    else:
        risk_reasons = str(reasons or "")

    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "participant_id": participant_id,
        "turn_id": meta.get("turn_id", ""),
        "pet_name": (meta.get("pet_profile", {}) or {}).get("pet_name", ""),
        "user_message": (user_message or "").strip(),
        "assistant_message": (assistant_message or "").strip(),
        "emotion_sadness": emotion.get("sadness", ""),
        "emotion_loneliness": emotion.get("loneliness", ""),
        "emotion_yearning": emotion.get("yearning", ""),
        "emotion_guilt": emotion.get("guilt", ""),
        "emotion_anger": emotion.get("anger", ""),
        "emotion_numbness": emotion.get("numbness", ""),
        "grief_stage": meta.get("grief_stage", ""),
        "risk_level": risk.get("level", ""),
        "risk_action": risk.get("action", ""),
        "risk_reasons": risk_reasons,
        "strategy_name": strategy.get("name", ""),
        "guidance_mode": strategy.get("guidance_mode", ""),
        "guidance_focus": strategy.get("guidance_focus", ""),
        "memory_retrieval_allowed": meta.get("memory_retrieval_allowed", ""),
        "memory_retrieval_count": meta.get("memory_retrieval_count", ""),
        "safety_template_used": meta.get("safety_template_used", ""),
    }
    _append_csv(CHAT_TRANSCRIPT_PATH, row, CHAT_TRANSCRIPT_FIELDNAMES)
    return row
