import csv
import json
import os
from datetime import date, datetime
from typing import Any


RESEARCH_DIR = os.path.join("data", "research")
CHECKIN_PATH = os.path.join(RESEARCH_DIR, "daily_checkins.csv")
PLAN_PROGRESS_PATH = os.path.join(RESEARCH_DIR, "companion_plan_progress.json")
MEMORIAL_SETTINGS_PATH = os.path.join(RESEARCH_DIR, "memorial_settings.json")

CHECKIN_FIELDNAMES = [
    "created_at",
    "checkin_date",
    "emotion_intensity",
    "yearning_intensity",
    "guilt_intensity",
    "anger_intensity",
    "numbness_intensity",
    "sleep_quality",
    "appetite_quality",
    "support_need",
    "safety_thoughts",
    "safety_level",
    "notes",
]


def _ensure_dir() -> None:
    os.makedirs(RESEARCH_DIR, exist_ok=True)


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, data: Any) -> None:
    _ensure_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_daily_checkin(payload: dict) -> None:
    _ensure_dir()
    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkin_date": payload.get("checkin_date") or date.today().isoformat(),
        "emotion_intensity": payload.get("emotion_intensity", ""),
        "yearning_intensity": payload.get("yearning_intensity", ""),
        "guilt_intensity": payload.get("guilt_intensity", ""),
        "anger_intensity": payload.get("anger_intensity", ""),
        "numbness_intensity": payload.get("numbness_intensity", ""),
        "sleep_quality": payload.get("sleep_quality", ""),
        "appetite_quality": payload.get("appetite_quality", ""),
        "support_need": payload.get("support_need", ""),
        "safety_thoughts": "yes" if payload.get("safety_thoughts") else "no",
        "safety_level": payload.get("safety_level", ""),
        "notes": (payload.get("notes") or "").strip(),
    }

    file_exists = os.path.exists(CHECKIN_PATH)
    with open(CHECKIN_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CHECKIN_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_daily_checkins() -> list[dict]:
    if not os.path.exists(CHECKIN_PATH):
        return []
    try:
        with open(CHECKIN_PATH, "r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def latest_checkin() -> dict | None:
    rows = load_daily_checkins()
    if not rows:
        return None
    return rows[-1]


def load_plan_progress() -> dict:
    data = _read_json(PLAN_PROGRESS_PATH, {})
    return data if isinstance(data, dict) else {}


def save_plan_progress(progress: dict) -> None:
    clean = {str(k): bool(v) for k, v in (progress or {}).items()}
    _write_json(PLAN_PROGRESS_PATH, clean)


def load_memorial_settings() -> dict:
    data = _read_json(MEMORIAL_SETTINGS_PATH, {})
    return data if isinstance(data, dict) else {}


def save_memorial_settings(settings: dict) -> None:
    clean = {
        "memorial_date": settings.get("memorial_date", ""),
        "label": (settings.get("label") or "纪念日").strip(),
        "note": (settings.get("note") or "").strip(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(MEMORIAL_SETTINGS_PATH, clean)


def clear_companion_plan_data() -> None:
    for path in [CHECKIN_PATH, PLAN_PROGRESS_PATH, MEMORIAL_SETTINGS_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
