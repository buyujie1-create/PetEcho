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
    "participant_id",
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


def normalize_checkin_payload(payload: dict) -> dict:
    return {
        "created_at": payload.get("created_at") or datetime.now().isoformat(timespec="seconds"),
        "participant_id": payload.get("participant_id", ""),
        "checkin_date": payload.get("checkin_date") or date.today().isoformat(),
        "emotion_intensity": payload.get("emotion_intensity", ""),
        "yearning_intensity": payload.get("yearning_intensity", ""),
        "guilt_intensity": payload.get("guilt_intensity", ""),
        "anger_intensity": payload.get("anger_intensity", ""),
        "numbness_intensity": payload.get("numbness_intensity", ""),
        "sleep_quality": payload.get("sleep_quality", ""),
        "appetite_quality": payload.get("appetite_quality", ""),
        "support_need": payload.get("support_need", ""),
        "safety_thoughts": "yes" if payload.get("safety_thoughts") in [True, "yes", "true", "1", 1] else "no",
        "safety_level": payload.get("safety_level", ""),
        "notes": (payload.get("notes") or "").strip(),
    }


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
    row = normalize_checkin_payload(payload)

    file_exists = os.path.exists(CHECKIN_PATH)
    if file_exists:
        try:
            with open(CHECKIN_PATH, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
                existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames != CHECKIN_FIELDNAMES:
                with open(CHECKIN_PATH, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=CHECKIN_FIELDNAMES)
                    writer.writeheader()
                    for existing_row in existing_rows:
                        writer.writerow({name: existing_row.get(name, "") for name in CHECKIN_FIELDNAMES})
        except Exception:
            pass
    with open(CHECKIN_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CHECKIN_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return row


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


def normalize_memorial_settings(settings: dict) -> dict:
    return {
        "memorial_date": settings.get("memorial_date", ""),
        "label": (settings.get("label") or "纪念日").strip(),
        "note": (settings.get("note") or "").strip(),
        "updated_at": settings.get("updated_at") or datetime.now().isoformat(timespec="seconds"),
    }


def save_memorial_settings(settings: dict) -> None:
    clean = normalize_memorial_settings(settings)
    _write_json(MEMORIAL_SETTINGS_PATH, clean)
    return clean


def build_companion_state_package(
    checkins: list[dict],
    plan_progress: dict,
    memorial_settings: dict,
) -> dict:
    return {
        "package_type": "petecho_companion_state",
        "version": 1,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "daily_checkins": [
            normalize_checkin_payload(row)
            for row in (checkins or [])
            if isinstance(row, dict)
        ],
        "plan_progress": {str(k): bool(v) for k, v in (plan_progress or {}).items()},
        "memorial_settings": normalize_memorial_settings(memorial_settings or {}),
    }


def companion_state_package_bytes(package: dict) -> bytes:
    return json.dumps(package, ensure_ascii=False, indent=2).encode("utf-8")


def load_companion_state_package(data: bytes) -> dict:
    package = json.loads(data.decode("utf-8-sig"))
    if not isinstance(package, dict):
        raise ValueError("状态包格式不正确。")
    if package.get("package_type") != "petecho_companion_state":
        raise ValueError("这不是 PetEcho 7 日状态包。")
    if package.get("version") != 1:
        raise ValueError("状态包版本暂不支持。")
    return build_companion_state_package(
        package.get("daily_checkins", []),
        package.get("plan_progress", {}),
        package.get("memorial_settings", {}),
    )


def clear_companion_plan_data() -> None:
    for path in [CHECKIN_PATH, PLAN_PROGRESS_PATH, MEMORIAL_SETTINGS_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
