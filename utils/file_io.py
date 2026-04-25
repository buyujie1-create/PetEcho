import json
import os
from typing import Any

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")

PROFILE_PATH = os.path.join(DATA_DIR, "pet_profile.json")
MEMORY_PATH = os.path.join(DATA_DIR, "pet_memories.txt")
CHAT_HISTORY_PATH = os.path.join(DATA_DIR, "chat_history.json")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)


def _safe_write_text(path: str, text: str):
    ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _safe_read_text(path: str, default: str = "") -> str:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


def _safe_write_json(path: str, data: Any):
    ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_read_json(path: str, default: Any):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# ---------------------------
# 宠物档案
# ---------------------------
def save_pet_profile(profile: dict):
    """
    保存宠物档案信息。
    建议字段：
    - pet_name
    - pet_personality
    - pet_appearance
    """
    clean_profile = {
        "pet_name": (profile.get("pet_name", "") or "").strip(),
        "pet_personality": (profile.get("pet_personality", "") or "").strip(),
        "pet_appearance": (profile.get("pet_appearance", "") or "").strip(),
    }
    _safe_write_json(PROFILE_PATH, clean_profile)


def load_pet_profile() -> dict | None:
    data = _safe_read_json(PROFILE_PATH, None)
    if not data:
        return None
    return data


# ---------------------------
# 宠物回忆
# ---------------------------
def save_pet_memories(text: str):
    _safe_write_text(MEMORY_PATH, (text or "").strip())


def load_pet_memories() -> str:
    return _safe_read_text(MEMORY_PATH, "")


# ---------------------------
# 宠物图片
# ---------------------------
def _clear_old_images():
    ensure_data_dir()
    for old_name in os.listdir(IMAGE_DIR):
        old_path = os.path.join(IMAGE_DIR, old_name)
        if os.path.isfile(old_path):
            try:
                os.remove(old_path)
            except Exception:
                pass


def save_pet_image(uploaded_file) -> str:
    """
    保存上传的宠物图片。
    返回保存后的本地路径。
    """
    ensure_data_dir()

    filename = (uploaded_file.name or "").lower()
    ext = ".png"
    if filename.endswith(".jpg"):
        ext = ".jpg"
    elif filename.endswith(".jpeg"):
        ext = ".jpeg"
    elif filename.endswith(".png"):
        ext = ".png"

    _clear_old_images()

    image_path = os.path.join(IMAGE_DIR, f"pet_image{ext}")
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return image_path


def load_pet_image_path() -> str | None:
    if not os.path.exists(IMAGE_DIR):
        return None

    for filename in os.listdir(IMAGE_DIR):
        path = os.path.join(IMAGE_DIR, filename)
        if os.path.isfile(path) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
            return path

    return None


# ---------------------------
# 聊天历史
# ---------------------------
def save_chat_history(history: list):
    if not isinstance(history, list):
        history = []
    _safe_write_json(CHAT_HISTORY_PATH, history)


def load_chat_history() -> list:
    data = _safe_read_json(CHAT_HISTORY_PATH, [])
    if isinstance(data, list):
        return data
    return []


# ---------------------------
# 清理工具（可选）
# ---------------------------
def reset_pet_data():
    """
    清空宠物档案、回忆、聊天记录和图片。
    适合你后面做“重新开始体验”按钮。
    """
    for path in [PROFILE_PATH, MEMORY_PATH, CHAT_HISTORY_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    if os.path.exists(IMAGE_DIR):
        for filename in os.listdir(IMAGE_DIR):
            path = os.path.join(IMAGE_DIR, filename)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except Exception:
                    pass