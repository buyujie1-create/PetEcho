import base64
import html
import os
import re
from datetime import datetime
from urllib.parse import quote

import streamlit as st

from utils.file_io import (
    load_chat_history,
    load_generated_pet_avatar_path,
    load_pet_image_path,
    load_pet_memories,
    load_pet_profile,
    reset_pet_data,
    save_chat_history,
    save_pet_memories,
    save_pet_profile,
)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join(ROOT_DIR, "assets")
HOME_BACKGROUND_PATH = os.path.join(ASSET_DIR, "home_warm_pet_background.png")
DECOR_ILLUSTRATION_PATH = os.path.join(ASSET_DIR, "warm_pet_companions.png")


def asset_data_uri(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower().replace(".", "") or "png"
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{encoded}"
    except Exception:
        return ""


def svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def page_href(page_name: str) -> str:
    return "/" + quote(page_name)


def now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def reset_runtime_state() -> None:
    keys_to_clear = [
        "chat_history",
        "last_reply",
        "last_meta",
        "recent_memory_contexts",
        "pet_name_input",
        "pet_personality_input",
        "pet_appearance_input",
        "pet_memories_input",
        "chat_input",
        "pending_chat_input",
        "pending_pet_appearance",
        "pet_image_uploader",
        "reflection_exercise_text",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def reset_all_pet_data() -> None:
    from utils.companion_plan_io import clear_companion_plan_data

    reset_pet_data()
    clear_companion_plan_data()
    reset_runtime_state()
    save_chat_history([])


def load_demo_data() -> None:
    from modules.rag import build_vectorstore

    demo_profile = {
        "pet_name": "咪咪",
        "pet_personality": "很黏人、特别乖，喜欢安静地靠近主人，开心时会轻轻蹭过来。",
        "pet_appearance": "一只毛发蓬松的长毛猫，身体大部分是浅色，脸部和耳朵颜色偏深，眼睛圆圆的，看起来安静又温柔。",
    }
    demo_memories = (
        "咪咪每天早上都会跳上床叫我起床，站在我枕头边安静地看着我。"
        "它最喜欢傍晚陪我在客厅和阳台之间走来走去，安安静静待在我身边。"
        "下雨天的时候，它会蜷在我的脚边，像一团软绵绵的小棉花。"
        "每次我回家，它都会先跑到门口等我，然后慢慢蹭我的腿。"
    )

    save_pet_profile(demo_profile)
    save_pet_memories(demo_memories)
    save_chat_history([])

    try:
        build_vectorstore(demo_memories)
    except Exception:
        pass

    reset_runtime_state()
    st.session_state["pet_name_input"] = demo_profile["pet_name"]
    st.session_state["pet_personality_input"] = demo_profile["pet_personality"]
    st.session_state["pet_appearance_input"] = demo_profile["pet_appearance"]
    st.session_state["pet_memories_input"] = demo_memories
    st.session_state["chat_history"] = []
    st.session_state["last_reply"] = ""
    st.session_state["last_meta"] = {}
    st.session_state["recent_memory_contexts"] = []
    st.session_state["chat_input"] = ""


def ensure_session_state() -> None:
    saved_profile = load_pet_profile() or {}
    saved_memories = load_pet_memories()

    st.session_state.setdefault("chat_history", load_chat_history())
    st.session_state.setdefault("last_reply", "")
    st.session_state.setdefault("last_meta", {})
    st.session_state.setdefault("recent_memory_contexts", [])
    st.session_state.setdefault("pet_name_input", saved_profile.get("pet_name", ""))
    st.session_state.setdefault("pet_personality_input", saved_profile.get("pet_personality", ""))
    st.session_state.setdefault("pet_appearance_input", saved_profile.get("pet_appearance", ""))
    st.session_state.setdefault("pet_memories_input", saved_memories if saved_memories else "")
    st.session_state.setdefault("chat_input", "")

    if "pending_pet_appearance" in st.session_state:
        st.session_state["pet_appearance_input"] = st.session_state.pop("pending_pet_appearance")

    if "pending_chat_input" in st.session_state:
        st.session_state["chat_input"] = st.session_state.pop("pending_chat_input")


def image_file_to_data_uri(image_path: str | None) -> str:
    return asset_data_uri(image_path or "")


def get_default_pet_avatar_data_uri() -> str:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="112" height="112" viewBox="0 0 112 112">
      <defs>
        <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#ffd6a5"/>
          <stop offset="58%" stop-color="#ffc8dd"/>
          <stop offset="100%" stop-color="#cdeac0"/>
        </linearGradient>
      </defs>
      <rect width="112" height="112" rx="56" fill="url(#bg)"/>
      <polygon points="28,39 42,16 49,42" fill="#f5a3b7"/>
      <polygon points="84,39 70,16 63,42" fill="#f5a3b7"/>
      <circle cx="56" cy="61" r="33" fill="#fff8f1"/>
      <circle cx="43" cy="58" r="5.6" fill="#3f3432"/>
      <circle cx="69" cy="58" r="5.6" fill="#3f3432"/>
      <circle cx="41.4" cy="56.4" r="1.8" fill="#ffffff"/>
      <circle cx="67.4" cy="56.4" r="1.8" fill="#ffffff"/>
      <path d="M56 64 L51 70 L61 70 Z" fill="#f08aa6"/>
      <path d="M51 73 Q56 78 61 73" stroke="#7d5a5a" stroke-width="2.8" fill="none" stroke-linecap="round"/>
      <path d="M30 66 Q42 64 49 67" stroke="#b88b8b" stroke-width="2.3" fill="none" stroke-linecap="round"/>
      <path d="M63 67 Q70 64 82 66" stroke="#b88b8b" stroke-width="2.3" fill="none" stroke-linecap="round"/>
      <path d="M31 74 Q42 72 49 75" stroke="#b88b8b" stroke-width="2.3" fill="none" stroke-linecap="round"/>
      <path d="M63 75 Q70 72 81 74" stroke="#b88b8b" stroke-width="2.3" fill="none" stroke-linecap="round"/>
      <circle cx="86" cy="84" r="9" fill="#fff3c4"/>
      <circle cx="78" cy="76" r="3.4" fill="#fff3c4"/>
      <circle cx="89" cy="72" r="3.4" fill="#fff3c4"/>
      <circle cx="96" cy="80" r="3.4" fill="#fff3c4"/>
    </svg>
    """
    return svg_data_uri(svg)


def get_user_avatar_data_uri() -> str:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="112" height="112" viewBox="0 0 112 112">
      <defs>
        <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#bfe4ff"/>
          <stop offset="100%" stop-color="#e7f0ff"/>
        </linearGradient>
      </defs>
      <rect width="112" height="112" rx="56" fill="url(#bg)"/>
      <circle cx="56" cy="44" r="18" fill="#ffffff"/>
      <path d="M27 91 Q56 64 85 91" fill="#ffffff"/>
      <circle cx="49" cy="43" r="2.8" fill="#4b5563"/>
      <circle cx="63" cy="43" r="2.8" fill="#4b5563"/>
      <path d="M49 54 Q56 59 63 54" stroke="#6b7280" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      <path d="M24 30 C31 18, 46 16, 56 23 C68 14, 86 20, 89 36" stroke="#ffffff" stroke-width="5" fill="none" stroke-linecap="round"/>
    </svg>
    """
    return svg_data_uri(svg)


def get_pet_avatar_data_uri() -> str:
    generated_uri = image_file_to_data_uri(load_generated_pet_avatar_path())
    if generated_uri:
        return generated_uri

    uploaded_uri = image_file_to_data_uri(load_pet_image_path())
    if uploaded_uri:
        return uploaded_uri

    return get_default_pet_avatar_data_uri()


def clean_reply_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = text.replace("【回复】", "").replace("【数字宠物回复】", "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def apply_global_style(page: str = "default") -> None:
    st.markdown(
        """
        <style>
        :root {
            --pe-ink: #2f3a46;
            --pe-heading: #523229;
            --pe-body: #6c5b54;
            --pe-muted: #89746d;
            --pe-coral: #ef856c;
            --pe-rose: #f28aa1;
            --pe-sage: #a9c9af;
            --pe-mist: #d8eaf0;
            --pe-line: rgba(226, 156, 126, 0.36);
            --pe-line-strong: rgba(226, 137, 104, 0.58);
            --pe-surface: rgba(255, 252, 248, 0.88);
            --pe-surface-strong: rgba(255, 255, 255, 0.94);
            --pe-shadow: 0 18px 42px rgba(86, 61, 48, 0.09);
            --pe-shadow-soft: 0 10px 28px rgba(86, 61, 48, 0.07);
        }

        html, body, [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(118deg, rgba(255, 238, 224, 0.72) 0%, rgba(255, 250, 247, 0.84) 42%, rgba(240, 249, 244, 0.72) 100%),
                linear-gradient(180deg, #fffaf4 0%, #fffdfb 50%, #f3fbf7 100%) !important;
            color: var(--pe-body);
        }

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 247, 240, 0.96) 0%, rgba(255, 251, 249, 0.96) 55%, rgba(242, 250, 246, 0.96) 100%) !important;
            border-right: 1px solid rgba(219, 178, 156, 0.28);
            box-shadow: 10px 0 32px rgba(86, 61, 48, 0.04);
        }

        [data-testid="stSidebar"] * {
            letter-spacing: 0 !important;
        }

        [data-testid="stSidebar"] a {
            border-radius: 10px !important;
            color: #5f514d !important;
        }

        [data-testid="stSidebar"] a[aria-current="page"],
        [data-testid="stSidebar"] a:hover {
            background: rgba(91, 117, 126, 0.09) !important;
            color: var(--pe-heading) !important;
        }

        h1, h2, h3, p, div, span, label, button {
            letter-spacing: 0 !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--pe-line) !important;
            border-radius: 18px !important;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(255, 250, 246, 0.86)) !important;
            box-shadow: var(--pe-shadow-soft) !important;
        }

        .stButton button,
        .stDownloadButton button,
        .stFormSubmitButton button {
            border-radius: 999px !important;
            border: 1px solid rgba(222, 125, 93, 0.34) !important;
            background: linear-gradient(135deg, #ffb07c 0%, #ef7f8f 100%) !important;
            color: #4f2d27 !important;
            font-weight: 850 !important;
            box-shadow: 0 12px 24px rgba(209, 106, 85, 0.16) !important;
            min-height: 44px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton button:hover,
        .stDownloadButton button:hover,
        .stFormSubmitButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 28px rgba(209, 106, 85, 0.2) !important;
        }

        .stButton button:disabled {
            background: #f2e8e2 !important;
            color: #9a8279 !important;
            box-shadow: none !important;
        }

        .stTextArea textarea,
        .stTextInput input,
        .stSelectbox [data-baseweb="select"],
        .stSlider {
            border-radius: 12px !important;
        }

        .stTextArea textarea,
        .stTextInput input,
        .stSelectbox [data-baseweb="select"] > div {
            border-color: rgba(214, 160, 135, 0.48) !important;
            background: rgba(255, 253, 249, 0.96) !important;
        }

        .soft-panel {
            border: 1px solid var(--pe-line);
            background: var(--pe-surface);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: var(--pe-shadow-soft);
        }

        .page-hero {
            position: relative;
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(220px, 0.65fr);
            gap: 22px;
            align-items: center;
            min-height: 215px;
            margin-top: 0.25rem;
            margin-bottom: 1.55rem;
            padding: 28px 30px;
            border: 1px solid var(--pe-line-strong);
            border-radius: 22px;
            background:
                linear-gradient(115deg, rgba(255, 245, 236, 0.96), rgba(255, 250, 249, 0.94) 48%, rgba(241, 250, 246, 0.94)),
                linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,255,255,0));
            box-shadow: var(--pe-shadow);
            overflow: hidden;
        }

        .page-hero::before {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            top: 0;
            height: 3px;
            opacity: 0.86;
            background: linear-gradient(90deg, var(--pe-coral), var(--pe-rose), var(--pe-sage), var(--pe-mist));
        }

        .page-title {
            position: relative;
            z-index: 1;
            color: var(--pe-heading);
            font-size: 2.08rem;
            font-weight: 900;
            line-height: 1.35;
            margin: 0 0 0.42rem 0;
        }

        .page-subtitle {
            position: relative;
            z-index: 1;
            color: var(--pe-body);
            font-size: 1.02rem;
            line-height: 1.78;
            margin: 0;
            max-width: 760px;
        }

        .hero-eyebrow {
            position: relative;
            z-index: 1;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #884739;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(226, 150, 119, 0.42);
            border-radius: 999px;
            padding: 6px 11px;
            font-size: 0.88rem;
            font-weight: 800;
            margin-bottom: 10px;
        }

        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
        }

        .hero-badge {
            position: relative;
            z-index: 1;
            display: inline-flex;
            align-items: center;
            padding: 7px 11px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(148, 178, 160, 0.32);
            color: #6c5048;
            font-size: 0.88rem;
            font-weight: 800;
        }

        .hero-art {
            position: relative;
            z-index: 1;
            justify-self: end;
            width: min(270px, 100%);
            border-radius: 18px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(226, 167, 139, 0.42);
            box-shadow: 0 16px 32px rgba(86, 61, 48, 0.1);
        }

        .hero-art img {
            width: 100%;
            display: block;
        }

        .section-head {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin: 0.15rem 0 0.95rem 0;
        }

        .section-icon {
            flex: 0 0 auto;
            width: 36px;
            height: 36px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, #ffe2ce, #edf7f1);
            color: #7a453b;
            font-weight: 900;
            border: 1px solid rgba(226, 150, 119, 0.28);
        }

        .section-title {
            font-size: 1.18rem;
            font-weight: 900;
            color: var(--pe-heading);
            margin: 0;
            line-height: 1.35;
        }

        .section-desc {
            color: var(--pe-body);
            font-size: 0.94rem;
            margin-top: 0.2rem;
            line-height: 1.65;
        }

        .tiny-muted {
            color: var(--pe-muted);
            font-size: 0.9rem;
            line-height: 1.68;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin: 18px 0 6px 0;
        }

        .feature-card,
        .metric-card,
        .explain-card {
            border: 1px solid var(--pe-line);
            border-radius: 16px;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,250,247,0.82));
            padding: 17px 18px;
            box-shadow: var(--pe-shadow-soft);
            min-height: 100%;
        }

        .feature-icon {
            width: 40px;
            height: 40px;
            display: grid;
            place-items: center;
            margin-bottom: 12px;
            border-radius: 50%;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.45rem;
            font-style: italic;
            font-weight: 800;
            color: #73443a;
            background: linear-gradient(135deg, rgba(255, 224, 200, 0.96), rgba(235, 247, 240, 0.9));
            border: 1px solid rgba(255, 255, 255, 0.72);
            box-shadow: 0 8px 18px rgba(207, 104, 82, 0.12);
        }

        .feature-title,
        .metric-title,
        .explain-title {
            color: var(--pe-heading);
            font-weight: 900;
            margin-bottom: 6px;
        }

        .feature-body,
        .metric-body,
        .explain-body {
            color: var(--pe-body);
            font-size: 0.93rem;
            line-height: 1.68;
        }

        .metric-value {
            font-size: 1.08rem;
            font-weight: 900;
            color: #283a4d;
            line-height: 1.45;
        }

        .info-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 7px 11px;
            border-radius: 999px;
            background: rgba(255, 247, 239, 0.86);
            border: 1px solid rgba(226, 156, 126, 0.38);
            color: #72483f;
            font-size: 0.9rem;
            font-weight: 800;
            margin: 4px 7px 4px 0;
        }

        .nav-link-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 44px;
            padding: 0 16px;
            border-radius: 999px;
            color: #59372f !important;
            text-decoration: none !important;
            font-weight: 900;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(226, 156, 126, 0.44);
            box-shadow: 0 8px 18px rgba(86, 61, 48, 0.08);
            margin: 4px 8px 4px 0;
        }

        .decor-strip {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 12px 0 4px;
        }

        .decor-dot {
            width: 42px;
            height: 12px;
            border-radius: 999px;
            background: #ffd277;
        }

        .decor-dot:nth-child(2) { background: #ffa7b8; }
        .decor-dot:nth-child(3) { background: #bfe7bf; }
        .decor-dot:nth-child(4) { background: #bde3ff; }

        .chat-board {
            min-height: 430px;
            max-height: 58vh;
            overflow-y: auto;
            border: 1px solid var(--pe-line);
            border-radius: 18px;
            padding: 22px 18px;
            background:
                linear-gradient(180deg, rgba(255, 253, 249, 0.96), rgba(247, 252, 248, 0.92));
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.82);
        }

        .message-row {
            display: flex;
            gap: 12px;
            align-items: flex-start;
            margin: 0.8rem 0;
            animation: fadeUp 0.18s ease-out;
        }

        .message-row.right {
            justify-content: flex-end;
        }

        .avatar {
            flex: 0 0 auto;
            width: 54px;
            height: 54px;
            border-radius: 18px;
            padding: 3px;
            background: #ffffff;
            border: 1px solid rgba(226, 156, 126, 0.42);
            box-shadow: 0 8px 18px rgba(86, 61, 48, 0.1);
            overflow: hidden;
        }

        .avatar img {
            width: 100%;
            height: 100%;
            border-radius: 15px;
            object-fit: cover;
            display: block;
        }

        .message-col-left,
        .message-col-right {
            max-width: min(74%, 820px);
        }

        .meta-left,
        .meta-right {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #94756d;
            font-size: 0.82rem;
            margin-bottom: 5px;
        }

        .meta-right {
            justify-content: flex-end;
        }

        .pet-name,
        .user-name {
            color: #5f3931;
            font-weight: 900;
        }

        .pet-bubble,
        .user-bubble {
            position: relative;
            padding: 15px 17px;
            border-radius: 18px;
            line-height: 1.76;
            font-size: 1rem;
            word-break: break-word;
            white-space: normal;
        }

        .pet-bubble {
            color: #54352f;
            background: linear-gradient(180deg, #fff3f1, #fff9f5);
            border: 1px solid rgba(239, 157, 132, 0.36);
            border-bottom-left-radius: 7px;
            box-shadow: 0 10px 22px rgba(209, 106, 85, 0.08);
        }

        .pet-bubble::after {
            content: "";
            position: absolute;
            right: 15px;
            bottom: -7px;
            width: 18px;
            height: 2px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--pe-coral), var(--pe-sage));
            opacity: 0.64;
        }

        .user-bubble {
            color: #254154;
            background: linear-gradient(180deg, #eef8fa, #f7fcfb);
            border: 1px solid rgba(160, 199, 205, 0.46);
            border-bottom-right-radius: 7px;
            box-shadow: 0 10px 22px rgba(99, 164, 211, 0.08);
        }

        .empty-chat-box,
        .empty-soft-box {
            border: 1px dashed rgba(226, 156, 126, 0.45);
            border-radius: 16px;
            color: #8a6a62;
            background: rgba(255, 255, 255, 0.72);
            padding: 20px 18px;
            line-height: 1.75;
            text-align: center;
        }

        .exercise-box {
            border: 1px solid rgba(226, 156, 126, 0.38);
            background: linear-gradient(135deg, #fff7ef, #fffdf9 55%, #f2faf5);
            border-radius: 16px;
            padding: 16px 17px;
            color: #684139;
            line-height: 1.72;
        }

        .memory-card {
            position: relative;
            border: 1px solid var(--pe-line);
            border-radius: 16px;
            padding: 16px 18px 15px 18px;
            margin-bottom: 12px;
            background:
                linear-gradient(90deg, rgba(255, 210, 145, 0.22), transparent 16%),
                linear-gradient(180deg, #ffffff 0%, #fff9f4 100%);
            box-shadow: var(--pe-shadow-soft);
            overflow: hidden;
        }

        .memory-card::after {
            content: "";
            position: absolute;
            right: 16px;
            top: 15px;
            width: 44px;
            height: 24px;
            opacity: 0.5;
            background: linear-gradient(90deg, var(--pe-coral), var(--pe-rose), var(--pe-sage));
            border-radius: 999px;
            height: 3px;
            width: 54px;
        }

        .memory-card-title {
            color: #65372f;
            font-weight: 900;
            margin-bottom: 4px;
        }

        .memory-card-meta {
            font-size: 0.88rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .memory-card-text {
            color: #6f5650;
            line-height: 1.72;
        }

        .memorial-card-ui {
            position: relative;
            min-height: 420px;
            border-radius: 22px;
            padding: 28px;
            overflow: hidden;
            box-shadow: 0 20px 44px rgba(86, 61, 48, 0.14);
        }

        .memorial-card-ui::before {
            content: "";
            position: absolute;
            inset: 0;
            opacity: 0.9;
            background-size: cover;
            background-position: center;
            pointer-events: none;
        }

        .memorial-content {
            position: relative;
            z-index: 1;
            height: 100%;
        }

        .memorial-label {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(255, 255, 255, 0.88);
            color: #70453b;
            font-weight: 900;
            font-size: 0.88rem;
        }

        .memorial-name {
            color: #5f332c;
            font-size: 2rem;
            line-height: 1.32;
            font-weight: 950;
            margin: 14px 0 10px;
        }

        .memorial-line {
            color: #6a4c45;
            line-height: 1.76;
            font-size: 1rem;
        }

        .memorial-mini-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin: 18px 0;
        }

        .memorial-note {
            border-radius: 16px;
            padding: 15px 16px;
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.86);
            color: #65423a;
            line-height: 1.7;
        }

        .memorial-note-title {
            color: #7b4036;
            font-weight: 900;
            margin-bottom: 5px;
        }

        .memorial-card-ui.journal .memorial-content {
            display: grid;
            grid-template-columns: 0.86fr 1.14fr;
            gap: 18px;
            align-items: stretch;
        }

        .memorial-card-ui.night-light {
            text-align: center;
            display: grid;
            align-items: center;
        }

        .memorial-card-ui.night-light .memorial-mini-grid {
            grid-template-columns: 1fr;
            max-width: 620px;
            margin-left: auto;
            margin-right: auto;
        }

        .scale-guide {
            border-left: 4px solid var(--pe-rose);
            background: rgba(255, 255, 255, 0.78);
            border-radius: 14px;
            padding: 14px 16px;
            color: #6f5650;
            line-height: 1.72;
        }

        .console-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(7px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 980px) {
            .page-hero {
                grid-template-columns: 1fr;
                padding: 24px 22px;
            }
            .hero-art {
                justify-self: start;
                width: min(230px, 100%);
            }
            .feature-grid,
            .console-grid,
            .memorial-mini-grid,
            .memorial-card-ui.journal .memorial-content {
                grid-template-columns: 1fr;
            }
            .message-col-left,
            .message-col-right {
                max-width: calc(100vw - 130px);
            }
            .page-title {
                font-size: 1.72rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if page == "home":
        bg_uri = asset_data_uri(HOME_BACKGROUND_PATH) or asset_data_uri(DECOR_ILLUSTRATION_PATH)
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image:
                    linear-gradient(90deg, rgba(255, 250, 244, 0.86) 0%, rgba(255, 249, 243, 0.66) 43%, rgba(255, 250, 244, 0.24) 100%),
                    linear-gradient(180deg, rgba(255, 255, 255, 0.1) 0%, rgba(244, 252, 248, 0.3) 100%),
                    url("{bg_uri}") !important;
                background-size: cover !important;
                background-position: center center !important;
                background-attachment: fixed !important;
            }}

            [data-testid="stHeader"] {{
                background: rgba(255, 255, 255, 0.18) !important;
                backdrop-filter: blur(8px);
            }}

            .block-container {{
                max-width: 1300px;
                padding-top: 4.2rem;
                padding-bottom: 4.5rem;
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, rgba(255, 246, 238, 0.94) 0%, rgba(255, 251, 249, 0.9) 55%, rgba(240, 249, 244, 0.9) 100%) !important;
                backdrop-filter: blur(12px);
                border-right: 1px solid rgba(219, 178, 156, 0.34);
            }}

            .feature-grid {{
                margin-top: 10px;
                gap: 18px;
            }}

            .feature-card {{
                background: rgba(255, 253, 249, 0.72);
                backdrop-filter: blur(8px);
                border-color: rgba(255, 236, 226, 0.72);
                box-shadow: 0 18px 38px rgba(74, 54, 45, 0.09);
                padding: 18px 20px 20px;
            }}

            .feature-title {{
                font-size: 1.05rem;
            }}

            .feature-body {{
                color: #70564f;
            }}

            @media (max-width: 980px) {{
                [data-testid="stAppViewContainer"] {{
                    background-position: center right !important;
                }}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )


def render_home_hero() -> None:
    start_href = page_href("宠物档案")
    plan_href = page_href("今日陪伴计划")
    st.markdown(
        f"""
        <style>
        .home-hero {{
            position: relative;
            min-height: 56vh;
            display: flex;
            align-items: center;
            padding: 34px 44px;
            margin-bottom: 12px;
            border-radius: 24px;
            background: transparent;
        }}
        .home-hero::before {{
            content: "";
            position: absolute;
            left: 44px;
            bottom: 18px;
            width: min(420px, 44%);
            height: 3px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--pe-coral), var(--pe-sage), transparent);
            opacity: 0.72;
        }}
        .home-title {{
            color: #283746;
            font-size: clamp(3rem, 5.8vw, 5.4rem);
            line-height: 1.06;
            font-weight: 950;
            margin: 0 0 14px;
            text-shadow: 0 3px 24px rgba(255, 255, 255, 0.82);
        }}
        .home-subtitle {{
            max-width: 710px;
            color: #59473f;
            font-size: 1.17rem;
            line-height: 1.82;
            font-weight: 760;
            text-shadow: 0 2px 20px rgba(255, 255, 255, 0.86);
        }}
        .home-actions {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 24px;
        }}
        .hero-start,
        .hero-secondary {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 48px;
            padding: 0 20px;
            border-radius: 999px;
            font-weight: 900;
            text-decoration: none !important;
            box-shadow: 0 14px 28px rgba(86, 61, 48, 0.12);
        }}
        .hero-start {{
            color: #4e2d26 !important;
            background: linear-gradient(135deg, #ffb07c, #ef7f8f);
            border: 1px solid rgba(255, 255, 255, 0.5);
        }}
        .hero-secondary {{
            color: #3f4c53 !important;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(174, 198, 190, 0.44);
        }}
        .home-paw {{
            display: inline-grid;
            place-items: center;
            width: 29px;
            height: 29px;
            margin-right: 8px;
            border-radius: 50%;
            background: rgba(255,255,255,0.5);
            font-size: 0.9rem;
        }}
        @media (max-width: 780px) {{
            .home-hero {{
                min-height: 520px;
                padding: 26px 18px;
            }}
            .home-subtitle {{
                font-size: 1rem;
            }}
        }}
        </style>
        <section class="home-hero">
            <div>
                <div class="hero-eyebrow">温暖纪念 · 哀伤支持 · 安全边界</div>
                <h1 class="home-title">PetEcho</h1>
                <div class="home-subtitle">
                    一个面向宠物离别后的数字宠物哀伤支持系统：把纪念档案、每日状态记录、支持对话、低负担练习、纪念内容和安全边界放进同一条连续的陪伴流程里。
                </div>
                <div class="home-actions">
                    <a class="hero-start" href="{start_href}" target="_self"><span class="home-paw">🐾</span>开始建立档案</a>
                    <a class="hero-secondary" href="{plan_href}" target="_self">进入今日陪伴计划</a>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_page_hero(
    title: str,
    subtitle: str,
    eyebrow: str = "",
    badges: list[str] | None = None,
    art_uri: str | None = None,
) -> None:
    badge_html = "".join(f'<span class="hero-badge">{html.escape(badge)}</span>' for badge in (badges or []))
    art_html = ""
    if art_uri:
        art_html = f'<div class="hero-art"><img src="{art_uri}" alt=""></div>'
    elif asset_data_uri(DECOR_ILLUSTRATION_PATH):
        art_html = f'<div class="hero-art"><img src="{asset_data_uri(DECOR_ILLUSTRATION_PATH)}" alt=""></div>'

    eyebrow_html = f'<div class="hero-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""

    st.markdown(
        f"""
        <div class="page-hero">
            <div>
                {eyebrow_html}
                <div class="page-title">{html.escape(title)}</div>
                <div class="page-subtitle">{html.escape(subtitle)}</div>
                <div class="hero-badges">{badge_html}</div>
            </div>
            {art_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, desc: str = "", icon: str = "paw") -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-icon">{html.escape(icon)}</div>
            <div>
                <div class="section-title">{html.escape(title)}</div>
                <div class="section-desc">{html.escape(desc)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_grid(items: list[dict]) -> None:
    cards = []
    for item in items:
        cards.append(
            f'<div class="feature-card">'
            f'<div class="feature-icon">{html.escape(item.get("icon", ""))}</div>'
            f'<div class="feature-title">{html.escape(item.get("title", ""))}</div>'
            f'<div class="feature-body">{html.escape(item.get("body", ""))}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="feature-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_metric_card(title: str, value: str, body: str = "") -> None:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-title">{html.escape(title)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-body">{html.escape(body)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_pill(label: str, value: str) -> None:
    st.markdown(
        f'<span class="info-pill"><b>{html.escape(label)}</b>：{html.escape(value)}</span>',
        unsafe_allow_html=True,
    )


def message_html(role: str, text: str, name: str, timestamp: str) -> str:
    safe_text = html.escape(text or "").replace("\n", "<br>")
    safe_name = html.escape(name or "")
    safe_time = html.escape(timestamp or "")

    if role == "user":
        avatar_uri = get_user_avatar_data_uri()
        return (
            f'<div class="message-row right">'
            f'<div class="message-col-right">'
            f'<div class="meta-right"><span class="msg-time">{safe_time}</span>'
            f'<span class="user-name">{safe_name}</span></div>'
            f'<div class="user-bubble">{safe_text}</div>'
            f'</div>'
            f'<div class="avatar"><img src="{avatar_uri}" alt="user avatar"></div>'
            f'</div>'
        )

    avatar_uri = get_pet_avatar_data_uri()
    return (
        f'<div class="message-row left">'
        f'<div class="avatar"><img src="{avatar_uri}" alt="pet avatar"></div>'
        f'<div class="message-col-left">'
        f'<div class="meta-left"><span class="pet-name">{safe_name}</span>'
        f'<span class="msg-time">{safe_time}</span></div>'
        f'<div class="pet-bubble">{safe_text}</div>'
        f'</div>'
        f'</div>'
    )


def render_message(role: str, text: str, name: str, timestamp: str) -> None:
    st.markdown(message_html(role, text, name, timestamp), unsafe_allow_html=True)


def render_chat_board(messages: list[dict], pet_name: str) -> None:
    if not messages:
        body = '<div class="empty-chat-box">还没有开始对话。你可以先输入一句话，例如：“我很想你”。</div>'
    else:
        rows = []
        for msg in messages:
            role = msg.get("role", "assistant")
            name = "你" if role == "user" else pet_name
            rows.append(message_html(role, msg.get("content", ""), name, msg.get("timestamp", "")))
        body = "".join(rows)
    st.markdown(f'<div class="chat-board">{body}</div>', unsafe_allow_html=True)


def memorial_pattern_data_uri(symbol: str) -> str:
    if symbol == "journal":
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg" width="900" height="620" viewBox="0 0 900 620">
          <rect width="900" height="620" fill="#fff8dc"/>
          <path d="M90 90 H810 M90 154 H810 M90 218 H810 M90 282 H810 M90 346 H810 M90 410 H810 M90 474 H810" stroke="#f5d7a6" stroke-width="3" opacity=".55"/>
          <path d="M110 70 C170 95 178 142 136 160 C92 178 52 129 74 96" fill="none" stroke="#f0a98f" stroke-width="5" opacity=".45"/>
          <path d="M720 470 C766 428 822 455 808 510 C789 572 706 548 720 470Z" fill="none" stroke="#a5d6a7" stroke-width="6" opacity=".42"/>
          <rect x="340" y="34" width="210" height="46" rx="12" fill="#ffd6a5" opacity=".64" transform="rotate(-3 445 57)"/>
          <circle cx="764" cy="104" r="18" fill="#ffc8dd" opacity=".54"/>
          <circle cx="798" cy="134" r="11" fill="#ffd166" opacity=".56"/>
        </svg>
        """
    elif symbol == "night_light":
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg" width="900" height="620" viewBox="0 0 900 620">
          <rect width="900" height="620" fill="#f4f0ff"/>
          <circle cx="450" cy="270" r="180" fill="#fff6ed" opacity=".62"/>
          <path d="M690 130 C628 146 604 90 642 52 C656 105 704 96 690 130Z" fill="#ffe8a3" opacity=".78"/>
          <path d="M170 170 L182 200 L214 204 L190 224 L197 255 L170 238 L143 255 L150 224 L126 204 L158 200Z" fill="#ffd166" opacity=".42"/>
          <path d="M710 428 L719 450 L743 453 L725 468 L730 492 L710 479 L690 492 L695 468 L677 453 L701 450Z" fill="#ffc8dd" opacity=".5"/>
          <path d="M86 520 C180 470 250 492 330 542 C420 600 530 583 636 535 C716 499 782 504 846 534" fill="none" stroke="#b8d8ba" stroke-width="7" opacity=".42"/>
        </svg>
        """
    else:
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg" width="900" height="620" viewBox="0 0 900 620">
          <rect width="900" height="620" fill="#fff5ed"/>
          <rect x="74" y="72" width="250" height="174" rx="26" fill="none" stroke="#ffc8a8" stroke-width="8" opacity=".5"/>
          <rect x="572" y="330" width="244" height="172" rx="28" fill="none" stroke="#bfe7bf" stroke-width="8" opacity=".5"/>
          <path d="M420 88 C448 62 492 75 502 112 C538 102 566 128 556 164 C544 210 473 222 424 176 C388 142 388 116 420 88Z" fill="#ffc8dd" opacity=".4"/>
          <path d="M152 470 C190 422 256 440 262 496 C228 540 174 532 152 470Z" fill="#ffd166" opacity=".4"/>
          <circle cx="716" cy="132" r="34" fill="#bfe7bf" opacity=".5"/>
          <circle cx="766" cy="156" r="16" fill="#ffd166" opacity=".46"/>
        </svg>
        """
    return svg_data_uri(svg)


def render_memorial_card(memorial: dict) -> None:
    config = memorial.get("style_config", {})
    symbol = config.get("symbol", "soft_album")
    pattern = memorial_pattern_data_uri(symbol)
    class_name = {
        "journal": "journal",
        "night_light": "night-light",
    }.get(symbol, "album")
    bg = config.get("background", "linear-gradient(135deg, #fff5ed, #fffafd)")
    border = config.get("border", "#ffd7c2")
    accent = config.get("accent", "#e76f51")

    pet_name = html.escape(memorial.get("pet_name", "它"))
    label = html.escape(config.get("label", memorial.get("style", "纪念卡")))
    personality = html.escape(memorial.get("personality", "温柔地陪伴过你"))
    memory_title = html.escape(memorial.get("memory_title", "被认真记住的一小幕"))
    memory_line = html.escape(memorial.get("memory_line", ""))
    connection = html.escape(memorial.get("connection_practice", ""))
    closing = html.escape(memorial.get("closing", ""))

    if class_name == "journal":
        content = f"""
        <div>
            <div class="memorial-label">tape · {label}</div>
            <div class="memorial-name" style="color:{accent};">{pet_name}</div>
            <div class="memorial-line">{closing}</div>
        </div>
        <div>
            <div class="memorial-note">
                <div class="memorial-note-title">今天夹进手账的一页</div>
                {personality}
            </div>
            <div class="memorial-note" style="margin-top:12px;">
                <div class="memorial-note-title">{memory_title}</div>
                {memory_line}
            </div>
            <div class="memorial-note" style="margin-top:12px; color:{accent};">
                <div class="memorial-note-title">今日联结小仪式</div>
                {connection}
            </div>
        </div>
        """
    elif class_name == "night-light":
        content = f"""
        <div>
            <div class="memorial-label">light · {label}</div>
            <div class="memorial-name" style="color:{accent};">{pet_name}</div>
            <div class="memorial-line">{closing}</div>
            <div class="memorial-mini-grid">
                <div class="memorial-note">
                    <div class="memorial-note-title">{memory_title}</div>
                    {memory_line}
                </div>
                <div class="memorial-note" style="color:{accent};">
                    <div class="memorial-note-title">今晚可以做的小事</div>
                    {connection}
                </div>
            </div>
        </div>
        """
    else:
        content = f"""
        <div class="memorial-label">album · {label}</div>
        <div class="memorial-name" style="color:{accent};">{pet_name}</div>
        <div class="memorial-mini-grid">
            <div class="memorial-note">
                <div class="memorial-note-title">它给你的感觉</div>
                {personality}
            </div>
            <div class="memorial-note">
                <div class="memorial-note-title">{memory_title}</div>
                {memory_line}
            </div>
        </div>
        <div class="memorial-note" style="color:{accent};">
            <div class="memorial-note-title">今日联结小仪式</div>
            {connection}
        </div>
        <div class="memorial-line" style="margin-top:16px;">{closing}</div>
        """

    st.markdown(
        f"""
        <div class="memorial-card-ui {class_name}" style="background:{bg}; border:1px solid {border};">
            <style>
            .memorial-card-ui.{class_name}::before {{
                background-image: url("{pattern}");
            }}
            </style>
            <div class="memorial-content">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
