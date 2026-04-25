import os
import re
import html
import base64
from datetime import datetime
from urllib.parse import quote

import streamlit as st
from utils.file_io import (
    save_pet_profile,
    load_pet_profile,
    save_pet_memories,
    load_pet_memories,
    save_pet_image,
    load_pet_image_path,
    reset_pet_data,
)
from modules.rag import build_vectorstore, retrieve_memories
from modules.emotion import detect_emotion
from modules.grief_stage import detect_grief_stage
from modules.risk import detect_risk
from modules.strategy import choose_strategy
from modules.prompt_builder import build_prompt
from modules.llm_api import call_llm
from modules.vision_caption import generate_pet_appearance_caption

st.set_page_config(page_title="PetEcho", page_icon="🐾", layout="wide")

# ---------------------------
# 页面样式
# ---------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2.35rem;
    padding-bottom: 2.8rem;
    max-width: 1260px;
}

.title-wrap {
    padding-top: 0.35rem;
    margin-bottom: 1.45rem;
    overflow: visible;
}

.main-title {
    display: block;
    font-size: 2.18rem;
    font-weight: 800;
    color: #23324a;
    margin: 0 0 0.32rem 0;
    letter-spacing: -0.01em;
    line-height: 1.42;
    padding-top: 0.1rem;
    padding-bottom: 0.1rem;
    overflow: visible;
}

.sub-title {
    color: #667085;
    font-size: 1.01rem;
    margin: 0;
    line-height: 1.72;
}

.major-section-title {
    font-size: 1.28rem;
    font-weight: 800;
    color: #23324a;
    margin: 0.2rem 0 0.9rem 0;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #23324a;
    margin: 0;
}

.section-desc {
    color: #667085;
    font-size: 0.92rem;
    margin-top: 0.2rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #e8edf5 !important;
    border-radius: 20px !important;
    background: linear-gradient(180deg, #ffffff 0%, #fcfdff 100%) !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04) !important;
    padding: 10px 12px !important;
}

.card-head {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 40px;
    margin-bottom: 0.95rem;
}

.info-pill {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: #f6f8fc;
    border: 1px solid #e7ebf3;
    margin: 4px 8px 4px 0;
    font-size: 0.92rem;
    color: #344054;
}

.metric-card {
    border: 1px solid #e8edf5;
    background: #ffffff;
    border-radius: 18px;
    padding: 14px 16px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
    height: 100%;
}

.metric-title {
    font-size: 0.9rem;
    color: #667085;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 1.12rem;
    font-weight: 700;
    color: #25324a;
}

.small-muted {
    color: #667085;
    font-size: 0.92rem;
    line-height: 1.65;
}

hr {
    border: none;
    border-top: 1px solid #edf1f7;
    margin: 1.45rem 0;
}

.stTextArea textarea,
.stTextInput input {
    border-radius: 14px !important;
}

.chat-container {
    margin-top: 0.55rem;
    margin-bottom: 0.45rem;
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.message-row {
    display: flex;
    gap: 10px;
    margin: 16px 0;
    animation: fadeUp 0.28s ease-out;
}

.message-row.left {
    justify-content: flex-start;
    align-items: flex-start;
}

.message-row.right {
    justify-content: flex-end;
    align-items: flex-start;
}

.message-col-left,
.message-col-right {
    max-width: 76%;
    display: flex;
    flex-direction: column;
}

.message-col-left {
    align-items: flex-start;
}

.message-col-right {
    align-items: flex-end;
}

.avatar {
    width: 46px;
    height: 46px;
    min-width: 46px;
    border-radius: 50%;
    overflow: hidden;
    border: 2px solid #ffffff;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.08);
    background: #fff;
}

.avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.meta-left,
.meta-right {
    font-size: 0.82rem;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.meta-left {
    margin-left: 4px;
}

.meta-right {
    margin-right: 4px;
}

.pet-name {
    color: #c65c7b;
    font-weight: 700;
}

.user-name {
    color: #4d79d8;
    font-weight: 700;
}

.msg-time {
    color: #98a2b3;
    font-size: 0.78rem;
}

.pet-bubble {
    background: #fff7f8;
    color: #2b2b2b;
    border: 1px solid #f5d8df;
    border-radius: 18px 18px 18px 8px;
    padding: 12px 15px;
    line-height: 1.82;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    white-space: pre-wrap;
    word-break: break-word;
}

.user-bubble {
    background: #eef4ff;
    color: #22324d;
    border: 1px solid #d8e4ff;
    border-radius: 18px 18px 8px 18px;
    padding: 12px 15px;
    line-height: 1.82;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    white-space: pre-wrap;
    word-break: break-word;
}

.empty-chat-box {
    border: 1px dashed #d8dee9;
    background: #fbfcfe;
    color: #667085;
    border-radius: 16px;
    padding: 16px 18px;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="title-wrap">
        <div class="main-title">PetEcho：数字宠物哀伤支持系统</div>
        <div class="sub-title">基于宠物记忆检索、心理策略控制与大语言模型生成的阶段化哀伤支持原型</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# 工具函数
# ---------------------------
def now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def reset_runtime_state() -> None:
    keys_to_clear = [
        "chat_history",
        "last_reply",
        "last_meta",
        "pet_name_input",
        "pet_personality_input",
        "pet_appearance_input",
        "pet_memories_input",
        "chat_input",
        "pending_pet_appearance",
        "pending_chat_input",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def load_demo_data() -> None:
    demo_profile = {
        "pet_name": "咪咪",
        "pet_personality": "很黏人、特别乖、喜欢安静地靠近主人，开心时会轻轻蹭过来。",
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

    try:
        build_vectorstore(demo_memories)
    except Exception:
        pass

    reset_runtime_state()
    st.session_state["pet_name_input"] = demo_profile["pet_name"]
    st.session_state["pet_personality_input"] = demo_profile["pet_personality"]
    st.session_state["pet_appearance_input"] = demo_profile["pet_appearance"]
    st.session_state["pet_memories_input"] = demo_memories
    st.session_state["chat_input"] = ""
    st.session_state["chat_history"] = []
    st.session_state["last_reply"] = ""
    st.session_state["last_meta"] = {}


def render_tag(label: str, value: str) -> None:
    st.markdown(
        f'<span class="info-pill"><b>{label}</b>：{value}</span>',
        unsafe_allow_html=True,
    )


def image_file_to_data_uri(image_path: str):
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower().replace(".", "")
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{encoded}"
    except Exception:
        return None


def get_default_cat_avatar_data_uri() -> str:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
      <defs>
        <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#ffd6a5"/>
          <stop offset="100%" stop-color="#ffc8dd"/>
        </linearGradient>
      </defs>
      <rect width="96" height="96" rx="48" fill="url(#bg)"/>
      <polygon points="24,34 36,16 42,36" fill="#f5a3b7"/>
      <polygon points="72,34 60,16 54,36" fill="#f5a3b7"/>
      <circle cx="48" cy="52" r="28" fill="#fff6f0"/>
      <circle cx="37" cy="50" r="5" fill="#333"/>
      <circle cx="59" cy="50" r="5" fill="#333"/>
      <circle cx="35.5" cy="48.5" r="1.6" fill="#fff"/>
      <circle cx="57.5" cy="48.5" r="1.6" fill="#fff"/>
      <path d="M48 55 L44 60 L52 60 Z" fill="#f28ba8"/>
      <path d="M44 63 Q48 67 52 63" stroke="#7d5a5a" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      <path d="M29 58 Q38 57 42 59" stroke="#b88b8b" stroke-width="2" fill="none" stroke-linecap="round"/>
      <path d="M54 59 Q58 57 67 58" stroke="#b88b8b" stroke-width="2" fill="none" stroke-linecap="round"/>
      <path d="M29 64 Q38 63 42 65" stroke="#b88b8b" stroke-width="2" fill="none" stroke-linecap="round"/>
      <path d="M54 65 Q58 63 67 64" stroke="#b88b8b" stroke-width="2" fill="none" stroke-linecap="round"/>
    </svg>
    """
    return "data:image/svg+xml;utf8," + quote(svg)


def get_pet_avatar_data_uri() -> str:
    pet_image_path = load_pet_image_path()
    uploaded_uri = image_file_to_data_uri(pet_image_path)
    if uploaded_uri:
        return uploaded_uri
    return get_default_cat_avatar_data_uri()


def get_user_avatar_data_uri() -> str:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
      <defs>
        <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#bfd7ff"/>
          <stop offset="100%" stop-color="#dbeafe"/>
        </linearGradient>
      </defs>
      <rect width="96" height="96" rx="48" fill="url(#bg)"/>
      <circle cx="48" cy="38" r="16" fill="#ffffff"/>
      <path d="M24 78 Q48 56 72 78" fill="#ffffff"/>
      <circle cx="42" cy="37" r="2.4" fill="#4b5563"/>
      <circle cx="54" cy="37" r="2.4" fill="#4b5563"/>
      <path d="M42 46 Q48 50 54 46" stroke="#6b7280" stroke-width="2.2" fill="none" stroke-linecap="round"/>
    </svg>
    """
    return "data:image/svg+xml;utf8," + quote(svg)


def clean_reply_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = text.replace("【回复】", "").replace("【数字宠物回复】", "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def render_message(role: str, text: str, name: str, timestamp: str) -> None:
    safe_text = html.escape(text).replace("\n", "<br>")
    safe_name = html.escape(name)
    safe_time = html.escape(timestamp or "")

    if role == "user":
        avatar_uri = get_user_avatar_data_uri()
        st.markdown(
            f"""
            <div class="chat-container">
                <div class="message-row right">
                    <div class="message-col-right">
                        <div class="meta-right">
                            <span class="msg-time">{safe_time}</span>
                            <span class="user-name">{safe_name}</span>
                        </div>
                        <div class="user-bubble">{safe_text}</div>
                    </div>
                    <div class="avatar"><img src="{avatar_uri}" alt="user avatar"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        avatar_uri = get_pet_avatar_data_uri()
        st.markdown(
            f"""
            <div class="chat-container">
                <div class="message-row left">
                    <div class="avatar"><img src="{avatar_uri}" alt="pet avatar"></div>
                    <div class="message-col-left">
                        <div class="meta-left">
                            <span class="pet-name">{safe_name}</span>
                            <span class="msg-time">{safe_time}</span>
                        </div>
                        <div class="pet-bubble">{safe_text}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------
# 侧边栏
# ---------------------------
with st.sidebar:
    st.header("项目控制台")
    st.write("用于快速演示、调试与重置数据。")

    debug_mode = st.checkbox("显示调试信息", value=True)

    st.markdown("---")

    if st.button("载入演示数据", use_container_width=True):
        load_demo_data()
        st.success("已载入演示数据。")
        st.rerun()

    if st.button("重置所有宠物数据", use_container_width=True):
        reset_pet_data()
        reset_runtime_state()
        st.success("已重置所有数据。")
        st.rerun()

    st.markdown("---")
    st.caption("本系统仅作为哀伤支持工具，不替代真实关系和专业帮助。")

# ---------------------------
# 读取已保存数据
# ---------------------------
saved_profile = load_pet_profile() or {}
saved_memories = load_pet_memories()

# ---------------------------
# Session State 初始化
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "last_reply" not in st.session_state:
    st.session_state["last_reply"] = ""

if "last_meta" not in st.session_state:
    st.session_state["last_meta"] = {}

if "pet_name_input" not in st.session_state:
    st.session_state["pet_name_input"] = saved_profile.get("pet_name", "")

if "pet_personality_input" not in st.session_state:
    st.session_state["pet_personality_input"] = saved_profile.get("pet_personality", "")

if "pet_appearance_input" not in st.session_state:
    st.session_state["pet_appearance_input"] = saved_profile.get("pet_appearance", "")

if "pet_memories_input" not in st.session_state:
    st.session_state["pet_memories_input"] = saved_memories if saved_memories else ""

if "chat_input" not in st.session_state:
    st.session_state["chat_input"] = ""

if "pending_pet_appearance" in st.session_state:
    st.session_state["pet_appearance_input"] = st.session_state["pending_pet_appearance"]
    del st.session_state["pending_pet_appearance"]

if "pending_chat_input" in st.session_state:
    st.session_state["chat_input"] = st.session_state["pending_chat_input"]
    del st.session_state["pending_chat_input"]

# ---------------------------
# 1）建立宠物档案
# ---------------------------
st.markdown('<div class="major-section-title">1）建立宠物档案</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([2.1, 1], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-head">
                <div class="section-title">建立宠物档案</div>
                <div class="section-desc">填写宠物名字、性格、外观与重要回忆，构建数字宠物身份。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pet_name = st.text_input("宠物名字", key="pet_name_input")

        pet_personality = st.text_area(
            "宠物性格",
            placeholder="例如：很黏人，喜欢散步，见到我会很兴奋",
            key="pet_personality_input",
        )

        pet_appearance = st.text_area(
            "宠物外观描述（可手动填写，也可由照片自动生成）",
            placeholder="例如：毛很蓬松，白色身体，脸和耳朵偏灰，眼睛圆圆的，像一团棉花",
            key="pet_appearance_input",
        )

        pet_memories = st.text_area(
            "宠物回忆",
            placeholder="例如：它最喜欢傍晚陪我去公园，雨天会缩在我的脚边",
            key="pet_memories_input",
            height=150,
        )

        uploaded_image = st.file_uploader(
            "上传宠物照片",
            type=["jpg", "jpeg", "png"],
            key="pet_image_uploader",
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("根据当前照片生成外观描述", use_container_width=True):
                if uploaded_image is None:
                    st.warning("请先上传宠物照片。")
                else:
                    image_path = save_pet_image(uploaded_image)
                    try:
                        with st.spinner("正在根据照片生成外观描述..."):
                            auto_caption = generate_pet_appearance_caption(image_path)
                        st.session_state["pending_pet_appearance"] = auto_caption
                        st.success("已根据当前照片生成外观描述。")
                        st.rerun()
                    except Exception as e:
                        st.error(f"自动生成外观描述失败：{e}")

        with btn_col2:
            if st.button("保存宠物档案", use_container_width=True):
                if uploaded_image is not None:
                    save_pet_image(uploaded_image)

                profile = {
                    "pet_name": pet_name.strip(),
                    "pet_personality": pet_personality.strip(),
                    "pet_appearance": pet_appearance.strip(),
                }

                save_pet_profile(profile)
                save_pet_memories(pet_memories)

                try:
                    if pet_memories.strip():
                        with st.spinner("正在建立记忆库..."):
                            build_vectorstore(pet_memories)
                        st.success("宠物档案、照片和记忆库已保存。")
                    else:
                        st.warning("宠物档案已保存，但你还没有填写宠物回忆，因此未建立记忆库。")
                except Exception as e:
                    st.warning("宠物档案已保存，但记忆库暂未建立成功。")
                    if debug_mode:
                        st.error(f"RAG 模块错误：{e}")

                st.rerun()

with right_col:
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-head">
                <div class="section-title">宠物档案预览</div>
                <div class="section-desc">实时查看宠物头像与档案信息。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        preview_image_path = load_pet_image_path()

        if uploaded_image is not None:
            st.image(uploaded_image, caption="当前上传图片", use_container_width=True)
        elif preview_image_path and os.path.exists(preview_image_path):
            st.image(preview_image_path, caption="宠物头像", use_container_width=True)
        else:
            st.info("还没有上传宠物照片")

        preview_name = st.session_state.get("pet_name_input", "")
        preview_personality = st.session_state.get("pet_personality_input", "")
        preview_appearance = st.session_state.get("pet_appearance_input", "")

        st.markdown(f"**名字：** {preview_name if preview_name else '未填写'}")
        st.markdown(f"**性格：** {preview_personality if preview_personality else '未填写'}")
        st.markdown(f"**外观：** {preview_appearance if preview_appearance else '未填写'}")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------
# 2）开始对话
# ---------------------------
st.markdown('<div class="major-section-title">2）开始对话</div>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown(
        """
        <div class="card-head">
            <div class="section-title">数字宠物对话区</div>
            <div class="section-desc">输入你想对它说的话，系统会结合宠物记忆和心理状态生成回复。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pet_profile_for_chat = load_pet_profile() or {}
    pet_name_for_chat = pet_profile_for_chat.get("pet_name", "数字宠物")

    if st.session_state["chat_history"]:
        for msg in st.session_state["chat_history"]:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            if role == "user":
                render_message("user", content, "你", timestamp)
            else:
                render_message("assistant", content, pet_name_for_chat, timestamp)
    else:
        st.markdown(
            '<div class="empty-chat-box">还没有开始对话。你可以先输入一句话，例如：“我很想你”。</div>',
            unsafe_allow_html=True,
        )

    user_input = st.text_area("你想对它说什么？", height=110, key="chat_input")

    col_send, col_clear = st.columns([1, 1])

    with col_send:
        send_clicked = st.button("发送", use_container_width=True)

    with col_clear:
        clear_clicked = st.button("清空聊天历史", use_container_width=True)

    if clear_clicked:
        st.session_state["chat_history"] = []
        st.session_state["last_reply"] = ""
        st.session_state["last_meta"] = {}
        st.session_state["pending_chat_input"] = ""
        st.rerun()

    if send_clicked:
        profile = load_pet_profile()

        if not profile:
            st.error("请先保存宠物档案。")
        elif not user_input.strip():
            st.warning("请输入内容。")
        else:
            emotion = detect_emotion(user_input)
            grief_stage = detect_grief_stage(user_input, emotion)
            risk = detect_risk(user_input)
            strategy = choose_strategy(grief_stage, risk, emotion, user_input)

            with st.spinner("数字宠物正在想一想该怎么回应你..."):
                try:
                    memory_context = retrieve_memories(user_input, top_k=3)
                except Exception as e:
                    memory_context = []
                    if debug_mode:
                        st.caption(f"RAG 检索错误：{e}")

                recent_history = st.session_state["chat_history"][-4:]

                try:
                    prompt = build_prompt(
                        pet_profile=profile,
                        memory_context=memory_context,
                        emotion=emotion,
                        grief_stage=grief_stage,
                        risk=risk,
                        strategy=strategy,
                        user_input=user_input,
                    )

                    reply = clean_reply_text(call_llm(prompt))
                    current_time = now_hhmm()

                    st.session_state["chat_history"].append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": current_time,
                    })
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": reply,
                        "timestamp": current_time,
                    })

                    st.session_state["last_reply"] = reply
                    st.session_state["last_meta"] = {
                        "emotion": emotion,
                        "grief_stage": grief_stage,
                        "risk": risk,
                        "strategy": strategy,
                        "memory_context": memory_context,
                        "pet_profile": profile,
                        "recent_history": recent_history,
                        "guidance_mode": strategy.get("guidance_mode", "none"),
                        "guidance_focus": strategy.get("guidance_focus", ""),
                    }
                    st.session_state["pending_chat_input"] = ""
                    st.rerun()

                except Exception as e:
                    st.error(f"生成回复时发生错误：{e}")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------
# 3）系统判定
# ---------------------------
if st.session_state.get("last_meta"):
    meta = st.session_state["last_meta"]

    st.markdown('<div class="major-section-title">3）本轮系统判定</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            """
            <div class="card-head">
                <div class="section-title">系统状态分析</div>
                <div class="section-desc">系统会先判断用户当前情绪、哀伤阶段与风险状态，再决定是否调用记忆，并生成更自然的数字宠物回复。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">当前阶段</div>
                    <div class="metric-value">{meta['grief_stage']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">风险等级</div>
                    <div class="metric-value">{meta['risk']['level']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">支持策略</div>
                    <div class="metric-value">{meta['strategy']['name']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c4:
            mode_label = meta.get("guidance_mode", "none")
            display_value = "建议型支持" if mode_label == "coping_guidance" else ("常规支持" if mode_label in {"", "none"} else mode_label)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">回应模式</div>
                    <div class="metric-value">{display_value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if meta["risk"]["level"] == "high":
            st.warning("当前检测到较高风险表达，系统应优先提供现实支持与安全提醒。")
        elif meta["risk"]["level"] == "medium":
            st.info("当前情绪负荷较高，系统会减少沉浸式回忆，优先提供稳定支持。")
        elif meta.get("guidance_mode") == "coping_guidance":
            st.caption("当前用户在主动寻求建议，系统会在共情基础上给出更具体、低负担的建议。")
        else:
            st.caption("当前以常规支持模式运行。")

        if debug_mode:
            with st.expander("查看调试信息"):
                st.json(meta)
