import os
import html

import streamlit as st

from modules.image_avatar import IMAGE_STYLE_PROMPTS, generate_virtual_pet_avatar, image_generation_status
from modules.rag import build_vectorstore
from modules.ui_components import (
    DECOR_ILLUSTRATION_PATH,
    apply_global_style,
    asset_data_uri,
    ensure_session_state,
    get_pet_avatar_data_uri,
    page_href,
    render_page_hero,
    render_section_header,
)
from utils.file_io import (
    load_generated_pet_avatar_path,
    load_pet_image_path,
    load_pet_memories,
    load_pet_profile,
    save_pet_image,
    save_pet_memories,
    save_pet_profile,
)


st.set_page_config(page_title="宠物档案 - PetEcho", page_icon="🐾", layout="wide")

apply_global_style("profile")
ensure_session_state()

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.55rem;
        max-width: 1280px;
    }

    .page-hero {
        margin-top: 1.05rem;
        margin-bottom: 1.85rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(226, 156, 126, 0.42) !important;
        background:
            linear-gradient(135deg, rgba(255, 253, 249, 0.95), rgba(255, 250, 247, 0.9) 52%, rgba(243, 251, 247, 0.9)) !important;
        box-shadow: 0 16px 34px rgba(86, 61, 48, 0.08) !important;
    }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stMarkdown"] .section-head) {
        border: 1px solid rgba(226, 156, 126, 0.42) !important;
        border-radius: 18px !important;
        background:
            linear-gradient(135deg, rgba(255, 253, 249, 0.95), rgba(255, 250, 247, 0.9) 52%, rgba(243, 251, 247, 0.9)) !important;
        box-shadow: 0 16px 34px rgba(86, 61, 48, 0.08) !important;
    }

    .section-head {
        margin-bottom: 1.15rem;
    }

    .profile-note-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 4px 0 18px;
    }

    .profile-note {
        border: 1px solid rgba(226, 156, 126, 0.34);
        border-radius: 14px;
        padding: 12px 13px;
        background: rgba(255, 255, 255, 0.68);
    }

    .profile-note b {
        display: block;
        color: #6b3d33;
        font-size: 0.92rem;
        margin-bottom: 3px;
    }

    .profile-note span {
        color: #80645d;
        font-size: 0.86rem;
        line-height: 1.58;
    }

    .profile-subhead {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #6b3d33;
        font-size: 1rem;
        font-weight: 900;
        margin: 6px 0 12px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid rgba(226, 156, 126, 0.38);
    }

    .profile-subhead::before {
        content: "";
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: linear-gradient(135deg, #ffb07c, #ef8796);
        box-shadow: 0 0 0 4px rgba(226, 156, 126, 0.13);
    }

    .profile-hint {
        color: #85675f;
        font-size: 0.88rem;
        line-height: 1.65;
        margin: -3px 0 12px;
    }

    .profile-divider {
        height: 1px;
        margin: 18px 0;
        background: linear-gradient(90deg, transparent, rgba(255, 185, 160, 0.84), rgba(196, 229, 185, 0.72), transparent);
    }

    .profile-save-band {
        margin: 22px 0 10px;
        padding-top: 18px;
        border-top: 1px solid rgba(255, 185, 160, 0.74);
    }

    .profile-save-band .profile-hint {
        text-align: center;
        margin-bottom: 12px;
    }

    .stButton button {
        min-height: 50px !important;
        font-size: 1rem !important;
        font-weight: 950 !important;
    }

    .stButton button p {
        font-weight: 950 !important;
    }

    .stTextInput input,
    .stTextArea textarea {
        background: rgba(255, 248, 242, 0.96) !important;
        border: 1px solid rgba(226, 156, 126, 0.46) !important;
        color: #5e3a32 !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: rgba(255, 142, 155, 0.95) !important;
        box-shadow: 0 0 0 3px rgba(255, 170, 185, 0.18) !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #b2938b !important;
    }

    .stSelectbox [data-baseweb="select"] > div {
        background: rgba(255, 248, 242, 0.96) !important;
        border-color: rgba(226, 156, 126, 0.46) !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background:
            linear-gradient(135deg, rgba(255, 249, 243, 0.98), rgba(244, 253, 247, 0.94)) !important;
        border: 1px dashed rgba(226, 156, 126, 0.54) !important;
        border-radius: 16px !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: rgba(255, 255, 255, 0.78) !important;
        border-color: rgba(226, 156, 126, 0.42) !important;
        color: #624038 !important;
        box-shadow: none !important;
    }

    div[data-testid="stImage"] img {
        border-radius: 18px;
        border: 1px solid rgba(255, 205, 184, 0.9);
        box-shadow: 0 14px 30px rgba(143, 91, 63, 0.1);
    }

    .empty-soft-box {
        background:
            rgba(255, 250, 246, 0.86);
        border-color: rgba(226, 156, 126, 0.38);
    }

    .profile-preview-horizontal {
        display: grid;
        grid-template-columns: minmax(170px, 0.78fr) minmax(0, 1.1fr) minmax(0, 1.1fr) minmax(170px, 0.82fr);
        gap: 14px;
        align-items: stretch;
        margin: 4px 0 6px;
    }

    .profile-preview-card {
        min-height: 156px;
        border-radius: 18px;
        border: 1px solid rgba(226, 156, 126, 0.34);
        background:
            rgba(255, 255, 255, 0.6);
        padding: 15px 16px;
    }

    .profile-preview-label {
        color: #9b6456;
        font-size: 0.8rem;
        font-weight: 900;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    .profile-preview-value {
        color: #5f3a32;
        font-size: 0.94rem;
        line-height: 1.62;
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    .profile-preview-identity {
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 10px;
        min-height: 156px;
        border-radius: 18px;
        background: rgba(255, 248, 242, 0.72);
        border: 1px solid rgba(255, 205, 184, 0.86);
        padding: 15px 16px;
    }

    .profile-preview-avatar {
        width: 82px;
        height: 82px;
        border-radius: 16px;
        object-fit: cover;
        border: 1px solid rgba(226, 156, 126, 0.42);
        box-shadow: 0 10px 22px rgba(143, 91, 63, 0.12);
    }

    .profile-preview-status {
        color: #73564f;
        font-size: 0.88rem;
        line-height: 1.6;
        font-weight: 700;
    }

    .profile-page-footer {
        display: flex;
        justify-content: center;
        margin: 26px 0 16px;
    }

    .profile-chat-action {
        min-width: min(520px, 100%);
        min-height: 58px;
        padding: 0 32px;
        color: #48251f !important;
        background: linear-gradient(135deg, #ffb07c 0%, #ef7f8f 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.66) !important;
        box-shadow: 0 18px 34px rgba(222, 93, 84, 0.24) !important;
        font-size: 1.06rem;
        font-weight: 950;
    }

    @media (max-width: 980px) {
        .profile-note-strip {
            grid-template-columns: 1fr;
        }
        .profile-preview-horizontal {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

render_page_hero(
    "宠物档案",
    "用于建立宠物的基础资料、照片素材与记忆文本，为后续个性化回应、记忆检索、数字宠物头像和纪念空间提供依据。",
    eyebrow="第一步 · 建立纪念素材",
    badges=["照片可作对话头像", "纪念形象可选", "记忆用于相关检索"],
    art_uri=asset_data_uri(DECOR_ILLUSTRATION_PATH),
)

profile = load_pet_profile() or {}
saved_memories = load_pet_memories()

with st.container(border=True):
    render_section_header(
        "纪念素材整理",
        "集中整理基础资料、照片头像和重要回忆，形成后续个性化回应与纪念空间的素材基础。",
        "🐾",
    )
    st.markdown(
        """
        <div class="profile-note-strip">
            <div class="profile-note"><b>身份资料</b><span>名称与基本特征会成为数字宠物回应的基础设定。</span></div>
            <div class="profile-note"><b>个性边界</b><span>性格描述用于减少泛化表达，使回应更贴合具体宠物。</span></div>
            <div class="profile-note"><b>记忆检索</b><span>重要回忆会进入记忆库，并在相关性足够时被自然引用。</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    material_col, photo_col = st.columns([1.25, 0.75], gap="large")

    with material_col:
        st.markdown('<div class="profile-subhead">宠物基础资料</div>', unsafe_allow_html=True)
        pet_name = st.text_input("宠物名字", key="pet_name_input", placeholder="例如：咪咪")

        trait_col, look_col = st.columns([1, 1], gap="medium")
        with trait_col:
            pet_personality = st.text_area(
                "性格特征",
                placeholder="例如：很黏人，喜欢安静地靠近人，听见钥匙声会跑过来",
                key="pet_personality_input",
                height=140,
            )

        with look_col:
            pet_appearance = st.text_area(
                "外观特征",
                placeholder="例如：白色身体，脸和耳朵偏灰，眼睛圆圆的，毛很蓬松",
                key="pet_appearance_input",
                height=140,
            )

        st.markdown('<div class="profile-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="profile-subhead">重要回忆</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="profile-hint">记录具体、可回想的生活片段。系统会根据语义相关性选择是否调用记忆，避免强行插入。</div>',
            unsafe_allow_html=True,
        )
        pet_memories = st.text_area(
            "宠物回忆",
            placeholder="例如：它最喜欢傍晚陪我去公园，雨天会缩在我的脚边",
            key="pet_memories_input",
            height=180,
        )

        st.markdown('<div class="profile-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="profile-subhead">生成数字纪念形象</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="profile-hint">生成结果作为纪念性头像素材，用于增强温暖陪伴感，不作为真实重现。</div>',
            unsafe_allow_html=True,
        )
        avatar_style = st.selectbox("形象风格", list(IMAGE_STYLE_PROMPTS.keys()), key="virtual_avatar_style")
        image_status = image_generation_status()

        if image_status["ready"]:
            st.caption(f"图像生成已启用：{image_status['model']}，尺寸 {image_status['size']}")
        else:
            st.caption("当前环境未开启图像生成；上传照片仍可作为对话头像。")

        generate_clicked = st.button(
            "生成虚拟纪念形象",
            use_container_width=True,
            disabled=not image_status["ready"],
        )

        if generate_clicked:
            uploaded_for_avatar = st.session_state.get("pet_image_uploader")
            if uploaded_for_avatar is not None:
                reference_image_path = save_pet_image(uploaded_for_avatar)
            else:
                reference_image_path = load_pet_image_path()

            profile_for_avatar = {
                "pet_name": (pet_name or "").strip(),
                "pet_personality": (pet_personality or "").strip(),
                "pet_appearance": (pet_appearance or "").strip(),
            }

            try:
                with st.spinner("正在生成数字纪念形象..."):
                    generate_virtual_pet_avatar(reference_image_path, profile_for_avatar, avatar_style)
                st.success("已生成数字纪念形象，并会优先作为聊天头像使用。")
                st.rerun()
            except Exception as e:
                st.warning(f"生成纪念形象失败：{e}")

            st.rerun()

    with photo_col:
        st.markdown('<div class="profile-subhead">照片与头像设置</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="profile-hint">上传照片可作为对话头像；生成纪念形象后，系统会优先使用纪念形象。</div>',
            unsafe_allow_html=True,
        )

        uploaded_image = st.file_uploader(
            "上传宠物照片",
            type=["jpg", "jpeg", "png"],
            key="pet_image_uploader",
        )

        preview_path = load_pet_image_path()
        generated_path = load_generated_pet_avatar_path()

        if generated_path and os.path.exists(generated_path):
            st.image(generated_path, caption="数字纪念形象", use_container_width=True)
        elif uploaded_image is not None:
            st.image(uploaded_image, caption="当前上传图片", use_container_width=True)
        elif preview_path and os.path.exists(preview_path):
            st.image(preview_path, caption="宠物头像", use_container_width=True)
        else:
            st.markdown(
                '<div class="empty-soft-box">还没有上传照片。上传后，它会优先成为对话页里的数字宠物头像。</div>',
                unsafe_allow_html=True,
            )

        st.button("图片自动描述暂未开放", use_container_width=True, disabled=True)
        st.caption("当前版本暂不从图片自动生成外观描述，外观特征以左侧填写内容为准。")

    st.markdown(
        '<div class="profile-save-band"><div class="profile-hint">保存后将同步更新档案、照片和记忆库，用于后续对话与纪念空间。</div></div>',
        unsafe_allow_html=True,
    )
    save_left, save_center, save_right = st.columns([0.18, 0.64, 0.18], gap="medium")
    with save_center:
        save_clicked = st.button("保存档案并建立记忆库", use_container_width=True)

    if save_clicked:
        uploaded_for_save = st.session_state.get("pet_image_uploader")
        if uploaded_for_save is not None:
            save_pet_image(uploaded_for_save)

        new_profile = {
            "pet_name": (pet_name or "").strip(),
            "pet_personality": (pet_personality or "").strip(),
            "pet_appearance": (pet_appearance or "").strip(),
        }
        save_pet_profile(new_profile)
        save_pet_memories(pet_memories)
        st.session_state["recent_memory_contexts"] = []

        try:
            if pet_memories.strip():
                with st.spinner("正在建立记忆库..."):
                    build_vectorstore(pet_memories)
                st.success("宠物档案、照片和记忆库已保存。")
            else:
                st.warning("宠物档案已保存，但还没有填写回忆，因此暂未建立记忆库。")
        except Exception as e:
            st.warning(f"档案已保存，但记忆库暂未建立成功：{e}")

        st.rerun()

with st.container(border=True):
    render_section_header("档案预览", "汇总当前已保存的关键档案素材，便于确认个性化信息是否完整。", "•")
    current = load_pet_profile() or profile
    current_memories = st.session_state.get("pet_memories_input") or saved_memories
    preview_name = current.get("pet_name") or st.session_state.get("pet_name_input") or "未填写"
    preview_personality = current.get("pet_personality") or st.session_state.get("pet_personality_input") or "未填写"
    preview_appearance = current.get("pet_appearance") or st.session_state.get("pet_appearance_input") or "未填写"
    preview_memory_count = str(len(current_memories or ""))
    preview_avatar = get_pet_avatar_data_uri()
    st.markdown(
        f"""
        <div class="profile-preview-horizontal">
            <div class="profile-preview-identity">
                <img class="profile-preview-avatar" src="{preview_avatar}" alt="">
                <div>
                    <div class="profile-preview-label">宠物名字</div>
                    <div class="profile-preview-value">{html.escape(preview_name)}</div>
                </div>
                <div>
                    <div class="profile-preview-label">回忆字数</div>
                    <div class="profile-preview-value">{html.escape(preview_memory_count)} 字</div>
                </div>
            </div>
            <div class="profile-preview-card">
                <div class="profile-preview-label">性格特征</div>
                <div class="profile-preview-value">{html.escape(preview_personality)}</div>
            </div>
            <div class="profile-preview-card">
                <div class="profile-preview-label">外观特征</div>
                <div class="profile-preview-value">{html.escape(preview_appearance)}</div>
            </div>
            <div class="profile-preview-card">
                <div class="profile-preview-label">头像使用</div>
                <div class="profile-preview-status">
                    当前头像会用于对话页展示。生成纪念形象后，系统将自动优先使用纪念形象。
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="profile-page-footer">
        <a class="nav-link-button profile-chat-action" href="{page_href("今日陪伴计划")}" target="_self">进入今日陪伴计划</a>
    </div>
    """,
    unsafe_allow_html=True,
)
