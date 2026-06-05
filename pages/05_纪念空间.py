import html

import streamlit as st

from modules.memorial_card import STYLE_CONFIG, build_memorial_card
from modules.memory_timeline import build_memory_cards
from modules.ui_components import (
    apply_global_style,
    asset_data_uri,
    ensure_session_state,
    page_href,
    render_memorial_card,
    render_page_hero,
    render_section_header,
)
from utils.file_io import load_pet_image_path, load_pet_memories, load_pet_profile


st.set_page_config(page_title="纪念空间 - PetEcho", page_icon="🐾", layout="wide")

apply_global_style("memorial")
ensure_session_state()

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.55rem;
        max-width: 1260px;
    }

    .page-hero {
        margin-top: 1.05rem;
        margin-bottom: 1.65rem;
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

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stMarkdown"] .memory-panel-marker),
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stMarkdown"] .memorial-panel-marker) {
        min-height: 0 !important;
    }

    .memory-panel-marker,
    .memorial-panel-marker {
        display: none;
    }

    .memory-card {
        border: 1px solid rgba(226, 156, 126, 0.46) !important;
        background:
            linear-gradient(135deg, rgba(255, 248, 239, 0.96), rgba(255, 252, 245, 0.94) 58%, rgba(242, 253, 245, 0.92));
        box-shadow: 0 12px 26px rgba(86, 61, 48, 0.07);
    }

    .memory-card::after {
        opacity: 0.7;
    }

    .photo-memory-card {
        display: grid;
        grid-template-columns: 1fr;
        gap: 13px;
        align-items: stretch;
        min-height: 398px;
    }

    .photo-memory-preview {
        width: 100%;
        height: 235px;
        min-height: 235px;
        border-radius: 14px;
        border: 1px solid rgba(226, 156, 126, 0.38);
        object-fit: cover;
        background:
            linear-gradient(135deg, rgba(255, 248, 242, 0.95), rgba(241, 252, 246, 0.92));
    }

    .photo-memory-empty {
        display: grid;
        place-items: center;
        text-align: center;
        color: #8a6a62;
        line-height: 1.68;
        padding: 18px;
    }

    .photo-memory-body {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        gap: 8px;
    }

    .photo-memory-body .memory-card-text {
        font-size: 0.94rem;
    }

    .memorial-card-ui {
        border: 1px solid rgba(226, 156, 126, 0.48) !important;
        min-height: 610px;
        padding: 26px;
        box-shadow: 0 18px 40px rgba(86, 61, 48, 0.11);
    }

    .memorial-card-ui.album {
        min-height: 590px;
        padding-bottom: 22px;
    }

    .memorial-label,
    .memorial-note {
        border-color: rgba(226, 156, 126, 0.38) !important;
    }

    .memorial-name {
        font-size: 1.75rem;
        margin: 10px 0 8px;
    }

    .memorial-line {
        font-size: 0.94rem;
        line-height: 1.6;
    }

    .memorial-mini-grid {
        gap: 10px;
        margin: 14px 0;
    }

    .memorial-note {
        padding: 12px 13px;
        font-size: 0.92rem;
        line-height: 1.58;
    }

    .stSelectbox [data-baseweb="select"] > div {
        background: rgba(255, 248, 242, 0.96) !important;
        border-color: rgba(226, 156, 126, 0.46) !important;
    }

    .memorial-action-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 14px;
        margin-bottom: 27px;
    }

    .memorial-action-row .nav-link-button {
        min-height: 54px;
        width: 100%;
        margin: 0;
        border: 1px solid rgba(239, 133, 108, 0.62);
        box-shadow: 0 14px 28px rgba(214, 105, 82, 0.14);
        font-size: 1rem;
    }

    .memorial-action-row .memorial-action-primary {
        background: linear-gradient(135deg, #ffb07c 0%, #ef7f8f 100%);
        color: #4c2b25 !important;
    }

    .memorial-action-row .memorial-action-secondary {
        background: rgba(255, 255, 255, 0.74);
        color: #63372e !important;
    }

    @media (max-width: 980px) {
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stMarkdown"] .memory-panel-marker),
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stMarkdown"] .memorial-panel-marker) {
            min-height: unset !important;
        }
        .photo-memory-card,
        .memorial-action-row {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

render_page_hero(
    "纪念空间",
    "本页面用于整理宠物记忆与纪念卡片，将回忆转化为可保存、可回看的纪念内容，并保留一件当下可以完成的小联结。",
    eyebrow="温暖纪念 · 意义重建",
    badges=["记忆卡片", "主题化纪念卡", "持续性联结", "生活重连"],
)

profile = load_pet_profile() or {}
memories = st.session_state.get("pet_memories_input") or load_pet_memories()
memory_cards = build_memory_cards(memories)
pet_image_path = load_pet_image_path()
pet_image_uri = asset_data_uri(pet_image_path or "")

memory_col, card_col = st.columns([1.08, 1], gap="large")

with memory_col:
    with st.container(border=True):
        render_section_header("宠物记忆卡片", "将已保存的回忆整理为可浏览的小片段，保留具体场景和温和的阅读节奏。", "✿")
        st.markdown('<div class="memory-panel-marker"></div>', unsafe_allow_html=True)
        if memory_cards:
            for idx, card in enumerate(memory_cards[:8], start=1):
                tone_color = "#c65c7b" if card["heavy"] else "#5d8a66"
                st.markdown(
                    f"""
                    <div class="memory-card">
                        <div class="memory-card-title">小小片段 {idx} · {html.escape(card["title"])}｜{html.escape(card["category"])}</div>
                        <div class="memory-card-meta" style="color:{tone_color};">{html.escape(card["tone"])}</div>
                        <div class="memory-card-text">{html.escape(card["text"])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"""
                <div class="empty-soft-box">
                保存宠物回忆后，这里会自动生成记忆卡片。<br>
                <a class="nav-link-button" href="{page_href("宠物档案")}" target="_self">返回档案页补充回忆</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

        photo_idx = min(len(memory_cards), 8) + 1 if memory_cards else 1
        if pet_image_uri:
            photo_preview = f'<img class="photo-memory-preview" src="{pet_image_uri}" alt="">'
            photo_meta = "已识别到照片素材"
            photo_text = "档案中已有上传照片。该照片可作为纪念空间中的视觉线索，也会在未生成纪念形象时作为数字宠物头像。"
        else:
            photo_preview = '<div class="photo-memory-preview photo-memory-empty">暂无照片素材<br>等待在档案页上传</div>'
            photo_meta = "等待上传"
            photo_text = "上传宠物照片后，这里会自动显示照片片段；当前版本不从图片自动生成外观描述。"

        st.markdown(
            f"""
            <div class="memory-card photo-memory-card">
                {photo_preview}
                <div class="photo-memory-body">
                    <div class="memory-card-title">小小片段 {photo_idx} · 照片记忆｜照片素材</div>
                    <div class="memory-card-meta" style="color:#5d8a66;">{html.escape(photo_meta)}</div>
                    <div class="memory-card-text">{html.escape(photo_text)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with card_col:
    with st.container(border=True):
        render_section_header("数字宠物纪念卡", "纪念卡以简笔画和线稿式背景承载文字内容，减少视觉干扰，突出回忆与联结。", "✧")
        st.markdown('<div class="memorial-panel-marker"></div>', unsafe_allow_html=True)

        selected_style = st.selectbox("纪念卡主题", list(STYLE_CONFIG.keys()), key="memorial_card_style")
        memorial = build_memorial_card(profile, memories, selected_style)
        render_memorial_card(memorial)

        st.markdown(
            f"""
            <div class="decor-strip">
                <span class="decor-dot"></span><span class="decor-dot"></span><span class="decor-dot"></span><span class="decor-dot"></span>
            </div>
            <div class="memorial-action-row">
                <a class="nav-link-button memorial-action-secondary" href="{page_href("今日陪伴计划")}" target="_self">回到今日陪伴计划</a>
                <a class="nav-link-button memorial-action-primary" href="{page_href("用户测试与反馈")}" target="_self">记录体验反馈</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
