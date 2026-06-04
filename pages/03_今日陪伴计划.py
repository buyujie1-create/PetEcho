import os
from datetime import date, datetime

import pandas as pd
import streamlit as st

from modules.memory_timeline import build_memory_cards
from modules.ui_components import (
    apply_global_style,
    asset_data_uri,
    ensure_session_state,
    page_href,
    render_page_hero,
    render_section_header,
)
from utils.companion_plan_io import (
    build_companion_state_package,
    companion_state_package_bytes,
    load_daily_checkins,
    load_companion_state_package,
    load_memorial_settings,
    load_plan_progress,
    normalize_checkin_payload,
    normalize_memorial_settings,
    save_daily_checkin,
    save_memorial_settings,
    save_plan_progress,
)
from utils.file_io import load_pet_image_path, load_pet_memories, load_pet_profile
from utils.research_io import REFLECTION_PATH, save_reflection_entry


st.set_page_config(page_title="今日陪伴计划 - PetEcho", page_icon="🐾", layout="wide")

apply_global_style("companion")
ensure_session_state()


PLAN_DAYS = [
    {
        "key": "day_1",
        "day": "Day 1",
        "title": "先让情绪落地",
        "body": "记录今日状态，做一件照顾身体的小事，例如喝水、拉开窗帘或坐稳一分钟。",
    },
    {
        "key": "day_2",
        "day": "Day 2",
        "title": "留下一句话",
        "body": "写下一句今天最想对它说的话，不需要完整，也不需要解释。",
    },
    {
        "key": "day_3",
        "day": "Day 3",
        "title": "整理一个温暖片段",
        "body": "选择一段不太沉重的回忆，放进纪念册或纪念空间。",
    },
    {
        "key": "day_4",
        "day": "Day 4",
        "title": "靠近一个现实支点",
        "body": "给可信任的人发一句短消息，或安排一次很轻的现实连接。",
    },
    {
        "key": "day_5",
        "day": "Day 5",
        "title": "松动一点自责",
        "body": "写下曾经认真照顾它的一件小事，让遗憾不全部变成自我惩罚。",
    },
    {
        "key": "day_6",
        "day": "Day 6",
        "title": "做一个小纪念动作",
        "body": "为它留一张照片、一句话或一个小角落，把想念放进可承载的形式里。",
    },
    {
        "key": "day_7",
        "day": "Day 7",
        "title": "带着回忆继续生活",
        "body": "回看这一周的记录，写下它留给生活的一点提醒或力量。",
    },
]


def _as_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _row_is_safety_priority(row: dict | None) -> bool:
    if not row:
        return False
    return row.get("safety_thoughts") == "yes" or _as_int(row.get("safety_level")) >= 4


def _next_plan_day(progress: dict) -> dict:
    for item in PLAN_DAYS:
        if not progress.get(item["key"]):
            return item
    return PLAN_DAYS[-1]


def _recommend_route(row: dict | None, progress: dict, has_memories: bool) -> dict:
    if not row:
        return {
            "title": "先完成今日状态记录",
            "body": "记录后系统会根据情绪强度、思念程度和安全信号给出今日建议路径。",
            "primary": "保存今日状态",
            "href": "",
            "tone": "neutral",
        }

    if _row_is_safety_priority(row):
        return {
            "title": "今日优先：安全资源卡",
            "body": "当前记录出现安全压力信号，建议先联系现实中的可信任支持，再进入普通对话或纪念内容。",
            "primary": "查看安全资源",
            "href": "#safety-card",
            "tone": "safety",
        }

    emotion = _as_int(row.get("emotion_intensity"))
    yearning = _as_int(row.get("yearning_intensity"))
    sleep = _as_int(row.get("sleep_quality"))
    appetite = _as_int(row.get("appetite_quality"))
    need = row.get("support_need", "")

    if emotion >= 6 or sleep <= 2 or appetite <= 2:
        return {
            "title": "今日优先：低负担练习",
            "body": "当前状态较重，先从一个小动作开始，比马上整理很多回忆更稳。",
            "primary": "进入支持对话",
            "href": page_href("哀伤支持对话"),
            "tone": "care",
        }

    if "回忆" in need or (yearning >= 5 and has_memories):
        return {
            "title": "今日优先：更新纪念内容",
            "body": "思念较明显，可以把一段温暖记忆放进纪念册或纪念空间，让关系有一个可回看的位置。",
            "primary": "进入纪念空间",
            "href": page_href("纪念空间"),
            "tone": "memory",
        }

    if "建议" in need:
        return {
            "title": "今日优先：获取一点具体支持",
            "body": "当前更适合进入支持对话，系统会先承接情绪，再给出一两个低负担建议。",
            "primary": "进入支持对话",
            "href": page_href("哀伤支持对话"),
            "tone": "care",
        }

    next_day = _next_plan_day(progress)
    return {
        "title": f"今日优先：{next_day['title']}",
        "body": next_day["body"],
        "primary": "查看七天计划",
        "href": "#seven-day-plan",
        "tone": "memory",
    }


def _load_reflections(limit: int = 5) -> list[dict]:
    if not REFLECTION_PATH or not os.path.exists(REFLECTION_PATH):
        return []
    try:
        rows = pd.read_csv(REFLECTION_PATH, encoding="utf-8-sig").fillna("").to_dict("records")
    except Exception:
        return []
    return rows[-limit:][::-1]


def _activate_personal_state(
    checkin_rows: list[dict] | None = None,
    plan_state: dict | None = None,
    memorial_state: dict | None = None,
) -> None:
    st.session_state["companion_personal_mode"] = True
    st.session_state["companion_personal_checkins"] = [
        normalize_checkin_payload(row)
        for row in (checkin_rows or [])
        if isinstance(row, dict)
    ]
    st.session_state["companion_personal_progress"] = {
        str(k): bool(v) for k, v in (plan_state or {}).items()
    }
    st.session_state["companion_personal_memorial"] = normalize_memorial_settings(memorial_state or {})


def _using_personal_state() -> bool:
    return bool(st.session_state.get("companion_personal_mode"))


def _current_companion_state() -> tuple[list[dict], dict, dict]:
    if _using_personal_state():
        return (
            st.session_state.get("companion_personal_checkins", []),
            st.session_state.get("companion_personal_progress", {}),
            st.session_state.get("companion_personal_memorial", {}),
        )
    return load_daily_checkins(), load_plan_progress(), load_memorial_settings()


def _days_until_memorial(memorial_date: str) -> int | None:
    if not memorial_date:
        return None
    try:
        raw = date.fromisoformat(memorial_date)
    except Exception:
        return None
    today = date.today()
    next_date = raw.replace(year=today.year)
    if next_date < today:
        next_date = raw.replace(year=today.year + 1)
    return (next_date - today).days


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
        padding-bottom: 24px !important;
        background:
            linear-gradient(135deg, rgba(255, 253, 249, 0.95), rgba(255, 250, 247, 0.9) 52%, rgba(243, 251, 247, 0.9)) !important;
        box-shadow: 0 16px 34px rgba(86, 61, 48, 0.08) !important;
    }

    .section-head {
        margin-bottom: 1.15rem;
    }

    .companion-kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 16px;
        margin-bottom: 20px;
    }

    .companion-kpi {
        min-height: 112px;
        border: 1px solid rgba(226, 156, 126, 0.36);
        border-radius: 16px;
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(248, 252, 249, 0.82));
        padding: 15px 16px;
        box-shadow: 0 10px 24px rgba(86, 61, 48, 0.06);
    }

    .companion-kpi-title {
        color: #816156;
        font-size: 0.84rem;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .companion-kpi-value {
        color: #263a48;
        font-size: 1.42rem;
        font-weight: 950;
        line-height: 1.28;
    }

    .companion-kpi-body {
        color: #6f5e57;
        font-size: 0.9rem;
        line-height: 1.55;
        margin-top: 7px;
    }

    .route-card {
        position: relative;
        overflow: hidden;
        min-height: 218px;
        border: 1px solid rgba(239, 133, 108, 0.66);
        border-radius: 18px;
        padding: 20px 20px 18px;
        background:
            linear-gradient(135deg, rgba(255, 244, 235, 0.98), rgba(255, 253, 249, 0.94) 48%, rgba(244, 251, 247, 0.92));
        box-shadow: 0 16px 32px rgba(86, 61, 48, 0.09);
    }

    .route-card.safety {
        border-color: rgba(218, 83, 99, 0.68);
        background:
            linear-gradient(135deg, rgba(255, 244, 244, 0.98), rgba(255, 250, 245, 0.94));
    }

    .route-label {
        display: inline-flex;
        padding: 6px 10px;
        border-radius: 999px;
        color: #79483d;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(226, 156, 126, 0.42);
        font-size: 0.84rem;
        font-weight: 900;
        margin-bottom: 12px;
    }

    .route-title {
        color: #523229;
        font-size: 1.32rem;
        font-weight: 950;
        line-height: 1.35;
        margin-bottom: 8px;
    }

    .route-body {
        color: #6f5e57;
        line-height: 1.72;
        margin-bottom: 16px;
    }

    .route-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .plan-grid {
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        gap: 12px;
        margin: 10px 0 20px;
    }

    .plan-day-card {
        min-height: 196px;
        border: 1px solid rgba(226, 156, 126, 0.36);
        border-radius: 16px;
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(248, 252, 249, 0.76));
        padding: 14px 13px;
        box-shadow: 0 8px 20px rgba(86, 61, 48, 0.05);
    }

    .plan-day-tag {
        color: #cf6d59;
        font-size: 0.78rem;
        font-weight: 950;
        margin-bottom: 7px;
    }

    .plan-day-title {
        color: #523229;
        font-weight: 950;
        line-height: 1.36;
        margin-bottom: 8px;
    }

    .plan-day-body {
        color: #6f5e57;
        font-size: 0.88rem;
        line-height: 1.62;
    }

    .safety-card {
        border: 1px solid rgba(218, 83, 99, 0.62);
        border-radius: 18px;
        padding: 18px 19px;
        background:
            rgba(255, 255, 255, 0.78);
        color: #65423a;
        line-height: 1.72;
        box-shadow: 0 12px 28px rgba(196, 91, 91, 0.08);
    }

    .book-grid {
        display: grid;
        grid-template-columns: minmax(0, 0.82fr) minmax(0, 1.18fr);
        gap: 16px;
        align-items: stretch;
    }

    .book-cover {
        min-height: 270px;
        border: 1px solid rgba(226, 156, 126, 0.38);
        border-radius: 18px;
        padding: 18px;
        background:
            linear-gradient(135deg, rgba(255, 248, 242, 0.92), rgba(242, 252, 245, 0.86));
    }

    .book-photo {
        width: 100%;
        height: 158px;
        object-fit: cover;
        border-radius: 14px;
        border: 1px solid rgba(226, 156, 126, 0.38);
        margin-bottom: 12px;
    }

    .book-empty-photo {
        display: grid;
        place-items: center;
        height: 158px;
        border-radius: 14px;
        border: 1px dashed rgba(226, 156, 126, 0.48);
        color: #8a6a62;
        background: rgba(255, 250, 246, 0.72);
        margin-bottom: 12px;
        text-align: center;
        line-height: 1.62;
    }

    .book-title {
        color: #523229;
        font-size: 1.16rem;
        font-weight: 950;
        line-height: 1.38;
        margin-bottom: 6px;
    }

    .book-body,
    .reminder-body {
        color: #6f5e57;
        line-height: 1.68;
        font-size: 0.94rem;
    }

    .book-note-stack {
        display: grid;
        gap: 10px;
    }

    .book-note {
        border: 1px solid rgba(226, 156, 126, 0.34);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.72);
        padding: 13px 14px;
        color: #76584f;
        line-height: 1.62;
    }

    .book-note b {
        display: block;
        color: #523229;
        margin-bottom: 4px;
    }

    .stSlider {
        padding-bottom: 2px;
    }

    .stTextArea textarea,
    .stTextInput input,
    .stSelectbox [data-baseweb="select"] > div {
        background: rgba(255, 248, 242, 0.96) !important;
        border-color: rgba(226, 156, 126, 0.46) !important;
        color: #5e3a32 !important;
    }

    @media (max-width: 1100px) {
        .plan-grid,
        .companion-kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .book-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 700px) {
        .plan-grid,
        .companion-kpi-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


render_page_hero(
    "今日陪伴计划",
    "本页面把每日状态记录、支持路径、七天温和陪伴计划、状态趋势和纪念册整理到同一条连续流程中，使 PetEcho 不只完成一次对话，而是支持用户下次继续回来。",
    eyebrow="每日入口 · 长期陪伴闭环",
    badges=["今日 check-in", "七天陪伴计划", "状态趋势", "安全资源卡", "我的纪念册"],
)

profile = load_pet_profile() or {}
memories = load_pet_memories()
memory_cards = build_memory_cards(memories)
image_path = load_pet_image_path()
image_uri = asset_data_uri(image_path or "")
checkins, progress, memorial_settings = _current_companion_state()
latest = checkins[-1] if checkins else None
completed_count = sum(1 for item in PLAN_DAYS if progress.get(item["key"]))
route = _recommend_route(latest, progress, bool(memories.strip()))

latest_date = latest.get("checkin_date", "尚未记录") if latest else "尚未记录"
latest_emotion = latest.get("emotion_intensity", "-") if latest else "-"
latest_need = latest.get("support_need", "完成今日记录后生成") if latest else "完成今日记录后生成"

with st.container(border=True):
    render_section_header(
        "个人 7 日状态包",
        "用于在线测试时延续自己的状态记录。首次使用可开始新状态包；下次回来先导入上次下载的 JSON，再继续记录。",
        "⇩",
    )
    mode_text = "个人状态包模式已启用。" if _using_personal_state() else "当前使用在线临时记录；多人测试时建议先开始新的个人状态包。"
    st.markdown(f'<div class="scale-guide">{mode_text}</div>', unsafe_allow_html=True)

    package_cols = st.columns([1.25, 1, 1], gap="medium")
    with package_cols[0]:
        uploaded_package = st.file_uploader(
            "导入上次状态包",
            type=["json"],
            help="上传上次下载的 petecho_7day_state_*.json 后，可继续自己的 7 日状态跟踪。",
        )
        if uploaded_package is not None and st.button("应用导入的状态包", use_container_width=True):
            try:
                package = load_companion_state_package(uploaded_package.getvalue())
                _activate_personal_state(
                    package.get("daily_checkins", []),
                    package.get("plan_progress", {}),
                    package.get("memorial_settings", {}),
                )
                st.success("状态包已导入。")
                st.rerun()
            except Exception as exc:
                st.error(f"状态包导入失败：{exc}")
    with package_cols[1]:
        if st.button("开始新的 7 日状态包", use_container_width=True):
            _activate_personal_state([], {}, {})
            st.success("已开始新的个人状态包。")
            st.rerun()
    with package_cols[2]:
        export_package = build_companion_state_package(checkins, progress, memorial_settings)
        st.download_button(
            "下载我的 7 日状态包",
            data=companion_state_package_bytes(export_package),
            file_name=f"petecho_7day_state_{date.today().isoformat()}.json",
            mime="application/json",
            use_container_width=True,
        )

st.markdown(
    f"""
    <div class="companion-kpi-grid">
        <div class="companion-kpi">
            <div class="companion-kpi-title">最近记录</div>
            <div class="companion-kpi-value">{latest_date}</div>
            <div class="companion-kpi-body">今日状态会成为后续建议路径的依据。</div>
        </div>
        <div class="companion-kpi">
            <div class="companion-kpi-title">情绪强度</div>
            <div class="companion-kpi-value">{latest_emotion}</div>
            <div class="companion-kpi-body">1-7 分，数值越高代表当前情绪负荷越重。</div>
        </div>
        <div class="companion-kpi">
            <div class="companion-kpi-title">七天计划</div>
            <div class="companion-kpi-value">{completed_count}/7</div>
            <div class="companion-kpi-body">完成情况用于提示下次回来继续的位置。</div>
        </div>
        <div class="companion-kpi">
            <div class="companion-kpi-title">今日支持需要</div>
            <div class="companion-kpi-value" style="font-size:1.08rem;">{latest_need}</div>
            <div class="companion-kpi-body">系统会据此推荐对话、练习或纪念内容。</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.12, 0.88], gap="large")

with left_col:
    with st.container(border=True):
        render_section_header("今日状态 check-in", "用一组低负担问题记录当前状态，为今日支持路径和趋势图提供依据。", "✎")
        with st.form("daily_checkin_form", clear_on_submit=False):
            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                emotion_intensity = st.slider("今日情绪强度", 1, 7, _as_int(latest.get("emotion_intensity"), 4) if latest else 4)
                yearning_intensity = st.slider("今日思念强度", 1, 7, _as_int(latest.get("yearning_intensity"), 4) if latest else 4)
            with c2:
                guilt_intensity = st.slider("愧疚/自责", 1, 7, _as_int(latest.get("guilt_intensity"), 2) if latest else 2)
                anger_intensity = st.slider("愤怒/不公平感", 1, 7, _as_int(latest.get("anger_intensity"), 2) if latest else 2)
            with c3:
                numbness_intensity = st.slider("麻木/不真实感", 1, 7, _as_int(latest.get("numbness_intensity"), 2) if latest else 2)
                safety_level = st.slider("安全压力", 1, 7, _as_int(latest.get("safety_level"), 1) if latest else 1)

            d1, d2, d3 = st.columns([0.8, 0.8, 1.15], gap="medium")
            with d1:
                sleep_quality = st.slider("睡眠状态", 1, 7, _as_int(latest.get("sleep_quality"), 4) if latest else 4)
            with d2:
                appetite_quality = st.slider("进食状态", 1, 7, _as_int(latest.get("appetite_quality"), 4) if latest else 4)
            with d3:
                support_need = st.selectbox(
                    "今日更需要什么",
                    ["先稳定情绪", "想说说它", "整理一段回忆", "需要一点建议", "只想安静一下"],
                    index=0,
                )

            safety_thoughts = st.checkbox("今日出现过伤害自己、不想活下去或无法保证安全的念头")
            notes = st.text_area("今日备注（可选）", height=90, placeholder="例如：今天看到它的照片后很想它，晚上睡得不太好。")
            submitted = st.form_submit_button("保存今日状态", use_container_width=True)

        if submitted:
            checkin_payload = {
                "checkin_date": date.today().isoformat(),
                "emotion_intensity": emotion_intensity,
                "yearning_intensity": yearning_intensity,
                "guilt_intensity": guilt_intensity,
                "anger_intensity": anger_intensity,
                "numbness_intensity": numbness_intensity,
                "sleep_quality": sleep_quality,
                "appetite_quality": appetite_quality,
                "support_need": support_need,
                "safety_thoughts": safety_thoughts,
                "safety_level": safety_level,
                "notes": notes,
            }
            if _using_personal_state():
                st.session_state.setdefault("companion_personal_checkins", []).append(
                    normalize_checkin_payload(checkin_payload)
                )
            else:
                save_daily_checkin(checkin_payload)
            st.success("今日状态已保存。")
            st.rerun()

    with st.container(border=True):
        render_section_header("状态变化趋势", "根据每日 check-in 记录观察情绪强度、思念强度和安全压力的变化。", "↗")
        if checkins:
            df = pd.DataFrame(checkins)
            for col in [
                "emotion_intensity",
                "yearning_intensity",
                "guilt_intensity",
                "anger_intensity",
                "numbness_intensity",
                "safety_level",
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["记录时间"] = pd.to_datetime(df["created_at"], errors="coerce")
            chart_df = (
                df.sort_values("记录时间")
                .set_index("记录时间")[
                    [
                        "emotion_intensity",
                        "yearning_intensity",
                        "guilt_intensity",
                        "anger_intensity",
                        "numbness_intensity",
                        "safety_level",
                    ]
                ]
                .rename(
                    columns={
                        "emotion_intensity": "情绪强度",
                        "yearning_intensity": "思念强度",
                        "guilt_intensity": "愧疚自责",
                        "anger_intensity": "愤怒不甘",
                        "numbness_intensity": "麻木感",
                        "safety_level": "安全压力",
                    }
                )
            )
            st.line_chart(chart_df, height=280)
        else:
            st.markdown(
                '<div class="empty-soft-box">保存一次今日状态后，这里会显示情绪变化趋势。</div>',
                unsafe_allow_html=True,
            )

with right_col:
    with st.container(border=True):
        render_section_header("今日建议路径", "根据最近一次状态记录，推荐更适合当前的下一步。", "✓")
        route_class = "safety" if route["tone"] == "safety" else ""
        href = route["href"]
        action_html = (
            f'<a class="nav-link-button" href="{href}" target="_self">{route["primary"]}</a>'
            if href and not href.startswith("#")
            else f'<a class="nav-link-button" href="{href or "#"}">{route["primary"]}</a>'
        )
        st.markdown(
            f"""
            <div class="route-card {route_class}">
                <div class="route-label">下一步建议</div>
                <div class="route-title">{route["title"]}</div>
                <div class="route-body">{route["body"]}</div>
                <div class="route-actions">
                    {action_html}
                    <a class="nav-link-button" href="{page_href("哀伤支持对话")}" target="_self">进入支持对话</a>
                    <a class="nav-link-button" href="{page_href("纪念空间")}" target="_self">查看纪念空间</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        render_section_header("下次回来继续", "保留最近状态和七天计划进度，作为下一次进入系统时的起点。", "↻")
        next_day = _next_plan_day(progress)
        if latest:
            st.markdown(
                f"""
                <div class="scale-guide">
                最近一次记录为 {latest.get("checkin_date", "")}，情绪强度 {latest.get("emotion_intensity", "-")} / 7，
                支持需要为“{latest.get("support_need", "")}”。下一步可继续：{next_day["day"]} · {next_day["title"]}。
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="scale-guide">尚未保存今日状态。完成一次 check-in 后，系统会在这里显示继续入口。</div>',
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        render_section_header("纪念日提醒", "应用内记录重要日期，打开项目时即可看到距离下一次纪念日还有多久。", "◇")
        settings = memorial_settings
        default_date = date.today()
        if settings.get("memorial_date"):
            try:
                default_date = date.fromisoformat(settings["memorial_date"])
            except Exception:
                default_date = date.today()

        with st.form("memorial_settings_form", clear_on_submit=False):
            memorial_label = st.text_input("日期名称", value=settings.get("label", "纪念日"))
            memorial_date = st.date_input("重要日期", value=default_date)
            memorial_note = st.text_area(
                "提醒备注（可选）",
                value=settings.get("note", ""),
                height=74,
                placeholder="例如：这一天适合翻看一张照片，写一句很短的话。",
            )
            saved_memorial = st.form_submit_button("保存纪念日提醒", use_container_width=True)

        if saved_memorial:
            memorial_payload = {
                "label": memorial_label,
                "memorial_date": memorial_date.isoformat(),
                "note": memorial_note,
            }
            if _using_personal_state():
                st.session_state["companion_personal_memorial"] = normalize_memorial_settings(memorial_payload)
            else:
                save_memorial_settings(memorial_payload)
            st.success("纪念日提醒已保存。")
            st.rerun()

        days_left = _days_until_memorial(settings.get("memorial_date", ""))
        if days_left is not None:
            if days_left == 0:
                reminder = f"今天是{settings.get('label', '纪念日')}。可以只做一个很小的纪念动作。"
            else:
                reminder = f"距离下一次{settings.get('label', '纪念日')}还有 {days_left} 天。"
            st.markdown(f'<div class="reminder-body">{reminder}</div>', unsafe_allow_html=True)

st.markdown('<span id="seven-day-plan"></span>', unsafe_allow_html=True)
with st.container(border=True):
    render_section_header("七天温和陪伴计划", "以低负担、可持续的小任务支持用户从一次对话延展到一周陪伴。", "7")
    st.progress(completed_count / len(PLAN_DAYS))
    plan_cards_html = "".join(
        '<div class="plan-day-card">'
        f'<div class="plan-day-tag">{item["day"]}</div>'
        f'<div class="plan-day-title">{item["title"]}</div>'
        f'<div class="plan-day-body">{item["body"]}</div>'
        "</div>"
        for item in PLAN_DAYS
    )
    st.markdown(f'<div class="plan-grid">{plan_cards_html}</div>', unsafe_allow_html=True)

    with st.form("plan_progress_form", clear_on_submit=False):
        cols = st.columns(7, gap="small")
        new_progress = {}
        for idx, item in enumerate(PLAN_DAYS):
            with cols[idx]:
                new_progress[item["key"]] = st.checkbox(item["day"], value=bool(progress.get(item["key"])))
        progress_saved = st.form_submit_button("保存七天计划进度", use_container_width=True)
    if progress_saved:
        if _using_personal_state():
            st.session_state["companion_personal_progress"] = {str(k): bool(v) for k, v in new_progress.items()}
        else:
            save_plan_progress(new_progress)
        st.success("陪伴计划进度已保存。")
        st.rerun()

st.markdown('<span id="safety-card"></span>', unsafe_allow_html=True)
with st.container(border=True):
    render_section_header("安全资源卡", "当状态记录或对话中出现安全风险时，系统会优先提示现实支持。", "!")
    priority_text = "当前最近一次记录未显示高安全压力。"
    if _row_is_safety_priority(latest):
        priority_text = "最近一次记录显示安全压力较高，建议先联系现实中的可信任支持。"
    st.markdown(
        f"""
        <div class="safety-card">
            <b>{priority_text}</b><br>
            如出现伤害自己、不想活下去、无法保证安全、已经准备自伤方式等情况，应立即让现实中的人靠近，联系可信任亲友、学校/社区支持人员或当地急救与危机援助服务。PetEcho 可以提供稳定提示，但不能替代现实中的紧急帮助。
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.container(border=True):
    render_section_header("我的纪念册", "汇总宠物档案、回忆片段、陪伴计划进度和练习记录，形成可持续更新的纪念内容。", "□")
    pet_name = profile.get("pet_name") or "尚未命名"
    personality = profile.get("pet_personality") or "尚未填写性格特征"
    recent_reflections = _load_reflections()
    book_photo = (
        f'<img class="book-photo" src="{image_uri}" alt="">'
        if image_uri
        else '<div class="book-empty-photo">暂无照片素材<br>可在宠物档案页上传</div>'
    )
    st.markdown(
        f"""
        <div class="book-grid">
            <div class="book-cover">
                {book_photo}
                <div class="book-title">{pet_name} 的纪念册</div>
                <div class="book-body">
                    已整理 {len(memory_cards)} 个记忆片段；七天计划完成 {completed_count}/7 天。<br>
                    {personality}
                </div>
                <div style="margin-top:12px;">
                    <a class="nav-link-button" href="{page_href("纪念空间")}" target="_self">进入完整纪念空间</a>
                </div>
            </div>
            <div class="book-note-stack">
        """,
        unsafe_allow_html=True,
    )

    if memory_cards:
        for card in memory_cards[:2]:
            st.markdown(
                f"""
                <div class="book-note">
                    <b>{card.get("title", "记忆片段")} · {card.get("category", "")}</b>
                    {card.get("text", "")}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="book-note"><b>记忆片段</b>保存宠物回忆后，纪念册会自动收录温暖片段。</div>',
            unsafe_allow_html=True,
        )

    if recent_reflections:
        for row in recent_reflections[:2]:
            st.markdown(
                f"""
                <div class="book-note">
                    <b>{row.get("exercise_title", "练习记录")}</b>
                    {row.get("response_text", "")}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="book-note"><b>近期练习</b>完成支持对话后的低负担练习，记录会出现在这里。</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div></div>", unsafe_allow_html=True)

    with st.form("daily_memorial_note_form", clear_on_submit=True):
        note_text = st.text_area(
            "写入今日纪念短笺",
            height=90,
            placeholder="例如：今天又想起它等我回家的样子，我会把这份想念慢慢放好。",
        )
        note_saved = st.form_submit_button("保存到纪念册", use_container_width=True)
    if note_saved:
        if note_text.strip():
            save_reflection_entry(
                {
                    "key": "daily_companion_note",
                    "title": "今日纪念短笺",
                    "category": "持续性联结",
                },
                note_text.strip(),
                {},
            )
            st.success("今日纪念短笺已保存。")
            st.rerun()
        else:
            st.warning("可以先写下一句话，再保存到纪念册。")
