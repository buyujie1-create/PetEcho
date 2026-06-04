import csv
import os

import streamlit as st

from modules.image_avatar import image_generation_status
from modules.support_display import build_support_panel
from modules.ui_components import (
    apply_global_style,
    ensure_session_state,
    load_demo_data,
    render_metric_card,
    render_page_hero,
    render_section_header,
    reset_all_pet_data,
)
from utils.companion_plan_io import (
    CHECKIN_PATH,
    CHECKIN_FIELDNAMES,
    MEMORIAL_SETTINGS_PATH,
    PLAN_PROGRESS_PATH,
    load_daily_checkins,
    load_memorial_settings,
    load_plan_progress,
)
from utils.file_io import (
    CHAT_HISTORY_PATH,
    GENERATED_AVATAR_PATH,
    MEMORY_PATH,
    PROFILE_PATH,
    load_chat_history,
    load_pet_image_path,
    load_pet_memories,
    load_pet_profile,
)
from utils.research_io import (
    REFLECTION_PATH,
    USER_TEST_FIELDNAMES,
    USER_TEST_PATH,
    csv_file_bytes,
)


st.set_page_config(page_title="研究者控制台 - PetEcho", page_icon="🐾", layout="wide")

apply_global_style("console")
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
        padding-bottom: 28px !important;
        background:
            linear-gradient(135deg, rgba(255, 253, 249, 0.95), rgba(255, 250, 247, 0.9) 52%, rgba(243, 251, 247, 0.9)) !important;
        box-shadow: 0 16px 34px rgba(86, 61, 48, 0.08) !important;
    }

    .section-head {
        margin-bottom: 1.15rem;
    }

    .console-panel-gap {
        height: 18px;
    }

    .console-section-bottom-space {
        height: 18px;
    }

    .console-status-wrap {
        margin-top: 2px;
    }

    .console-card-row-gap {
        height: 14px;
    }

    .metric-card {
        height: 142px;
        min-height: 142px;
        box-sizing: border-box;
        border-color: rgba(226, 156, 126, 0.36);
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(248, 252, 249, 0.78));
    }

    .metric-body {
        overflow-wrap: anywhere;
        word-break: break-word;
    }

    .console-check-stack {
        display: grid;
        gap: 10px;
        padding-bottom: 18px;
    }

    .console-check-stack .explain-card {
        min-height: 94px;
        border-color: rgba(226, 156, 126, 0.34);
        background: rgba(255, 255, 255, 0.72);
    }

    .download-note {
        color: #76584f;
        line-height: 1.68;
        margin: -2px 0 14px;
    }

    .scale-guide {
        border-left-color: #ef8796;
        background: rgba(255, 255, 255, 0.78);
    }

    details[data-testid="stExpander"],
    div[data-testid="stExpander"] {
        border: 1px solid rgba(226, 156, 126, 0.38) !important;
        border-radius: 16px !important;
        background:
            rgba(255, 250, 246, 0.78) !important;
        box-shadow: 0 10px 22px rgba(86, 61, 48, 0.06) !important;
    }

    details[data-testid="stExpander"] summary {
        color: #65372f !important;
        font-weight: 900 !important;
    }

    div[data-testid="stAlert"] {
        border-color: rgba(226, 156, 126, 0.38) !important;
        border-radius: 16px !important;
        background: rgba(255, 250, 246, 0.78) !important;
    }

    .stButton button {
        min-height: 50px !important;
        font-size: 1rem !important;
        font-weight: 950 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

render_page_hero(
    "研究者控制台",
    "集中查看样例数据、运行状态、研究记录和最近一次模型判定，用于快速确认系统链路完整性与研究材料状态。",
    eyebrow="研究与运行管理",
    badges=["样例数据", "运行状态", "模型判定", "研究记录"],
)


def csv_row_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)
    except Exception:
        return 0


profile = load_pet_profile() or {}
memories = load_pet_memories()
history = st.session_state.get("chat_history") or load_chat_history()
image_path = load_pet_image_path()
image_status = image_generation_status()
last_meta = st.session_state.get("last_meta") or {}
checkins = load_daily_checkins()
plan_progress = load_plan_progress()
memorial_settings = load_memorial_settings()
plan_done_count = sum(1 for idx in range(1, 8) if plan_progress.get(f"day_{idx}"))

with st.container(border=True):
    render_section_header("快速操作", "用于准备或清理本地样例数据。重置会清空本地宠物档案、回忆、聊天、图片和陪伴计划记录。", "⚙")
    op1, op2, op3 = st.columns([1, 1, 2], gap="medium")
    with op1:
        if st.button("载入演示数据", use_container_width=True):
            load_demo_data()
            st.success("已载入演示数据。")
            st.rerun()
    with op2:
        if st.button("重置所有宠物数据", use_container_width=True):
            reset_all_pet_data()
            st.success("已重置所有宠物数据。")
            st.rerun()
    with op3:
        st.markdown(
            """
            <div class="scale-guide" style="padding:11px 14px;">
            当前测试版本保持关闭 BLIP 图片描述；Embedding、RAG、情绪识别、阶段识别、风险识别、策略选择和 LLM 回复生成仍为核心链路。
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="console-section-bottom-space"></div>', unsafe_allow_html=True)

st.markdown('<div class="console-panel-gap"></div>', unsafe_allow_html=True)

with st.container(border=True):
    render_section_header("运行状态总览", "汇总本地数据、素材状态、研究记录和外部接口状态。", "i")
    st.markdown('<div class="console-status-wrap"></div>', unsafe_allow_html=True)
    status_cols = st.columns(4, gap="medium")
    with status_cols[0]:
        render_metric_card("宠物档案", "已保存" if profile else "未保存", PROFILE_PATH)
    with status_cols[1]:
        render_metric_card("回忆素材", f"{len(memories)} 字", MEMORY_PATH)
    with status_cols[2]:
        render_metric_card("聊天记录", f"{len(history)} 条", CHAT_HISTORY_PATH)
    with status_cols[3]:
        render_metric_card("宠物照片", "已上传" if image_path else "未上传", image_path or "data/images")

    st.markdown('<div class="console-card-row-gap"></div>', unsafe_allow_html=True)

    status_cols2 = st.columns(4, gap="medium")
    with status_cols2[0]:
        render_metric_card("虚拟纪念形象", "已生成" if os.path.exists(GENERATED_AVATAR_PATH) else "未生成", GENERATED_AVATAR_PATH)
    with status_cols2[1]:
        render_metric_card("图像生成接口", "可用" if image_status["ready"] else "未启用", image_status.get("model", ""))
    with status_cols2[2]:
        render_metric_card("用户反馈记录", f"{csv_row_count(USER_TEST_PATH)} 条", USER_TEST_PATH)
    with status_cols2[3]:
        render_metric_card("练习记录", f"{csv_row_count(REFLECTION_PATH)} 条", REFLECTION_PATH)

    st.markdown('<div class="console-card-row-gap"></div>', unsafe_allow_html=True)

    status_cols3 = st.columns(4, gap="medium")
    with status_cols3[0]:
        render_metric_card("今日状态记录", f"{len(checkins)} 条", CHECKIN_PATH)
    with status_cols3[1]:
        render_metric_card("七天计划进度", f"{plan_done_count}/7 天", PLAN_PROGRESS_PATH)
    with status_cols3[2]:
        render_metric_card(
            "纪念日提醒",
            "已设置" if memorial_settings.get("date") else "未设置",
            MEMORIAL_SETTINGS_PATH,
        )
    with status_cols3[3]:
        render_metric_card("陪伴计划页面", "已启用", "pages/03_今日陪伴计划.py")

    st.markdown('<div class="console-section-bottom-space"></div>', unsafe_allow_html=True)

st.markdown('<div class="console-panel-gap"></div>', unsafe_allow_html=True)

with st.container(border=True):
    render_section_header("测试数据导出", "用于在 Streamlit 在线测试后手动下载 CSV，避免云端临时文件重启后丢失。", "⇩")
    st.markdown(
        '<div class="download-note">建议每轮测试结束后立即导出。在线环境中的本地 CSV 不会自动同步回 GitHub 或电脑。</div>',
        unsafe_allow_html=True,
    )
    download_cols = st.columns(3, gap="medium")
    with download_cols[0]:
        st.download_button(
            "下载用户反馈 CSV",
            data=csv_file_bytes(USER_TEST_PATH, USER_TEST_FIELDNAMES),
            file_name="petecho_user_test_feedback.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_cols[1]:
        st.download_button(
            "下载每日状态 CSV",
            data=csv_file_bytes(CHECKIN_PATH, CHECKIN_FIELDNAMES),
            file_name="petecho_daily_checkins.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_cols[2]:
        st.download_button(
            "下载练习记录 CSV",
            data=csv_file_bytes(REFLECTION_PATH),
            file_name="petecho_reflection_entries.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown('<div class="console-section-bottom-space"></div>', unsafe_allow_html=True)

st.markdown('<div class="console-panel-gap"></div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:
    with st.container(border=True):
        render_section_header("最近一次心理支持判定", "用于复核情绪、阶段、风险和策略选择是否与当前输入一致。", "✦")
        if last_meta:
            support_panel = build_support_panel(last_meta)
            c1, c2 = st.columns(2)
            with c1:
                render_metric_card("支持重点", support_panel["support_focus"])
                render_metric_card("回应策略", support_panel["response_strategy"])
                render_metric_card("记忆调用", support_panel["memory_mode"])
            with c2:
                render_metric_card("情绪负荷", support_panel["emotion_load"])
                render_metric_card("安全状态", support_panel["safety_status"])
                render_metric_card("调用条数", str(last_meta.get("memory_retrieval_count", 0)))

            st.markdown(
                f"""
                <div class="scale-guide" style="margin-top:12px;">
                {support_panel["explanation"]}
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("查看完整判定数据"):
                st.json(last_meta)
        else:
            st.info("尚未完成对话。哀伤支持对话页面产生消息后，此处会显示最近一次判定。")

with right_col:
    with st.container(border=True):
        render_section_header("研究与运行检查清单", "用于确认核心链路、边界策略和验证材料是否完整。", "✓")
        st.markdown(
            """
            <div class="console-check-stack">
            <div class="explain-card">
                <div class="explain-title">心理支持链路</div>
                <div class="explain-body">情绪识别、哀伤阶段、风险等级、策略选择、RAG 调用和 LLM 生成均可在对话页与此页展示。</div>
            </div>
            <div class="explain-card">
                <div class="explain-title">安全边界</div>
                <div class="explain-body">高风险表达会跳过普通沉浸式回忆，优先使用安全回应模板和现实求助提示。</div>
            </div>
            <div class="explain-card">
                <div class="explain-title">用户测试记录</div>
                <div class="explain-body">反馈页记录体验前后情绪、支持感、不适感、推荐意愿和 PBQ/ICG 辅助维度。</div>
            </div>
            <div class="explain-card">
                <div class="explain-title">规则评估</div>
                <div class="explain-body">可使用 scripts/evaluate_rules.py 汇总系统判断与专业标注之间的一致性系数。</div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
