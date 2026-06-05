import streamlit as st

from modules.ui_components import (
    apply_global_style,
    ensure_session_state,
    render_feature_grid,
    render_page_hero,
    render_section_header,
)
from utils.participant import render_participant_control
from utils.research_io import USER_TEST_FIELDNAMES, rows_to_csv_bytes, save_user_test_feedback


st.set_page_config(page_title="用户测试与反馈 - PetEcho", page_icon="🐾", layout="wide")

apply_global_style("feedback")
ensure_session_state()
participant_id = render_participant_control()

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

    .feature-grid {
        align-items: stretch;
        margin: 16px 0 1.75rem;
    }

    .feature-card {
        min-height: 172px;
        border-color: rgba(226, 156, 126, 0.36);
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(248, 252, 249, 0.78));
    }

    .feedback-panel-gap {
        height: 18px;
    }

    .feedback-explain-stack {
        display: grid;
        gap: 10px;
        padding-bottom: 18px;
    }

    .feedback-explain-stack .explain-card {
        min-height: 88px;
        border-color: rgba(226, 156, 126, 0.34);
        background: rgba(255, 255, 255, 0.72);
    }

    .scale-guide {
        border-left-color: #ef8796;
        background: rgba(255, 255, 255, 0.78);
        margin-bottom: 8px;
    }

    details[data-testid="stExpander"],
    div[data-testid="stExpander"] {
        border: 1px solid rgba(226, 156, 126, 0.38) !important;
        border-radius: 16px !important;
        background:
            rgba(255, 250, 246, 0.78) !important;
        box-shadow: 0 10px 22px rgba(86, 61, 48, 0.06) !important;
    }

    div[data-testid="stExpander"] details {
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

    div[data-testid="stForm"] {
        border: 1px solid rgba(226, 156, 126, 0.38) !important;
        border-radius: 16px !important;
        background:
            rgba(255, 255, 255, 0.46) !important;
        padding: 18px 16px 16px !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75) !important;
    }

    .stTextArea textarea {
        background: rgba(255, 248, 242, 0.96) !important;
        border: 1px solid rgba(226, 156, 126, 0.46) !important;
        color: #5e3a32 !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(255, 142, 155, 0.95) !important;
        box-shadow: 0 0 0 3px rgba(255, 170, 185, 0.18) !important;
    }

    .stFormSubmitButton button {
        min-height: 50px !important;
        font-size: 1rem !important;
        font-weight: 950 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

render_page_hero(
    "用户测试与反馈",
    "集中记录真实体验、安全性反馈与量表辅助维度，形成“体验前后变化、主观支持感、风险反馈、量表参考”的研究闭环。",
    eyebrow="真实用户测试 · 安全性验证",
    badges=["匿名反馈", "体验前后对比", "PBQ/ICG 辅助校准", "非诊断用途"],
)

render_feature_grid(
    [
        {
            "icon": "A",
            "title": "页面定位",
            "body": "用于沉淀真实体验数据，支持产品迭代、安全性复核与后续研究分析。",
        },
        {
            "icon": "B",
            "title": "量表字段怎么用",
            "body": "PBQ/ICG 维度用于后续与系统情绪、阶段和风险判断做一致性或相关性分析，不是临床诊断。",
        },
        {
            "icon": "C",
            "title": "测试流程",
            "body": "体验前记录情绪强度；完成档案建立和一次支持对话后，再记录理解感、支持感、不适感和备注。",
        },
        {
            "icon": "D",
            "title": "研究价值",
            "body": "用于证明系统不只是生成对话，也具备可验证、可复盘、可持续迭代的评估路径。",
        },
    ]
)

left_col, right_col = st.columns([1.25, 0.85], gap="large")

with left_col:
    with st.container(border=True):
        render_section_header("匿名体验反馈表", "不记录姓名；数据用于体验评估、产品迭代与安全性验证。", "✎")

        with st.form("user_test_feedback_form", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                pre_emotion = st.slider("体验前情绪强度", 1, 7, 4)
                post_emotion = st.slider("体验后情绪强度", 1, 7, 4)
            with c2:
                understood = st.slider("被理解感", 1, 7, 5)
                naturalness = st.slider("回复自然度", 1, 7, 5)
            with c3:
                supportiveness = st.slider("心理支持感", 1, 7, 5)
                discomfort = st.slider("不适感", 1, 7, 1)

            willingness = st.slider("继续使用或推荐意愿", 1, 7, 5)

            with st.expander("量表辅助校准字段（可选）", expanded=True):
                st.markdown(
                    """
                    <div class="scale-guide">
                    0-4 分表示从“没有或很轻”到“非常明显”。该字段不用于临床诊断，仅用于把主观状态与系统识别结果进行辅助校准：
                    当系统识别出明显的愧疚、高思念或功能受损倾向时，相关维度应呈现大致对应趋势。
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                s1, s2, s3, s4, s5 = st.columns(5)
                with s1:
                    pbq_grief = st.slider("PBQ 悲伤", 0, 4, 0, help="宠物离别后的悲伤、空落、痛感强度。")
                with s2:
                    pbq_guilt = st.slider("PBQ 愧疚", 0, 4, 0, help="是否有自责、后悔、觉得自己做得不够。")
                with s3:
                    pbq_anger = st.slider("PBQ 愤怒", 0, 4, 0, help="是否有不公平、愤怒、埋怨或难以接受。")
                with s4:
                    icg_yearning = st.slider("ICG 思念", 0, 4, 0, help="是否强烈想念、反复想见到它或被回忆牵动。")
                with s5:
                    icg_impairment = st.slider("ICG 功能受损", 0, 4, 0, help="是否影响睡眠、进食、学习、工作或社交。")

            notes = st.text_area("备注（可选）", height=95, placeholder="例如：哪一句回复带来了被理解感？哪处内容造成不适？")
            feedback_submitted = st.form_submit_button("保存匿名反馈")

        if feedback_submitted:
            saved_feedback = save_user_test_feedback(
                {
                    "participant_id": participant_id,
                    "pre_emotion": pre_emotion,
                    "post_emotion": post_emotion,
                    "understood": understood,
                    "naturalness": naturalness,
                    "supportiveness": supportiveness,
                    "discomfort": discomfort,
                    "willingness": willingness,
                    "pbq_grief": pbq_grief,
                    "pbq_guilt": pbq_guilt,
                    "pbq_anger": pbq_anger,
                    "icg_yearning": icg_yearning,
                    "icg_impairment": icg_impairment,
                    "notes": notes.strip(),
                }
            )
            st.success("匿名反馈已保存。")
            st.download_button(
                "下载本次反馈 CSV",
                data=rows_to_csv_bytes([saved_feedback], USER_TEST_FIELDNAMES),
                file_name=f"petecho_feedback_{saved_feedback['created_at'].replace(':', '-')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

with right_col:
    with st.container(border=True):
        render_section_header("PBQ / ICG 维度说明", "说明各维度的记录含义，避免缩写影响理解。", "i")
        st.markdown(
            """
            <div class="feedback-explain-stack">
            <div class="explain-card">
                <div class="explain-title">PBQ 悲伤</div>
                <div class="explain-body">宠物离别后的悲伤、空落、难受程度。</div>
            </div>
            <div class="explain-card">
                <div class="explain-title">PBQ 愧疚</div>
                <div class="explain-body">是否觉得“是不是我没做好”“如果当时怎样就好了”。</div>
            </div>
            <div class="explain-card">
                <div class="explain-title">PBQ 愤怒</div>
                <div class="explain-body">对离别、不公平、疾病或外界因素的愤怒与不甘。</div>
            </div>
            <div class="explain-card">
                <div class="explain-title">ICG 思念</div>
                <div class="explain-body">强烈想念、反复想见到它、被回忆持续牵动。</div>
            </div>
            <div class="explain-card">
                <div class="explain-title">ICG 功能受损</div>
                <div class="explain-body">悲伤是否影响睡眠、进食、学习、工作或社交。</div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="feedback-panel-gap"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        render_section_header("安全性验证记录建议", "真实测试阶段可持续记录不适感、风险反馈与系统处置情况。", "!")
        st.markdown(
            """
            <div class="scale-guide">
            当“不适感”评分较高时，建议进一步记录造成不适的内容类型，例如过度沉浸、记忆调用突兀、风险提示不足或回复不贴合事实。后续可将这些反馈与系统日志中的风险等级、策略选择和记忆调用次数进行联合分析。
            </div>
            """,
            unsafe_allow_html=True,
        )
