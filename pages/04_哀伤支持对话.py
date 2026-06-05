import html

import streamlit as st

from modules.emotion import detect_emotion
from modules.grief_stage import detect_grief_stage
from modules.interventions import recommend_intervention
from modules.llm_api import call_llm
from modules.prompt_builder import build_prompt
from modules.rag import retrieve_memories, should_retrieve_memories
from modules.risk import detect_risk
from modules.safety_response import build_safety_reply, should_use_safety_template
from modules.strategy import choose_strategy
from modules.support_display import build_support_panel
from modules.ui_components import (
    apply_global_style,
    clean_reply_text,
    ensure_session_state,
    get_pet_avatar_data_uri,
    now_hhmm,
    page_href,
    render_chat_board,
    render_metric_card,
    render_page_hero,
    render_section_header,
)
from utils.file_io import load_pet_profile, save_chat_history
from utils.participant import render_participant_control
from utils.research_io import save_chat_transcript_turn, save_reflection_entry


st.set_page_config(page_title="哀伤支持对话 - PetEcho", page_icon="🐾", layout="wide")

apply_global_style("chat")
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
        background:
            linear-gradient(135deg, rgba(255, 253, 249, 0.95), rgba(255, 250, 247, 0.9) 52%, rgba(243, 251, 247, 0.9)) !important;
        box-shadow: 0 16px 34px rgba(86, 61, 48, 0.08) !important;
    }

    .chat-board {
        min-height: 408px;
        max-height: 50vh;
    }

    .chat-status-card {
        display: flex;
        align-items: center;
        gap: 13px;
        margin-bottom: 14px;
        padding: 12px 14px;
        border: 1px solid rgba(226, 156, 126, 0.38);
        border-radius: 16px;
        background:
            rgba(255, 255, 255, 0.66);
        box-shadow: 0 10px 22px rgba(86, 61, 48, 0.06);
    }

    .chat-status-card img {
        width: 58px;
        height: 58px;
        border-radius: 19px;
        object-fit: cover;
        border: 1px solid #ffd8c7;
        flex: 0 0 auto;
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

    .next-step-panel {
        position: relative;
        overflow: hidden;
        width: 100%;
        box-sizing: border-box;
        min-height: 248px;
        border: 1px solid rgba(239, 133, 108, 0.66);
        border-radius: 18px;
        padding: 18px 18px 16px;
        background:
            linear-gradient(135deg, rgba(255, 241, 231, 0.98), rgba(255, 250, 245, 0.94));
        box-shadow: 0 14px 30px rgba(86, 61, 48, 0.1);
        margin: 2px 0 16px;
    }

    .next-step-panel::after {
        content: "";
        position: absolute;
        right: 16px;
        top: 16px;
        width: 54px;
        height: 30px;
        opacity: 0.35;
        border-radius: 999px;
        height: 3px;
        background: linear-gradient(90deg, #ef856c, #ef8796, #a9c9af);
    }

    .next-step-title {
        position: relative;
        z-index: 1;
        color: #63372e;
        font-size: 1.08rem;
        font-weight: 950;
        line-height: 1.35;
        margin-bottom: 6px;
    }

    .next-step-desc {
        position: relative;
        z-index: 1;
        color: #76584f;
        font-size: 0.94rem;
        line-height: 1.68;
        margin-bottom: 11px;
        max-width: 92%;
    }

    .next-step-actions {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: 1fr;
        gap: 10px;
    }

    .next-primary,
    .next-secondary {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 42px;
        border-radius: 999px;
        text-decoration: none !important;
        font-weight: 950;
        border: 1px solid rgba(255, 255, 255, 0.66);
    }

    .next-primary {
        color: #4c2b25 !important;
        background: linear-gradient(135deg, #ffb07c 0%, #ef7f8f 100%);
        box-shadow: 0 14px 26px rgba(222, 93, 84, 0.2);
    }

    .next-secondary {
        color: #69433b !important;
        background: rgba(255, 255, 255, 0.72);
        border-color: rgba(226, 156, 126, 0.4);
    }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stMarkdown"] .support-panel-marker) {
        min-height: 304px !important;
    }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stMarkdown"] .exercise-panel-marker) {
        min-height: 306px !important;
    }

    .support-panel-marker,
    .exercise-panel-marker {
        display: none;
    }

    .support-empty-box,
    .exercise-empty-box {
        border: 1px dashed rgba(226, 156, 126, 0.45);
        border-radius: 16px;
        color: #7c5d55;
        background:
            rgba(255, 255, 255, 0.7);
        padding: 20px 18px;
        line-height: 1.75;
        text-align: center;
    }

    .support-empty-box {
        min-height: 118px;
        display: grid;
        place-items: center;
        margin-top: 8px;
    }

    .exercise-empty-box {
        min-height: 132px;
        display: grid;
        place-items: center;
        margin-top: 10px;
    }

    .exercise-box {
        min-height: 156px;
    }

    .scale-guide {
        border-left-color: #ff9fac;
        margin-bottom: 4px;
    }

    @media (max-width: 980px) {
        .chat-board {
            min-height: 320px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

render_page_hero(
    "哀伤支持对话",
    "本页面承载 PetEcho 的核心支持流程：系统会先识别情绪负荷、哀伤阶段倾向和安全风险，再选择心理支持策略，按需调用记忆，生成贴合事实和当下状态的回应。",
    eyebrow="核心功能 · 温暖但有边界",
    badges=["风险优先", "记忆按需调用", "事实贴合审校", "支持当下与未来"],
)

profile = load_pet_profile()
pet_name = (profile or {}).get("pet_name", "数字宠物")

if not profile:
    st.warning("还没有保存宠物档案。可以先进入档案页填写基本信息，也可以载入演示数据后测试。")
    st.markdown(
        f'<a class="nav-link-button" href="{page_href("宠物档案")}" target="_self">先去建立宠物档案</a>',
        unsafe_allow_html=True,
    )

chat_col, insight_col = st.columns([1.46, 0.92], gap="large")

with chat_col:
    with st.container(border=True):
        render_section_header(
            "支持对话区",
            "数字宠物头像会优先使用生成的纪念形象；未生成时，使用已上传的宠物照片。",
            "💬",
        )
        st.markdown(
            f"""
            <div class="chat-status-card">
                <img src="{get_pet_avatar_data_uri()}" alt="">
                <div>
                    <div class="section-title" style="font-size:1.04rem;">{html.escape(pet_name)}</div>
                    <div class="section-desc">回应会先承接情绪，再在合适时把注意力带回当下、身体感受和现实联结。</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_chat_board(st.session_state.get("chat_history", []), pet_name)

        with st.form("chat_message_form", clear_on_submit=False):
            user_input = st.text_area(
                "你想对它说什么？",
                height=145,
                key="chat_input",
                placeholder="例如：今天又想它了，回家看到门口空空的，我很难受。",
            )

            send_clicked = st.form_submit_button("发送", use_container_width=True)

        clear_clicked = st.button("清空聊天历史", use_container_width=True)

        if clear_clicked:
            st.session_state["chat_history"] = []
            st.session_state["last_reply"] = ""
            st.session_state["last_meta"] = {}
            st.session_state["recent_memory_contexts"] = []
            st.session_state["pending_chat_input"] = ""
            save_chat_history([])
            st.rerun()

        if send_clicked:
            if not profile:
                st.error("请先保存宠物档案。")
            elif not user_input.strip():
                st.warning("请输入内容。")
            else:
                emotion = detect_emotion(user_input)
                grief_stage = detect_grief_stage(user_input, emotion)
                risk = detect_risk(user_input)
                strategy = choose_strategy(grief_stage, risk, emotion, user_input)
                recent_history = st.session_state["chat_history"][-4:]

                with st.spinner("数字宠物正在想一想该怎么回应你..."):
                    try:
                        memory_context = []
                        memory_retrieval_allowed = False

                        if should_use_safety_template(risk):
                            reply = clean_reply_text(build_safety_reply(risk, user_input))
                        else:
                            memory_retrieval_allowed = should_retrieve_memories(
                                user_input,
                                strategy=strategy,
                                emotion=emotion,
                            )
                            if memory_retrieval_allowed:
                                try:
                                    memory_context = retrieve_memories(
                                        user_input,
                                        top_k=2,
                                        avoid_texts=st.session_state.get("recent_memory_contexts", []),
                                        strategy=strategy,
                                        emotion=emotion,
                                    )
                                except Exception:
                                    memory_context = []

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
                        history = st.session_state["chat_history"]
                        turn_id = f"{participant_id}-{(len(history) // 2) + 1:03d}"
                        meta = {
                            "participant_id": participant_id,
                            "turn_id": turn_id,
                            "emotion": emotion,
                            "grief_stage": grief_stage,
                            "risk": risk,
                            "strategy": strategy,
                            "memory_context": memory_context,
                            "memory_retrieval_allowed": memory_retrieval_allowed,
                            "memory_retrieval_count": len(memory_context),
                            "pet_profile": profile,
                            "recent_history": recent_history,
                            "guidance_mode": strategy.get("guidance_mode", "none"),
                            "guidance_focus": strategy.get("guidance_focus", ""),
                            "safety_template_used": should_use_safety_template(risk),
                        }
                        history.append({
                            "role": "user",
                            "content": user_input,
                            "timestamp": current_time,
                            "participant_id": participant_id,
                            "turn_id": turn_id,
                        })
                        history.append({
                            "role": "assistant",
                            "content": reply,
                            "timestamp": current_time,
                            "participant_id": participant_id,
                            "turn_id": turn_id,
                            "meta": meta,
                        })
                        st.session_state["chat_history"] = history
                        save_chat_history(history)
                        save_chat_transcript_turn(participant_id, user_input, reply, meta)

                        st.session_state["last_reply"] = reply
                        st.session_state["last_meta"] = meta

                        if memory_context:
                            recent_contexts = st.session_state.get("recent_memory_contexts", [])
                            st.session_state["recent_memory_contexts"] = (recent_contexts + memory_context)[-8:]

                        st.session_state["pending_chat_input"] = ""
                        st.rerun()
                    except Exception as e:
                        st.error(f"生成回复时发生错误：{e}")

with insight_col:
    with st.container(border=True):
        render_section_header("本轮支持解释", "概括本轮识别结果、支持重点、风险状态和记忆调用方式。", "✦")
        st.markdown('<div class="support-panel-marker"></div>', unsafe_allow_html=True)
        meta = st.session_state.get("last_meta")
        if meta:
            support_panel = build_support_panel(meta)
            c1, c2 = st.columns(2)
            with c1:
                render_metric_card("支持重点", support_panel["support_focus"])
                render_metric_card("回应策略", support_panel["response_strategy"])
            with c2:
                render_metric_card("情绪负荷", support_panel["emotion_load"])
                render_metric_card("安全状态", support_panel["safety_status"])

            st.markdown(
                f"""
                <div class="scale-guide" style="margin-top:12px;">
                <b>记忆调用：</b>{html.escape(support_panel["memory_mode"])}<br>
                {html.escape(support_panel["explanation"])}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="support-empty-box">发送一条消息后，这里会显示情绪负荷、风险等级、回应策略和记忆调用方式。</div>',
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        render_section_header("低负担小练习", "在悲伤较强时，提供轻量练习，帮助情绪从回忆慢慢回到当下和现实联结。", "☀")
        st.markdown('<div class="exercise-panel-marker"></div>', unsafe_allow_html=True)
        meta = st.session_state.get("last_meta")
        if meta:
            exercise = recommend_intervention(meta)
            st.markdown(
                f"""
                <div class="exercise-box">
                    <b>{html.escape(exercise["title"])}</b>
                    <br><span class="tiny-muted">类别：{html.escape(exercise["category"])}｜{html.escape(exercise["rationale"])}</span>
                    <br><br>{html.escape(exercise["prompt"])}
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.form("reflection_exercise_form", clear_on_submit=True):
                reflection_text = st.text_area(
                    "写在这里",
                    placeholder=exercise["placeholder"],
                    height=95,
                    key="reflection_exercise_text",
                )
                reflection_submitted = st.form_submit_button("保存这次练习")

            if reflection_submitted:
                if reflection_text.strip():
                    save_reflection_entry(exercise, reflection_text.strip(), meta)
                    st.success("已保存这次练习。")
                else:
                    st.warning("可以先写下一句话，再保存。")
        else:
            st.markdown(
                '<div class="exercise-empty-box">完成一次对话后，这里会根据本轮状态推荐练习。</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="next-step-panel">
            <div class="next-step-title">下一步</div>
            <div class="next-step-desc">完成一次对话后，可以把体验沉淀为纪念内容，也可以回到今日计划继续记录状态变化。</div>
            <div class="next-step-actions">
                <a class="next-primary" href="{page_href("纪念空间")}" target="_self">进入纪念空间</a>
                <a class="next-secondary" href="{page_href("今日陪伴计划")}" target="_self">回到今日陪伴计划</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
