from datetime import datetime
from uuid import uuid4

import re
import streamlit as st


def _new_participant_id(prefix: str = "P") -> str:
    stamp = datetime.now().strftime("%m%d%H%M")
    suffix = uuid4().hex[:4].upper()
    return f"{prefix}{stamp}-{suffix}"


def normalize_participant_id(value: str | None) -> str:
    text = (value or "").strip()
    text = re.sub(r"[^0-9A-Za-z_\-]", "", text)
    return text[:32]


def ensure_participant_id() -> str:
    current = normalize_participant_id(st.session_state.get("participant_id"))
    if not current:
        current = _new_participant_id()
        st.session_state["participant_id"] = current
    return current


def get_participant_id() -> str:
    return ensure_participant_id()


def render_participant_control() -> str:
    participant_id = ensure_participant_id()
    with st.sidebar:
        st.markdown("### 测试编号")
        edited = st.text_input(
            "participant_id",
            value=participant_id,
            label_visibility="collapsed",
            help="用于区分不同测试用户。可使用系统自动生成编号，也可以手动填写 U001、SIM001 等。",
        )
        cleaned = normalize_participant_id(edited)
        if cleaned and cleaned != participant_id:
            st.session_state["participant_id"] = cleaned
            participant_id = cleaned
        if st.button("生成新测试编号", use_container_width=True):
            participant_id = _new_participant_id()
            st.session_state["participant_id"] = participant_id
            st.rerun()
        st.caption(f"当前编号：{participant_id}")
    return participant_id
