import streamlit as st

from modules.ui_components import (
    apply_global_style,
    ensure_session_state,
    render_feature_grid,
    render_home_hero,
)


st.set_page_config(page_title="PetEcho", page_icon="🐾", layout="wide")

apply_global_style("home")
ensure_session_state()

render_home_hero()

render_feature_grid(
    [
        {
            "icon": "1",
            "title": "建立宠物档案",
            "body": "用名字、照片、性格和回忆建立纪念素材，为后续记忆检索和支持回应提供依据。",
        },
        {
            "icon": "2",
            "title": "今日陪伴计划",
            "body": "完成当日状态 check-in，查看建议路径、七天陪伴任务、纪念日提醒和状态趋势。",
        },
        {
            "icon": "3",
            "title": "支持对话与纪念",
            "body": "在安全边界内完成支持对话，并把回忆整理成可保存、可回看的纪念内容。",
        },
        {
            "icon": "4",
            "title": "验证与研究记录",
            "body": "记录体验反馈、量表辅助校准和安全性验证线索，为后续研究与迭代提供依据。",
        },
    ]
)
