STAGE_LABELS = {
    "acute_grief": "承接当下悲伤",
    "shock": "先稳定突如其来的冲击",
    "denial": "温柔保持现实边界",
    "yearning": "承接强烈思念",
    "guilt": "松动愧疚与自责",
    "anger": "承接愤怒与不公平感",
    "numbness": "稳定麻木与不真实感",
    "depressive_withdrawal": "低负担现实支撑",
    "integration": "带着回忆重新连接生活",
}

STRATEGY_LABELS = {
    "crisis_safety": "危机安全支持",
    "stabilize_and_support": "稳定情绪与现实支持",
    "emotional_holding": "当下悲伤承接",
    "stabilization": "情绪稳定",
    "gentle_acceptance": "温柔接纳与现实边界",
    "memory_activation": "轻量记忆唤起",
    "reconnection_support": "持续性联结",
    "coping_guidance": "低负担应对建议",
    "reconnection_guidance": "生活重连建议",
    "self_compassion": "自我宽恕支持",
    "anger_validation": "愤怒与不公平感承接",
    "numbness_grounding": "麻木感稳定支持",
    "behavioral_activation_support": "低负担行动支持",
}

RISK_LABELS = {
    "low": "常规支持",
    "medium": "需要现实支持",
    "high": "高风险提醒",
    "imminent": "即时安全优先",
}

MEMORY_USAGE_LABELS = {
    "none": "暂不调用回忆",
    "none_or_low": "必要时轻量调用",
    "low": "轻量调用贴切记忆",
    "low_or_medium": "适度调用贴切记忆",
    "medium": "调用一段温暖记忆",
}


def _emotion_load_label(emotion: dict, risk_level: str) -> str:
    if risk_level in {"high", "imminent"}:
        return "高，需要优先安全支持"

    sadness = emotion.get("sadness", 0.0)
    loneliness = emotion.get("loneliness", 0.0)
    yearning = emotion.get("yearning", 0.0)
    guilt = emotion.get("guilt", 0.0)
    anger = emotion.get("anger", 0.0)
    numbness = emotion.get("numbness", 0.0)
    strongest = max(sadness, loneliness, yearning, guilt, anger, numbness)

    if strongest >= 0.75:
        return "较高，需要放慢回应节奏"
    if strongest >= 0.45:
        return "中等，需要温和承接"
    return "较低，适合常规陪伴"


def _memory_mode_label(strategy: dict, memory_context: list | None = None) -> str:
    if not memory_context:
        return "本轮未调用具体回忆"
    usage = strategy.get("memory_usage", "low")
    if strategy.get("memory_style") == "forbid":
        return "暂不调用回忆"
    return MEMORY_USAGE_LABELS.get(usage, "按贴合程度调用")


def _support_explanation(
    grief_stage: str,
    risk: dict,
    strategy: dict,
    emotion: dict,
) -> str:
    risk_level = risk.get("level", "low")
    strategy_name = strategy.get("name", "")

    if risk_level == "imminent":
        return "系统检测到可能存在即时危险的表达，因此跳过沉浸式回忆，优先给出清晰的现实求助建议。"
    if risk_level == "high":
        return "系统检测到较高风险表达，因此减少普通安慰和记忆展开，优先提醒用户联系现实支持。"
    if risk_level == "medium":
        return "当前表达包含较重痛苦或功能受损信号，系统会先稳定情绪，再给出低负担的现实支持方向。"
    if strategy.get("guidance_mode") == "coping_guidance":
        return "用户正在主动寻求办法，系统会先承接情绪，再提供一两个具体、低负担的行动建议。"
    if strategy_name == "self_compassion":
        return "当前表达中有明显愧疚或自责，系统会先承接遗憾，再温柔松动过度自我归因。"
    if strategy_name == "anger_validation":
        return "当前表达中有不公平感或愤怒，系统会允许这种感受被表达，而不是急着劝用户接受。"
    if strategy_name == "numbness_grounding":
        return "当前表达中有麻木或不真实感，系统会减少复杂分析，优先提供稳定和落地感。"
    if strategy_name == "behavioral_activation_support":
        return "当前表达中有低落退缩或日常功能受损信号，系统会减少回忆沉浸，优先给出很小的现实支撑。"
    if strategy_name == "emotional_holding":
        return "当前表达更像当下的悲伤和说不出口的痛感，系统会先安静承接，而不是强行调取回忆。"
    if grief_stage == "yearning":
        return "当前表达以想念和回忆为主，系统会轻轻调用贴切记忆，避免把沉重细节强行展开。"
    if grief_stage == "integration":
        return "当前表达已有一点重新连接生活的倾向，系统会支持用户带着回忆继续照顾自己。"
    if strategy_name == "stabilization":
        return "当前更需要先缓一缓，系统会以短句和稳定陪伴为主，不急着推动接受。"

    return "系统会根据情绪负荷、风险等级和记忆贴合度，选择较温和的支持方式。"


def build_support_panel(meta: dict) -> dict:
    meta = meta or {}
    emotion = meta.get("emotion", {}) or {}
    grief_stage = meta.get("grief_stage", "")
    risk = meta.get("risk", {}) or {}
    strategy = meta.get("strategy", {}) or {}
    risk_level = risk.get("level", "low")
    strategy_name = strategy.get("name", "")

    if risk_level in {"high", "imminent"}:
        support_focus = "安全与现实支持优先"
    elif risk_level == "medium":
        support_focus = "稳定痛苦与现实支持"
    elif strategy_name == "self_compassion":
        support_focus = "松动愧疚与自责"
    elif strategy_name == "anger_validation":
        support_focus = "承接不公平感"
    elif strategy_name == "numbness_grounding":
        support_focus = "稳定麻木与不真实感"
    elif strategy_name == "behavioral_activation_support":
        support_focus = "低负担现实行动"
    elif strategy_name == "emotional_holding":
        support_focus = "承接当下悲伤"
    elif strategy.get("guidance_mode") == "coping_guidance":
        support_focus = "低负担应对支持"
    else:
        support_focus = STAGE_LABELS.get(grief_stage, "温和陪伴与情绪承接")

    return {
        "support_focus": support_focus,
        "emotion_load": _emotion_load_label(emotion, risk_level),
        "response_strategy": STRATEGY_LABELS.get(strategy_name, "温和支持"),
        "memory_mode": _memory_mode_label(strategy, meta.get("memory_context")),
        "safety_status": RISK_LABELS.get(risk_level, risk_level),
        "explanation": _support_explanation(grief_stage, risk, strategy, emotion),
    }
