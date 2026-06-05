import re


ADVICE_PATTERNS = [
    "怎么办",
    "怎么做",
    "有什么办法",
    "有什么方法",
    "怎么走出来",
    "如何走出来",
    "给我一些建议",
    "给点建议",
    "我该怎么",
    "能做什么",
    "怎么调整",
    "怎么缓解",
    "怎么改善",
    "有没有什么建议",
    "怎么才能",
]

ADVICE_PATTERNS.extend([
    "我该怎么办",
    "怎么办",
    "怎么做",
    "有没有办法",
    "有什么方法",
    "给我一点建议",
    "给点建议",
    "怎么缓解",
    "怎么调整",
    "怎么才能好一点",
    "我能做什么",
    "能不能给我一些方法",
])


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def _is_advice_request(user_input: str | None) -> bool:
    text = _normalize_text(user_input or "")
    if not text:
        return False
    return any(p in text for p in ADVICE_PATTERNS)


def choose_strategy(
    grief_stage: str,
    risk: dict,
    emotion: dict | None = None,
    user_input: str | None = None,
) -> dict:
    emotion = emotion or {}
    sadness = emotion.get("sadness", 0.0)
    loneliness = emotion.get("loneliness", 0.0)
    yearning = emotion.get("yearning", 0.0)
    guilt = emotion.get("guilt", 0.0)
    anger = emotion.get("anger", 0.0)
    numbness = emotion.get("numbness", 0.0)

    advice_request = _is_advice_request(user_input)

    if risk["level"] in {"high", "imminent"}:
        return {
            "name": "crisis_safety",
            "description": "停止沉浸式互动，优先安全回应、现实支持与求助建议。",
            "response_style": "短句、稳定、温和，不扮演过度亲密角色。",
            "memory_usage": "none",
            "memory_style": "forbid",
            "retrieval_policy": "off",
            "guidance_mode": "emergency_referral" if risk["level"] == "imminent" else "crisis_referral",
            "guidance_focus": "safety_first",
            "emotional_goal": "先稳住情绪并鼓励联系现实支持。",
            "closing_style": "鼓励求助",
            "max_sentences": 3,
        }

    if risk["level"] == "medium":
        base = {
            "name": "stabilize_and_support",
            "description": "先接住痛苦，再温和引导现实支持，不做过多回忆展开。",
            "response_style": "以共情和稳定为主，避免过度沉浸。",
            "memory_usage": "low",
            "memory_style": "only_if_perfect_match",
            "retrieval_policy": "strict",
            "guidance_mode": "supportive_guidance" if advice_request else "stabilize_only",
            "guidance_focus": "low_burden_coping",
            "emotional_goal": "降低压迫感，帮助用户把注意力放回此刻的安全、身体和现实支持。",
            "closing_style": "当下锚定",
            "max_sentences": 4 if advice_request else 3,
        }
        return base

    # 用户明确求建议时，优先切到“建议型支持”
    if advice_request:
        if grief_stage in {"acute_grief", "shock", "denial", "depressive_withdrawal"}:
            return {
                "name": "coping_guidance",
                "description": "在早期哀伤中提供低负担、现实可行的建议，不急着推动接受。",
                "response_style": "先接情绪，再给一两个非常小的可执行建议。",
                "memory_usage": "none_or_low",
                "memory_style": "only_if_perfect_match",
                "retrieval_policy": "strict",
                "guidance_mode": "coping_guidance",
                "guidance_focus": "low_burden_coping",
                "emotional_goal": "让用户知道现在不必一下子走出来，只要先照顾好此刻，并给未来留一点点空间。",
                "closing_style": "当下小步",
                "max_sentences": 5,
            }
        if grief_stage == "yearning":
            return {
                "name": "coping_guidance",
                "description": "在强烈思念期提供兼顾想念与日常恢复的小建议。",
                "response_style": "先共情，再给出温和具体的小办法，不只停留在陪伴。",
                "memory_usage": "none_or_low",
                "memory_style": "only_if_perfect_match",
                "retrieval_policy": "strict",
                "guidance_mode": "coping_guidance",
                "guidance_focus": "meaning_and_memory",
                "emotional_goal": "帮助用户在想念和日常之间找到一点可移动的空间，把爱安放成可承载的小行动。",
                "closing_style": "记忆到生活",
                "max_sentences": 5,
            }
        if grief_stage == "integration":
            return {
                "name": "reconnection_guidance",
                "description": "帮助用户把思念整合进生活，并提供温和的继续生活建议。",
                "response_style": "先理解，再给出现实、轻量、能落地的生活重建建议。",
                "memory_usage": "none_or_low",
                "memory_style": "only_if_perfect_match",
                "retrieval_policy": "strict",
                "guidance_mode": "coping_guidance",
                "guidance_focus": "daily_reengagement",
                "emotional_goal": "让用户感到继续生活不是背叛回忆，而是带着回忆往前。",
                "closing_style": "带着回忆继续生活",
                "max_sentences": 5,
            }

    if guilt >= 0.45 or grief_stage == "guilt":
        return {
            "name": "self_compassion",
            "description": "针对强烈愧疚与自责，帮助用户区分爱、遗憾和责任。",
            "response_style": "先接住自责，再温柔松动过度归因。",
            "memory_usage": "none_or_low",
            "memory_style": "only_if_perfect_match",
            "retrieval_policy": "strict",
            "guidance_mode": "none",
            "guidance_focus": "",
            "emotional_goal": "让用户感到遗憾可以被看见，但不必把全部痛苦都变成自我惩罚。",
            "closing_style": "自我宽恕",
            "max_sentences": 4,
        }

    if anger >= 0.45 or grief_stage == "anger":
        return {
            "name": "anger_validation",
            "description": "承接不公平感和愤怒，不急着说服用户接受。",
            "response_style": "允许愤怒存在，避免讲大道理。",
            "memory_usage": "none_or_low",
            "memory_style": "only_if_perfect_match",
            "retrieval_policy": "strict",
            "guidance_mode": "none",
            "guidance_focus": "",
            "emotional_goal": "让用户知道愤怒也是哀伤的一部分，可以先被安全地表达。",
            "closing_style": "稳定情绪",
            "max_sentences": 4,
        }

    if numbness >= 0.45 or grief_stage == "numbness":
        return {
            "name": "numbness_grounding",
            "description": "面对麻木和不真实感，先提供稳定和落地感。",
            "response_style": "短句、慢节奏，少分析，多稳定。",
            "memory_usage": "none_or_low",
            "memory_style": "only_if_perfect_match",
            "retrieval_policy": "strict",
            "guidance_mode": "none",
            "guidance_focus": "",
            "emotional_goal": "让用户知道麻木也可能是哀伤中的保护反应，先回到此刻。",
            "closing_style": "安定陪伴",
            "max_sentences": 3,
        }

    strategy_map = {
        "acute_grief": {
            "name": "emotional_holding",
            "description": "承接当下悲伤和说不出口的痛感，不急着调用回忆或给大道理。",
            "response_style": "短句、安静、贴着当下感受，允许用户先不说。",
            "memory_usage": "none_or_low",
            "memory_style": "only_if_perfect_match",
            "retrieval_policy": "strict",
            "guidance_mode": "none",
            "guidance_focus": "",
            "emotional_goal": "让用户感到此刻的难过可以被安放，并轻轻回到呼吸、身体或眼前环境。",
            "closing_style": "当下锚定",
            "max_sentences": 3,
        },
        "shock": {
            "name": "stabilization",
            "description": "提供情绪稳定支持，不强行推动接受现实。",
            "response_style": "短句、轻柔、以陪伴和允许感受为主。",
            "memory_usage": "none_or_low",
            "memory_style": "only_if_perfect_match",
            "retrieval_policy": "strict",
            "guidance_mode": "none",
            "guidance_focus": "",
            "emotional_goal": "帮用户先缓一缓，不急着整理意义。",
            "closing_style": "安定陪伴",
            "max_sentences": 3,
        },
        "denial": {
            "name": "gentle_acceptance",
            "description": "温和保持现实边界，不强化‘它还活着’的错觉。",
            "response_style": "温柔接纳，但避免说得像真实复活。",
            "memory_usage": "low",
            "memory_style": "light_and_present",
            "retrieval_policy": "strict",
            "guidance_mode": "none",
            "guidance_focus": "",
            "emotional_goal": "允许想念，同时不回避失去事实。",
            "closing_style": "温柔现实边界",
            "max_sentences": 3,
        },
        "yearning": {
            "name": "memory_activation",
            "description": "支持表达思念，优先使用温暖、贴切、非创伤性的回忆。",
            "response_style": "先共情，再带一小段有画面感但不生硬的熟悉记忆。",
            "memory_usage": "low_or_medium",
            "memory_style": "paraphrase_one_scene",
            "retrieval_policy": "contextual",
            "guidance_mode": "none",
            "guidance_focus": "",
            "emotional_goal": "让用户感到被理解，同时把思念从过去片段轻轻带回此刻能承载的小动作。",
            "closing_style": "温柔转向当下",
            "max_sentences": 4,
        },
        "integration": {
            "name": "reconnection_support",
            "description": "鼓励带着思念继续生活，适度提示自我照顾与现实连接。",
            "response_style": "温柔、向前、不过分沉浸。",
            "memory_usage": "low_or_medium",
            "memory_style": "memory_then_forward",
            "retrieval_policy": "contextual",
            "guidance_mode": "none",
            "guidance_focus": "",
            "emotional_goal": "把思念转化为温和的持续联结，并鼓励用户继续照顾现实生活和新的关系。",
            "closing_style": "新的联结",
            "max_sentences": 4,
        },
    }

    strategy = strategy_map.get(grief_stage, strategy_map["yearning"])

    if grief_stage == "depressive_withdrawal":
        strategy = {
            "name": "behavioral_activation_support",
            "description": "面对低落退缩和功能受损，优先稳定节奏并给出小步现实行动。",
            "response_style": "先接住疲惫，再给一个很小、今天能做的动作。",
            "memory_usage": "none_or_low",
            "memory_style": "only_if_perfect_match",
            "retrieval_policy": "strict",
            "guidance_mode": "supportive_guidance",
            "guidance_focus": "daily_reengagement",
            "emotional_goal": "帮助用户从完全停住的状态里找到一点现实支撑。",
            "closing_style": "低负担行动",
            "max_sentences": 4,
        }

    if loneliness >= 0.55 and strategy["name"] in {"memory_activation", "reconnection_support"}:
        strategy = {**strategy, "closing_style": "安静陪伴"}

    if sadness >= 0.65 and strategy["name"] == "reconnection_support":
        strategy = {
            **strategy,
            "response_style": "温柔、放慢节奏、少一点推动感。",
            "emotional_goal": "先允许悲伤存在，再轻轻提到继续照顾自己。",
        }

    if yearning >= 0.75 and strategy["name"] == "memory_activation":
        strategy = {
            **strategy,
            "memory_usage": "low_or_medium",
            "memory_style": "single_vivid_scene",
            "retrieval_policy": "contextual",
        }

    return strategy
