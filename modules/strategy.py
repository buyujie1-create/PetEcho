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

    advice_request = _is_advice_request(user_input)

    if risk["level"] == "high":
        return {
            "name": "crisis_safety",
            "description": "停止沉浸式互动，优先安全回应、现实支持与求助建议。",
            "response_style": "短句、稳定、温和，不扮演过度亲密角色。",
            "memory_usage": "none",
            "memory_style": "forbid",
            "guidance_mode": "crisis_referral",
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
            "guidance_mode": "supportive_guidance" if advice_request else "stabilize_only",
            "guidance_focus": "low_burden_coping",
            "emotional_goal": "降低压迫感，给出一点现实支撑感。",
            "closing_style": "轻度现实支持",
            "max_sentences": 4 if advice_request else 3,
        }
        return base

    # 用户明确求建议时，优先切到“建议型支持”
    if advice_request:
        if grief_stage in {"shock", "denial"}:
            return {
                "name": "coping_guidance",
                "description": "在早期哀伤中提供低负担、现实可行的建议，不急着推动接受。",
                "response_style": "先接情绪，再给一两个非常小的可执行建议。",
                "memory_usage": "low",
                "memory_style": "only_if_perfect_match",
                "guidance_mode": "coping_guidance",
                "guidance_focus": "low_burden_coping",
                "emotional_goal": "让用户知道现在不必一下子走出来，只要先照顾好此刻。",
                "closing_style": "轻柔建议收束",
                "max_sentences": 5,
            }
        if grief_stage == "yearning":
            return {
                "name": "coping_guidance",
                "description": "在强烈思念期提供兼顾想念与日常恢复的小建议。",
                "response_style": "先共情，再给出温和具体的小办法，不只停留在陪伴。",
                "memory_usage": "low_or_medium",
                "memory_style": "memory_then_forward",
                "guidance_mode": "coping_guidance",
                "guidance_focus": "meaning_and_memory",
                "emotional_goal": "帮助用户在想念和日常之间找到一点可移动的空间。",
                "closing_style": "带一点方向感的收束",
                "max_sentences": 5,
            }
        if grief_stage == "integration":
            return {
                "name": "reconnection_guidance",
                "description": "帮助用户把思念整合进生活，并提供温和的继续生活建议。",
                "response_style": "先理解，再给出现实、轻量、能落地的生活重建建议。",
                "memory_usage": "low_or_medium",
                "memory_style": "memory_then_forward",
                "guidance_mode": "coping_guidance",
                "guidance_focus": "daily_reengagement",
                "emotional_goal": "让用户感到继续生活不是背叛回忆，而是带着回忆往前。",
                "closing_style": "带着回忆继续生活",
                "max_sentences": 5,
            }

    strategy_map = {
        "shock": {
            "name": "stabilization",
            "description": "提供情绪稳定支持，不强行推动接受现实。",
            "response_style": "短句、轻柔、以陪伴和允许感受为主。",
            "memory_usage": "none_or_low",
            "memory_style": "only_if_perfect_match",
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
            "memory_usage": "medium",
            "memory_style": "paraphrase_one_scene",
            "guidance_mode": "none",
            "guidance_focus": "",
            "emotional_goal": "让用户感到被理解，而不是被记忆条目淹没。",
            "closing_style": "温柔收束",
            "max_sentences": 4,
        },
        "integration": {
            "name": "reconnection_support",
            "description": "鼓励带着思念继续生活，适度提示自我照顾与现实连接。",
            "response_style": "温柔、向前、不过分沉浸。",
            "memory_usage": "low_or_medium",
            "memory_style": "memory_then_forward",
            "guidance_mode": "none",
            "guidance_focus": "",
            "emotional_goal": "把思念转化为温和的持续联结，而非停留在痛苦里。",
            "closing_style": "带着回忆继续生活",
            "max_sentences": 4,
        },
    }

    strategy = strategy_map.get(grief_stage, strategy_map["yearning"])

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
            "memory_usage": "medium",
            "memory_style": "single_vivid_scene",
        }

    return strategy