import re
from typing import Dict


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def _hit_score(text: str, weighted_terms: dict[str, float]) -> float:
    score = 0.0
    for term, weight in weighted_terms.items():
        if term in text:
            score += weight
    return score


def _regex_score(text: str, weighted_patterns: list[tuple[str, float]]) -> float:
    score = 0.0
    for pattern, weight in weighted_patterns:
        if re.search(pattern, text):
            score += weight
    return score


def _clamp(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 2)


def detect_emotion(text: str) -> Dict[str, float]:
    text = _normalize_text(text)

    if not text:
        return {"sadness": 0.0, "loneliness": 0.0, "yearning": 0.0}

    sadness_terms = {
        "难过": 0.30,
        "伤心": 0.34,
        "舍不得": 0.40,
        "好难受": 0.42,
        "痛苦": 0.48,
        "崩溃": 0.58,
        "想哭": 0.32,
        "哭": 0.18,
        "失落": 0.28,
        "低落": 0.28,
        "心里空": 0.34,
        "心里很空": 0.42,
        "空空的": 0.28,
        "不在了": 0.26,
        "离开了": 0.24,
        "再也": 0.18,
        "受不了": 0.24,
        "压得喘不过气": 0.46,
        "提不起劲": 0.28,
        "没有力气": 0.28,
        "很累": 0.18,
    }

    loneliness_terms = {
        "孤独": 0.50,
        "一个人": 0.30,
        "空落落": 0.52,
        "空荡荡": 0.42,
        "没人陪": 0.58,
        "冷清": 0.30,
        "安静得可怕": 0.42,
        "没有陪我": 0.36,
        "没人在": 0.24,
        "回家很空": 0.34,
        "好安静": 0.24,
        "家里太安静": 0.40,
    }

    yearning_terms = {
        "想你": 0.62,
        "很想你": 0.78,
        "好想你": 0.82,
        "我想你": 0.72,
        "想它": 0.70,
        "很想它": 0.80,
        "好想它": 0.82,
        "想念": 0.58,
        "怀念": 0.50,
        "惦记": 0.34,
        "想起": 0.20,
        "记得": 0.14,
        "以前": 0.16,
        "那时候": 0.18,
        "回忆": 0.18,
        "又想起": 0.26,
        "总会想起": 0.30,
    }

    sadness = _hit_score(text, sadness_terms)
    loneliness = _hit_score(text, loneliness_terms)
    yearning = _hit_score(text, yearning_terms)

    sadness += _regex_score(text, [
        (r"(再也|已经).{0,4}(见不到|摸不到|听不到|不能)", 0.28),
        (r"(睡不着|吃不下|没胃口)", 0.16),
        (r"(怎么会|为什么会这样)", 0.12),
    ])

    loneliness += _regex_score(text, [
        (r"(没有|没).{0,4}(陪|在身边)", 0.26),
        (r"(回家|晚上|一个人).{0,6}(很空|好空|空落落|冷清)", 0.24),
    ])

    yearning += _regex_score(text, [
        (r"(想|想念|怀念).{0,2}(你|它)", 0.28),
        (r"(还记得|记不记得|以前|那时候)", 0.10),
        (r"(如果你还在|要是你还在)", 0.16),
    ])

    # 强度词修正
    intensity_terms = ["很", "特别", "真的", "一直", "总是", "太", "非常", "一下子"]
    if any(term in text for term in intensity_terms):
        if sadness > 0:
            sadness += 0.05
        if loneliness > 0:
            loneliness += 0.05
        if yearning > 0:
            yearning += 0.06

    # 联动关系：思念往往伴随一定悲伤；强孤单也会带一点悲伤
    if yearning >= 0.55 and sadness < 0.24:
        sadness += 0.14
    if loneliness >= 0.50 and sadness < 0.24:
        sadness += 0.10

    # 如果用户主要在叙事回忆，不一定非常痛，但会有思念
    if yearning == 0 and re.search(r"(记得|以前|那时候|总会|每次)", text):
        yearning += 0.18

    return {
        "sadness": _clamp(sadness),
        "loneliness": _clamp(loneliness),
        "yearning": _clamp(yearning),
    }