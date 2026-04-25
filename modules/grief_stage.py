import re
from typing import Dict


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def _score_hits(text: str, keywords: list[str], base: float) -> float:
    return sum(base for k in keywords if k in text)


def detect_grief_stage(text: str, emotion_result: Dict[str, float]) -> str:
    text = _normalize_text(text)

    sadness = emotion_result.get("sadness", 0.0)
    loneliness = emotion_result.get("loneliness", 0.0)
    yearning = emotion_result.get("yearning", 0.0)

    scores = {
        "shock": 0.0,
        "denial": 0.0,
        "yearning": 0.0,
        "integration": 0.0,
    }

    shock_keywords = [
        "不敢相信", "太突然", "怎么会", "无法接受", "不可能",
        "没反应过来", "像做梦", "是真的吗", "一下子", "还缓不过来"
    ]
    denial_keywords = [
        "还在", "没离开", "会回来", "像还在一样", "总觉得它还在",
        "感觉它还会出现", "好像只是出门了", "还会听到", "还是会下意识找它"
    ]
    yearning_keywords = [
        "想你", "想它", "怀念", "想念", "记得", "以前", "回忆",
        "又想起", "很想", "舍不得", "总会想起"
    ]
    integration_keywords = [
        "慢慢接受", "开始适应", "重新开始", "恢复生活", "接受事实",
        "我会好好的", "慢慢走出来", "想照顾好自己", "开始好一点了",
        "带着它的回忆继续", "继续生活", "慢慢往前"
    ]

    scores["shock"] += _score_hits(text, shock_keywords, 0.38)
    scores["denial"] += _score_hits(text, denial_keywords, 0.40)
    scores["yearning"] += _score_hits(text, yearning_keywords, 0.18)
    scores["integration"] += _score_hits(text, integration_keywords, 0.36)

    # 情绪驱动
    scores["yearning"] += yearning * 1.00 + sadness * 0.28 + loneliness * 0.18
    scores["shock"] += sadness * 0.14 if _contains_any(text, ["突然", "不敢相信", "怎么会"]) else 0.0
    scores["denial"] += 0.18 if _contains_any(text, ["还在", "会回来", "没离开", "像还在一样"]) else 0.0
    scores["integration"] += 0.22 if _contains_any(text, ["照顾自己", "继续生活", "会慢慢好", "往前走"]) else 0.0

    # 语义修正
    if re.search(r"(如果|要是).{0,4}(你|它).{0,4}还在", text):
        scores["yearning"] += 0.24

    if re.search(r"(好像|总觉得).{0,5}(还在|会回来|没走)", text):
        scores["denial"] += 0.26

    if re.search(r"(开始|慢慢|努力).{0,6}(适应|接受|往前|恢复)", text):
        scores["integration"] += 0.28

    if re.search(r"(家里|回家|晚上).{0,6}(好安静|空落落|空荡荡)", text):
        scores["yearning"] += 0.16

    # 判定优先级：强震惊 / 强否认优先，否则在 yearning 与 integration 中比较
    if scores["shock"] >= 0.82:
        return "shock"
    if scores["denial"] >= 0.82 and scores["shock"] < 0.82:
        return "denial"

    # 如果明确整合，但思念仍强，优先视为 integration，避免系统总停留在 yearning
    if scores["integration"] >= 0.72 and _contains_any(text, ["慢慢接受", "开始适应", "继续生活", "往前走"]):
        return "integration"

    return max(scores, key=scores.get)