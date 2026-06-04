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
    guilt = emotion_result.get("guilt", 0.0)
    anger = emotion_result.get("anger", 0.0)
    numbness = emotion_result.get("numbness", 0.0)

    scores = {
        "acute_grief": 0.0,
        "shock": 0.0,
        "denial": 0.0,
        "yearning": 0.0,
        "guilt": 0.0,
        "anger": 0.0,
        "numbness": 0.0,
        "depressive_withdrawal": 0.0,
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
    acute_grief_keywords = [
        "难过", "伤心", "痛苦", "心里很沉", "哭", "低落", "很疼",
        "不想说话", "说不出来", "缓不过来"
    ]
    integration_keywords = [
        "慢慢接受", "开始适应", "重新开始", "恢复生活", "接受事实",
        "我会好好的", "慢慢走出来", "想照顾好自己", "开始好一点了",
        "带着它的回忆继续", "继续生活", "慢慢往前"
    ]
    guilt_keywords = [
        "愧疚", "内疚", "自责", "后悔", "都是我的错", "是我的错",
        "我没有照顾好", "早知道", "对不起", "如果我"
    ]
    anger_keywords = [
        "不公平", "为什么偏偏", "为什么是它", "生气", "愤怒",
        "怨", "恨", "太残忍", "接受不了"
    ]
    numbness_keywords = [
        "麻木", "不真实", "像假的", "像做梦", "空白", "哭不出来", "没有反应"
    ]
    depressive_keywords = [
        "什么都不想做", "提不起劲", "没有力气", "睡不着", "吃不下",
        "不想见人", "不想出门", "每天都很难熬", "生活没有意义"
    ]

    scores["acute_grief"] += _score_hits(text, acute_grief_keywords, 0.24)
    scores["shock"] += _score_hits(text, shock_keywords, 0.38)
    scores["denial"] += _score_hits(text, denial_keywords, 0.40)
    scores["yearning"] += _score_hits(text, yearning_keywords, 0.18)
    scores["integration"] += _score_hits(text, integration_keywords, 0.36)
    scores["guilt"] += _score_hits(text, guilt_keywords, 0.34)
    scores["anger"] += _score_hits(text, anger_keywords, 0.34)
    scores["numbness"] += _score_hits(text, numbness_keywords, 0.34)
    scores["depressive_withdrawal"] += _score_hits(text, depressive_keywords, 0.28)

    # 情绪驱动
    scores["yearning"] += yearning * 1.00 + sadness * 0.28 + loneliness * 0.18
    scores["acute_grief"] += sadness * 0.90 + loneliness * 0.10
    scores["guilt"] += guilt * 1.05 + sadness * 0.12
    scores["anger"] += anger * 1.05
    scores["numbness"] += numbness * 1.05
    scores["depressive_withdrawal"] += sadness * 0.20 + loneliness * 0.22
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

    if re.search(r"(我).{0,8}(没有照顾好|害了|耽误了|没救到|没保护好)", text):
        scores["guilt"] += 0.28

    if re.search(r"(为什么|凭什么).{0,8}(它|这样|这么快|离开)", text):
        scores["anger"] += 0.24

    if re.search(r"(像|好像).{0,4}(做梦|假的|不是真的)", text):
        scores["numbness"] += 0.24

    # 判定优先级：即时的强烈体验优先；阶段仅用于策略，不作为诊断展示。
    if scores["shock"] >= 0.82:
        return "shock"
    if scores["denial"] >= 0.82 and scores["shock"] < 0.82:
        return "denial"
    if scores["guilt"] >= 0.72:
        return "guilt"
    if scores["anger"] >= 0.72:
        return "anger"
    if scores["numbness"] >= 0.72:
        return "numbness"
    if scores["depressive_withdrawal"] >= 0.72:
        return "depressive_withdrawal"
    if (
        scores["acute_grief"] >= 0.34
        and scores["yearning"] < 0.55
        and not _contains_any(text, yearning_keywords)
    ):
        return "acute_grief"

    # 如果明确整合，但思念仍强，优先视为 integration，避免系统总停留在 yearning
    if scores["integration"] >= 0.72 and _contains_any(text, ["慢慢接受", "开始适应", "继续生活", "往前走"]):
        return "integration"

    return max(scores, key=scores.get)
