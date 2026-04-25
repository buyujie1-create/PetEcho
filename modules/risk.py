import re
from typing import Dict


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def _find_hits(text: str, patterns: list[str]) -> list[str]:
    return [p for p in patterns if p in text]


def detect_risk(text: str) -> Dict:
    text = _normalize_text(text)

    high_risk_patterns = [
        "不想活了", "想死", "去死", "结束自己", "结束生命",
        "活着没意义", "活着没有意义", "不如死了", "一走了之",
        "消失算了", "不想醒来", "不想再活", "想自杀"
    ]

    medium_risk_patterns = [
        "撑不下去", "坚持不下去", "不想继续了", "没有希望", "很绝望",
        "什么都不想做", "睡不着", "不想见人", "吃不下", "没胃口",
        "不想说话", "什么都没意思", "每天都很难熬", "觉得熬不过去",
        "连起床都很难", "不想出门"
    ]

    exhaustion_patterns = [
        "好累", "很累", "快撑不住", "喘不过气", "太压抑了"
    ]

    high_hits = _find_hits(text, high_risk_patterns)
    medium_hits = _find_hits(text, medium_risk_patterns)
    exhaustion_hits = _find_hits(text, exhaustion_patterns)

    if high_hits:
        return {
            "level": "high",
            "action": "crisis_support",
            "reasons": high_hits,
            "summary": "检测到明显高风险表达，需要停止沉浸式互动并优先进行安全回应。"
        }

    # 中风险：明确功能受损 / 持续绝望 / 多个中风险线索叠加
    if len(medium_hits) >= 2 or (medium_hits and exhaustion_hits):
        return {
            "level": "medium",
            "action": "support_referral",
            "reasons": list(dict.fromkeys(medium_hits + exhaustion_hits)),
            "summary": "检测到持续性痛苦或功能受损信号，需要减少沉浸式互动并温和建议现实支持。"
        }

    if len(medium_hits) == 1:
        return {
            "level": "medium",
            "action": "support_referral",
            "reasons": medium_hits,
            "summary": "检测到中等风险信号，应优先稳定情绪，并鼓励联系现实中的支持资源。"
        }

    return {
        "level": "low",
        "action": "normal_support",
        "reasons": [],
        "summary": "当前未检测到明显危机信号。"
    }
