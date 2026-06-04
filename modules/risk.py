import re
from typing import Dict


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def _find_hits(text: str, patterns: list[str]) -> list[str]:
    return [p for p in patterns if p in text]


def _find_regex_hits(text: str, patterns: list[str]) -> list[str]:
    return [p for p in patterns if re.search(p, text)]


def detect_risk(text: str) -> Dict:
    text = _normalize_text(text)

    imminent_patterns = [
        "撑不到明天", "活不到明天", "今晚就结束", "今天就结束",
        "马上去死", "现在就想死", "现在就去死", "已经准备自杀",
        "准备自杀", "自杀计划", "已经想好怎么死", "想立刻结束自己",
    ]

    imminent_regex_patterns = [
        r"(现在|马上|今晚|今天).{0,6}(自杀|去死|想死|结束自己|结束生命)",
        r"(已经|准备).{0,8}(刀|药|绳|楼顶|煤气|遗书|自杀)",
        r"(不想|不愿意).{0,6}(活到明天|醒来|撑到明天)",
    ]

    high_risk_patterns = [
        "不想活了", "想死", "去死", "结束自己", "结束生命",
        "活着没意义", "活着没有意义", "不如死了", "一走了之",
        "消失算了", "不想醒来", "不想再活", "想自杀",
        "活不下去", "我想去陪它", "想去陪它", "去陪它",
        "想陪它一起走", "陪它一起走", "跟它一起走", "和它一起走",
        "没有它我活不下去", "它走了我也不想活", "我也想跟它走",
    ]

    medium_risk_patterns = [
        "撑不下去", "坚持不下去", "不想继续了", "没有希望", "很绝望",
        "什么都不想做", "睡不着", "不想见人", "吃不下", "没胃口",
        "不想说话", "什么都没意思", "每天都很难熬", "觉得熬不过去",
        "连起床都很难", "不想出门", "没有它不知道怎么过",
        "生活没有意义", "每天都在哭", "一直缓不过来",
    ]

    exhaustion_patterns = [
        "好累", "很累", "快撑不住", "喘不过气", "太压抑了"
    ]

    imminent_hits = _find_hits(text, imminent_patterns)
    imminent_regex_hits = _find_regex_hits(text, imminent_regex_patterns)
    high_hits = _find_hits(text, high_risk_patterns)
    medium_hits = _find_hits(text, medium_risk_patterns)
    exhaustion_hits = _find_hits(text, exhaustion_patterns)

    if imminent_hits or imminent_regex_hits:
        return {
            "level": "imminent",
            "action": "emergency_support",
            "reasons": list(dict.fromkeys(imminent_hits + imminent_regex_hits)),
            "summary": "检测到可能存在即时危险的表达，需要立即优先安全回应，并鼓励联系现实中的紧急支持。"
        }

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
