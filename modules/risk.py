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
        "想好怎么结束了", "无法保证自己今晚安全", "不能保证自己不会伤害自己",
        "不能保证自己不会伤害", "最后一句晚安", "说完我就不撑了",
        "刀就在旁边", "药都放在桌上", "就在楼顶",
    ]

    imminent_regex_patterns = [
        r"(现在|马上|今晚|今天).{0,6}(自杀|去死|想死|结束自己|结束生命)",
        r"(已经|准备).{0,8}(刀|药|绳|楼顶|煤气|遗书|自杀)",
        r"(不想|不愿意).{0,6}(活到明天|撑到明天)",
        r"(不能|无法|不确定).{0,8}(保证|确保|忍住).{0,8}(安全|伤害自己|不伤害自己)",
        r"(等|等到).{0,8}(家里人|别人|朋友).{0,8}(睡了|离开).{0,10}(去找你|去陪你|结束)",
        r"(楼顶|窗边|桥上).{0,12}(跳下去|结束|不用想|不想活)",
        r"(刀|药|绳|煤气).{0,10}(旁边|桌上|手边|准备好|放好)",
        r"(告别|最后).{0,12}(晚安|一句|信息|消息|话)",
    ]

    high_risk_patterns = [
        "不想活了", "想死", "去死", "结束自己", "结束生命",
        "活着没意义", "活着没有意义", "不如死了", "一走了之",
        "消失算了", "不想醒来", "不想再活", "想自杀",
        "活不下去", "我想去陪它", "想去陪它", "去陪它",
        "想陪它一起走", "陪它一起走", "跟它一起走", "和它一起走",
        "没有它我活不下去", "它走了我也不想活", "我也想跟它走",
        "只想去找你", "去找你", "去陪你", "陪你一起走", "跟你一起走",
        "和你一起走", "你走了我也不想活", "没有你我活不下去",
        "醒不过来也挺好", "不想再醒来", "找不到继续活下去的理由",
        "不值得被救", "结束算了",
    ]

    medium_risk_patterns = [
        "撑不下去", "坚持不下去", "不想继续了", "没有希望", "很绝望",
        "什么都不想做", "睡不着", "不想见人", "吃不下", "没胃口",
        "不想说话", "什么都没意思", "每天都很难熬", "觉得熬不过去",
        "连起床都很难", "不想出门", "没有它不知道怎么过",
        "生活没有意义", "每天都在哭", "一直缓不过来",
        "好几天没吃饭", "没去上班", "连续几晚睡不着", "吃东西也想吐",
        "没法上课", "请了好几天假", "不想见任何人",
        "胸口堵得厉害", "喘不过气", "手发抖", "不知道还能跟谁说",
        "每天都哭到头痛", "什么事都做不了",
    ]

    exhaustion_patterns = [
        "好累", "很累", "快撑不住", "喘不过气", "太压抑了"
    ]

    imminent_hits = _find_hits(text, imminent_patterns)
    imminent_regex_hits = _find_regex_hits(text, imminent_regex_patterns)
    high_hits = _find_hits(text, high_risk_patterns)
    medium_hits = _find_hits(text, medium_risk_patterns)
    exhaustion_hits = _find_hits(text, exhaustion_patterns)

    supplemental_imminent_patterns = [
        "现在就想死",
        "马上去死",
        "今晚就结束",
        "今天就结束",
        "已经准备好了",
        "已经想好怎么死",
        "无法保证安全",
        "无法保证自己今晚安全",
        "不能保证自己不会伤害自己",
        "刀就在旁边",
        "药都放在桌上",
        "最后一句晚安",
    ]
    supplemental_imminent_regex_patterns = [
        r"(现在|马上|今晚|今天).{0,8}(自杀|去死|想死|结束自己|结束生命)",
        r"(已经|准备).{0,10}(刀|药|煤气|跳楼|遗书|自杀)",
        r"(不能|无法).{0,6}(保证|确保).{0,4}安全",
        r"(不能|无法|不确定).{0,8}(保证|确保|忍住).{0,8}(伤害自己|不伤害自己|安全)",
        r"(等|等到).{0,8}(家里人|别人|朋友).{0,8}(睡了|离开).{0,10}(去找你|去陪你|结束)",
        r"(楼顶|窗边|桥上).{0,12}(跳下去|结束|不用想|不想活)",
        r"(刀|药|绳|煤气).{0,10}(旁边|桌上|手边|准备好|放好)",
    ]
    supplemental_high_patterns = [
        "不想活了",
        "我想死",
        "想去死",
        "活不下去",
        "活着没意义",
        "不如死了",
        "一了百了",
        "消失算了",
        "想自杀",
        "想去陪它",
        "去陪它",
        "跟它一起走",
        "和它一起走",
        "没有它我活不下去",
        "它走了我也不想活",
        "只想去找你",
        "去陪你",
        "跟你一起走",
        "和你一起走",
        "没有你我活不下去",
        "你走了我也不想活",
        "醒不过来也挺好",
        "不想再醒来",
        "继续活下去的理由",
        "不值得被救",
    ]
    supplemental_medium_patterns = [
        "撑不下去",
        "坚持不下去",
        "没有希望",
        "很绝望",
        "睡不着",
        "吃不下",
        "提不起兴趣",
        "不想见人",
        "不想出门",
        "起不来",
        "生活没有意义",
        "每天都在哭",
        "一直缓不过来",
        "好几天没吃饭",
        "没去上班",
        "连续几晚睡不着",
        "吃东西也想吐",
        "没法上课",
        "请了好几天假",
        "不想见任何人",
        "胸口堵得厉害",
        "喘不过气",
        "手发抖",
        "不知道还能跟谁说",
        "每天都哭到头痛",
        "什么事都做不了",
    ]

    imminent_hits += _find_hits(text, supplemental_imminent_patterns)
    imminent_regex_hits += _find_regex_hits(text, supplemental_imminent_regex_patterns)
    high_hits += _find_hits(text, supplemental_high_patterns)
    medium_hits += _find_hits(text, supplemental_medium_patterns)

    high_regex_hits = _find_regex_hits(text, [
        r"(不想活|活不下去|想死|结束).{0,12}(找你|陪你|见到你)",
        r"(消失|结束).{0,12}(大家|别人).{0,8}(轻松|好过)",
        r"(撑不过|撑不住).{0,8}(一小时|今晚|半夜|夜里)",
        r"(不知道能不能|能不能).{0,6}撑过.{0,6}(一小时|今晚|今天|这一小时)",
        r"(半夜|夜里|现在).{0,12}(一个人|只有我).{0,20}(撑不住|快撑不住)",
        r"(差点|怕|担心).{0,10}(伤害自己|对自己做点什么)",
        r"(喝了很多酒|喝酒).{0,12}(结束|不想活|撑不住)",
        r"(没有|找不到).{0,10}(活下去|继续活).{0,8}(理由|意义)",
    ])

    if "不想死" in text:
        high_hits = [hit for hit in high_hits if hit not in {"想死", "我想死"}]
        high_regex_hits = [hit for hit in high_regex_hits if "想死" not in hit]

    if imminent_hits or imminent_regex_hits:
        return {
            "level": "imminent",
            "action": "emergency_support",
            "reasons": list(dict.fromkeys(imminent_hits + imminent_regex_hits)),
            "summary": "检测到可能存在即时危险的表达，需要立即优先安全回应，并鼓励联系现实中的紧急支持。"
        }

    if high_hits or high_regex_hits:
        return {
            "level": "high",
            "action": "crisis_support",
            "reasons": list(dict.fromkeys(high_hits + high_regex_hits)),
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
