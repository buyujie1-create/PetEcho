import re


HEAVY_TERMS = ["去世", "离开", "最后", "医院", "生病", "抢救", "走丢", "不在了", "再也"]

CATEGORY_RULES = [
    ("初见与相遇", ["第一次", "初见", "遇见", "带回家", "领养", "刚来"]),
    ("日常陪伴", ["每天", "回家", "门口", "睡", "起床", "散步", "陪", "一起", "脚边", "床边", "蹭"]),
    ("难忘瞬间", ["最喜欢", "最难忘", "记得", "那次", "有一次", "总会想起"]),
    ("想念表达", ["想你", "想它", "想念", "怀念", "舍不得"]),
    ("沉重记忆", HEAVY_TERMS),
    ("意义与感谢", ["谢谢", "感谢", "教会", "留下", "意义", "会好好的", "继续生活"]),
]


def split_memories(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        parts = [text]

    cards = []
    current = ""
    for part in parts:
        if len(current) + len(part) <= 90:
            current += part
        else:
            if current:
                cards.append(current.strip())
            current = part
    if current:
        cards.append(current.strip())

    unique = []
    seen = set()
    for item in cards:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def classify_memory(text: str) -> str:
    for label, terms in CATEGORY_RULES:
        if any(term in text for term in terms):
            return label
    return "温柔片段"


def is_heavy_memory(text: str) -> bool:
    return any(term in text for term in HEAVY_TERMS)


def build_memory_cards(memories_text: str) -> list[dict]:
    cards = []
    for idx, item in enumerate(split_memories(memories_text), start=1):
        category = classify_memory(item)
        heavy = is_heavy_memory(item)
        cards.append({
            "id": idx,
            "title": f"记忆 {idx}",
            "category": category,
            "text": item,
            "tone": "谨慎调用" if heavy else "适合温和调用",
            "heavy": heavy,
        })
    return cards


def extract_memory_keywords(memories_text: str, limit: int = 6) -> list[str]:
    text = memories_text or ""
    candidates = [
        "陪伴", "回家", "门口", "起床", "散步", "脚边", "安静", "撒娇",
        "想念", "感谢", "温柔", "黏人", "下雨天", "阳台", "客厅", "床边",
    ]
    result = [word for word in candidates if word in text]
    if not result and text:
        result = re.findall(r"[\u4e00-\u9fff]{2,4}", text)[:limit]
    return result[:limit]
