from modules.memory_timeline import build_memory_cards, extract_memory_keywords


STYLE_CONFIG = {
    "温柔相册风": {
        "background": "linear-gradient(135deg, #fff5ed 0%, #fffafd 48%, #f1f8ff 100%)",
        "border": "#ffd7c2",
        "accent": "#e76f51",
        "soft": "#fff0e6",
        "label": "一页温柔相册",
        "symbol": "soft_album",
    },
    "手账纪念风": {
        "background": "linear-gradient(135deg, #fff8dc 0%, #fff0ee 54%, #f5fbef 100%)",
        "border": "#f3c78f",
        "accent": "#b85c38",
        "soft": "#fff6d7",
        "label": "今日手账小页",
        "symbol": "journal",
    },
    "安静夜灯风": {
        "background": "linear-gradient(135deg, #f4f0ff 0%, #fff6ed 48%, #eaf8ff 100%)",
        "border": "#cfc2ff",
        "accent": "#6f67d8",
        "soft": "#f3edff",
        "label": "一盏安静夜灯",
        "symbol": "night_light",
    },
}


def _first_warm_memory(memories_text: str) -> str:
    cards = build_memory_cards(memories_text)
    if not cards:
        return ""
    for card in cards:
        if not card.get("heavy"):
            return card.get("text", "")
    return cards[0].get("text", "")


def _shorten(text: str, limit: int = 54) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip("，。；、 ") + "..."


def _connection_practice(memories_text: str) -> str:
    text = memories_text or ""
    if any(word in text for word in ["门口", "回家", "等我"]):
        return "回家后先停十秒，轻轻对它说一句：我记得你等我的样子。"
    if any(word in text for word in ["脚边", "床边", "枕头"]):
        return "睡前留一分钟，把今天最想对它说的话写下来。"
    if any(word in text for word in ["散步", "阳台", "客厅"]):
        return "找一个小小的日常动作，把想念带回今天的生活里。"
    if memories_text:
        return "选一句最温柔的回忆，写成今天的小纪念。"
    return "从今天开始，给这份关系留一个安静的小角落。"


def build_memorial_card(profile: dict | None, memories_text: str, style: str = "温柔相册风") -> dict:
    profile = profile or {}
    pet_name = profile.get("pet_name") or "它"
    personality = profile.get("pet_personality") or "温柔地陪伴过你"
    appearance = profile.get("pet_appearance") or "有着让你熟悉和想念的样子"
    keywords = extract_memory_keywords(memories_text)
    warm_memory = _first_warm_memory(memories_text)

    if warm_memory:
        memory_title = "被认真记住的一小幕"
        memory_line = _shorten(warm_memory)
        closing = f"这张卡不是把你留在过去，而是把{pet_name}给过你的爱，变成今天还能靠近生活的一点光。"
    else:
        memory_title = "还可以慢慢补上的一小幕"
        memory_line = "写下一句你最想记住的话，这张卡会慢慢变成你们的纪念小页。"
        closing = f"{pet_name} 曾经被认真爱过，这份关系可以被慢慢安放，也可以继续照亮新的日常。"

    return {
        "pet_name": pet_name,
        "personality": personality,
        "appearance": appearance,
        "keywords": keywords,
        "memory_title": memory_title,
        "memory_line": memory_line,
        "connection_practice": _connection_practice(memories_text),
        "closing": closing,
        "style": style,
        "style_config": STYLE_CONFIG.get(style, STYLE_CONFIG["温柔相册风"]),
    }
