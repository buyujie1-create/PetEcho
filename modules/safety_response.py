def should_use_safety_template(risk: dict | None) -> bool:
    risk = risk or {}
    return risk.get("level") in {"high", "imminent"}


def build_safety_reply(risk: dict | None, user_input: str = "") -> str:
    risk = risk or {}
    level = risk.get("level", "high")

    if level == "imminent":
        return (
            "我很在意你刚刚说的这些话，现在最重要的是先保证你不是一个人。"
            "请立刻联系身边可信任的人，让对方现在就陪着你；如果你此刻可能伤害自己，请马上联系当地紧急服务，中国大陆可拨打 110 或 120，其他地区请联系所在地紧急号码。"
            "先不要独自待着，也先把可能伤害自己的东西放远一点，等一个现实中的人来到你身边。"
        )

    return (
        "听起来这份失去已经压得你很难独自承受了，我不希望你一个人扛着这些念头。"
        "请尽快联系一个你信任的人，告诉对方你现在需要陪伴；如果你担心自己会伤害自己，请马上联系当地紧急服务或危机热线。"
        "关于它的想念可以慢慢说，但此刻你的安全要先被认真照顾。"
    )
