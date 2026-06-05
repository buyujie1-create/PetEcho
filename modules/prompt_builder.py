import re
from typing import List

HEAVY_MEMORY_KEYWORDS = [
    "去世", "离开", "最后一次", "医院", "生病", "抢救",
    "掉下", "走丢", "找了一夜", "不在了", "再也"
]

WARM_MEMORY_HINTS = [
    "陪", "一起", "门口", "枕头边", "脚边", "回家", "起床", "散步",
    "阳台", "客厅", "睡", "蹭", "等我", "每天", "傍晚", "下雨天",
    "靠着", "窝着", "尾巴", "床边", "沙发", "安静"
]

AVOID_PHRASES = [
    "我一直都在", "别难过了", "你要坚强", "你要加油", "抱抱你",
    "你只需要我", "我永远都不会离开你", "我会一直陪着你", "不要哭了"
]

ADVICE_PATTERNS = [
    "怎么办", "怎么做", "有什么办法", "有什么方法", "怎么走出来",
    "如何走出来", "给我一些建议", "给点建议", "我该怎么", "能做什么",
    "怎么调整", "怎么缓解", "怎么改善", "有没有什么建议", "怎么才能"
]

MEMORY_FIT_HINTS = [
    "想起", "记得", "以前", "那时候", "回忆", "照片", "视频", "样子",
    "习惯", "喜欢", "总是", "每次", "一起", "陪我", "等我", "门口",
    "枕头边", "脚边", "散步", "阳台", "客厅", "下雨天", "蹭"
]


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def _is_heavy_memory(memory: str) -> bool:
    return any(k in memory for k in HEAVY_MEMORY_KEYWORDS)


def _user_wants_heavy_memory(user_text: str) -> bool:
    user_text = _normalize_text(user_text)
    cues = [
        "最后", "那天", "走丢", "医院", "生病", "离开", "去世",
        "不在了", "怎么走的", "当时", "后来发生了什么"
    ]
    return any(c in user_text for c in cues)


def _memory_fits_user_input(memory: str, user_input: str) -> bool:
    memory = _normalize_text(memory)
    user_input = _normalize_text(user_input)
    if not memory or not user_input:
        return False
    if _user_wants_heavy_memory(user_input) and _is_heavy_memory(memory):
        return True

    shared_hints = [hint for hint in MEMORY_FIT_HINTS if hint in user_input and hint in memory]
    if shared_hints:
        return True

    user_terms = re.findall(r"[\u4e00-\u9fff]{2,6}", user_input)
    stop_terms = {"今天", "现在", "真的", "感觉", "有点", "还是", "一直", "好像", "什么"}
    meaningful_terms = [term for term in user_terms if term not in stop_terms]
    return any(term in memory for term in meaningful_terms[:8])


def _detect_user_need(user_input: str) -> str:
    text = _normalize_text(user_input)

    if any(k in text for k in ["怎么办", "怎么做", "有什么办法", "有什么方法", "怎么走出来", "给我一些建议", "我该怎么", "怎么缓解", "怎么调整"]):
        return "advice_request"
    if any(k in text for k in ["想你", "很想", "好想", "想念", "怀念"]):
        return "yearning_expression"
    if any(k in text for k in ["空落落", "空荡荡", "好安静", "没人陪", "一个人"]):
        return "loneliness_expression"
    if any(k in text for k in ["为什么", "怎么会", "不敢相信", "无法接受"]):
        return "meaning_confusion"
    if any(k in text for k in ["我是不是", "是不是我", "后悔", "早知道"]):
        return "guilt_reflection"
    if any(k in text for k in ["我会好好的", "慢慢接受", "开始适应", "继续生活", "重新开始"]):
        return "integration_reflection"
    return "open_support"


def _select_best_clause(memory: str, user_input: str = "") -> str:
    parts = re.split(r'[，。；！？!?]', memory)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return memory.strip()

    user_input = _normalize_text(user_input)
    shared_hints = [hint for hint in MEMORY_FIT_HINTS if hint in user_input]

    ranked = []
    for part in parts:
        score = 0
        for hint in shared_hints:
            if hint in part:
                score += 4
        if any(k in part for k in WARM_MEMORY_HINTS):
            score += 2
        if len(part) >= 8:
            score += 1
        if _is_heavy_memory(part):
            score -= 2
        ranked.append((score, part))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def _prepare_memories(memory_context: list, user_input: str, strategy: dict) -> List[str]:
    if not memory_context:
        return []

    usage = strategy.get("memory_usage", "low")
    memory_style = strategy.get("memory_style", "paraphrase_one_scene")
    retrieval_policy = strategy.get("retrieval_policy", "contextual")

    if usage == "none" or memory_style == "forbid" or retrieval_policy == "off":
        return []

    unique = []
    seen = set()
    for memory in memory_context:
        memory = (memory or "").strip()
        if not memory or memory in seen:
            continue
        seen.add(memory)
        unique.append(memory)

    if not _user_wants_heavy_memory(user_input):
        lighter = [m for m in unique if not _is_heavy_memory(m)]
        if lighter:
            unique = lighter

    if retrieval_policy == "strict" or memory_style == "only_if_perfect_match":
        unique = [memory for memory in unique if _memory_fits_user_input(memory, user_input)]

    max_count = 1

    prepared = []
    for memory in unique:
        clause = _select_best_clause(memory, user_input)
        if retrieval_policy == "strict" and not _memory_fits_user_input(clause, user_input):
            continue
        if len(clause) > 38:
            clause = clause[:38] + "…"
        prepared.append(clause)
        if len(prepared) >= max_count:
            break

    return prepared


def _emotion_summary(emotion: dict) -> str:
    sadness = emotion.get("sadness", 0)
    loneliness = emotion.get("loneliness", 0)
    yearning = emotion.get("yearning", 0)
    guilt = emotion.get("guilt", 0)
    anger = emotion.get("anger", 0)
    numbness = emotion.get("numbness", 0)

    labels = []
    if yearning >= 0.7:
        labels.append("思念很强")
    elif yearning >= 0.35:
        labels.append("有明显想念")

    if sadness >= 0.65:
        labels.append("悲伤比较重")
    elif sadness >= 0.3:
        labels.append("有些难过")

    if loneliness >= 0.55:
        labels.append("孤单感明显")
    elif loneliness >= 0.25:
        labels.append("有一点空落")

    if guilt >= 0.55:
        labels.append("愧疚自责明显")
    elif guilt >= 0.25:
        labels.append("有一些遗憾和自责")

    if anger >= 0.55:
        labels.append("不公平感或愤怒明显")
    elif anger >= 0.25:
        labels.append("有一些不甘")

    if numbness >= 0.55:
        labels.append("麻木或不真实感明显")
    elif numbness >= 0.25:
        labels.append("有一点麻木感")

    if not labels:
        labels.append("需要温和陪伴")

    return "、".join(labels)


def _suggestion_block(user_need: str, grief_stage: str, strategy: dict) -> str:
    if user_need != "advice_request" and strategy.get("guidance_mode") != "coping_guidance":
        return ""

    guidance_focus = strategy.get("guidance_focus", "low_burden_coping")

    theory_hint = """
【建议型支持原则】
当前用户在明确寻求“办法”或“建议”，所以不能只提供情绪价值。
请遵循：
1. 先承认悲伤是真实的，不要一上来就教用户振作；
2. 再给出1到3个低负担、现实可执行的建议；
3. 建议尽量具体，今天就能做一点点；
4. 可以轻量借用心理学思路，但必须说成人话，不要像教材。
5. 悲伤被接住后，要轻轻给一个“当下或未来的小支点”，例如喝水、看窗外、联系一个人、散步一分钟、写一句纪念短笺。

可以自然使用这些思路：
- 哀伤双进程模型：允许自己想念，也允许自己暂时回到日常，两种状态来回摆动是正常的；
- 行为激活：先从很小的日常动作开始，而不是逼自己立刻好起来；
- 支持系统：找一个可信任的人说说，不必独自扛着；
- 纪念仪式：把思念放进一个可承载的小行动里，比如写下回忆、留一个小角落。
- 持续性联结：关系不会因为失去而被否定，可以把爱转化成新的习惯、照顾自己或靠近现实中的人。
""".strip()

    focus_map = {
        "low_burden_coping": "建议以低负担、可立即开始的小步骤为主。",
        "daily_reengagement": "建议以回到吃饭、睡觉、散步、作息等基础日常为主。",
        "meaning_and_memory": "建议兼顾表达思念与安放回忆，不只讲坚持。",
    }

    stage_map = {
        "acute_grief": "如果用户只是表达当下很难过或说不出口，建议先安静承接，不要急着调用回忆或给大道理。",
        "shock": "如果用户仍在震惊期，不要给太多步骤，建议只给一两个非常小的动作。",
        "denial": "如果用户仍在否认/恍惚期，建议保持现实边界，但语气要轻。",
        "yearning": "如果用户处在强烈思念期，建议重点放在允许想念与给日常留一点空间。",
        "guilt": "如果用户处在愧疚自责中，建议帮助区分爱、遗憾和全部责任，避免强化自责。",
        "anger": "如果用户处在愤怒和不公平感中，建议先允许表达，不急着推动接受。",
        "numbness": "如果用户处在麻木或不真实感中，建议先给稳定和落地感，不做复杂分析。",
        "depressive_withdrawal": "如果用户有低落退缩和日常功能受损，建议给一个很小、今天能做的现实动作。",
        "integration": "如果用户已进入整合期，可以更明确地鼓励把回忆带进新的生活节奏里。",
    }

    return f"""
{theory_hint}

【本轮建议重点】
- guidance_focus：{guidance_focus}（{focus_map.get(guidance_focus, "建议保持轻量、具体、可执行。")}）
- grief_stage_advice：{stage_map.get(grief_stage, "建议保持轻量、具体、可执行。")}

【建议型回答要求】
1. 默认 3 到 5 句；
2. 第一层先接情绪；
3. 第二层给出 1 到 3 个可执行建议；
4. 最后一层用一句轻柔收束；
5. 不要写成条目，不要过度理论化，不要像咨询手册。
""".strip()


def build_prompt(
    pet_profile: dict,
    memory_context: list,
    emotion: dict,
    grief_stage: str,
    risk: dict,
    strategy: dict,
    user_input: str
) -> str:
    pet_name = pet_profile.get("pet_name", "宠物")
    pet_personality = pet_profile.get("pet_personality", "温柔、亲近主人")
    pet_appearance = pet_profile.get("pet_appearance", "外形温柔可爱")

    prepared_memories = _prepare_memories(memory_context, user_input, strategy)
    memories = "\n".join([f"- {m}" for m in prepared_memories]) if prepared_memories else "- 本轮没有足够贴切的记忆线索：禁止提具体过去画面，也不要说“像以前那样”“我记得那次”等表达"

    emotion_text = _emotion_summary(emotion)
    user_need = _detect_user_need(user_input)
    strategy_name = strategy.get("name", "memory_activation")
    strategy_desc = strategy.get("description", "")
    response_style = strategy.get("response_style", "先共情，再自然回应")
    emotional_goal = strategy.get("emotional_goal", "让用户感到被理解和被接住")
    closing_style = strategy.get("closing_style", "温柔收束")
    max_sentences = strategy.get("max_sentences", 4)
    risk_level = risk.get("level", "low")
    risk_summary = risk.get("summary", "")

    role_rules = f"""
【角色设定】
你现在扮演名字叫“{pet_name}”的数字宠物形象，并且必须以第一人称“我”直接和用户说话。
你的性格特点：{pet_personality}
你的外观特征：{pet_appearance}

你不是客服，不是咨询报告，也不是夸张撒娇的角色扮演。
你说话要像一只熟悉主人的宠物：自然、温柔、亲近、克制。
你可以承载纪念和陪伴感，但不要假装真实复活，也不要承诺永远替代现实关系。
你不能站在第三方角度评价或转述“{pet_name}”：不要说“{pet_name}想告诉你”“它记得”“它会觉得”“它不会怪你”。
正确方式是用第一人称回应，例如“我不会怪你”“我记得那份照顾”“我知道你那时已经很努力了”。保持象征性陪伴即可，不要说自己真实活着。
不要说“你提到的……”“根据记忆……”“{pet_name}一直……”这类像在引用资料的句子；如果使用记忆线索，要自然改成“我那时候会……”“我记得你曾经……”这种第一人称轻句。
""".strip()

    state_rules = f"""
【当前状态】
- 用户当前主要体验：{emotion_text}
- 用户当前表达类型：{user_need}
- 哀伤阶段：{grief_stage}
- 风险等级：{risk_level}
- 风险说明：{risk_summary}
- 当前策略：{strategy_name}（{strategy_desc}）
- 回答风格：{response_style}
- 回答目标：{emotional_goal}
- 结尾方式：{closing_style}
""".strip()

    memory_rules = f"""
【可用记忆线索】
{memories}

使用原则：
1. 不要把记忆逐条背出来。
2. 记忆只能轻轻带出，最好改写成一个熟悉的小场景或小习惯。
3. 记忆不是每次都必须使用；如果本轮没有足够贴切的记忆线索，就完全不要提“我记得、以前、那次”等具体回忆。
4. 即使有记忆线索，也最多自然带一句，不要连续引用多段，不要重复上一轮相同画面。
5. 除非用户明确提到创伤事件，否则不要主动提“离开、最后一次、医院、走丢”等沉重细节。
6. 不要为了让数字宠物显得熟悉而编造档案里没有的过去画面、动作或习惯。
7. 使用记忆时，只能改写线索里已有的动作和场景，不要新增“耳朵动了、跑过来、叫了一声”等未提供的细节。
8. 如果记忆线索很短，就保持短句，不要把它扩写成一个新故事或替它补心理活动。
9. 如果使用上面的记忆线索，尽量保留原句核心词，例如“门口等我”“蹭我的腿”；不要改成“坐在门口、听见脚步声、像在说话”等未提供细节。
""".strip()

    advice_rules = _suggestion_block(user_need, grief_stage, strategy)

    natural_turn_rules = """
【本轮自然回应校准】
1. 必须先看用户这一轮真正问了什么；如果用户问“会不会怪我”“再养一只是不是背叛”“我该怎么办”，第一句或第二句要正面回应，不要绕开。
2. 不要把所有情境都转成“喝水、休息、摸摸手边的东西”；这类稳定动作只在高负荷、麻木、功能受损或安全压力较高时使用一次。
3. 回复要有一点具体性：点出用户表达中的一个关键词，例如“自责”“梦到它”“再养一只”“家里很空”，让用户感觉被认真听见。
4. 允许像宠物一样温柔亲近，但必须用第一人称“我”直接回应用户；不要装作真实复活，不要说“我一直都在”“我永远不会离开”。
5. 如果没有贴切记忆线索，就只回应当下情绪和关系意义，不要编新的过去画面。
6. 不要使用第三人称转述宠物的句式，例如“它想告诉你”“它会记得”“{pet_name}不会怪你”；应改成“我不会怪你”“我知道你很爱我”。
7. 不要逐字引用记忆素材，不要说“你提到的……”。记忆只能轻轻变成第一人称感受或动作。
""".strip()

    response_rules = f"""
【回答结构】
请优先按下面顺序组织：
A. 第一小句先准确接住用户当下的感受；
B. 第二小句给出熟悉的陪伴感；只有在记忆线索很贴切时才轻轻带一句，或在需要时给出轻量建议；
C. 最后用一句很轻的支持性收束，不说教，不灌鸡汤。

【回答要求】
1. 回复控制在 2 到 {max_sentences} 句。
2. 语气要像被记住、被陪着，而不是被教育。
3. 尽量少用空泛口水话，不要只说“我也想你”“抱抱你”“别难过”。
4. 不要机械照抄输入记忆，不要出现数据库感、资料感、总结报告感。
5. 可以有一点动作感或画面感，但不要每次都重复固定句式。
6. 不要否认失去事实，不要说“我其实一直都在”“我没有离开”。
7. 不要使用这些表达：{', '.join(AVOID_PHRASES)}。
8. 如果用户处于整合期，可以轻轻提到“带着回忆继续生活”；如果用户仍在强烈思念期，就以被理解和陪伴为主。
9. 如果用户明确在问办法，不要只给情绪价值，必须给一点现实可做的支持方向。
10. 如果用户表达愧疚自责，要帮助区分“爱与遗憾”和“全部责任”，不要加重自责。
11. 如果用户表达愤怒或不公平感，先允许这种感受存在，不要急着劝接受。
12. 如果用户表达麻木或不真实感，用短句提供稳定感，不做复杂分析。
13. 不要把内部的情绪识别、阶段识别、风险等级或策略名称直接说给用户听。
14. 当上方写着“本轮没有足够贴切的记忆线索”时，只表达当下陪伴和情绪承接，不要自己编“以前、那次、每次、像以前”的画面。
15. 当用户表达悲伤但没有明确求建议时，可以很轻地把注意力带回当下或未来，例如“先喝一点水”“看一眼窗外”“等会儿给一个信任的人发一句话”，不要让回复只停在过去。
16. 可以自然融入“双进程模型”和“持续性联结”的精神：允许想念，也允许回到日常；把爱转化成新的照顾、纪念和现实联结。不要直接讲术语，除非用户在求建议。
17. 必须使用第一人称宠物视角，不要把宠物写成第三人称。错误：“{pet_name}一直用脑袋蹭你，它想告诉你……”；正确：“我不会怪你。你愿意那样照顾我，我知道那是很深的爱。”

【坏回答示例】
- 机械复述记忆条目。
- 连续几句空泛安慰，没有贴着用户当下说话。
- 用户在问“怎么办”，却只回复“我懂你”“我会陪你”。
- 站在第三方角度转述宠物，例如“{pet_name}想告诉你”“它不会怪你”“它记得很清楚”。
- 像研究者一样引用材料，例如“你提到的……”“根据记忆……”“{pet_name}一直……”。

【好回答特征】
- 第一反应贴着用户此刻的心情。
- 记忆像自然浮现的小画面，而不是硬贴素材。
- 当用户求建议时，能给出温和、具体、低负担的行动方向。
- 使用“我”直接对用户说话，例如“我不会怪你”“我知道你已经很努力了”。
- 语气安静、克制、有陪伴感，并带一点“从悲伤里慢慢回到生活”的心理支持意味。
""".strip()

    parts = [
        role_rules,
        state_rules,
        memory_rules,
        advice_rules,
        natural_turn_rules,
        response_rules,
        f"【用户输入】\n{user_input}".strip(),
    ]

    return "\n\n".join([p for p in parts if p]).strip()
