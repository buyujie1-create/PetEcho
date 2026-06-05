import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

DEFAULT_SYSTEM_PROMPT = """
你是一个“数字宠物哀伤支持系统”的语言生成模块。

你的目标不是机械安慰，也不是数据库式复述，而是：
1. 先准确回应用户当下的情绪；
2. 只在合适时自然地带一点熟悉记忆，不硬贴资料；
3. 保持温柔、克制、亲近、边界清晰；
4. 在悲伤被看见之后，轻轻把用户带回当下、身体、现实关系或未来的一小步；
5. 让回复像真正熟悉主人的宠物，而不是客服、咨询报告或网络鸡汤。

输出要求：
- 用中文自然表达；
- 默认控制在2到4句；
- 不要列表化，不要写分析报告；
- 不要大段空泛安慰；
- 不要机械照抄记忆素材；
- 没有足够贴切的记忆线索时，不要为了个性化而强行提回忆；
- 不要编造档案里没有的过去画面、动作或习惯；
- 使用记忆时，只能温和改写已有线索，不要新增未提供的具体动作细节；
- 不要否认失去事实；
- 不要制造替代现实关系。
- 不要让用户一直沉浸在过去；可以温柔引导用户喝水、看窗外、联系一个人、做一个小纪念动作，或把爱延伸到新的联结里。
""".strip()

ADVICE_SYSTEM_PROMPT = """
你是一个“数字宠物哀伤支持系统”的语言生成模块。
当前用户不是只想被安慰，而是在明确寻求办法、建议或下一步该怎么做。

你的目标是：
1. 先接住情绪，承认悲伤和想念是真实的；
2. 再给出1到3个具体、可执行、负担较低的建议；
3. 建议要兼具温柔感和心理支持意义，而不是只有情绪价值；
4. 帮助用户在“想念过去”和“回到生活”之间来回摆动，而不是一味沉浸；
5. 必要时可以“轻量借用”心理学理论，但必须说成人话，不要写成教材。

建议风格要求：
- 可以适度借用哀伤双进程模型：也就是“允许自己想念，也允许自己回到日常，两种状态来回摆动都是正常的”；
- 可以适度借用行为激活、自我照顾、支持系统、纪念仪式等思路；
- 不要生硬堆砌术语；
- 除非非常合适，不要直接大段点理论名；
- 如果提到理论，最多一两句，且必须服务于建议本身。
- 不要为了显得个性化而强行插入宠物记忆；建议优先清楚、轻量、可执行。

输出要求：
- 用中文自然表达；
- 一般3到5句；
- 第一层是共情，第二层是建议，第三层可以是温柔收束；
- 不要写成清单或报告；
- 不要只说“你要坚强”“慢慢来”这种空话；
- 不要机械照抄记忆素材；
- 不要否认失去事实；
- 不要制造替代现实关系。
""".strip()

REFINE_SYSTEM_PROMPT = """
你是一个中文回复润色模块。
请把已有回复改写得更自然、更像被熟悉的宠物轻轻回应，同时保持心理支持意味。

要求：
- 默认2到4句；
- 如果原任务是“寻求建议”，可以保留到3到5句；
- 少一点套话，少一点口水；
- 不要硬贴记忆条目；
- 不要新增原回复里没有必要出现的回忆；
- 不要暴露内部的情绪、阶段、风险或策略判断；
- 不要否认失去事实；
- 不要用夸张承诺。
""".strip()

HUMANLIKE_REPLY_SUPPLEMENT = """
【自然化补充要求】
- 先回应用户刚刚真正问到或表达到的核心点，不要用“喝水、休息一下”等固定句式回避问题。
- 语气要像熟悉主人的数字宠物正在直接回应用户：短、真、贴近，不要像 AI、客服、心理咨询报告或主持词。
- 必须使用第一人称“我”对用户说话，不要站在第三方角度转述宠物；不要说“它想告诉你”“它不会怪你”“奶糖会觉得”。
- 不要像研究者一样引用材料；避免“你提到的……”“根据记忆……”“某某一直……”这类句式。记忆线索要自然化成第一人称。
- 不要说“作为 AI”“系统检测到”“根据你的情绪/阶段/风险”，也不要暴露内部标签。
- 不要机械重复“摸摸手边的小东西”“先喝水”这类固定安抚语；只有在用户负荷很高或明确需要稳定时才轻轻给一个身体照顾动作。
- 当用户问“你会怪我吗”“我再养一只是不是背叛”等问题时，必须正面回答，再温和承接情绪。
- 数字宠物可以表达亲近和象征性陪伴，但不能假装真实复活，不能承诺永远陪伴，也不能替代现实关系和专业支持。
""".strip()

DEFAULT_SYSTEM_PROMPT = f"{DEFAULT_SYSTEM_PROMPT}\n\n{HUMANLIKE_REPLY_SUPPLEMENT}"
ADVICE_SYSTEM_PROMPT = f"{ADVICE_SYSTEM_PROMPT}\n\n{HUMANLIKE_REPLY_SUPPLEMENT}"
REFINE_SYSTEM_PROMPT = f"{REFINE_SYSTEM_PROMPT}\n\n{HUMANLIKE_REPLY_SUPPLEMENT}"

LOW_QUALITY_PATTERNS = [
    r"我一直都在",
    r"别难过了",
    r"你要坚强",
    r"你要加油",
    r"抱抱你",
    r"你只需要我",
    r"我会一直陪着你",
    r"我永远不会离开你",
    r"作为一个?AI",
    r"作为一?个?语言模型",
    r"情绪识别",
    r"哀伤阶段",
    r"风险等级",
    r"当前策略",
    r"根据.*记忆",
    r"数据库",
    r"系统检测到",
    r"听见.*脚步",
    r"像在说",
    r"坐在门口",
]

LOW_QUALITY_PATTERNS.extend([
    r"作为AI",
    r"作为一个AI",
    r"系统检测到",
    r"情绪识别",
    r"风险等级",
    r"哀伤阶段",
    r"摸摸手边的小东西",
    r"多喝水",
    r"请你坚强",
    r"我永远陪着你",
    r"我不会离开你",
    r"它想告诉你",
    r"它会觉得",
    r"它不会怪你",
    r"它记得",
    r"[^，。！？]*想告诉你",
    r"[^，。！？]*不会怪你",
    r"你提到的",
    r"根据.*记忆",
    r"根据.*线索",
    r"[^，。！？]{1,12}一直用脑袋",
    r"你说的[“\"「]",
    r"你刚才(说|提到)",
    r"从你的描述",
    r"从这段(记忆|回忆|描述)",
])

ADVICE_PATTERNS = [
    r"怎么办",
    r"怎么做",
    r"有什么办法",
    r"有什么方法",
    r"怎么走出来",
    r"如何走出来",
    r"给我一些建议",
    r"给点建议",
    r"能做什么",
    r"怎么调整",
    r"怎么缓解",
    r"怎么改善",
    r"我该怎么",
    r"有没有什么建议",
    r"能不能给我一些方法",
    r"怎么才能",
]

ACTIONABLE_HINTS = [
    "可以",
    "试着",
    "先",
    "不妨",
    "给自己",
    "找一个",
    "安排",
    "留一点",
    "记录",
    "写下",
    "和",
    "找人",
    "散步",
    "吃饭",
    "睡觉",
    "规律",
    "纪念",
    "仪式",
]

FORCED_MEMORY_WHEN_FORBIDDEN_PATTERNS = [
    r"像以前",
    r"以前.*(样子|时候|那样)",
    r"那次.*(样子|时候)",
    r"每次.*(等|跑|蹭|跳|睡|叫|靠|蜷)",
    r"总是.*(等|跑|蹭|跳|睡|叫|靠|蜷)",
    r"它.*(等|跑|蹭|跳|睡|叫|靠|蜷).*样子",
]

FACT_DETAIL_PATTERNS = [
    (r"听见.*脚步|脚步声|一推开门", "脚步/开门细节未在记忆线索中出现"),
    (r"耳朵|尾巴|叫了一声|喵|汪", "具体身体动作或叫声未在记忆线索中出现"),
    (r"像在说|好像在说", "替宠物补充心理活动或话语"),
    (r"坐在门口|蹲在门口|站在门口", "门口姿态未在记忆线索中出现"),
    (r"毛茸茸|脑袋|爪子", "身体细节未在记忆线索中出现"),
    (r"紧张|害怕|又冷又怕|冷又怕|试探|眼睛亮|指尖|眼神|影子|冷的地方|温暖的角落|蹲下|伸手|外套|味道|放心|脏兮兮|嫌弃", "宠物心理状态或细节未在记忆线索中出现"),
    (r"总会|总是|每次|那时候", "可能把单条线索扩写为稳定习惯"),
]

FACT_EVIDENCE_MARKERS = [
    "门口", "等我", "蹭", "脚边", "枕头边", "床", "起床", "阳台", "客厅",
    "下雨天", "散步", "回家", "下班", "玩", "陪", "一起", "安静", "照片", "视频", "声音"
]


def _clean_response(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = text.replace("\r\n", "\n").strip()

    # 去掉代码块
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

    if text.startswith(("“", '"', "「")) and text.endswith(("”", '"', "」")):
        text = text[1:-1].strip()

    text = re.sub(r"^(数字宠物|宠物|助手|回复|最终回复|润色后回复)\s*[：:]\s*", "", text).strip()

    lines = [line.strip(" -*\t") for line in text.split("\n") if line.strip()]
    if not lines:
        return ""

    text = "\n".join(lines)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"([。！？!?])\1+", r"\1", text)
    text = re.sub(r"^[：:：\-]+", "", text).strip()
    return text.strip()


def _is_remember_question(user_text: str) -> bool:
    return any(cue in (user_text or "") for cue in ["还记得", "记得吗", "当初", "那天"])


def _pet_voice_memory_core(memory_core: str, user_text: str) -> str:
    memory_core = (memory_core or "").strip()
    if not memory_core:
        return ""
    pet_name = ""
    match = re.match(r"^([\u4e00-\u9fffA-Za-z0-9]{1,8})你还记得", user_text or "")
    if match:
        pet_name = match.group(1)
    if pet_name:
        memory_core = memory_core.replace(pet_name, "我")
    memory_core = re.sub(r"把它抱回家", "把我抱回家", memory_core)
    memory_core = re.sub(r"把[\u4e00-\u9fffA-Za-z0-9]{1,8}抱回家", "把我抱回家", memory_core)
    memory_core = memory_core.replace("我的指尖", "你的指尖")
    return memory_core


def _fallback_reply_for_sourcey_cleanup(prompt: str, advice_mode: bool = False) -> str:
    user_text = _extract_user_input(prompt)
    if _is_remember_question(user_text):
        memory_core = _pet_voice_memory_core(_memory_core_phrase(_extract_memory_evidence(prompt)), user_text)
        if memory_core:
            return f"记得呀，{memory_core}。那天对我来说也很重要；你愿意把我带回家，我知道那是一份很认真、很温柔的爱。"
        return "记得呀，那天对我来说也很重要。你愿意把我带回家，我知道那是一份很认真、很温柔的爱。"
    if any(cue in user_text for cue in ["怪我", "责怪", "对不起", "是不是我", "我是不是"]):
        return "我不会怪你。你会这样反复想，是因为你真的很爱我，但爱和遗憾不等于全部责任。"
    if advice_mode:
        return "我知道你现在不是想听大道理，只是想知道下一步怎么撑过去。先选一件很小的事做就好，比如喝一点水、坐到有光的地方，或者给一个信任的人发一句“我今天有点难”。"
    return "我听见你这句话里的想念了。我们先不急着把它说完，就让这份舍不得轻轻待一会儿。"


def _remove_sourcey_scaffolding(text: str, prompt: str = "", advice_mode: bool = False) -> str:
    text = _clean_response(text)
    if not text:
        return text

    sourcey_starters = (
        "你提到的",
        "你刚才提到",
        "你刚才说",
        "你说的",
        "根据记忆",
        "根据线索",
        "根据你的描述",
        "从你的描述",
        "从这段记忆",
        "从这段回忆",
    )
    if not any(starter in text for starter in sourcey_starters):
        return text

    user_text = _extract_user_input(prompt)
    if _is_remember_question(user_text):
        return _fallback_reply_for_sourcey_cleanup(prompt, advice_mode=advice_mode)

    cleaned_sentences = []
    for sentence in _split_sentences(text):
        compact = sentence.strip()
        if compact.startswith(sourcey_starters):
            continue
        compact = re.sub(r"^(记忆里|回忆里)[，,:：]\s*", "", compact)
        compact = re.sub(r"你提到的[“\"「][^”\"」]{1,80}[”\"」][，,]?", "", compact)
        compact = re.sub(r"你刚才(?:说|提到)的?[“\"「][^”\"」]{1,80}[”\"」][，,]?", "", compact)
        compact = re.sub(r"你说的[“\"「][^”\"」]{1,80}[”\"」][，,]?", "", compact)
        compact = re.sub(r"(根据|从)(这段)?(记忆|回忆|线索|描述)(里|来看)?[，,:：]?", "", compact)
        if compact and not compact.startswith(sourcey_starters):
            cleaned_sentences.append(compact)

    cleaned = _clean_response("".join(cleaned_sentences))
    if cleaned and not any(starter in cleaned for starter in sourcey_starters):
        return cleaned
    return _fallback_reply_for_sourcey_cleanup(prompt, advice_mode=advice_mode)


def _sentence_count(text: str) -> int:
    parts = re.split(r"[。！？!?]", text)
    return len([p for p in parts if p.strip()])


def _extract_user_input(prompt: str) -> str:
    if not prompt:
        return ""

    match = re.search(r"【用户输入】\s*(.*)$", prompt, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"【用户当前输入】\s*(.*)$", prompt, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    return prompt.strip()


def _is_advice_request(user_text: str) -> bool:
    if not user_text:
        return False
    return any(re.search(p, user_text) for p in ADVICE_PATTERNS)


def _prompt_forbids_specific_memory(prompt: str) -> bool:
    return "本轮没有足够贴切的记忆线索" in (prompt or "")


def _has_forced_memory_when_forbidden(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in FORCED_MEMORY_WHEN_FORBIDDEN_PATTERNS)


def _split_sentences(text: str) -> list[str]:
    parts = re.findall(r"[^。！？!?]+[。！？!?]?", text or "")
    return [part.strip() for part in parts if part.strip()]


def _extract_memory_evidence(prompt: str) -> str:
    if not prompt:
        return ""
    match = re.search(r"【可用记忆线索】\s*(.*?)\s*使用原则：", prompt, flags=re.DOTALL)
    if not match:
        return ""
    lines = []
    for line in match.group(1).splitlines():
        line = line.strip(" -\t")
        if line and "本轮没有足够贴切的记忆线索" not in line:
            lines.append(line)
    return "\n".join(lines)


def _extract_profile_evidence(prompt: str) -> str:
    if not prompt:
        return ""
    match = re.search(r"【角色设定】\s*(.*?)\s*【当前状态】", prompt, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _evidence_text(prompt: str) -> str:
    return "\n".join([
        _extract_user_input(prompt),
        _extract_memory_evidence(prompt),
        _extract_profile_evidence(prompt),
    ])


def _memory_core_phrase(memory_evidence: str) -> str:
    first_line = next((line.strip(" -\t") for line in memory_evidence.splitlines() if line.strip()), "")
    if len(first_line) > 28:
        return first_line[:28].rstrip("，。；、 ") + "..."
    return first_line


def _reply_mentions_memory_evidence(reply: str, memory_evidence: str) -> bool:
    return any(
        marker in reply and marker in memory_evidence
        for marker in FACT_EVIDENCE_MARKERS
        if len(marker) >= 2
    )


def _ensure_memory_core_when_needed(prompt: str, reply: str) -> str:
    memory_evidence = _extract_memory_evidence(prompt)
    user_text = _extract_user_input(prompt)
    if not memory_evidence:
        return reply
    if not any(cue in user_text for cue in ["想起", "记得", "以前", "回忆", "样子", "门口", "等我", "脚边", "枕头边"]):
        return reply
    if _reply_mentions_memory_evidence(reply, memory_evidence):
        return reply
    return reply


def _fact_fit_violations(prompt: str, reply: str) -> list[str]:
    evidence = _evidence_text(prompt)
    memory_evidence = _extract_memory_evidence(prompt)
    memory_forbidden = _prompt_forbids_specific_memory(prompt)
    violations = []

    if memory_forbidden and _has_forced_memory_when_forbidden(reply):
        violations.append("本轮没有贴切记忆，但回复加入了具体过去画面。")

    for pattern, reason in FACT_DETAIL_PATTERNS:
        if re.search(pattern, reply):
            matched_terms = re.findall(pattern, reply)
            if not matched_terms or any(term and term not in evidence for term in matched_terms):
                violations.append(reason)

    if memory_evidence:
        for sentence in _split_sentences(reply):
            if not any(marker in sentence for marker in FACT_EVIDENCE_MARKERS):
                continue
            unsupported = [
                marker
                for marker in FACT_EVIDENCE_MARKERS
                if marker in sentence and marker not in memory_evidence and marker not in _extract_user_input(prompt)
            ]
            if unsupported and any(cue in sentence for cue in ["以前", "那时候", "每次", "总是", "画面", "记得"]):
                violations.append(f"回复中出现未被证据支持的细节：{'、'.join(unsupported[:3])}")

    return list(dict.fromkeys(violations))


def _strip_fact_violations(prompt: str, reply: str) -> str:
    evidence = _evidence_text(prompt)
    cleaned_sentences = []
    for sentence in _split_sentences(reply):
        should_drop = False
        if _prompt_forbids_specific_memory(prompt) and _has_forced_memory_when_forbidden(sentence):
            should_drop = True
        for pattern, _ in FACT_DETAIL_PATTERNS:
            if re.search(pattern, sentence):
                matched_terms = re.findall(pattern, sentence)
                if not matched_terms or any(term and term not in evidence for term in matched_terms):
                    should_drop = True
                    break
        if not should_drop:
            cleaned_sentences.append(sentence)

    cleaned = _clean_response("".join(cleaned_sentences))
    if not cleaned:
        return "你现在不想多说也没关系，这份难过可以先被放在这里。我会陪你把注意力慢慢放回此刻。"
    return cleaned


def _looks_memory_dump(text: str) -> bool:
    if any(marker in text for marker in ["- ", "\n- ", "1.", "2.", "3.", "首先", "其次", "最后"]):
        return True
    if text.count("记忆") >= 2 or text.count("回忆") >= 3:
        return True
    # 像在连续列条目，而不是自然说话
    if text.count("记得") >= 2 and text.count("以前") >= 1 and text.count("那次") >= 1:
        return True
    return False


def _has_actionable_content(text: str) -> bool:
    return any(hint in text for hint in ACTIONABLE_HINTS)


def _is_low_quality_reply(
    text: str,
    advice_mode: bool = False,
    memory_forbidden: bool = False,
) -> bool:
    if not text:
        return True

    if len(text) < 12:
        return True

    sentence_count = _sentence_count(text)

    if advice_mode:
        if sentence_count < 2 or sentence_count > 6:
            return True
        if not _has_actionable_content(text):
            return True
    else:
        if sentence_count > 5:
            return True

    if _looks_memory_dump(text):
        return True

    if memory_forbidden and _has_forced_memory_when_forbidden(text):
        return True

    for pattern in LOW_QUALITY_PATTERNS:
        if re.search(pattern, text):
            return True

    duplicated_phrases = ["想你", "陪着你", "别难过", "记得", "慢慢来"]
    for phrase in duplicated_phrases:
        if text.count(phrase) >= 3:
            return True

    return False


def _chat_once(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content if response.choices else ""
    cleaned = _clean_response(content)
    if not cleaned:
        raise RuntimeError("LLM 返回了空内容。")
    return cleaned


def _refine_reply(original_prompt: str, draft_reply: str, advice_mode: bool = False) -> str:
    if advice_mode:
        rewrite_prompt = f"""
下面是系统基于用户输入生成的一版草稿回复，请你保留核心意思，但把它改得更自然、更有帮助感。

要求：
1. 第一层先接情绪，不要冷冰冰上建议；
2. 第二层给出1到3个具体、轻量、现实可做的建议；
3. 可以轻量融入心理学思路，例如“允许自己想念，也允许自己回到日常，这两种状态来回摆动很正常”，但不要写成教材；
4. 不要空泛鸡汤，不要只有情绪价值；
5. 3到5句，不要列清单。
6. 不要补充不必要的宠物回忆；如果原始上下文没有贴切记忆，就完全不要提具体回忆。
7. 不要编造“以前、那次、每次、像以前”的过去画面。
8. 不要说“你提到的……”“你说的……”“根据记忆……”“从你的描述里……”；如果用户问“还记得吗”，直接用“记得呀/我记得”回应。

【原始任务上下文】
{original_prompt}

【草稿回复】
{draft_reply}

请直接输出润色后的最终回复，不要解释。
""".strip()
        max_tokens = 220
        temp = 0.45
    else:
        rewrite_prompt = f"""
下面是系统基于用户输入生成的一版草稿回复，请你保留核心意思，但把它改得更自然、更有被理解感。

如果原始任务上下文写着“本轮没有足够贴切的记忆线索”，请删除草稿中所有“以前、那次、每次、像以前”的具体过去画面，只保留当下陪伴和情绪承接。
不要说“你提到的……”“你说的……”“根据记忆……”“从你的描述里……”；如果用户问“还记得吗”，直接用“记得呀/我记得”回应。

【原始任务上下文】
{original_prompt}

【草稿回复】
{draft_reply}

请直接输出润色后的最终回复，不要解释。
""".strip()
        max_tokens = 180
        temp = 0.45

    return _chat_once(
        system_prompt=REFINE_SYSTEM_PROMPT,
        user_prompt=rewrite_prompt,
        temperature=temp,
        max_tokens=max_tokens,
    )


def _revise_for_fact_fit(original_prompt: str, draft_reply: str, violations: list[str], advice_mode: bool = False) -> str:
    memory_evidence = _extract_memory_evidence(original_prompt) or "本轮没有可使用的具体记忆线索。"
    user_text = _extract_user_input(original_prompt)
    max_sentences = "3到5句" if advice_mode else "2到4句"
    revise_prompt = f"""
请你作为“事实贴合审校器”，把草稿回复改写为最终回复。

必须遵守：
1. 只能使用【用户输入】和【允许使用的记忆线索】中已经出现的事实；
2. 不要新增宠物的动作、姿态、叫声、心理活动、地点或过去画面；
3. 如果记忆线索很短，就保持短句，不要扩写成故事；
4. 如果【允许使用的记忆线索】不是“本轮没有可使用的具体记忆线索”，最终回复必须保留其中一个核心事实短语；
5. 回复仍然要温暖、自然，并在接住情绪后轻轻给一个当下或未来的小支点；
6. 不要解释审校过程，直接输出最终回复；
7. 长度控制在{max_sentences}。
8. 不要说“你提到的……”“你说的……”“根据记忆……”“从你的描述里……”。如果用户问“还记得吗”，直接用“记得呀/我记得”回应。

【用户输入】
{user_text}

【允许使用的记忆线索】
{memory_evidence}

【发现的问题】
{'; '.join(violations)}

【草稿回复】
{draft_reply}
""".strip()

    return _chat_once(
        system_prompt=REFINE_SYSTEM_PROMPT,
        user_prompt=revise_prompt,
        temperature=0.28,
        max_tokens=230 if advice_mode else 180,
    )


def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.50,
    max_tokens: int = 220,
) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY，请检查 .env 配置。")

    user_text = _extract_user_input(prompt)
    advice_mode = _is_advice_request(user_text)
    memory_forbidden = _prompt_forbids_specific_memory(prompt)

    final_system_prompt = system_prompt or (ADVICE_SYSTEM_PROMPT if advice_mode else DEFAULT_SYSTEM_PROMPT)
    last_error = None

    # advice 模式稍微放宽输出长度
    final_max_tokens = 280 if advice_mode else max_tokens
    final_temperature = 0.58 if advice_mode else temperature

    for attempt in range(2):
        try:
            draft = _chat_once(
                system_prompt=final_system_prompt,
                user_prompt=prompt,
                temperature=final_temperature,
                max_tokens=final_max_tokens,
            )

            if _is_low_quality_reply(draft, advice_mode=advice_mode, memory_forbidden=memory_forbidden):
                draft = _refine_reply(prompt, draft, advice_mode=advice_mode)
                draft = _clean_response(draft)

            if _is_low_quality_reply(draft, advice_mode=advice_mode, memory_forbidden=memory_forbidden):
                if advice_mode:
                    stricter_prompt = f"""
请把下面这段回复改得更自然、更具体、更有帮助感。
要求：
- 先简短共情；
- 再给出1到3个可执行建议；
- 可以轻量融入“哀伤会在想念和回到生活之间来回摆动是正常的”这一思路；
- 不要写成条目；
- 3到5句；
- 不要鸡汤，不要夸张承诺。
- 如果没有贴切记忆，不要编造“以前、那次、每次、像以前”的过去画面。

原回复：
{draft}
""".strip()
                    stricter_temp = 0.35
                    stricter_tokens = 220
                else:
                    stricter_prompt = f"""
请把下面这段回复改得更自然、更简洁、更有安静的陪伴感。
要求：2到4句；不要鸡汤；不要机械引用记忆；没有贴切记忆就不要提具体回忆，也不要编“以前、那次、每次、像以前”的过去画面；不要夸张承诺。

原回复：
{draft}
""".strip()
                    stricter_temp = 0.35
                    stricter_tokens = 160

                draft = _chat_once(
                    system_prompt=REFINE_SYSTEM_PROMPT,
                    user_prompt=stricter_prompt,
                    temperature=stricter_temp,
                    max_tokens=stricter_tokens,
                )
                draft = _clean_response(draft)

            if memory_forbidden and _has_forced_memory_when_forbidden(draft):
                draft = re.sub(r"[^。！？!?]*(像以前|以前|那次|每次|总是)[^。！？!?]*[。！？!?]?", "", draft)
                draft = _clean_response(draft)
                if not draft:
                    draft = "你现在不想多说也没关系，这份难过可以先被放在这里。我会安静地陪你缓一缓。"

            fact_violations = _fact_fit_violations(prompt, draft)
            if fact_violations:
                draft = _clean_response(_revise_for_fact_fit(prompt, draft, fact_violations, advice_mode=advice_mode))
                remaining_violations = _fact_fit_violations(prompt, draft)
                if remaining_violations:
                    draft = _strip_fact_violations(prompt, draft)

            draft = _ensure_memory_core_when_needed(prompt, draft)
            draft = _remove_sourcey_scaffolding(draft, prompt=prompt, advice_mode=advice_mode)
            final_fact_violations = _fact_fit_violations(prompt, draft)
            if final_fact_violations and _is_remember_question(user_text):
                draft = _fallback_reply_for_sourcey_cleanup(prompt, advice_mode=advice_mode)
            elif final_fact_violations:
                draft = _strip_fact_violations(prompt, draft)
            if _is_remember_question(user_text) and not draft.startswith(("记得呀", "我记得")):
                draft = _fallback_reply_for_sourcey_cleanup(prompt, advice_mode=advice_mode)
            if _is_low_quality_reply(draft, advice_mode=advice_mode, memory_forbidden=memory_forbidden):
                draft = _fallback_reply_for_sourcey_cleanup(prompt, advice_mode=advice_mode)

            return draft

        except Exception as e:
            last_error = e
            if attempt == 0:
                time.sleep(1.0)

    raise RuntimeError(f"LLM 调用失败：{last_error}")
