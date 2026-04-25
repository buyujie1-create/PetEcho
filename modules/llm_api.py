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
4. 让回复像真正熟悉主人的宠物，而不是客服、咨询报告或网络鸡汤。

输出要求：
- 用中文自然表达；
- 默认控制在2到4句；
- 不要列表化，不要写分析报告；
- 不要大段空泛安慰；
- 不要机械照抄记忆素材；
- 不要否认失去事实；
- 不要制造替代现实关系。
""".strip()

ADVICE_SYSTEM_PROMPT = """
你是一个“数字宠物哀伤支持系统”的语言生成模块。
当前用户不是只想被安慰，而是在明确寻求办法、建议或下一步该怎么做。

你的目标是：
1. 先接住情绪，承认悲伤和想念是真实的；
2. 再给出1到3个具体、可执行、负担较低的建议；
3. 建议要兼具温柔感和心理支持意义，而不是只有情绪价值；
4. 必要时可以“轻量借用”心理学理论，但必须说成人话，不要写成教材。

建议风格要求：
- 可以适度借用哀伤双进程模型：也就是“允许自己想念，也允许自己回到日常，两种状态来回摆动都是正常的”；
- 可以适度借用行为激活、自我照顾、支持系统、纪念仪式等思路；
- 不要生硬堆砌术语；
- 除非非常合适，不要直接大段点理论名；
- 如果提到理论，最多一两句，且必须服务于建议本身。

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
- 不要否认失去事实；
- 不要用夸张承诺。
""".strip()

LOW_QUALITY_PATTERNS = [
    r"我一直都在",
    r"别难过了",
    r"你要坚强",
    r"你要加油",
    r"抱抱你",
    r"你只需要我",
    r"我会一直陪着你",
    r"我永远不会离开你",
]

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


def _clean_response(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = text.replace("\r\n", "\n").strip()

    # 去掉代码块
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

    if text.startswith(("“", '"', "「")) and text.endswith(("”", '"', "」")):
        text = text[1:-1].strip()

    lines = [line.strip(" -*\t") for line in text.split("\n") if line.strip()]
    if not lines:
        return ""

    text = "\n".join(lines)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"([。！？!?])\1+", r"\1", text)
    text = re.sub(r"^[：:：\-]+", "", text).strip()
    return text.strip()


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


def _looks_memory_dump(text: str) -> bool:
    if any(marker in text for marker in ["- ", "\n- ", "1.", "2.", "3.", "首先", "其次", "最后"]):
        return True
    # 像在连续列条目，而不是自然说话
    if text.count("记得") >= 2 and text.count("以前") >= 1 and text.count("那次") >= 1:
        return True
    return False


def _has_actionable_content(text: str) -> bool:
    return any(hint in text for hint in ACTIONABLE_HINTS)


def _is_low_quality_reply(text: str, advice_mode: bool = False) -> bool:
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


def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.62,
    max_tokens: int = 220,
) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY，请检查 .env 配置。")

    user_text = _extract_user_input(prompt)
    advice_mode = _is_advice_request(user_text)

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

            if _is_low_quality_reply(draft, advice_mode=advice_mode):
                draft = _refine_reply(prompt, draft, advice_mode=advice_mode)
                draft = _clean_response(draft)

            if _is_low_quality_reply(draft, advice_mode=advice_mode):
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

原回复：
{draft}
""".strip()
                    stricter_temp = 0.35
                    stricter_tokens = 220
                else:
                    stricter_prompt = f"""
请把下面这段回复改得更自然、更简洁、更有安静的陪伴感。
要求：2到4句；不要鸡汤；不要机械引用记忆；不要夸张承诺。

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

            return draft

        except Exception as e:
            last_error = e
            if attempt == 0:
                time.sleep(1.0)

    raise RuntimeError(f"LLM 调用失败：{last_error}")
