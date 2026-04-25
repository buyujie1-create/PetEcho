from functools import lru_cache
import traceback

from PIL import Image
from transformers.utils import logging as hf_logging
import torch
import streamlit as st

from modules.llm_api import call_llm
from modules.model_loader import load_blip_model

hf_logging.set_verbosity_error()

BLIP_FALLBACK_TEXT = "图像理解模块当前未成功加载，请先手动填写宠物外观描述。"

# 记录最近一次 BLIP 错误，方便在前端调试或日志中查看
_LAST_BLIP_ERROR = None


def _set_last_blip_error(msg: str) -> None:
    global _LAST_BLIP_ERROR
    _LAST_BLIP_ERROR = msg
    print(f"[BLIP ERROR] {msg}")


def get_last_blip_error() -> str | None:
    return _LAST_BLIP_ERROR


@lru_cache(maxsize=1)
def get_blip_models():
    try:
        return load_blip_model()
    except Exception as e:
        error_msg = f"BLIP 模型加载失败：{type(e).__name__}: {e}"
        _set_last_blip_error(error_msg)
        raise RuntimeError(error_msg) from e


def _generate_caption(raw_image, prompt: str | None = None) -> str:
    processor, model = get_blip_models()

    if prompt:
        inputs = processor(raw_image, prompt, return_tensors="pt")
    else:
        inputs = processor(images=raw_image, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=48,
            num_beams=5,
            no_repeat_ngram_size=2
        )

    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    return caption


def generate_raw_blip_signals(image_path: str) -> list[str]:
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        error_msg = f"图片读取失败：{type(e).__name__}: {e}"
        _set_last_blip_error(error_msg)
        return []

    prompts = [
        None,
        "a detailed description of a pet's fur color and coat",
        "a close-up description of a pet's face, eyes, ears, and expression",
        "a short description of the pet's size and overall appearance",
    ]

    captions = []
    errors = []

    for p in prompts:
        try:
            cap = _generate_caption(raw_image, p)
            if cap and cap not in captions:
                captions.append(cap)
        except Exception as e:
            prompt_label = p if p is not None else "default"
            error_msg = f"BLIP caption generation failed. prompt={prompt_label}, error={type(e).__name__}: {e}"
            errors.append(error_msg)
            _set_last_blip_error(error_msg)
            print(traceback.format_exc())
            continue

    if not captions and errors:
        _set_last_blip_error("；".join(errors[:2]))

    return captions


def refine_caption_to_chinese(captions_en: list[str]) -> str:
    joined = "\n".join([f"- {c}" for c in captions_en]) if captions_en else "- no useful caption"

    prompt = f"""
你现在是一个宠物视觉描述助手。

下面是同一张宠物照片生成的若干英文视觉线索，请综合它们，写成适合“宠物档案”的中文外观描述。

要求：
1. 用自然中文，不要直译腔。
2. 尽量提到：毛发长短、主色/局部颜色、脸部、耳朵、眼睛、整体气质。
3. 如果信息不够确定，可用“看起来”“似乎”这类保守表达。
4. 如果原始英文信息不够具体，可以根据“宠物照片描述常见表达”做适度细化，但不要编造夸张内容。
5. 输出1到2句话，简洁但尽量具体。
6. 只输出最终中文描述，不要解释。

英文视觉线索：
{joined}
""".strip()

    try:
        result = call_llm(
            prompt,
            system_prompt="你是一个宠物外观描述生成助手，擅长把视觉线索整理成自然、具体、简洁的中文。",
            temperature=0.35,
            max_tokens=140
        )
        return result.strip()
    except Exception as e:
        error_msg = f"中文润色失败：{type(e).__name__}: {e}"
        _set_last_blip_error(error_msg)
        return "这是一只毛发比较蓬松、气质温和的小宠物。"


def generate_pet_appearance_caption(image_path: str) -> str:
    captions_en = generate_raw_blip_signals(image_path)
    if not captions_en:
        return BLIP_FALLBACK_TEXT
    return refine_caption_to_chinese(captions_en)


def show_blip_status_hint():
    """
    可在 app.py 中调用：
        from modules.vision_caption import show_blip_status_hint
        show_blip_status_hint()
    用于在页面上显示最近一次 BLIP 错误信息。
    """
    err = get_last_blip_error()
    if err:
        st.caption(f"图像理解调试信息：{err}")
