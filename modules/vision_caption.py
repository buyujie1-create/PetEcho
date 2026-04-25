from functools import lru_cache

from PIL import Image
from transformers.utils import logging as hf_logging
import torch

from modules.llm_api import call_llm
from modules.model_loader import load_blip_model

hf_logging.set_verbosity_error()


@lru_cache(maxsize=1)
def get_blip_models():
    return load_blip_model()


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
    raw_image = Image.open(image_path).convert("RGB")

    prompts = [
        None,
        "a detailed description of a pet's fur color and coat",
        "a close-up description of a pet's face, eyes, ears, and expression",
        "a short description of the pet's size and overall appearance",
    ]

    captions = []
    for p in prompts:
        try:
            cap = _generate_caption(raw_image, p)
            if cap and cap not in captions:
                captions.append(cap)
        except Exception:
            continue

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
    except Exception:
        return "这是一只毛发比较蓬松、气质温和的小宠物。"


def generate_pet_appearance_caption(image_path: str) -> str:
    captions_en = generate_raw_blip_signals(image_path)
    if not captions_en:
        return "这是一只外形可爱、气质温和的小宠物。"
    return refine_caption_to_chinese(captions_en)
