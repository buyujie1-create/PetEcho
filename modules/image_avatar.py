import base64
import os
from urllib.request import urlopen

from utils.file_io import save_generated_pet_avatar


IMAGE_STYLE_PROMPTS = {
    "温柔手绘风": "soft hand-drawn memorial illustration, warm gentle colors, clean background",
    "相册纪念风": "warm photo-album inspired digital portrait, soft light, commemorative feeling",
    "像素宠物风": "cute pixel art pet avatar, warm palette, simple background",
    "安静夜灯风": "calm night-light style illustration, soft glow, peaceful memorial mood",
}


def image_generation_status() -> dict:
    enabled = os.getenv("IMAGE_GEN_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "enabled": enabled,
        "has_key": has_key,
        "ready": enabled and has_key,
        "model": os.getenv("IMAGE_MODEL", "gpt-image-1"),
        "size": os.getenv("IMAGE_SIZE", "1024x1024"),
    }


def build_avatar_prompt(profile: dict, style: str) -> str:
    pet_name = (profile or {}).get("pet_name") or "the pet"
    personality = (profile or {}).get("pet_personality") or "gentle and close to its owner"
    appearance = (profile or {}).get("pet_appearance") or "keep the pet's main species, coat color, face shape, ears, eyes, and overall temperament"
    style_prompt = IMAGE_STYLE_PROMPTS.get(style, IMAGE_STYLE_PROMPTS["温柔手绘风"])

    return f"""
Create a comforting digital memorial avatar based on the reference pet photo.

Pet name: {pet_name}
Personality notes: {personality}
Appearance notes from the owner: {appearance}

Visual style: {style_prompt}.
Keep the pet's approximate species, coat colors, facial features, ears, eyes, and gentle temperament.
The image should feel warm, peaceful, and suitable for a pet grief support system.
Do not make it scary, uncanny, hyper-realistic, or imply that the pet has come back to life.
No text, no logo, no watermark.
""".strip()


def _extract_image_bytes(result) -> bytes:
    data = getattr(result, "data", None) or []
    if not data:
        raise RuntimeError("图像生成接口未返回图片。")

    item = data[0]
    b64_json = getattr(item, "b64_json", None)
    if b64_json:
        return base64.b64decode(b64_json)

    url = getattr(item, "url", None)
    if url:
        with urlopen(url, timeout=30) as response:
            return response.read()

    raise RuntimeError("图像生成接口返回了未知格式。")


def generate_virtual_pet_avatar(image_path: str, profile: dict, style: str = "温柔手绘风") -> str:
    status = image_generation_status()
    if not status["enabled"]:
        raise RuntimeError("当前未启用图像生成。请在环境变量中设置 IMAGE_GEN_ENABLED=true。")
    if not status["has_key"]:
        raise RuntimeError("未检测到 OPENAI_API_KEY，无法调用图像生成接口。")
    if not image_path or not os.path.exists(image_path):
        raise RuntimeError("请先上传或保存一张宠物照片。")

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = build_avatar_prompt(profile, style)

    with open(image_path, "rb") as image_file:
        result = client.images.edit(
            model=status["model"],
            image=image_file,
            prompt=prompt,
            size=status["size"],
        )

    image_bytes = _extract_image_bytes(result)
    return save_generated_pet_avatar(image_bytes)
