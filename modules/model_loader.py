import os
import warnings
from typing import Tuple

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import BlipForConditionalGeneration, BlipProcessor

EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
BLIP_MODEL_NAME = os.getenv(
    "BLIP_MODEL_NAME",
    "Salesforce/blip-image-captioning-base"
)

LOCAL_EMBED_PATH = os.getenv(
    "EMBED_MODEL_PATH",
    "models/paraphrase-multilingual-MiniLM-L12-v2"
)
LOCAL_BLIP_PATH = os.getenv(
    "BLIP_MODEL_PATH",
    "models/blip-image-captioning-base"
)


def _path_exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def _warn(msg: str) -> None:
    warnings.warn(msg, RuntimeWarning)


def _has_any_file(path: str, filenames: list[str]) -> bool:
    return any(os.path.exists(os.path.join(path, name)) for name in filenames)


def _is_valid_embed_dir(path: str) -> bool:
    """
    判断本地 embedding 模型目录是否完整到足以本地加载。
    这里不追求极严，只要关键配置和权重大概率齐全即可。
    """
    if not _path_exists(path):
        return False

    required_config_files = [
        "config.json",
        "modules.json",
        "sentence_bert_config.json",
        "config_sentence_transformers.json",
    ]
    weight_files = [
        "model.safetensors",
        "pytorch_model.bin",
    ]

    # sentence-transformers 的目录结构可能略有不同，因此放宽判断：
    # 只要有任一配置文件 + 任一权重文件，就认为可尝试本地加载
    has_config = _has_any_file(path, required_config_files)
    has_weight = _has_any_file(path, weight_files)

    # 某些 sentence-transformers 目录会把 transformer 放在 0_Transformer 子目录中
    if not has_weight:
        transformer_subdir = os.path.join(path, "0_Transformer")
        if _path_exists(transformer_subdir):
            has_weight = _has_any_file(transformer_subdir, weight_files)
            has_config = has_config or _has_any_file(
                transformer_subdir,
                ["config.json", "sentence_bert_config.json"]
            )

    return has_config and has_weight


def _is_valid_blip_dir(path: str) -> bool:
    """
    判断本地 BLIP 模型目录是否完整。
    必须至少同时具备：
    - 模型配置文件
    - 预处理配置文件
    - 权重文件
    """
    if not _path_exists(path):
        return False

    has_config = _has_any_file(path, ["config.json"])
    has_preprocessor = _has_any_file(
        path,
        ["preprocessor_config.json", "processor_config.json"]
    )
    has_weight = _has_any_file(path, ["model.safetensors", "pytorch_model.bin"])

    return has_config and has_preprocessor and has_weight


@st.cache_resource(show_spinner=False)
def load_embed_model() -> SentenceTransformer:
    """
    优先加载本地 embedding 模型；
    若本地模型目录不存在或不完整，则自动在线下载。
    """
    if _is_valid_embed_dir(LOCAL_EMBED_PATH):
        return SentenceTransformer(
            LOCAL_EMBED_PATH,
            local_files_only=True
        )

    if _path_exists(LOCAL_EMBED_PATH):
        _warn(
            f"本地 Embedding 模型目录存在但不完整：{LOCAL_EMBED_PATH}，"
            f"将改为在线下载：{EMBED_MODEL_NAME}"
        )
    else:
        _warn(
            f"本地 Embedding 模型未找到：{LOCAL_EMBED_PATH}，"
            f"将尝试在线下载：{EMBED_MODEL_NAME}"
        )

    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_blip_model() -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    """
    优先加载本地 BLIP 模型；
    若本地模型目录不存在或不完整，则自动在线下载。
    """
    if _is_valid_blip_dir(LOCAL_BLIP_PATH):
        processor = BlipProcessor.from_pretrained(
            LOCAL_BLIP_PATH,
            local_files_only=True
        )
        model = BlipForConditionalGeneration.from_pretrained(
            LOCAL_BLIP_PATH,
            local_files_only=True
        )
        model.eval()
        return processor, model

    if _path_exists(LOCAL_BLIP_PATH):
        _warn(
            f"本地 BLIP 模型目录存在但不完整：{LOCAL_BLIP_PATH}，"
            f"将改为在线下载：{BLIP_MODEL_NAME}"
        )
    else:
        _warn(
            f"本地 BLIP 模型未找到：{LOCAL_BLIP_PATH}，"
            f"将尝试在线下载：{BLIP_MODEL_NAME}"
        )

    processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
    model.eval()
    return processor, model