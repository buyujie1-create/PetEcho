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


@st.cache_resource(show_spinner=False)
def load_embed_model() -> SentenceTransformer:
    """优先加载本地 embedding 模型；若本地不存在，则自动在线下载。"""
    if _path_exists(LOCAL_EMBED_PATH):
        return SentenceTransformer(
            LOCAL_EMBED_PATH,
            local_files_only=True
        )

    _warn(
        f"本地 Embedding 模型未找到：{LOCAL_EMBED_PATH}，"
        f"将尝试在线下载：{EMBED_MODEL_NAME}"
    )
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_blip_model() -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    """优先加载本地 BLIP 模型；若本地不存在，则自动在线下载。"""
    if _path_exists(LOCAL_BLIP_PATH):
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

    _warn(
        f"本地 BLIP 模型未找到：{LOCAL_BLIP_PATH}，"
        f"将尝试在线下载：{BLIP_MODEL_NAME}"
    )
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
    model.eval()
    return processor, model
