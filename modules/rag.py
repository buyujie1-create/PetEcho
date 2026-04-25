import os
import re
import pickle
from typing import List

import faiss
import numpy as np

from modules.model_loader import load_embed_model

INDEX_PATH = "data/faiss.index"
TEXTS_PATH = "data/texts.pkl"

HEAVY_MEMORY_KEYWORDS = [
    "去世", "离开", "最后一次", "医院", "生病", "抢救",
    "掉下", "走丢", "找了一夜", "不在了", "再也"
]

WARM_MEMORY_KEYWORDS = [
    "陪", "一起", "门口", "枕头边", "脚边", "回家", "起床", "散步",
    "阳台", "客厅", "睡", "蹭", "等我", "每天", "傍晚", "下雨天"
]

_EMBED_MODEL = None


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def _extract_current_query(query: str) -> str:
    query = query or ""
    if "【用户当前输入】" in query:
        query = query.split("【用户当前输入】")[-1]
    if "【用户输入】" in query:
        query = query.split("【用户输入】")[-1]
    return _normalize_text(query)


def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = load_embed_model()
    return _EMBED_MODEL


def split_text(text: str, target_len: int = 70, max_len: int = 120) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    raw_sentences = re.split(r'(?<=[。！？!?；;\n])', text)
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not raw_sentences:
        raw_sentences = [text[i:i + target_len] for i in range(0, len(text), target_len)]

    chunks = []
    current = ""
    for sent in raw_sentences:
        if len(current) + len(sent) <= max_len:
            current += sent
        else:
            if current:
                chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())

    merged = []
    for chunk in chunks:
        if merged and len(chunk) < 16:
            merged[-1] += chunk
        else:
            merged.append(chunk)

    unique = []
    seen = set()
    for chunk in merged:
        chunk = chunk.strip()
        if chunk and chunk not in seen:
            seen.add(chunk)
            unique.append(chunk)
    return unique


def _extract_query_terms(query: str) -> list[str]:
    query = _extract_current_query(query)
    terms = re.findall(r'[\u4e00-\u9fff]{2,8}', query)
    stop_terms = {
        "最近对话", "用户当前输入", "用户输入", "数字宠物", "现在", "今天", "真的",
        "有点", "一下", "就是", "这个", "那个", "一下子", "感觉"
    }
    result = []
    for term in terms:
        if term not in stop_terms and len(term) >= 2:
            result.append(term)
    return sorted(set(result), key=len, reverse=True)[:8]


def _is_heavy_memory(text: str) -> bool:
    return any(k in text for k in HEAVY_MEMORY_KEYWORDS)


def _is_warm_memory(text: str) -> bool:
    return any(k in text for k in WARM_MEMORY_KEYWORDS)


def _query_wants_heavy_memory(query: str) -> bool:
    query = _extract_current_query(query)
    cues = [
        "最后", "那天", "走丢", "医院", "生病", "离开", "去世",
        "不在了", "怎么走的", "当时", "后来", "抢救"
    ]
    return any(c in query for c in cues)


def _query_is_simple_yearning(query: str) -> bool:
    query = _extract_current_query(query)
    yearning_cues = ["想你", "想它", "我很想", "好想", "想念", "怀念", "空落落"]
    heavy_cues = ["最后", "医院", "走丢", "去世", "不在了", "那天"]
    return any(c in query for c in yearning_cues) and not any(c in query for c in heavy_cues)


def build_vectorstore(text: str):
    chunks = split_text(text)
    if not chunks:
        return

    embed_model = get_embed_model()
    embeddings = embed_model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def retrieve_memories(query: str, top_k: int = 3) -> list[str]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(TEXTS_PATH):
        return []

    index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, "rb") as f:
        chunks = pickle.load(f)

    if not chunks:
        return []

    embed_model = get_embed_model()
    user_query = _extract_current_query(query)
    query_vec = embed_model.encode(
        [user_query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    search_k = min(max(top_k * 4, 8), len(chunks))
    scores, indices = index.search(query_vec, search_k)

    query_terms = _extract_query_terms(user_query)
    wants_heavy = _query_wants_heavy_memory(user_query)
    simple_yearning = _query_is_simple_yearning(user_query)

    candidates = []
    for raw_score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx].strip()
        if not chunk:
            continue

        score = float(raw_score)

        lexical_bonus = 0.0
        for term in query_terms:
            if term in chunk:
                lexical_bonus += 0.035
        lexical_bonus = min(lexical_bonus, 0.14)

        warmth_bonus = 0.10 if _is_warm_memory(chunk) else 0.0
        specificity_bonus = 0.05 if len(chunk) >= 18 else 0.0

        heavy_penalty = 0.0
        if _is_heavy_memory(chunk) and not wants_heavy:
            heavy_penalty = 0.22

        if simple_yearning and _is_warm_memory(chunk):
            warmth_bonus += 0.08

        final_score = score + lexical_bonus + warmth_bonus + specificity_bonus - heavy_penalty
        candidates.append((final_score, chunk))

    candidates.sort(key=lambda x: x[0], reverse=True)

    strong_candidates = [(s, c) for s, c in candidates if s >= 0.28]
    selected_pool = strong_candidates if strong_candidates else candidates[: max(top_k, 2)]

    results = []
    seen = set()
    for _, chunk in selected_pool:
        if chunk not in seen:
            seen.add(chunk)
            results.append(chunk)
        if len(results) >= top_k:
            break

    return results
