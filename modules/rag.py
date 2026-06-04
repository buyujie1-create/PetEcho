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

MEMORY_CUE_KEYWORDS = [
    "想起", "记得", "以前", "那时候", "回忆", "照片", "视频", "声音",
    "样子", "习惯", "喜欢", "总是", "每次", "一起", "陪我", "等我",
    "门口", "枕头边", "脚边", "散步", "阳台", "客厅", "下雨天", "蹭"
]

GENERIC_MEMORY_CUES = {"想起", "记得", "以前", "那时候", "回忆", "照片", "视频", "声音", "样子"}

ADVICE_KEYWORDS = [
    "怎么办", "怎么做", "有什么办法", "有什么方法", "怎么走出来", "如何走出来",
    "给我一些建议", "给点建议", "我该怎么", "能做什么", "怎么调整", "怎么缓解",
    "怎么改善", "有没有什么建议", "怎么才能"
]

GENERIC_DISTRESS_KEYWORDS = [
    "难过", "痛苦", "崩溃", "撑不住", "睡不着", "吃不下", "不想做",
    "麻木", "空落", "孤单", "自责", "愧疚", "生气", "不公平"
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


def _split_memory_clauses(text: str) -> list[str]:
    parts = re.split(r"[，。；！？!?]", text or "")
    return [part.strip() for part in parts if part.strip()]


def _select_relevant_clause(memory: str, query: str) -> str:
    clauses = _split_memory_clauses(memory)
    if not clauses:
        return (memory or "").strip()

    query_cues = _query_memory_cues(query)
    specific_query_cues = [cue for cue in query_cues if cue not in GENERIC_MEMORY_CUES]

    ranked = []
    for clause in clauses:
        score = 0
        for cue in specific_query_cues:
            if cue in clause:
                score += 5
        for cue in query_cues:
            if cue in clause:
                score += 2
        if _is_warm_memory(clause):
            score += 1
        if _is_heavy_memory(clause) and not _query_wants_heavy_memory(query):
            score -= 4
        ranked.append((score, clause))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


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


def _query_memory_cues(query: str) -> list[str]:
    query = _extract_current_query(query)
    return [cue for cue in MEMORY_CUE_KEYWORDS if cue in query]


def _query_is_advice_request(query: str) -> bool:
    query = _extract_current_query(query)
    return any(cue in query for cue in ADVICE_KEYWORDS)


def _query_is_generic_distress_only(query: str) -> bool:
    query = _extract_current_query(query)
    has_distress = any(cue in query for cue in GENERIC_DISTRESS_KEYWORDS)
    return has_distress and not _query_memory_cues(query) and not _query_is_simple_yearning(query)


def _memory_signature(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r"[，。！？!?；;、,.：:\-—“”\"'（）()【】\[\]\s]", "", text)
    return text[:48]


def _char_bigrams(text: str) -> set[str]:
    text = _memory_signature(text)
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _is_similar_memory(left: str, right: str) -> bool:
    left_sig = _memory_signature(left)
    right_sig = _memory_signature(right)
    if not left_sig or not right_sig:
        return False
    if len(left_sig) >= 16 and len(right_sig) >= 16 and (left_sig in right_sig or right_sig in left_sig):
        return True

    left_bigrams = _char_bigrams(left_sig)
    right_bigrams = _char_bigrams(right_sig)
    if not left_bigrams or not right_bigrams:
        return False
    overlap = len(left_bigrams & right_bigrams) / len(left_bigrams | right_bigrams)
    return overlap >= 0.72


def should_retrieve_memories(
    query: str,
    strategy: dict | None = None,
    emotion: dict | None = None,
) -> bool:
    strategy = strategy or {}
    emotion = emotion or {}
    user_query = _extract_current_query(query)
    if not user_query:
        return False

    usage = strategy.get("memory_usage", "low")
    memory_style = strategy.get("memory_style", "paraphrase_one_scene")
    retrieval_policy = strategy.get("retrieval_policy", "contextual")

    if usage == "none" or memory_style == "forbid" or retrieval_policy == "off":
        return False

    has_memory_cue = bool(_query_memory_cues(user_query))
    simple_yearning = _query_is_simple_yearning(user_query)
    advice_request = _query_is_advice_request(user_query)

    if retrieval_policy == "strict":
        return has_memory_cue or _query_wants_heavy_memory(user_query)

    if advice_request and not has_memory_cue:
        return False

    if usage in {"none_or_low", "low"} and not (has_memory_cue or simple_yearning):
        return False

    if _query_is_generic_distress_only(user_query):
        return False

    if emotion.get("yearning", 0.0) >= 0.55 and not advice_request:
        return True

    return has_memory_cue or simple_yearning


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


def retrieve_memories(
    query: str,
    top_k: int = 3,
    avoid_texts: list[str] | None = None,
    strategy: dict | None = None,
    emotion: dict | None = None,
    min_score: float | None = None,
) -> list[str]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(TEXTS_PATH):
        return []

    index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, "rb") as f:
        chunks = pickle.load(f)

    if not chunks:
        return []

    strategy = strategy or {}
    if not should_retrieve_memories(query, strategy=strategy, emotion=emotion):
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
    query_cues = _query_memory_cues(user_query)
    specific_query_cues = [cue for cue in query_cues if cue not in GENERIC_MEMORY_CUES]
    retrieval_policy = strategy.get("retrieval_policy", "contextual")
    memory_style = strategy.get("memory_style", "paraphrase_one_scene")

    if min_score is None:
        if retrieval_policy == "strict" or memory_style == "only_if_perfect_match":
            min_score = 0.38
        elif simple_yearning:
            min_score = 0.31
        else:
            min_score = 0.33

    avoid_texts = avoid_texts or []

    candidates = []
    for raw_score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx].strip()
        if not chunk:
            continue
        if any(_is_similar_memory(chunk, old_text) for old_text in avoid_texts):
            continue

        score = float(raw_score)

        lexical_bonus = 0.0
        for term in query_terms:
            if term in chunk:
                lexical_bonus += 0.035
        lexical_bonus = min(lexical_bonus, 0.14)

        cue_bonus = 0.0
        cue_overlap = 0
        specific_cue_overlap = 0
        for cue in query_cues:
            if cue in chunk:
                cue_overlap += 1
                cue_bonus += 0.06
                if cue in specific_query_cues:
                    specific_cue_overlap += 1
        cue_bonus = min(cue_bonus, 0.18)

        if specific_query_cues and specific_cue_overlap == 0:
            continue

        if retrieval_policy == "strict" and cue_overlap == 0 and lexical_bonus == 0 and not wants_heavy:
            continue

        warmth_bonus = 0.10 if _is_warm_memory(chunk) else 0.0
        specificity_bonus = 0.05 if len(chunk) >= 18 else 0.0

        heavy_penalty = 0.0
        if _is_heavy_memory(chunk) and not wants_heavy:
            heavy_penalty = 0.30

        if simple_yearning and _is_warm_memory(chunk):
            warmth_bonus += 0.08

        final_score = score + lexical_bonus + cue_bonus + warmth_bonus + specificity_bonus - heavy_penalty
        candidates.append((final_score, chunk))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected_pool = [(s, c) for s, c in candidates if s >= min_score]

    results = []
    seen = set()
    for _, chunk in selected_pool:
        chunk = _select_relevant_clause(chunk, user_query)
        signature = _memory_signature(chunk)
        if signature not in seen:
            seen.add(signature)
            results.append(chunk)
        if len(results) >= top_k:
            break

    return results
