import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.emotion import detect_emotion
from modules.grief_stage import detect_grief_stage
from modules.llm_api import call_llm
from modules.prompt_builder import build_prompt
from modules.rag import should_retrieve_memories
from modules.risk import detect_risk
from modules.strategy import choose_strategy


PROFILE = {
    "pet_name": "咪咪",
    "pet_personality": "很黏人、安静、会轻轻蹭主人。",
    "pet_appearance": "一只浅色长毛猫，眼睛圆圆的，看起来安静温柔。",
}

DEMO_MEMORY = "每次我回家，咪咪都会先跑到门口等我，然后慢慢蹭我的腿。"

CASES = [
    {
        "name": "generic_distress_without_forced_memory",
        "text": "我今天还是很难过，什么都不想说。",
        "memories": [],
    },
    {
        "name": "specific_memory_natural_use",
        "text": "我又想起它以前在门口等我的样子，真的很想它。",
        "memories": [DEMO_MEMORY],
    },
    {
        "name": "advice_with_low_burden_action",
        "text": "我每天都很想它，也不知道该怎么办，有什么办法能让我好一点吗？",
        "memories": [],
    },
]


def run_case(case: dict) -> dict:
    user_input = case["text"]
    emotion = detect_emotion(user_input)
    grief_stage = detect_grief_stage(user_input, emotion)
    risk = detect_risk(user_input)
    strategy = choose_strategy(grief_stage, risk, emotion, user_input)
    allow_memory = should_retrieve_memories(user_input, strategy, emotion)
    memory_context = case["memories"] if allow_memory else []
    prompt = build_prompt(
        pet_profile=PROFILE,
        memory_context=memory_context,
        emotion=emotion,
        grief_stage=grief_stage,
        risk=risk,
        strategy=strategy,
        user_input=user_input,
    )
    reply = call_llm(prompt)
    return {
        "case": case["name"],
        "stage": grief_stage,
        "strategy": strategy.get("name"),
        "allow_memory": allow_memory,
        "memory_count": len(memory_context),
        "reply": reply,
    }


def main():
    for case in CASES:
        result = run_case(case)
        print("\n==", result["case"], "==")
        print("stage:", result["stage"])
        print("strategy:", result["strategy"])
        print("allow_memory:", result["allow_memory"], "memory_count:", result["memory_count"])
        print("reply:", result["reply"])


if __name__ == "__main__":
    main()
