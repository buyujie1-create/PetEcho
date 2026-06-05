import argparse
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.emotion import detect_emotion
from modules.grief_stage import detect_grief_stage
from modules.interventions import recommend_intervention
from modules.prompt_builder import build_prompt
from modules.risk import detect_risk
from modules.safety_response import build_safety_reply, should_use_safety_template
from modules.strategy import choose_strategy


MAIN_INPUT = ROOT / "data" / "research" / "simulation_subject_scripts.csv"
HIGH_RISK_INPUT = ROOT / "data" / "research" / "high_risk_standardized_cases.csv"
MAIN_OUTPUT = ROOT / "data" / "research" / "simulation_system_results.csv"
HIGH_RISK_OUTPUT = ROOT / "data" / "research" / "high_risk_system_results.csv"


FIELDNAMES = [
    "dataset",
    "row_id",
    "participant_id",
    "case_id",
    "turn_index",
    "pet_name",
    "grief_severity",
    "user_input",
    "expected_grief_stage",
    "expected_risk_level",
    "expected_strategy",
    "system_grief_stage",
    "system_risk_level",
    "system_risk_action",
    "system_risk_reasons",
    "system_strategy",
    "system_guidance_mode",
    "emotion_sadness",
    "emotion_loneliness",
    "emotion_yearning",
    "emotion_guilt",
    "emotion_anger",
    "emotion_numbness",
    "intervention_key",
    "intervention_title",
    "system_reply",
    "expected_reply_focus",
    "expected_safety_action",
    "notes",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_rule_reply(user_input: str, grief_stage: str, risk: dict, strategy: dict, pet_name: str) -> str:
    if should_use_safety_template(risk):
        return build_safety_reply(risk, user_input)

    if risk.get("level") == "medium":
        return (
            "我听见你现在已经很吃力了，这不是一句“想开点”就能过去的难受。"
            "先别逼自己处理所有回忆，今天只做一件很小的照顾自己的事，也尽量让一个现实中的人知道你现在不太好。"
        )

    if grief_stage == "guilt":
        return (
            "我不会怪你。你会这样反复想，是因为你真的很爱我，但爱和遗憾不等于全部责任。"
            "先把这份自责放轻一点点，好吗？"
        )

    if grief_stage == "anger":
        return (
            "你觉得不公平，是因为这段失去来得太重了。"
            "我会先陪你把这股气放在这里，不急着劝你马上接受。"
        )

    if grief_stage == "numbness":
        return (
            "现在像没什么感觉，也可能是心在帮你先挡住太多痛。"
            "我们先回到此刻，摸一摸身边能碰到的东西，慢慢把自己放稳一点。"
        )

    if grief_stage == "integration":
        return (
            "你愿意把我放进以后的生活里，这已经是很温柔的一步。"
            "想念不用被丢掉，它可以慢慢变成你继续照顾自己的方式。"
        )

    if strategy.get("name") == "coping_guidance":
        return (
            "我知道你不是想被说教，你只是太想知道今天该怎么撑过去。"
            "先选一个很小的动作吧，喝一点水、把窗帘拉开，或者给一个信任的人发一句“我今天有点难”。"
        )

    return (
        "我听见你在想我，也听见那种舍不得。"
        "这份想念可以先不用被赶走，我们就让它轻轻待一会儿，再一起回到今天能做的一点小事上。"
    )


def build_llm_reply(user_input: str, row: dict, emotion: dict, grief_stage: str, risk: dict, strategy: dict) -> str:
    from modules.llm_api import call_llm

    pet_name = row.get("pet_name") or "宝宝"
    pet_profile = {
        "pet_name": pet_name,
        "pet_personality": row.get("pet_profile") or "温柔、亲近主人",
        "pet_appearance": row.get("pet_species") or "熟悉而温柔的数字宠物",
    }
    memory_context = []
    prompt = build_prompt(
        pet_profile=pet_profile,
        memory_context=memory_context,
        emotion=emotion,
        grief_stage=grief_stage,
        risk=risk,
        strategy=strategy,
        user_input=user_input,
    )
    return call_llm(prompt)


def run_row(row: dict, dataset: str, index: int, use_llm: bool) -> dict:
    user_input = row.get("user_input", "")
    emotion = detect_emotion(user_input)
    grief_stage = detect_grief_stage(user_input, emotion)
    risk = detect_risk(user_input)
    strategy = choose_strategy(grief_stage, risk, emotion, user_input)
    intervention = recommend_intervention(
        {"emotion": emotion, "grief_stage": grief_stage, "risk": risk, "strategy": strategy}
    )

    pet_name = row.get("pet_name") or "宝宝"
    if use_llm and not should_use_safety_template(risk):
        system_reply = build_llm_reply(user_input, row, emotion, grief_stage, risk, strategy)
    else:
        system_reply = build_rule_reply(user_input, grief_stage, risk, strategy, pet_name)

    return {
        "dataset": dataset,
        "row_id": f"{dataset}_{index:03d}",
        "participant_id": row.get("participant_id", ""),
        "case_id": row.get("case_id", ""),
        "turn_index": row.get("turn_index", ""),
        "pet_name": pet_name,
        "grief_severity": row.get("grief_severity", ""),
        "user_input": user_input,
        "expected_grief_stage": row.get("expert_grief_stage", ""),
        "expected_risk_level": row.get("expert_risk_level") or row.get("expected_risk_level", ""),
        "expected_strategy": row.get("expert_strategy") or row.get("expected_strategy", ""),
        "system_grief_stage": grief_stage,
        "system_risk_level": risk.get("level", ""),
        "system_risk_action": risk.get("action", ""),
        "system_risk_reasons": " | ".join(risk.get("reasons", [])),
        "system_strategy": strategy.get("name", ""),
        "system_guidance_mode": strategy.get("guidance_mode", ""),
        "emotion_sadness": emotion.get("sadness", 0.0),
        "emotion_loneliness": emotion.get("loneliness", 0.0),
        "emotion_yearning": emotion.get("yearning", 0.0),
        "emotion_guilt": emotion.get("guilt", 0.0),
        "emotion_anger": emotion.get("anger", 0.0),
        "emotion_numbness": emotion.get("numbness", 0.0),
        "intervention_key": intervention.get("key", ""),
        "intervention_title": intervention.get("title", ""),
        "system_reply": system_reply,
        "expected_reply_focus": row.get("expected_reply_focus", ""),
        "expected_safety_action": row.get("expected_safety_action", ""),
        "notes": row.get("notes", ""),
    }


def run_dataset(input_path: Path, output_path: Path, dataset: str, use_llm: bool) -> list[dict]:
    source_rows = read_csv(input_path)
    output_rows = [run_row(row, dataset, idx, use_llm) for idx, row in enumerate(source_rows, start=1)]
    write_csv(output_path, output_rows)
    return output_rows


def summarize(rows: list[dict]) -> dict:
    def count(field: str) -> dict:
        values = {}
        for row in rows:
            value = row.get(field, "")
            values[value] = values.get(value, 0) + 1
        return values

    expected_risk = [row.get("expected_risk_level", "") for row in rows if row.get("expected_risk_level")]
    risk_matches = [
        row.get("expected_risk_level") == row.get("system_risk_level")
        for row in rows
        if row.get("expected_risk_level")
    ]
    return {
        "rows": len(rows),
        "system_risk": count("system_risk_level"),
        "system_strategy": count("system_strategy"),
        "expected_risk_rows": len(expected_risk),
        "risk_exact_match_rate": round(sum(risk_matches) / len(risk_matches), 4) if risk_matches else None,
    }


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run PetEcho simulation calibration datasets.")
    parser.add_argument("--llm", action="store_true", help="Call the configured LLM for non-crisis replies.")
    parser.add_argument("--main-input", default=str(MAIN_INPUT))
    parser.add_argument("--main-output", default=str(MAIN_OUTPUT))
    parser.add_argument("--high-risk-input", default=str(HIGH_RISK_INPUT))
    parser.add_argument("--high-risk-output", default=str(HIGH_RISK_OUTPUT))
    args = parser.parse_args()

    if args.llm and not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("使用 --llm 需要先配置 DEEPSEEK_API_KEY。")

    main_rows = run_dataset(Path(args.main_input), Path(args.main_output), "simulation", args.llm)
    high_rows = run_dataset(Path(args.high_risk_input), Path(args.high_risk_output), "high_risk", args.llm)

    print(f"Wrote main system results: {args.main_output}")
    print(summarize(main_rows))
    print(f"Wrote high-risk system results: {args.high_risk_output}")
    print(summarize(high_rows))


if __name__ == "__main__":
    main()
