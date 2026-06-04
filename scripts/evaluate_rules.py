import argparse
import csv
import os
import sys
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.emotion import detect_emotion
from modules.grief_stage import detect_grief_stage
from modules.risk import detect_risk
from modules.strategy import choose_strategy


FIELDNAMES = [
    "case_id",
    "user_input",
    "system_grief_stage",
    "system_risk_level",
    "system_strategy",
    "emotion_sadness",
    "emotion_loneliness",
    "emotion_yearning",
    "emotion_guilt",
    "emotion_anger",
    "emotion_numbness",
    "expert_grief_stage",
    "expert_risk_level",
    "expert_strategy",
    "pbq_grief",
    "pbq_guilt",
    "pbq_anger",
    "icg_yearning",
    "icg_impairment",
    "notes",
]


def cohen_kappa(y_true: list[str], y_pred: list[str]) -> float | None:
    pairs = [(a, b) for a, b in zip(y_true, y_pred) if a and b]
    if not pairs:
        return None

    labels = sorted(set([a for a, _ in pairs] + [b for _, b in pairs]))
    total = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / total

    true_counts = Counter(a for a, _ in pairs)
    pred_counts = Counter(b for _, b in pairs)
    expected = sum((true_counts[label] / total) * (pred_counts[label] / total) for label in labels)

    if expected == 1:
        return 1.0
    return round((observed - expected) / (1 - expected), 4)


def confusion_counts(y_true: list[str], y_pred: list[str]) -> dict[tuple[str, str], int]:
    counts = Counter()
    for a, b in zip(y_true, y_pred):
        if a and b:
            counts[(a, b)] += 1
    return dict(counts)


def run_system(user_input: str) -> dict:
    emotion = detect_emotion(user_input)
    grief_stage = detect_grief_stage(user_input, emotion)
    risk = detect_risk(user_input)
    strategy = choose_strategy(grief_stage, risk, emotion, user_input)

    return {
        "system_grief_stage": grief_stage,
        "system_risk_level": risk.get("level", ""),
        "system_strategy": strategy.get("name", ""),
        "emotion_sadness": emotion.get("sadness", 0.0),
        "emotion_loneliness": emotion.get("loneliness", 0.0),
        "emotion_yearning": emotion.get("yearning", 0.0),
        "emotion_guilt": emotion.get("guilt", 0.0),
        "emotion_anger": emotion.get("anger", 0.0),
        "emotion_numbness": emotion.get("numbness", 0.0),
    }


def write_template(path: str):
    rows = [
        {
            "case_id": "case_001",
            "user_input": "我今天又想起它以前在门口等我的样子。",
            "expert_grief_stage": "",
            "expert_risk_level": "",
            "expert_strategy": "",
            "pbq_grief": "",
            "pbq_guilt": "",
            "pbq_anger": "",
            "icg_yearning": "",
            "icg_impairment": "",
            "notes": "",
        },
        {
            "case_id": "case_002",
            "user_input": "没有它我活不下去，我想去陪它。",
            "expert_grief_stage": "",
            "expert_risk_level": "",
            "expert_strategy": "",
            "pbq_grief": "",
            "pbq_guilt": "",
            "pbq_anger": "",
            "icg_yearning": "",
            "icg_impairment": "",
            "notes": "",
        },
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(input_path: str, output_path: str):
    with open(input_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    output_rows = []
    for idx, row in enumerate(rows, start=1):
        user_input = row.get("user_input", "")
        system = run_system(user_input)
        merged = {name: row.get(name, "") for name in FIELDNAMES}
        merged["case_id"] = merged["case_id"] or f"case_{idx:03d}"
        merged.update(system)
        output_rows.append(merged)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)

    expert_stage = [row.get("expert_grief_stage", "") for row in output_rows]
    system_stage = [row.get("system_grief_stage", "") for row in output_rows]
    expert_risk = [row.get("expert_risk_level", "") for row in output_rows]
    system_risk = [row.get("system_risk_level", "") for row in output_rows]
    expert_strategy = [row.get("expert_strategy", "") for row in output_rows]
    system_strategy = [row.get("system_strategy", "") for row in output_rows]

    summary = {
        "rows": len(output_rows),
        "grief_stage_kappa": cohen_kappa(expert_stage, system_stage),
        "risk_level_kappa": cohen_kappa(expert_risk, system_risk),
        "strategy_kappa": cohen_kappa(expert_strategy, system_strategy),
        "risk_confusion": confusion_counts(expert_risk, system_risk),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate PetEcho rule-based psychology modules.")
    parser.add_argument("--input", default="data/research/evaluation_cases.csv")
    parser.add_argument("--output", default="data/research/evaluation_results.csv")
    parser.add_argument("--template", action="store_true", help="Write a CSV template and exit.")
    args = parser.parse_args()

    if args.template:
        write_template(args.input)
        print(f"Wrote template: {args.input}")
        return

    summary = evaluate(args.input, args.output)
    print(f"Wrote results: {args.output}")
    print(summary)


if __name__ == "__main__":
    main()
