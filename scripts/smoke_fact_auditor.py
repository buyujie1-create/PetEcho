import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.llm_api import _fact_fit_violations, _strip_fact_violations


PROMPT_WITH_MEMORY = """
【可用记忆线索】
- 它都会先跑到门口等我

使用原则：

【用户输入】
我又想起它以前在门口等我的样子。
""".strip()

PROMPT_WITHOUT_MEMORY = """
【可用记忆线索】
- 本轮没有足够贴切的记忆线索：禁止提具体过去画面

使用原则：

【用户输入】
我今天很难过，什么都不想说。
""".strip()


def main():
    unsupported = "那时候它坐在门口，听见你的脚步声，就像在说你终于回来了。"
    violations = _fact_fit_violations(PROMPT_WITH_MEMORY, unsupported)
    print({"case": "unsupported_expansion", "violations": violations})
    if not violations:
        raise SystemExit("expected unsupported memory expansion to be flagged")

    stripped = _strip_fact_violations(PROMPT_WITH_MEMORY, unsupported)
    print({"case": "strip_unsupported", "result": stripped})
    if "脚步" in stripped or "像在说" in stripped:
        raise SystemExit("unsupported detail was not stripped")

    forbidden = "像以前那样，它每次都会蹭蹭你。"
    violations = _fact_fit_violations(PROMPT_WITHOUT_MEMORY, forbidden)
    print({"case": "forbidden_memory", "violations": violations})
    if not violations:
        raise SystemExit("expected forbidden memory use to be flagged")


if __name__ == "__main__":
    main()
