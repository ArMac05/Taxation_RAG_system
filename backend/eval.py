import json
from pathlib import Path
from typing import TypedDict

from ask import TOP_k, retrieve
from textnorm import normalize


class EvalItem(TypedDict):
    question: str
    expected_sections: list[str]
    expected_phrase: str


def main() -> None:
    items: list[EvalItem] = json.loads(Path("eval.json").read_text())
    section_hits: int = 0
    phrase_hits: int = 0

    for item in items:
        results: list[tuple[str, str]] = retrieve(item["question"], k=TOP_k)
        sections: list[str] = [sec for sec, _ in results]
        texts: list[str] = [normalize(text) for _, text in results]

        section_ok: bool = any(s in sections for s in item["expected_sections"])
        phrase_ok: bool = any(normalize(item["expected_phrase"]) in t for t in texts)
        section_hits += section_ok
        phrase_hits += phrase_ok

        print(f"[section {'OK ' if section_ok else 'MISS'}] "
              f"[phrase {'OK ' if phrase_ok else 'MISS'}] {item['question']}")

    n: int = len(items)
    print(f"\nSection hit rate@{TOP_k}: {section_hits}/{n} = {section_hits / n:.0%}")
    print(f"Phrase hit rate@{TOP_k}:  {phrase_hits}/{n} = {phrase_hits / n:.0%}")


if __name__ == "__main__":
    main()