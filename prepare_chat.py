"""Turn Alpaca + Dolly + OpenAssistant data into a simple chat-style train.txt.

Each example becomes:
    User: <question>
    Assistant: <answer><|endoftext|>

Expects (download once):
    data/alpaca.json  - https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
    data/dolly.jsonl  - https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
    data/oasst1.jsonl.gz - https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz

Usage: python3 prepare_chat.py [--max-chars 20000000]
"""

import argparse
import gzip
import json
import random
from pathlib import Path


def load_alpaca(path):
    for ex in json.loads(path.read_text(encoding="utf-8")):
        question = ex["instruction"]
        if ex["input"].strip():          # fold the extra input text into the question
            question += "\n" + ex["input"].strip()
        yield question, ex["output"]


def load_dolly(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        question = ex["instruction"]
        if ex["context"].strip():        # fold the context passage into the question
            question += "\n" + ex["context"].strip()
        yield question, ex["response"]


def load_oasst(path):
    """OpenAssistant trees: yield (prompter message, best-ranked assistant reply), English only."""
    def walk(node):
        if node.get("deleted") or node.get("lang") != "en":
            return
        if node["role"] == "prompter":
            replies = [r for r in node.get("replies", [])
                       if r["role"] == "assistant" and not r.get("deleted")]
            if replies:
                best = min(replies, key=lambda r: r.get("rank") if r.get("rank") is not None else 99)
                yield node["text"], best["text"]
        for child in node.get("replies", []):
            yield from walk(child)

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield from walk(json.loads(line)["prompt"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpaca", type=Path, default=Path("data/alpaca.json"))
    parser.add_argument("--dolly", type=Path, default=Path("data/dolly.jsonl"))
    parser.add_argument("--oasst", type=Path, default=Path("data/oasst1.jsonl.gz"))
    parser.add_argument("--output", type=Path, default=Path("train.txt"))
    parser.add_argument("--max-chars", type=int, default=20_000_000,
                        help="Stop after this many characters (default: effectively all data).")
    parser.add_argument("--max-answer-chars", type=int, default=1000,
                        help="Skip long answers; small models learn short ones better.")
    parser.add_argument("--max-question-chars", type=int, default=1500,
                        help="Skip very long questions/passages (context window is only 256 tokens).")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    pairs = []
    for path, loader in ((args.alpaca, load_alpaca), (args.dolly, load_dolly),
                         (args.oasst, load_oasst)):
        if path.is_file():
            pairs.extend(loader(path))
        else:
            print(f"Skipping missing file: {path}")

    # Shuffle so the 90/10 train/val split in train.py sees both datasets
    random.Random(args.seed).shuffle(pairs)

    parts, total = [], 0
    for question, answer in pairs:
        question, answer = question.strip(), answer.strip()
        if (not question or not answer or len(answer) > args.max_answer_chars
                or len(question) > args.max_question_chars):
            continue
        block = f"User: {question}\nAssistant: {answer}<|endoftext|>\n"
        parts.append(block)
        total += len(block)
        if total >= args.max_chars:
            break

    args.output.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {len(parts):,} examples ({total:,} characters) to {args.output}")


if __name__ == "__main__":
    main()
