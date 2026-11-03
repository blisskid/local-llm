"""Compare checkpoints: perplexity on held-out text, plus answers to fixed questions.

Usage:
    python3 evaluate.py model.pth gpt2_small.pth
    python3 evaluate.py model.pth gpt2_small.pth --no-answers      # perplexity only
"""

import argparse
import math
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from chat import END, generate_reply
from local_gpt import GPTModel, create_dataloader_v1, text_to_token_ids

QUESTIONS = [
    "What is the capital of France?",
    "What is the biggest lake in the world?",
    "Give three tips for staying healthy.",
    "Explain what a computer is in one sentence.",
    "Write a short greeting for a friend.",
    "What is 2 plus 3?",
    "Name three primary colors.",
    "Why is the sky blue?",
]


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = GPTModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def perplexity(model, cfg, text, device, window=256, batch_size=8):
    window = min(window, cfg["context_length"])
    loader = create_dataloader_v1(text, batch_size=batch_size, max_length=window, stride=window,
                                  shuffle=False, drop_last=False)
    total, count = 0.0, 0
    for x, y in loader:
        logits = model(x.to(device))
        total += F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten(), reduction="sum").item()
        count += y.numel()
    loss = total / count
    return loss, math.exp(loss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--chat-data", type=Path, default=Path("train.txt"),
                        help="Chat-format text. The last 10%% is used as held-out data.")
    parser.add_argument("--book-data", type=Path, default=Path("data/book.txt"),
                        help="Plain prose, to check general English ability.")
    parser.add_argument("--max-chars", type=int, default=1_000_000,
                        help="Limit each evaluation text (for speed).")
    parser.add_argument("--no-answers", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy, so answers are repeatable.")
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    texts = {}
    if args.chat_data.is_file():
        full = args.chat_data.read_text(encoding="utf-8")
        texts["chat (held-out)"] = full[int(0.9 * len(full)):][:args.max_chars]
    if args.book_data.is_file():
        texts["book prose"] = args.book_data.read_text(encoding="utf-8")[:args.max_chars]

    tokenizer = tiktoken.get_encoding("gpt2")
    end_id = tokenizer.encode(END, allowed_special={END})[0]
    results, answers = {}, {}

    for path in args.checkpoints:
        print(f"\n=== {path} ===", flush=True)
        model, cfg = load_model(path, device)
        params = sum(p.numel() for p in model.parameters())
        results[path.name] = {"params": params}
        for name, text in texts.items():
            loss, ppl = perplexity(model, cfg, text, device)
            results[path.name][name] = (loss, ppl)
            print(f"{name:18s} loss {loss:.3f}  perplexity {ppl:.1f}", flush=True)
        if not args.no_answers:
            answers[path.name] = []
            for q in QUESTIONS:
                prompt = f"User: {q}\nAssistant:"
                ids = text_to_token_ids(prompt, tokenizer).to(device)
                tokens = generate_reply(model, ids, 60, cfg["context_length"], end_id,
                                        args.temperature, 40, args.repetition_penalty)
                answers[path.name].append(tokenizer.decode(tokens).split("\nUser:")[0].strip())
        del model

    print("\n\n## Summary (lower perplexity is better)\n")
    names = list(texts)
    print("| model | params | " + " | ".join(f"{n} ppl" for n in names) + " |")
    print("|---|---|" + "---|" * len(names))
    for model_name, r in results.items():
        cells = " | ".join(f"{r[n][1]:.1f}" for n in names)
        print(f"| {model_name} | {r['params'] / 1e6:.0f}M | {cells} |")

    if answers:
        print("\n## Answers (greedy, repetition penalty)\n")
        for i, q in enumerate(QUESTIONS):
            print(f"Q: {q}")
            for model_name in answers:
                print(f"  [{model_name}] {answers[model_name][i][:200]!r}")
            print()


if __name__ == "__main__":
    main()

