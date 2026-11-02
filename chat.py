"""Simple chat loop for a checkpoint trained with train.py on chat-style data.

Uses sampling (temperature + top-k) and a repetition penalty instead of greedy
decoding, which otherwise tends to loop ("Open Assistant, Open Assistant, ...").

Usage: python3 chat.py --checkpoint model.pth
"""

import argparse
from pathlib import Path

import tiktoken
import torch

from local_gpt import GPTModel, text_to_token_ids

END = "<|endoftext|>"


def generate_reply(model, ids, max_new_tokens, context_size, end_id,
                   temperature, top_k, repetition_penalty):
    """Generate tokens one by one; stop at <|endoftext|>. Returns only the new token ids."""
    new_tokens = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(ids[:, -context_size:])[0, -1, :]   # (vocab_size,)

        # Make tokens that were already used less likely (avoids loops)
        if repetition_penalty != 1.0:
            for token in set(ids[0].tolist()):
                logits[token] = (logits[token] / repetition_penalty if logits[token] > 0
                                 else logits[token] * repetition_penalty)

        if temperature > 0:
            # Keep only the top-k candidates, then sample among them
            top_logits, top_idx = torch.topk(logits, top_k)
            probs = torch.softmax(top_logits / temperature, dim=-1)
            next_id = top_idx[torch.multinomial(probs, num_samples=1)]
        else:
            next_id = torch.argmax(logits, keepdim=True)        # greedy

        if next_id.item() == end_id:
            break
        new_tokens.append(next_id.item())
        ids = torch.cat((ids, next_id.view(1, 1)), dim=1)
    return new_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("model.pth"))
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Higher = more random, 0 = greedy.")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.3,
                        help="1.0 = off; 1.1-1.5 reduces repetition.")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = GPTModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    end_id = tokenizer.encode(END, allowed_special={END})[0]

    print("Type a message (empty line to quit).")
    while True:
        user = input("You: ").strip()
        if not user:
            break
        prompt = f"User: {user}\nAssistant:"
        ids = text_to_token_ids(prompt, tokenizer).to(device)
        tokens = generate_reply(model, ids, args.max_new_tokens, cfg["context_length"], end_id,
                                args.temperature, args.top_k, args.repetition_penalty)
        reply = tokenizer.decode(tokens).split("\nUser:")[0].strip()
        print(f"Bot: {reply}")


if __name__ == "__main__":
    main()
