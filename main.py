"""Run text generation with the GPT implementation in local_gpt.py."""

import argparse
from pathlib import Path

import tiktoken
import torch

from local_gpt import (
    GPTModel,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)


GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def select_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use --device cpu or auto.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available; use --device cpu or auto.")
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Every effort moves you")
    parser.add_argument("--max-new-tokens", type=nonnegative_int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Local checkpoint containing config and model_state_dict, or a raw state dict.",
    )
    parser.add_argument("--tiny", action="store_true", help="Use a small model for a quick smoke test.")
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    try:
        device = select_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(args.seed)
    config = dict(GPT_CONFIG)
    if args.tiny:
        config.update(context_length=128, emb_dim=64, n_heads=4, n_layers=2)

    state_dict = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" in checkpoint:
            config = checkpoint.get("config", config)
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    if config["vocab_size"] != GPT_CONFIG["vocab_size"]:
        parser.error("This runner requires the GPT-2 tokenizer vocabulary (50257 tokens).")

    model = GPTModel(config)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    else:
        print("Untrained model: random weights will produce meaningless text.")
    model.to(device).eval()
    print(f"Device: {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = text_to_token_ids(args.prompt, tokenizer).to(device)
    with torch.inference_mode():
        output = generate_text_simple(
            model, token_ids,
            max_new_tokens=args.max_new_tokens,
            context_size=config["context_length"],
        )
    print(token_ids_to_text(output.cpu(), tokenizer))


if __name__ == "__main__":
    main()
