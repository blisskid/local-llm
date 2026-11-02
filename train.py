"""Train the GPT implementation in local_gpt.py on a local text file.

Example:
    python train.py --data train.txt --tiny --epochs 20
    python main.py --checkpoint model.pth --prompt "Once upon a time"
"""

import argparse
import time
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from local_gpt import (
    GPTModel,
    create_dataloader_v1,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
TINY_CONFIG = dict(GPT_CONFIG, context_length=128, emb_dim=128, n_heads=4, n_layers=4)

# Model size presets: (config, default learning rate, default batch size)
# Measured on an M-series Mac with 24 GB (context 256):
#   tiny   ~14M params
#   small  ~162M params, batch 16: ~1.5 s/step, ~16 GB GPU memory
#   medium ~405M params, batch 4:  ~1.2 s/step, ~11 GB GPU memory
#          (batch 8 hit ~21 GB and slowed ~60x from memory pressure, so keep it at 4)
PRESETS = {
    "tiny": (TINY_CONFIG, 4e-4, 8),
    "small": (dict(GPT_CONFIG, context_length=256, emb_dim=768, n_heads=12, n_layers=12), 3e-4, 16),
    "medium": (dict(GPT_CONFIG, context_length=256, emb_dim=1024, n_heads=16, n_layers=24), 2e-4, 4),
}


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_loss(model, x, y, device):
    x, y = x.to(device), y.to(device)
    logits = model(x)  # (batch, seq_len, vocab_size)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = sum(batch_loss(model, x, y, device).item() for x, y in loader)
    model.train()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Plain-text training file.")
    parser.add_argument("--output", type=Path, default=Path("model.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None, help="Default depends on --size.")
    parser.add_argument("--lr", type=float, default=None, help="Default depends on --size.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prompt", default="Every effort moves you",
                        help="Prompt used for a sample generation after each epoch.")
    parser.add_argument("--size", choices=sorted(PRESETS), default="tiny", help="Model size preset.")
    parser.add_argument("--tiny", action="store_true", help="Alias for --size tiny.")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="Start from a pretrained checkpoint (e.g. gpt2_small.pth from "
                             "load_pretrained.py) instead of random weights. Overrides --size.")
    parser.add_argument("--window", type=int, default=None,
                        help="Tokens per training example. Default: model context length "
                             "(256 when using --init-from, to save memory).")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Data file does not exist: {args.data}")
    if args.init_from and not args.init_from.is_file():
        parser.error(f"Checkpoint does not exist: {args.init_from}")

    torch.manual_seed(args.seed)
    init_state = None
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        cfg, init_state = dict(ckpt["config"]), ckpt["model_state_dict"]
        cfg["drop_rate"] = 0.1                         # dropout is only active while training
        default_lr, default_batch = 5e-5, 8            # small learning rate: don't erase what GPT-2 knows
        args.window = args.window or 256
    else:
        cfg, default_lr, default_batch = PRESETS["tiny" if args.tiny else args.size]
        cfg = dict(cfg)
    args.lr = args.lr or default_lr
    args.batch_size = args.batch_size or default_batch
    window = args.window or cfg["context_length"]
    if window > cfg["context_length"]:
        parser.error(f"--window cannot exceed the model context length ({cfg['context_length']})")
    device = select_device()

    text = args.data.read_text(encoding="utf-8")
    split = int(args.train_ratio * len(text))
    loader_args = dict(batch_size=args.batch_size, max_length=window,
                       stride=window, num_workers=0)
    train_loader = create_dataloader_v1(text[:split], drop_last=True, shuffle=True, **loader_args)
    val_loader = create_dataloader_v1(text[split:], drop_last=False, shuffle=False, **loader_args)
    if len(train_loader) == 0 or len(val_loader) == 0:
        parser.error("Not enough text for the chosen context length / batch size.")

    model = GPTModel(cfg)
    if init_state is not None:
        model.load_state_dict(init_state)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Device: {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    best_val = float("inf")
    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        start = time.time()
        for step, (x, y) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            loss = batch_loss(model, x, y, device)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if step % 200 == 0:
                print(f"  step {step}/{len(train_loader)} | loss {running / step:.3f} "
                      f"| {time.time() - start:.0f}s", flush=True)
        val = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | train {running / len(train_loader):.3f} | val {val:.3f}")

        model.eval()
        ids = text_to_token_ids(args.prompt, tokenizer).to(device)
        with torch.no_grad():
            out = generate_text_simple(model, ids, 30, cfg["context_length"])
        print("  sample:", token_ids_to_text(out, tokenizer).replace("\n", " "))
        model.train()

        # Save only when validation loss improves, so the file is the best epoch
        if val < best_val:
            best_val = val
            torch.save({"config": cfg, "model_state_dict": model.state_dict()}, args.output)
            print(f"  new best val {val:.3f} -> saved {args.output}")

    print(f"Done. Best val loss {best_val:.3f} (checkpoint: {args.output})")


if __name__ == "__main__":
    main()

