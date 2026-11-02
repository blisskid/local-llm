"""Very simple training script: fine-tune GPT-2 on train.txt.

Run:  python3 train_simple.py
Then: python3 chat.py --checkpoint model_simple.pth
"""

import torch
import torch.nn.functional as F

from local_gpt import GPTModel, create_dataloader_v1

# ---- Settings (edit these) ----
DATA_FILE = "train.txt"
START_FROM = "gpt2_small.pth"    # made by: python3 load_pretrained.py   (set to None to train from scratch)
OUTPUT_FILE = "model_simple.pth"
EPOCHS = 3
BATCH_SIZE = 8
WINDOW = 256                     # tokens per training example
LEARNING_RATE = 5e-5             # small, so GPT-2's knowledge is kept

# 1. Pick the device (Apple GPU, NVIDIA GPU, or CPU)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# 2. Build the model (pretrained weights, or random if START_FROM is None)
if START_FROM:
    checkpoint = torch.load(START_FROM, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    config = {"vocab_size": 50257, "context_length": WINDOW, "emb_dim": 768, "n_heads": 12,
              "n_layers": 12, "drop_rate": 0.1, "qkv_bias": True}
    model = GPTModel(config)
model.to(device)

# 3. Load the text and split it: 90% to train on, 10% to check progress
text = open(DATA_FILE, encoding="utf-8").read()
cut = int(len(text) * 0.9)
train_loader = create_dataloader_v1(text[:cut], batch_size=BATCH_SIZE, max_length=WINDOW, stride=WINDOW)
val_loader = create_dataloader_v1(text[cut:], batch_size=BATCH_SIZE, max_length=WINDOW, stride=WINDOW,
                                  shuffle=False, drop_last=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)


def loss_of(x, y):
    logits = model(x.to(device))                                   # (batch, tokens, vocab)
    return F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten())


# 4. Train
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()                                                  # dropout on
    for step, (x, y) in enumerate(train_loader, 1):
        optimizer.zero_grad()                                      # clear old gradients
        loss = loss_of(x, y)                                       # how wrong is the model?
        loss.backward()                                            # work out how to fix each weight
        optimizer.step()                                           # nudge the weights
        if step % 100 == 0:
            print(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.3f}", flush=True)

    model.eval()                                                   # dropout off
    with torch.no_grad():
        val = sum(loss_of(x, y).item() for x, y in val_loader) / len(val_loader)
    print(f"== epoch {epoch} done | val loss {val:.3f}")

    if val < best_val:                                             # keep only the best model
        best_val = val
        torch.save({"config": config, "model_state_dict": model.state_dict()}, OUTPUT_FILE)
        print(f"   saved {OUTPUT_FILE}")

