# Local GPT

A small, readable PyTorch implementation of a GPT-2-style language model that you can
run, train and fine-tune on your own machine (CUDA, Apple Silicon MPS, or CPU).

The model code is in [`local_gpt.py`](local_gpt.py). The other scripts build on it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Files

| File | Purpose |
|---|---|
| `local_gpt.py` | Model (`GPTModel`), dataloader, text generation helpers |
| `load_pretrained.py` | Download open-source GPT-2 weights and convert them to a `.pth` checkpoint |
| `prepare_chat.py` | Build a chat-style `train.txt` from Alpaca, Dolly and OpenAssistant |
| `train_simple.py` | Minimal training script (settings are constants at the top) |
| `train.py` | Full training script with options (model size, resume from GPT-2, best-checkpoint saving) |
| `chat.py` | Chat loop with sampling, repetition penalty and stop token |
| `main.py` | Plain text generation from a prompt (greedy, fixed length) |

## Quick start: fine-tune GPT-2 into a simple chat model

```bash
# 1. Get pretrained GPT-2 weights (about 550 MB download) -> gpt2_small.pth
python3 load_pretrained.py

# 2. Build the training text -> train.txt
mkdir -p data
curl -L -o data/alpaca.json https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
curl -L -o data/dolly.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
curl -L -o data/oasst1.jsonl.gz https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz
python3 prepare_chat.py --max-answer-chars 1500 --max-chars 200000000

# 3. Fine-tune -> model_simple.pth
python3 train_simple.py

# 4. Chat
python3 chat.py --checkpoint model_simple.pth
```

Each answer in `train.txt` looks like this, which is the format `chat.py` expects:

```text
User: <question>
Assistant: <answer><|endoftext|>
```

## Scripts

### Generate text: `main.py`

```bash
python3 main.py --checkpoint gpt2_small.pth --prompt "Every effort moves you" --max-new-tokens 50
python3 main.py --tiny --device cpu --max-new-tokens 5     # quick pipeline check (random weights)
```

Uses greedy decoding for a fixed number of tokens, so it can repeat itself and never
stops early. Use `chat.py` for conversations. Run `python3 main.py --help` for all options.

### Chat: `chat.py`

```bash
python3 chat.py --checkpoint model_simple.pth
python3 chat.py --temperature 0.5 --repetition-penalty 1.2
```

Samples with temperature and top-k, penalizes repeated tokens and stops at `<|endoftext|>`.

### Train: `train.py`

```bash
# From scratch (tiny model, quick experiment)
python3 train.py --data train.txt --size tiny --epochs 20

# From scratch, larger models (see memory notes below)
python3 train.py --data train.txt --size small --epochs 6

# Fine-tune pretrained GPT-2
python3 train.py --data train.txt --init-from gpt2_small.pth --epochs 3
```

- Splits the text 90/10 into training and validation, and prints both losses each epoch.
- Saves `model.pth` only when the validation loss improves, so the file is the best epoch.
- Prints a sample generation after every epoch.

| `--size` | Parameters | Default batch | Notes (24 GB Mac, context 256) |
|---|---|---|---|
| `tiny` | 14M | 8 | Fast, for learning |
| `small` | 162M | 16 | About 1.5 s/step, about 16 GB GPU memory |
| `medium` | 405M | 4 | About 1.2 s/step, about 11 GB. Batch 8 runs out of memory and becomes very slow |

If the training loss keeps falling but the validation loss rises, the model is overfitting.
More data, a smaller model or fewer epochs will help.

### Pretrained weights: `load_pretrained.py`

```bash
python3 load_pretrained.py                  # GPT-2 small  (124M) -> gpt2_small.pth
python3 load_pretrained.py --model medium   # GPT-2 medium (355M) -> gpt2_medium.pth
```

Downloads OpenAI's GPT-2 weights from Hugging Face and maps them onto `GPTModel`.
Only GPT-2-family weights fit this architecture. Newer models (Llama, Qwen, Gemma, Mistral)
use a different design and would need their own model class.

## Checkpoints (`.pth` files)

A checkpoint is a dictionary saved with `torch.save`:

```python
torch.save({"config": config, "model_state_dict": model.state_dict()}, "model.pth")
```

`main.py`, `chat.py` and `train.py --init-from` load this format. All checkpoints must use
the GPT-2 tokenizer vocabulary (50257 tokens). Always load with `weights_only=True`, and only
open files you trust.

Checkpoints, `data/` and `train.txt` are large and listed in `.gitignore`.

## Notes

- A model trained from scratch on this data is a learning exercise. Expect short, often wrong
  answers. Fine-tuning pretrained GPT-2 gives noticeably better results.
- GPT-2 small is a 2019-era model. For a capable local assistant, run a modern instruct model
  with a tool such as Ollama, LM Studio or `mlx-lm`.
