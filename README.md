# Local GPT

Run the PyTorch GPT implementation in `local_gpt.py`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --prompt "Once upon a time" --max-new-tokens 50
```

The runner automatically selects CUDA, Apple Silicon MPS, or CPU. Override with
`--device cpu` if needed. The first tokenizer use may download its vocabulary.

**No trained weights are included.** By default this creates a randomly initialized
GPT-2-sized model, so the generated text will be meaningless until you load trained
weights. For a lightweight check of the generation pipeline:

```bash
python main.py --tiny --device cpu --max-new-tokens 5
```

To run weights saved from this implementation:

```bash
python main.py --checkpoint model.pth --prompt "Once upon a time"
```

Save your trained model with its configuration:

```python
torch.save({"config": config, "model_state_dict": model.state_dict()}, "model.pth")
```

Raw state dictionaries are also supported with the default architecture (or
`--tiny` for the tiny architecture). Checkpoints with a `config` use that
configuration instead. All checkpoints must use the GPT-2 tokenizer vocabulary.
Public GPT-2 checkpoints need their weight names and layouts converted to this
implementation first; `local_gpt.load_weights_into_gpt` supports the book's
parameter-dictionary format. This runner does not perform that conversion or
training. Generation uses greedy next-token selection, not a chat interface.

Run `python main.py --help` for all options.
