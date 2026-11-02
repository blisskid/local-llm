"""Download open-source GPT-2 weights (OpenAI, via Hugging Face) and convert them
into a checkpoint for the GPTModel in local_gpt.py.

Usage:
    python3 load_pretrained.py                      # GPT-2 small (124M) -> gpt2_small.pth
    python3 load_pretrained.py --model medium       # GPT-2 medium (355M) -> gpt2_medium.pth

Then:
    python3 main.py --checkpoint gpt2_small.pth --prompt "Every effort moves you"
    python3 train.py --data train.txt --init-from gpt2_small.pth   # fine-tune
"""

import argparse
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file

from local_gpt import GPTModel

MODELS = {
    # name: (Hugging Face repo, emb_dim, n_layers, n_heads)
    "small": ("openai-community/gpt2", 768, 12, 12),
    "medium": ("openai-community/gpt2-medium", 1024, 24, 16),
    "large": ("openai-community/gpt2-large", 1280, 36, 20),
}


def download(repo, dest):
    if dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
    print(f"Downloading {url}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def copy_(param, value):
    if param.shape != value.shape:
        raise ValueError(f"Shape mismatch: model {tuple(param.shape)} vs weights {tuple(value.shape)}")
    with torch.no_grad():
        param.copy_(value)


def convert(hf, model, n_layers):
    """Copy Hugging Face GPT-2 tensors into our model.

    Hugging Face stores linear layers as (in, out) "Conv1D" weights, ours are
    (out, in), hence the transposes. Q, K, V are stored fused in c_attn.
    """
    hf = {k.removeprefix("transformer."): v for k, v in hf.items()}

    copy_(model.tok_emb.weight, hf["wte.weight"])
    copy_(model.pos_emb.weight, hf["wpe.weight"])

    for i, block in enumerate(model.trf_blocks):
        p = f"h.{i}."
        q_w, k_w, v_w = hf[p + "attn.c_attn.weight"].chunk(3, dim=1)
        q_b, k_b, v_b = hf[p + "attn.c_attn.bias"].chunk(3, dim=0)
        copy_(block.att.W_query.weight, q_w.T); copy_(block.att.W_query.bias, q_b)
        copy_(block.att.W_key.weight, k_w.T);   copy_(block.att.W_key.bias, k_b)
        copy_(block.att.W_value.weight, v_w.T); copy_(block.att.W_value.bias, v_b)
        copy_(block.att.out_proj.weight, hf[p + "attn.c_proj.weight"].T)
        copy_(block.att.out_proj.bias, hf[p + "attn.c_proj.bias"])
        copy_(block.ff.layers[0].weight, hf[p + "mlp.c_fc.weight"].T)
        copy_(block.ff.layers[0].bias, hf[p + "mlp.c_fc.bias"])
        copy_(block.ff.layers[2].weight, hf[p + "mlp.c_proj.weight"].T)
        copy_(block.ff.layers[2].bias, hf[p + "mlp.c_proj.bias"])
        copy_(block.norm1.scale, hf[p + "ln_1.weight"]); copy_(block.norm1.shift, hf[p + "ln_1.bias"])
        copy_(block.norm2.scale, hf[p + "ln_2.weight"]); copy_(block.norm2.shift, hf[p + "ln_2.bias"])

    copy_(model.final_norm.scale, hf["ln_f.weight"])
    copy_(model.final_norm.shift, hf["ln_f.bias"])
    copy_(model.out_head.weight, hf["wte.weight"])   # GPT-2 ties the output head to the token embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="small")
    parser.add_argument("--output", type=Path, default=None, help="Default: gpt2_<model>.pth")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/pretrained"))
    args = parser.parse_args()

    repo, emb_dim, n_layers, n_heads = MODELS[args.model]
    output = args.output or Path(f"gpt2_{args.model}.pth")
    cfg = {"vocab_size": 50257, "context_length": 1024, "emb_dim": emb_dim,
           "n_heads": n_heads, "n_layers": n_layers, "drop_rate": 0.0, "qkv_bias": True}

    weights_file = args.cache_dir / f"gpt2_{args.model}.safetensors"
    download(repo, weights_file)

    model = GPTModel(cfg)
    convert(load_file(weights_file), model, n_layers)
    torch.save({"config": cfg, "model_state_dict": model.state_dict()}, output)
    print(f"Saved {output} ({sum(p.numel() for p in model.parameters()):,} parameters)")


if __name__ == "__main__":
    main()

