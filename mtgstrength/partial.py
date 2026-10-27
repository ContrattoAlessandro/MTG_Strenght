"""Fine-tune only the top k layers of a large text encoder on CPU.

The bottom L-k layers stay frozen, so their output for each card is computed
once and cached. During training those layers are replaced by no-ops and a
pre-hook on the first trainable layer swaps in the cached hidden states. The
rest of the forward pass (token embeddings, rotary tables, masks, per-layer
inputs, final norm) runs as the model defines it, so this works for any
encoder whose layers take the hidden states as their first argument.
``check_exact`` verifies that the result matches the full model.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch
from torch import nn


class _Stop(Exception):
    pass


class Skip(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


def load_encoder(name: str):
    """Return (tokenizer, text model with .layers, pooling mode)."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, dtype=torch.float32)
    text = text_submodule(model)
    pooling = "mean"
    try:
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(name, "1_Pooling/config.json")))
        if cfg.get("pooling_mode_lasttoken") or cfg.get("pooling_mode") == "lasttoken":
            pooling = "last"
    except Exception:
        pass
    text.eval()
    return tok, text, pooling


def text_submodule(model: nn.Module) -> nn.Module:
    """The module that owns the transformer layers and the token embeddings."""
    if hasattr(model, "layers") and hasattr(model, "embed_tokens"):
        return model
    for _, m in model.named_modules():
        if hasattr(m, "layers") and hasattr(m, "embed_tokens") and isinstance(m.layers, nn.ModuleList):
            return m
    raise ValueError("no text model with .layers and .embed_tokens found")


def pool(h: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "last":
        idx = mask.sum(1) - 1
        return h[torch.arange(len(h)), idx]
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


def _replace_first_arg(args, kwargs, value):
    if args:
        return (value, *args[1:]), kwargs
    kwargs = dict(kwargs)
    kwargs["hidden_states"] = value
    return args, kwargs


def _first_arg(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


@torch.no_grad()
def cache_hidden(name: str, texts: list[str], k: int, max_len: int, cache_dir: str,
                 batch_size: int = 16, log=print) -> list[np.ndarray]:
    """Hidden states entering the top k layers, one float16 array
    (n_tokens x d) per text. Cached on disk by model, k and texts."""
    key = hashlib.sha1(json.dumps([name, k, max_len, texts]).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{name.replace('/', '__')}_k{k}_{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        offs = np.concatenate([[0], np.cumsum(z["lens"])])
        flat = z["h"]
        return [flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
    tok, text, _ = load_encoder(name)
    n = len(text.layers)
    grabbed = {}

    def grab(_mod, args, kwargs):
        grabbed["h"] = _first_arg(args, kwargs)
        raise _Stop

    hook = text.layers[n - k].register_forward_pre_hook(grab, with_kwargs=True)
    ids = tok(texts, truncation=True, max_length=max_len)["input_ids"]
    order = np.argsort([len(x) for x in ids])
    out: list[np.ndarray | None] = [None] * len(texts)
    for bi, b in enumerate(np.array_split(order, max(1, len(order) // batch_size))):
        p = tok.pad({"input_ids": [ids[i] for i in b]}, return_tensors="pt", padding_side="right")
        try:
            text(input_ids=p["input_ids"], attention_mask=p["attention_mask"])
        except _Stop:
            pass
        hs = grabbed["h"]
        for j, i in enumerate(b):
            out[i] = hs[j, :len(ids[i])].to(torch.float16).numpy()
        if bi % 100 == 0:
            log(f"cache {name} k={k}: batch {bi}")
    hook.remove()
    lens = np.array([len(x) for x in out])
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, h=np.concatenate(out), lens=lens)
    return out


class TopLayers(nn.Module):
    """The top k layers of ``name`` (and everything after them), run on
    cached states. Only these parameters are trainable."""

    def __init__(self, name: str, k: int):
        super().__init__()
        self.tok, text, self.pooling = load_encoder(name)
        n = len(text.layers)
        for i in range(n - k):
            text.layers[i] = Skip()
        for p in text.parameters():
            p.requires_grad = False
        # Trainable: the top k layers and any module applied after them
        # (final norm, output projection).
        for p in text.layers[n - k:].parameters():
            p.requires_grad = True
        for nm, mod in text.named_children():
            if nm in ("norm", "final_norm", "embedding_projection", "final_layernorm"):
                for p in mod.parameters():
                    p.requires_grad = True
        self.text = text
        self._inject = None
        text.layers[n - k].register_forward_pre_hook(self._swap, with_kwargs=True)
        # Output width (can differ from hidden_size, e.g. an output projection).
        with torch.no_grad():
            ids = torch.tensor([[self.tok.pad_token_id or 0, 1]])
            d = text.config.hidden_size
            self._inject = torch.zeros(1, 2, d)
            self.hidden_size = text(input_ids=ids, attention_mask=torch.ones_like(ids)) \
                .last_hidden_state.shape[-1]
            self._inject = None
        self.text.train()

    def _swap(self, _mod, args, kwargs):
        return _replace_first_arg(args, kwargs, self._inject)

    def forward(self, input_ids: torch.Tensor, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._inject = h
        out = self.text(input_ids=input_ids, attention_mask=mask).last_hidden_state
        self._inject = None
        return pool(out, mask, self.pooling)


def pad_batch(states: list[np.ndarray], ids: list[list[int]], idx, pad_id: int = 0):
    L = max(len(states[i]) for i in idx)
    d = states[idx[0]].shape[1]
    h = torch.zeros(len(idx), L, d)
    m = torch.zeros(len(idx), L, dtype=torch.long)
    t = torch.full((len(idx), L), pad_id, dtype=torch.long)
    for j, i in enumerate(idx):
        n = len(states[i])
        h[j, :n] = torch.from_numpy(states[i].astype(np.float32))
        m[j, :n] = 1
        t[j, :n] = torch.tensor(ids[i][:n])
    return t, h, m


@torch.no_grad()
def check_exact(name: str, texts: list[str], k: int, cache_dir: str) -> float:
    """Relative max difference between the full model's pooled output and the
    top-k model run on cached states (float16 cache, so ~1e-3 is expected)."""
    tok, full, pooling = load_encoder(name)
    p = tok(texts, padding=True, return_tensors="pt", padding_side="right")
    ref = pool(full(**p).last_hidden_state, p["attention_mask"], pooling)
    states = cache_hidden(name, texts, k, 512, cache_dir)
    top = TopLayers(name, k).eval()
    ids = tok(texts)["input_ids"]
    t, h, m = pad_batch(states, ids, list(range(len(texts))), tok.pad_token_id or 0)
    got = top(t, h, m)
    return float((ref - got).abs().max() / ref.abs().max())
