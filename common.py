"""
Shared loaders and helpers for J-lens work.

The heavy lifting lives in the `jlens` library (github.com/anthropics/jacobian-lens).
This module is a thin, well-commented wrapper so the experiment scripts stay short:

    model, hf, tok = load_model(spec)          # HF weights -> jlens LensModel
    lens = load_lens(spec, kind="j")           # download + load the Jacobian lens
    check_provenance(spec)                      # guard against model/lens mismatch
    out = read(lens, model, prompt, positions) # {"jlens": {...}, "logit": {...}}

Confirmed jlens API used here:
    jlens.from_hf(hf_model, tokenizer) -> LensModel
    jlens.JacobianLens.from_pretrained(repo, filename=...) -> lens
        attrs: .jacobians {layer: [d,d]}, .source_layers, .n_prompts, .d_model
        lens.apply(model, prompt, positions=[...], use_jacobian=True|False)
            -> (lens_logits, model_logits, _)
               lens_logits is {layer: Tensor[num_positions, vocab]}
    use_jacobian=False gives the LOGIT-LENS baseline (skips the transport step).
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Sequence, Tuple

import torch
import transformers
import jlens
from huggingface_hub import hf_hub_download

from config import ModelSpec, LENS_REPO


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(spec: ModelSpec):
    """Load HF weights per `spec` and wrap them for the lens. Returns (model, hf, tok)."""
    kw = dict(trust_remote_code=spec.trust_remote_code)

    if spec.load_in_4bit:
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, spec.dtype),
        )
    else:
        # torch_dtype is the widely-supported name; newer transformers also accept `dtype`.
        kw["torch_dtype"] = getattr(torch, spec.dtype)

    if spec.device_map:
        kw["device_map"] = spec.device_map

    print(f"[load_model] {spec.hf_id}  (4bit={spec.load_in_4bit}, device_map={spec.device_map})")
    hf = transformers.AutoModelForCausalLM.from_pretrained(spec.hf_id, **kw)

    # If we did not let accelerate place the model, move it ourselves.
    if not spec.device_map and not spec.load_in_4bit:
        hf = hf.to("cuda" if torch.cuda.is_available() else "cpu")
    hf.eval()

    tok = transformers.AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=spec.trust_remote_code
    )
    model = jlens.from_hf(hf, tok)
    return model, hf, tok


# --------------------------------------------------------------------------- #
# Lens loading + provenance guard
# --------------------------------------------------------------------------- #
def load_lens(spec: ModelSpec, kind: str = "j"):
    """kind='j' -> Jacobian lens; kind='r' -> RelP lens."""
    filename = spec.lens_file if kind == "j" else spec.relp_lens_file
    print(f"[load_lens] {LENS_REPO}/{filename}")
    return jlens.JacobianLens.from_pretrained(LENS_REPO, filename=filename)


def peek_provenance(spec: ModelSpec, kind: str = "j") -> dict:
    """Load just the raw .pt to read its provenance dict (model_id, target_layer, ...)."""
    filename = spec.lens_file if kind == "j" else spec.relp_lens_file
    path = hf_hub_download(LENS_REPO, filename=filename)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return raw.get("provenance", {}) or {}


def check_provenance(spec: ModelSpec, kind: str = "j") -> dict:
    """Warn loudly if the configured hf_id disagrees with the lens's own provenance."""
    prov = peek_provenance(spec, kind)
    prov_id = prov.get("model_id")
    if prov_id and prov_id != spec.hf_id:
        warnings.warn(
            f"\n*** MODEL/LENS MISMATCH ***\n"
            f"  config hf_id      : {spec.hf_id}\n"
            f"  lens provenance   : {prov_id}\n"
            f"A lens is only valid on the model it was fit on. Trust the provenance:\n"
            f"  edit config.py so REGISTRY['{spec.key}'].hf_id = '{prov_id}'.\n",
            stacklevel=2,
        )
    else:
        print(f"[provenance] model_id={prov_id!r}  target_layer={prov.get('target_layer')}  "
              f"n_prompts={prov.get('n_prompts')}  -> matches config" if prov_id else
              "[provenance] no model_id recorded in lens; can't auto-verify.")
    return prov


# --------------------------------------------------------------------------- #
# Applying the lens + decoding
# --------------------------------------------------------------------------- #
def apply_lens(lens, model, prompt: str, positions: Sequence[int], use_jacobian: bool):
    """Return {layer: Tensor[num_positions, vocab]} for one prompt.

    use_jacobian=True  -> Jacobian lens
    use_jacobian=False -> logit-lens baseline (same code path, transport skipped)
    """
    try:
        lens_logits, _model_logits, _ = lens.apply(
            model, prompt, positions=list(positions), use_jacobian=use_jacobian
        )
    except TypeError:
        # Older/newer signatures may not take use_jacobian on apply(); fall back.
        lens_logits, _model_logits, _ = lens.apply(model, prompt, positions=list(positions))
    return lens_logits


def read(lens, model, prompt: str, positions: Sequence[int]) -> Dict[str, dict]:
    """Convenience: run both the Jacobian lens and the logit-lens baseline."""
    return {
        "jlens": apply_lens(lens, model, prompt, positions, use_jacobian=True),
        "logit": apply_lens(lens, model, prompt, positions, use_jacobian=False),
    }


def topk_tokens(logits_row: torch.Tensor, tok, k: int = 8) -> List[Tuple[str, float]]:
    """Top-k (token_string, score) for a single [vocab] logit row."""
    vals, idx = logits_row.float().topk(k)
    toks = [tok.decode([int(i)]) for i in idx.tolist()]
    return list(zip(toks, [float(v) for v in vals.tolist()]))


def numeric_token_rank(logits_row: torch.Tensor, tok, target_int: int) -> int:
    """Rank (0 = top) at which the exact string of `target_int` appears among sorted logits.
    Returns a large sentinel if the target's token id can't be resolved to a single token."""
    ids = tok.encode(str(target_int), add_special_tokens=False)
    if len(ids) != 1:
        # multi-token number; use " N" variant which often maps to one token
        ids = tok.encode(f" {target_int}", add_special_tokens=False)
        if len(ids) != 1:
            return 10**9
    order = torch.argsort(logits_row.float(), descending=True)
    pos = (order == ids[0]).nonzero(as_tuple=True)[0]
    return int(pos.item()) if len(pos) else 10**9
