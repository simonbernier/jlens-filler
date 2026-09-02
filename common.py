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
        lens.apply(model, prompt, positions=[...], use_jacobian=True|False,
                   max_seq_len=...)
            -> (lens_logits, model_logits, _)
               lens_logits is {layer: Tensor[num_positions, vocab]}
    use_jacobian=False gives the LOGIT-LENS baseline (skips the transport step).

Four guards live here, each protecting against a failure that is silent or
expensive rather than obvious:

  * the `encode` override in `load_model` — jlens tokenizes prompts with
    add_special_tokens=True, which on DeepSeek adds a second BOS in front of
    the one the chat template already rendered; the lens would then read a
    different sequence from the one the answer is generated from. `apply_lens`
    checks that the lens saw exactly the prompt's tokens.

  * `fit_seq_len` — jlens truncates at 512 tokens from the RIGHT by default, and
    our positions are negative indices computed on the untruncated text, so a
    long prompt silently shifts every readout instead of erroring. It sizes the
    window to the prompt instead.
  * `offload_vision_tower` — the dev model is a VL checkpoint whose vision
    encoder the lens never touches. Parking it in host RAM is what makes the dev
    model fit on a 12 GB card.
  * `describe_checkpoint` — refuses to stack bitsandbytes on a checkpoint that
    already ships quantized, which would dequantize it first and OOM a rented
    multi-GPU box after you have paid to download it.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import transformers
import jlens
from huggingface_hub import hf_hub_download

from config import ModelSpec, LENS_REPO


# Slack over the prompt's own token count, for a BOS (and friends) that jlens's
# `encode` may prepend that our offset-based counting does not see.
BOS_HEADROOM = 4

# Where HF keeps the vision encoder on the VL checkpoints we touch. Qwen VL uses
# `visual` (hung off the *ForConditionalGeneration or off `.model`); other
# families use `vision_tower` / `vision_model`.
VISION_ATTRS = ("visual", "vision_tower", "vision_model")


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def offload_vision_tower(hf) -> float:
    """Move any vision encoder off the GPU. Returns the GiB of VRAM freed.

    The dev model (Qwen3.5-4B) is a vision-language checkpoint: ~1.3 GB of its
    9.3 GB of bf16 weights is a vision encoder. The lens only ever sees
    `input_ids`, so that encoder is never called — but `from_pretrained` still
    puts it on the GPU. On a 12 GB card those 1.3 GB are the difference between
    a comfortable run and an OOM, so park it in host RAM.

    A no-op on text-only models (DeepSeek V4 Flash), and skipped entirely when
    accelerate placed the model (`device_map`), where moving a submodule by hand
    would fight the dispatch hooks.
    """
    freed = 0.0
    for holder in (hf, getattr(hf, "model", None)):
        if holder is None:
            continue
        for attr in VISION_ATTRS:
            tower = getattr(holder, attr, None)
            if not isinstance(tower, torch.nn.Module):
                continue
            on_gpu = [p for p in tower.parameters() if p.device.type == "cuda"]
            if not on_gpu:
                continue
            freed += sum(p.numel() * p.element_size() for p in on_gpu) / 2**30
            tower.to("cpu")
    if freed:
        torch.cuda.empty_cache()
        print(f"[load_model] vision tower -> CPU, freed {freed:.2f} GiB VRAM")
    return freed


def report_vram(note: str = "") -> None:
    """Print allocated / total VRAM. Cheap insurance before a long run."""
    if not torch.cuda.is_available():
        return
    used = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"[vram] {used:.2f} GiB allocated ({reserved:.2f} reserved) of "
          f"{total:.1f} GiB on {torch.cuda.get_device_name(0)} {note}")


def describe_checkpoint(spec: ModelSpec) -> dict:
    """Print the checkpoint's OWN quantization; refuse to stack bitsandbytes on it.

    DeepSeek V4 Flash ships pre-quantized — MoE expert weights in FP4, the
    attention/norm/router weights in FP8, ~160 GB across 46 shards. bitsandbytes
    cannot quantize that further: it would dequantize to bf16 first (284B params
    -> ~568 GB) and OOM any box you would plausibly rent, *after* you have paid
    for the download. The same holds for any checkpoint carrying a
    `quantization_config`: load it as published and let its own scheme do the
    work.

    Returns the checkpoint's quantization_config as a dict, `{}` if it has none.
    A config that cannot be read (offline, missing token) warns rather than
    blocks — this is a guard, not a gate.
    """
    try:
        cfg = transformers.AutoConfig.from_pretrained(
            spec.hf_id, trust_remote_code=spec.trust_remote_code)
    except Exception as exc:                       # offline, gated, bad id
        warnings.warn(f"could not read the config for {spec.hf_id} ({exc}); "
                      f"skipping the quantization preflight", stacklevel=2)
        return {}

    qc = getattr(cfg, "quantization_config", None) or {}
    if qc and not isinstance(qc, dict):
        qc = qc.to_dict() if hasattr(qc, "to_dict") else {"quant_method": str(qc)}
    if not qc:
        return {}

    print(f"[checkpoint] ships quantized: quant_method={qc.get('quant_method')} "
          f"fmt={qc.get('fmt')} weight_block_size={qc.get('weight_block_size')}")
    if spec.load_in_4bit:
        raise ValueError(
            f"{spec.hf_id} is ALREADY quantized ({qc.get('quant_method')}), but "
            f"config.py sets load_in_4bit=True for '{spec.key}'. bitsandbytes would "
            f"have to dequantize the whole checkpoint first and will OOM. Set "
            f"load_in_4bit=False and dtype='auto' so it loads as published."
        )
    return qc


def patch_fp8_tp_plan_bug() -> None:
    """Work around a transformers 5.16.x regression that breaks every FP8 MoE load.

    `FineGrainedFP8HfQuantizer.update_tp_plan` fetches per-kernel overrides with
    `_impl_tp_layer_overrides.get(impl)` — `None` for any experts implementation
    other than "deepgemm_megamoe" — and then calls `.get` on it, so loading
    DeepSeek V4 Flash dies with "'NoneType' object has no attribute 'get'"
    before a single shard is read. Fixed on transformers main (`.get(impl, {})`)
    but not yet released. With no overrides the fixed code leaves the plan as
    it was, so falling back to the untouched config is the same outcome; the
    plan only matters for tensor-parallel sharding, and we place layers with
    device_map="auto" instead. Harmless on versions without the bug.
    """
    try:
        from transformers.quantizers.quantizer_finegrained_fp8 import (
            FineGrainedFP8HfQuantizer)
    except ImportError:
        return
    original = FineGrainedFP8HfQuantizer.update_tp_plan
    if getattr(original, "_patched", False):
        return

    def update_tp_plan(self, config):
        try:
            return original(self, config)
        except AttributeError:        # the 5.16.x bug: no overrides for this impl
            return config

    update_tp_plan._patched = True
    FineGrainedFP8HfQuantizer.update_tp_plan = update_tp_plan


def collapse_hyper_connection_streams(hf) -> None:
    """Make jlens read a single residual vector per position on DeepSeek V4.

    V4 blocks use manifold-constrained hyper-connections: the residual is a
    stack of `hc_mult` (= 4) parallel streams, shape [B, S, 4, D], kept through
    every layer. Each block collapses the streams to one D-vector to feed
    attention/MLP and writes its output back across them; at the very end
    `model.hc_head` collapses them once more for the final norm + lm_head.
    jlens records block outputs and assumes [B, S, D], so on V4 it would
    unembed each stream separately and return [positions, 4, vocab].

    The published V4 lens is d_model=4096 — fit on one collapsed vector per
    position — so the streams must be collapsed BEFORE the lens sees them (the
    collapse weights depend on the streams, so it does not commute with the
    J transport). We use the model's own read-out, `hc_head`: at the last
    layer that is exactly what lm_head unembeds, and at earlier layers it keeps
    the logit lens's meaning of "what the model would say if it stopped here".
    The collapse happens on the recorder's copy; the block output the next
    layer consumes is untouched. No-op on models whose residual is 3-D.

    If the lens's provenance turns out to record a different collapse (e.g. the
    next block's own `attn_hc` pre-mapping, or a plain mean over streams),
    swap the one line marked below.
    """
    hc_head = getattr(getattr(hf, "model", None), "hc_head", None)
    if hc_head is None:
        return
    import jlens.lens
    import jlens.hooks

    class CollapsingRecorder(jlens.hooks.ActivationRecorder):
        def _make_hook(self, index):
            store = super()._make_hook(index)

            def hook(module, inputs, output):
                store(module, inputs, output)
                h = self.activations[index]
                if h.dim() == 4:                       # [B, S, hc_mult, D]
                    with torch.no_grad():
                        self.activations[index] = hc_head(h)   # <- the collapse
            return hook

    jlens.lens.ActivationRecorder = CollapsingRecorder
    print(f"[load_model] hyper-connection residual ({hf.config.hc_mult} streams): "
          f"collapsing with model.hc_head before every lens readout")


def load_model(spec: ModelSpec):
    """Load HF weights per `spec` and wrap them for the lens. Returns (model, hf, tok)."""
    describe_checkpoint(spec)
    patch_fp8_tp_plan_bug()
    kw = dict(trust_remote_code=spec.trust_remote_code)

    if spec.load_in_4bit:
        if spec.dtype == "auto":
            raise ValueError("load_in_4bit needs a concrete dtype for "
                             "bnb_4bit_compute_dtype, not 'auto'")
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, spec.dtype),
        )
    else:
        # torch_dtype is the widely-supported name; newer transformers also accept `dtype`.
        # "auto" defers to the checkpoint's own torch_dtype / quantization_config, which is
        # what a pre-quantized checkpoint needs — naming a dtype can force a silent upcast.
        kw["torch_dtype"] = "auto" if spec.dtype == "auto" else getattr(torch, spec.dtype)

    if spec.device_map:
        kw["device_map"] = spec.device_map

    print(f"[load_model] {spec.hf_id}  (4bit={spec.load_in_4bit}, device_map={spec.device_map})")
    hf = transformers.AutoModelForCausalLM.from_pretrained(spec.hf_id, **kw)

    # If we did not let accelerate place the model, move it ourselves.
    if not spec.device_map and not spec.load_in_4bit:
        hf = hf.to("cuda" if torch.cuda.is_available() else "cpu")
    hf.eval()

    # Reclaim the vision encoder's VRAM (see offload_vision_tower). Only safe on
    # the hand-placed path: with device_map, accelerate owns placement.
    if spec.offload_vision and not spec.device_map:
        offload_vision_tower(hf)
    report_vram("after load")

    tok = transformers.AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=spec.trust_remote_code
    )
    model = jlens.from_hf(hf, tok)
    collapse_hyper_connection_streams(hf)

    # jlens's own encode() tokenizes with add_special_tokens=True (and forces
    # add_bos_token on). Our prompts are rendered chat templates that already
    # carry whatever BOS the model wants — DeepSeek's template starts with one,
    # Qwen's has none — so that would prepend a SECOND BOS on DeepSeek, and the
    # lens would read a different sequence from the one `answer()` generates
    # from. Tokenize the rendered text verbatim instead, exactly as the greedy
    # generation in 20 does; apply_lens checks the token count agrees.
    def encode(text: str, *, max_length: int = 512) -> torch.Tensor:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_length,
                  add_special_tokens=False).input_ids
        return ids.to(model.input_device)
    model.encode = encode
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
    # Everything else the lens file says about itself (fit settings, notes) —
    # on DeepSeek V4 this is where a residual-stream collapse choice would show.
    extra = {k: v for k, v in prov.items()
             if k not in ("model_id", "target_layer", "n_prompts")
             and not isinstance(v, torch.Tensor)}
    if extra:
        print(f"[provenance] other fields: {extra}")
    return prov


# --------------------------------------------------------------------------- #
# Applying the lens + decoding
# --------------------------------------------------------------------------- #
def fit_seq_len(n_tok: int, max_seq_len: Optional[int] = None) -> int:
    """Resolve the truncation length for an n_tok-token prompt. Never truncates silently.

    `jlens.HFLensModel.encode` tokenizes with `truncation=True,
    max_length=max_seq_len` (default **512**), and HF truncates from the RIGHT.
    Our readout positions are NEGATIVE indices computed on the untruncated text,
    so a truncated prompt does not raise: it shifts every position by however
    many tokens were cut and quietly produces a wrong heatmap. A chat prompt with
    five few-shot examples is already past 512 by k=25, so the default is a trap.

    `max_seq_len=None` (the default everywhere in this repo) sizes the window to
    the prompt, so nothing is ever cut. An explicit value is honoured, but raises
    rather than truncate.
    """
    needed = n_tok + BOS_HEADROOM
    if max_seq_len is None:
        return needed
    if needed > max_seq_len:
        raise ValueError(
            f"prompt is {n_tok} tokens but max_seq_len={max_seq_len}: jlens would "
            f"truncate it from the right, and every negative position would then "
            f"point at the wrong token. Pass max_seq_len=None to size the window "
            f"to the prompt, or raise it above {needed}."
        )
    return max_seq_len


def apply_lens(lens, model, prompt: str, positions: Sequence[int],
               use_jacobian: bool, max_seq_len: Optional[int] = None):
    """Return {layer: Tensor[num_positions, vocab]} for one prompt.

    use_jacobian=True  -> Jacobian lens
    use_jacobian=False -> logit-lens baseline (same code path, transport skipped)
    max_seq_len=None   -> size the context window to this prompt (see fit_seq_len)
    """
    n_tok = len(model.tokenizer(prompt, add_special_tokens=False).input_ids)
    lens_logits, _model_logits, input_ids = lens.apply(
        model, prompt,
        positions=list(positions),
        use_jacobian=use_jacobian,
        max_seq_len=fit_seq_len(n_tok, max_seq_len),
    )
    # The lens must have run on exactly the tokens the positions were computed
    # on (see the encode override in load_model): a stray BOS or a truncation
    # would shift every negative position by one and never raise on its own.
    if input_ids.shape[-1] != n_tok:
        raise RuntimeError(
            f"the lens ran on {input_ids.shape[-1]} tokens but the prompt is "
            f"{n_tok}: the readout positions no longer line up with the text")
    return lens_logits


def read(lens, model, prompt: str, positions: Sequence[int],
         max_seq_len: Optional[int] = None) -> Dict[str, dict]:
    """Convenience: run both the Jacobian lens and the logit-lens baseline."""
    return {
        "jlens": apply_lens(lens, model, prompt, positions, True, max_seq_len),
        "logit": apply_lens(lens, model, prompt, positions, False, max_seq_len),
    }


def topk_tokens(logits_row: torch.Tensor, tok, k: int = 8) -> List[Tuple[str, float]]:
    """Top-k (token_string, score) for a single [vocab] logit row."""
    vals, idx = logits_row.float().topk(k)
    toks = [tok.decode([int(i)]) for i in idx.tolist()]
    return list(zip(toks, [float(v) for v in vals.tolist()]))
