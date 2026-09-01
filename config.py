"""
Model + lens registry for the filler-token / Jacobian-lens project.

Two entries:
  - "dev"      : Qwen3.5-4B. Small enough to develop and troubleshoot on a
                 single small GPU (or slowly on CPU). Has a published J-lens.
  - "deepseek" : DeepSeek V4 Flash (284B MoE). The real target. Ships already
                 quantized (FP4 experts + FP8 elsewhere, ~160 GB) and is loaded
                 as published — do NOT add bitsandbytes on top. Needs a rented
                 4x80GB box; see run_deepseek.md before using it.

IMPORTANT — model/lens matching:
  A Jacobian lens is a set of matrices tied to ONE model's residual-stream
  basis. You must apply a lens to the exact model it was fit on, or the
  readouts are meaningless. Each lens .pt stores its provenance; common.py
  reads lens["provenance"]["model_id"] and warns if it disagrees with hf_id
  below. If it warns, trust the provenance and update hf_id here.

Lens repo layout (huggingface.co/camilablank/workspace-lenses):
    <model>/j-lens/lens.pt   # Jacobian lens (this project)
    <model>/r-lens/lens.pt   # RelP lens (alternative; matched pair)
"""

from dataclasses import dataclass, field
from typing import Optional


LENS_REPO = "camilablank/workspace-lenses"


@dataclass
class ModelSpec:
    key: str
    hf_id: str                      # HuggingFace repo for the *weights*
    lens_dir: str                   # subdir inside LENS_REPO
    # --- how to load the weights (passed to AutoModelForCausalLM.from_pretrained) ---
    # "bfloat16" | "float16" | "float32", or "auto" to defer to the checkpoint's
    # own torch_dtype/quantization_config — which is what an already-quantized
    # checkpoint needs, since naming a dtype can force a silent upcast.
    dtype: str = "bfloat16"
    device_map: Optional[str] = None  # None = single device; "auto" = shard across GPUs
    # bitsandbytes nf4. ONLY for checkpoints published in bf16/fp16. On a
    # checkpoint that already carries a quantization_config, bnb must dequantize
    # to bf16 before re-quantizing, which OOMs; common.describe_checkpoint
    # refuses that combination up front.
    load_in_4bit: bool = False
    trust_remote_code: bool = True  # DeepSeek/Qwen custom modeling code
    # Park a VL checkpoint's vision encoder in host RAM after loading. The lens
    # only ever sees input_ids, so the encoder is dead weight on the GPU — but
    # it is ~1.3 of the dev model's 9.3 GB, which is what decides whether the
    # dev model fits on a 12 GB card. No-op on text-only models; ignored when
    # device_map is set (accelerate owns placement there). See
    # common.offload_vision_tower.
    offload_vision: bool = True
    notes: str = ""

    @property
    def lens_file(self) -> str:
        return f"{self.lens_dir}/j-lens/lens.pt"

    @property
    def relp_lens_file(self) -> str:
        return f"{self.lens_dir}/r-lens/lens.pt"


REGISTRY = {
    # ---------------------------------------------------------------------
    # Small dev model — run this first, iterate here, keep it cheap/fast.
    # ---------------------------------------------------------------------
    "dev": ModelSpec(
        key="dev",
        hf_id="Qwen/Qwen3.5-4B",     # if provenance warns, switch to Qwen/Qwen3.5-4B-Base
        lens_dir="qwen3.5-4b",
        dtype="bfloat16",
        device_map=None,             # fits on one GPU; on CPU set dtype="float32"
        load_in_4bit=False,
        offload_vision=True,         # VL checkpoint; the encoder is ~1.3GB we never use
        notes="9.3GB of bf16 weights on HF, of which ~1.3GB is a vision encoder the "
              "lens never touches -> ~8.0GB on the GPU with offload_vision. Fits a "
              "12GB card (RTX 4070 Super) with ~2GB to spare. CPU-runnable for smoke "
              "tests (slow). Ideal for debugging the pipeline. Do NOT quantize it for "
              "real numbers: the J matrices were fit in the bf16 residual basis.",
    ),

    # A couple of drop-in alternatives if you want a mid-size sanity check on a
    # single big GPU before paying for the 284B model. Both have published lenses.
    "dev-9b": ModelSpec(
        key="dev-9b", hf_id="Qwen/Qwen3.5-9B", lens_dir="qwen3.5-9b",
        notes="Fits on a single 24GB GPU in bf16-ish; good middle rung.",
    ),
    "gemma-27b": ModelSpec(
        key="gemma-27b", hf_id="google/gemma-3-27b-it", lens_dir="gemma-3-27b-it",
        device_map="auto", load_in_4bit=True,
        notes="Different family — useful for checking findings aren't Qwen-specific. "
              "Published in bf16, so bitsandbytes nf4 is legitimate here and is what "
              "gets 27B onto one card. Same caveat as the dev model though: the J "
              "matrices were fit in the bf16 residual basis, so treat 4-bit numbers "
              "as a shape check, not a headline. bf16 on one 80GB card (~54GB) if you "
              "have one.",
    ),

    # ---------------------------------------------------------------------
    # The real target — DeepSeek V4 Flash. Read run_deepseek.md first.
    # ---------------------------------------------------------------------
    "deepseek": ModelSpec(
        key="deepseek",
        hf_id="deepseek-ai/DeepSeek-V4-Flash",  # 284B MoE; verify the exact snapshot you rent for
        lens_dir="deepseek-v4-flash",
        dtype="auto",                # defer to the checkpoint's own FP8 quantization_config
        device_map="auto",           # shard across all visible GPUs
        load_in_4bit=False,          # it ALREADY ships quantized — see notes
        trust_remote_code=True,
        notes="284B MoE (13B active), 43 layers, d_model 4096, 256 experts. Ships "
              "pre-quantized: MoE expert weights FP4, attention/norm/router FP8, "
              "~160GB over 46 shards. Load it exactly as published — bitsandbytes on "
              "top would dequantize to bf16 (~568GB) and OOM. 160GB fits 4x80GB with "
              "2x headroom; go 8x80GB if transformers turns out to upcast the FP4 "
              "experts (check that for free before renting — see run_deepseek.md). "
              "Hopper or newer: Ampere has no FP8 units and would upcast. API access "
              "will NOT work: the lens needs the residual stream, which inference "
              "APIs do not expose.",
    ),
}


def get(key: str = "dev") -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key {key!r}. Options: {list(REGISTRY)}")
    return REGISTRY[key]
