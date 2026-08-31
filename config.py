"""
Model + lens registry for the filler-token / Jacobian-lens project.

Two entries:
  - "dev"      : Qwen3.5-4B. Small enough to develop and troubleshoot on a
                 single small GPU (or slowly on CPU). Has a published J-lens.
  - "deepseek" : DeepSeek V4 Flash (284B MoE). The real target. Needs rented
                 multi-GPU + 4-bit. See run_deepseek.md before using it.

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
    dtype: str = "bfloat16"         # "bfloat16" | "float16" | "float32"
    device_map: Optional[str] = None  # None = single device; "auto" = shard across GPUs
    load_in_4bit: bool = False      # bitsandbytes 4-bit (for the big model)
    trust_remote_code: bool = True  # DeepSeek/Qwen custom modeling code
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
        notes="~8GB in bf16. CPU-runnable for smoke tests (slow). Ideal for debugging the pipeline.",
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
        notes="Different family — useful for checking findings aren't Qwen-specific.",
    ),

    # ---------------------------------------------------------------------
    # The real target — DeepSeek V4 Flash. Read run_deepseek.md first.
    # ---------------------------------------------------------------------
    "deepseek": ModelSpec(
        key="deepseek",
        hf_id="deepseek-ai/DeepSeek-V4-Flash",  # 284B MoE; verify the exact snapshot you rent for
        lens_dir="deepseek-v4-flash",
        dtype="bfloat16",
        device_map="auto",           # shard across all visible GPUs
        load_in_4bit=True,           # ~4-bit to fit on a reasonable multi-GPU box
        trust_remote_code=True,
        notes="284B MoE. Needs multi-GPU (see run_deepseek.md). API access will NOT work: "
              "the lens needs the residual stream, which inference APIs do not expose.",
    ),
}


def get(key: str = "dev") -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key {key!r}. Options: {list(REGISTRY)}")
    return REGISTRY[key]
