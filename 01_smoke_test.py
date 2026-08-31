"""
Smoke test: prove the whole J-lens pipeline runs end to end on the small model.

What it checks:
  1. weights load, lens downloads + loads, provenance matches the model;
  2. lens.apply runs for BOTH the Jacobian lens and the logit-lens baseline;
  3. the two disagree at least somewhere (if identical, the transport step is a no-op
     and something is wrong).

Run:
    python 01_smoke_test.py                 # dev model (Qwen3.5-4B)
    python 01_smoke_test.py --model dev-9b
    python 01_smoke_test.py --model deepseek --prompt "..."   # only on a GPU box
"""
import argparse

import config
from common import load_model, load_lens, check_provenance, apply_lens, topk_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dev", help="registry key (config.py)")
    ap.add_argument("--prompt", default=(
        "Fact: The currency used in the country shaped like a boot is"))
    ap.add_argument("--position", type=int, default=-2,
                    help="token index to read (default -2: the token before the last).")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    spec = config.get(args.model)
    check_provenance(spec)                      # warns on mismatch, does not abort
    model, hf, tok = load_model(spec)
    lens = load_lens(spec, kind="j")

    print(f"\nlens: {len(lens.source_layers)} source layers "
          f"(d_model={lens.d_model}, n_prompts={lens.n_prompts})")
    print(f"prompt: {args.prompt!r}   position={args.position}\n")

    pos = [args.position]
    jl = apply_lens(lens, model, args.prompt, pos, use_jacobian=True)
    ll = apply_lens(lens, model, args.prompt, pos, use_jacobian=False)

    layers = sorted(jl.keys())
    print(f"{'layer':>6} | {'JACOBIAN lens top-'+str(args.topk):<45} | logit-lens top-"+str(args.topk))
    print("-" * 110)
    disagreements = 0
    for L in layers:
        j_top = topk_tokens(jl[L][0], tok, args.topk)
        l_top = topk_tokens(ll[L][0], tok, args.topk)
        j_str = " ".join(repr(t) for t, _ in j_top)
        l_str = " ".join(repr(t) for t, _ in l_top)
        if j_top and l_top and j_top[0][0] != l_top[0][0]:
            disagreements += 1
        print(f"{L:>6} | {j_str:<45} | {l_str}")

    # ---- assertions ----
    assert layers, "lens returned no layers"
    assert all(jl[L].shape[-1] > 1000 for L in layers), "vocab dim looks wrong"
    assert disagreements > 0, (
        "Jacobian lens == logit-lens at every layer — transport step may be a no-op.")
    print(f"\nPASS: pipeline works. J-lens and logit-lens differ at {disagreements}/"
          f"{len(layers)} layers (as expected).")


if __name__ == "__main__":
    main()
