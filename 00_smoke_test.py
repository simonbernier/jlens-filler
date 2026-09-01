"""
Smoke test — run this first on any new machine (laptop, rented GPU box).

Two independent checks, pick what the machine is for:

  python 00_smoke_test.py --api               # API path (stage 1): key, model id,
                                              # provider pin, reasoning-off, parsing
  python 00_smoke_test.py                     # GPU path (stages 2-4), dev model
  python 00_smoke_test.py --model deepseek    # GPU path on the real target

The GPU path proves the whole lens pipeline end to end:
  1. weights load, lens downloads + loads, provenance matches the model;
  2. prompt construction + filler-span location work on the REAL tokenizer
     (a rendered chat prompt round-trips to the right number of filler tokens);
  3. lens.apply runs for BOTH the Jacobian lens and the logit-lens baseline;
  4. the two disagree at least somewhere (if identical, the transport step is
     a no-op and something is wrong).
"""
import argparse

import paper_tasks as pt


def check(name: str, cond: bool):
    assert cond, f"FAIL: {name}"
    print(f"  ok  {name}")


def api_smoke():
    import api_common as api
    print(f"model: {api.API_MODEL}   provider pin: {api.PROVIDER}")
    print("live endpoints:")
    api.list_endpoints()
    api.smoke_call(api.make_client())
    print("\nPASS: API path works.")


def prompt_smoke(tok):
    """Prompt building + span location on the real tokenizer (no model needed)."""
    ex = pt.build_dataset(1, seed=0)[0]
    k = 10
    msgs = pt.build_messages(ex, "dots", k)
    check("message layout: system + 5 few-shot pairs + final user",
          len(msgs) == 12 and msgs[0]["role"] == "system"
          and msgs[-1]["role"] == "user")
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    filler = pt.make_filler("dots", k)
    c0, c1 = pt.final_filler_char_span(text, filler, ex.question)
    check("char span is exactly the filler", text[c0:c1] == filler)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    neg = pt.span_to_negative_positions(enc["offset_mapping"], c0, c1)
    check("filler tokens located, negative + contiguous",
          len(neg) >= k and all(n < 0 for n in neg)
          and neg == list(range(neg[0], neg[0] + len(neg))))
    nids = pt.numeric_token_ids(tok)
    check("numeric token table non-trivial", len(nids) > 100)
    return text


def lens_smoke(args):
    import config
    from common import load_model, load_lens, check_provenance, apply_lens, topk_tokens

    spec = config.get(args.model)
    check_provenance(spec)                      # warns on mismatch, does not abort
    model, hf, tok = load_model(spec)
    lens = load_lens(spec, kind="j")

    print("\n== prompt construction on the real tokenizer ==")
    prompt_smoke(tok)

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

    check("lens returned layers", bool(layers))
    check("vocab dim looks right", all(jl[L].shape[-1] > 1000 for L in layers))
    check("J-lens and logit-lens differ somewhere (transport not a no-op)",
          disagreements > 0)
    print(f"\nPASS: pipeline works. J-lens and logit-lens differ at "
          f"{disagreements}/{len(layers)} layers (as expected).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true",
                    help="check the API path (stage 1) instead of the GPU/lens path")
    ap.add_argument("--model", default="dev", help="registry key (config.py)")
    ap.add_argument("--prompt", default=(
        "Fact: The currency used in the country shaped like a boot is"))
    ap.add_argument("--position", type=int, default=-2,
                    help="token index to read (default -2: the token before the last).")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    if args.api:
        api_smoke()
    else:
        lens_smoke(args)


if __name__ == "__main__":
    main()
