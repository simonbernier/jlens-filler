"""
Smoke test — run this first on any new machine (laptop, rented GPU box).

Two independent checks, pick what the machine is for:

  python 00_smoke_test.py --api               # API path (stage 1): key, model id,
                                              # provider pin, reasoning-off, parsing
  python 00_smoke_test.py                     # GPU path (stages 2-4), dev model
  python 00_smoke_test.py --model deepseek    # GPU path on the real target
  python 00_smoke_test.py --model deepseek --tokenizer-only
                                              # template/tail/decode-mode checks on
                                              # the real tokenizer, no weights (laptop)

The GPU path proves the whole lens pipeline end to end:
  1. weights load, lens downloads + loads, provenance matches the model;
  2. prompt construction + filler-span location work on the REAL tokenizer
     (a rendered chat prompt round-trips to the right number of filler tokens);
  3. the numeric decode criterion this tokenizer supports is reported —
     "exact" where digits are grouped into single tokens (DeepSeek), "prefix"
     (first-token match) where they are split (Qwen, Llama 3). Not a failure,
     but headline stage-2/3 numbers should come from an exact-mode model;
  4. lens.apply runs for BOTH the Jacobian lens and the logit-lens baseline;
  5. the two disagree at least somewhere (if identical, the transport step is
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
    text = pt.render_chat(tok, msgs)     # raises if the template leaves <think> open
    check("generation prompt has no open <think> (reasoning off)",
          text.count("<think>") <= text.count("</think>"))
    filler = pt.make_filler("dots", k)
    c0, c1 = pt.final_filler_char_span(text, filler, ex.question)
    check("char span is exactly the filler", text[c0:c1] == filler)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    neg = pt.span_to_negative_positions(enc["offset_mapping"], c0, c1)
    check("filler tokens located, negative + contiguous",
          len(neg) >= k and all(n < 0 for n in neg)
          and neg == list(range(neg[0], neg[0] + len(neg))))
    ids = enc["input_ids"]
    print(f"  prompt: {len(ids)} tokens; post-filler tail read by 20: "
          f"{[tok.decode([ids[p]]) for p in range(neg[-1] + 1, 0)]}")
    # The rendered template already carries the model's BOS (if it has one);
    # tokenizing with add_special_tokens=True would add another. common.load_model
    # makes the lens tokenize verbatim, so the count must not change.
    n_special = len(tok(text, add_special_tokens=True).input_ids)
    if n_special != len(ids):
        print(f"  note: add_special_tokens=True would add {n_special - len(ids)} "
              f"token(s) (a second BOS) — the lens tokenizes verbatim instead")
    # Numeric readout: report which decode criterion this tokenizer supports.
    # This is NOT pass/fail — a digit-splitting tokenizer (Qwen, Llama 3) can
    # only do first-token matching, which is fine for pipe-cleaning and wrong
    # for headline numbers. Only an empty table is a real failure.
    numeric = pt.build_numeric_readout(tok)
    check("numeric token table non-empty", numeric.ids.size > 0)
    print("  " + numeric.describe())
    if numeric.mode == "prefix":
        print("  NOTE: run the headline decode (stages 2-3) on DeepSeek, whose "
              "tokenizer groups digits — this model can only pipe-clean.")
    return text


def lens_smoke(args):
    import config
    from common import load_model, load_lens, check_provenance, apply_lens, topk_tokens

    spec = config.get(args.model)
    check_provenance(spec)                      # warns on mismatch, does not abort
    model, hf, tok = load_model(spec)
    lens = load_lens(spec, kind="j")

    print("\n== prompt construction on the real tokenizer ==")
    chat_prompt = prompt_smoke(tok)

    # One greedy reply on the real task prompt. This is the check that catches
    # a reasoning-mode template: a hybrid reasoning model with thinking left on
    # answers "Thinking Process: 1." to every question, which parses as a
    # confident wrong answer unless the <think> tag is kept in the decode.
    print("\n== greedy answer on the task prompt (reasoning must be OFF) ==")
    import torch
    enc = tok(chat_prompt, add_special_tokens=False, return_tensors="pt").to(hf.device)
    with torch.no_grad():
        gen = hf.generate(**enc, max_new_tokens=6, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    reply = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=False)
    print(f"  reply: {reply!r} -> parsed {pt.parse_answer(reply)}")
    check("reply parses to an integer (no reasoning trace)",
          pt.parse_answer(reply) is not None)

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


def tokenizer_smoke(model_key: str):
    """Prompt checks on the real tokenizer only — no weights, no GPU.

    The free preflight for DeepSeek V4 Flash: the tokenizer download is a few
    MB, and it settles the three things that bit the dev model before renting
    anything — does the template honour reasoning-off, which tokens does the
    post-filler tail read, and which numeric decode mode do we get.
    """
    import transformers
    import config

    spec = config.get(model_key)
    tok = transformers.AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=spec.trust_remote_code)
    print(f"tokenizer: {spec.hf_id}\n\n== prompt construction on the real tokenizer ==")
    prompt_smoke(tok)
    print("\nPASS: tokenizer path works (weights not loaded).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true",
                    help="check the API path (stage 1) instead of the GPU/lens path")
    ap.add_argument("--tokenizer-only", action="store_true",
                    help="prompt/template checks on the tokenizer alone (no weights)")
    ap.add_argument("--model", default="dev", help="registry key (config.py)")
    ap.add_argument("--prompt", default=(
        "Fact: The currency used in the country shaped like a boot is"))
    ap.add_argument("--position", type=int, default=-2,
                    help="token index to read (default -2: the token before the last).")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    if args.api:
        api_smoke()
    elif args.tokenizer_only:
        tokenizer_smoke(args.model)
    else:
        lens_smoke(args)


if __name__ == "__main__":
    main()
