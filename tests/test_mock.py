"""
Mock tests for the filler/J-lens pipeline — no torch, no downloads, no GPU.

A fake word-level tokenizer + a fake lens exercise the plumbing that
actually goes wrong in practice: prompt construction, locating the filler
token span (with a BOS in the way), numeric-token bookkeeping, rank math,
McNemar, 04's per-example readout loop, and 05's aggregation end to end.

Run:  python tests/test_mock.py
"""
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_tasks as pt
import importlib
m04 = importlib.import_module("04_lens_readout")
m05 = importlib.import_module("05_analyze_lens")

PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print(f"  ok  {name}")


# --------------------------------------------------------------------------- #
# Fake tokenizer: every non-space run is one token; growable vocab.
# --------------------------------------------------------------------------- #
class FakeTok:
    def __init__(self):
        self.vocab = {}
        self.eos_token_id = 0
        self._id("<eos>")

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab)
        return self.vocab[w]

    def _words(self, text):
        return [(m.group(), m.start(), m.end())
                for m in re.finditer(r"\S+", text)]

    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w, _, _ in self._words(text)]

    def decode(self, ids, skip_special_tokens=False):
        rev = {v: k for k, v in self.vocab.items()}
        return " ".join(rev[i] for i in ids)

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False, return_tensors=None):
        words = self._words(text)
        enc = {"input_ids": [self._id(w) for w, _, _ in words]}
        if return_offsets_mapping:
            # simulate a leading BOS special token with a (0,0) offset
            enc["offset_mapping"] = [(0, 0)] + [(s, e) for _, s, e in words]
            enc["input_ids"] = [self.eos_token_id] + enc["input_ids"]
        return enc

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        parts = [f"<|{m['role']}|> {m['content']}" for m in msgs]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " \n".join(parts)


# --------------------------------------------------------------------------- #
print("== paper_tasks ==")
f = pt.make_filler("dots", 5)
check("dots filler", f == ". . . . .")
check("counting is a true count", pt.make_filler("counting", 4) == "1 2 3 4")
check("k=0 empty", pt.make_filler("dots", 0) == "")
cs = pt.make_filler("c-scram", 10)
check("c-scram same symbols, different order",
      sorted(cs.split()) == sorted(pt.make_filler("counting", 10).split())
      and cs != pt.make_filler("counting", 10))
check("scram deterministic", cs == pt.make_filler("c-scram", 10))

ds1, ds2 = pt.build_dataset(50, seed=0), pt.build_dataset(50, seed=0)
check("dataset deterministic",
      [(e.elem_a, e.elem_b) for e in ds1] == [(e.elem_a, e.elem_b) for e in ds2])
check("dataset excludes few-shot elements",
      all(e.elem_a not in pt.FEWSHOT_ELEMENTS and
          e.elem_b not in pt.FEWSHOT_ELEMENTS for e in ds1))
check("dataset differs across seeds",
      [(e.elem_a, e.elem_b) for e in pt.build_dataset(50, seed=1)]
      != [(e.elem_a, e.elem_b) for e in ds1])
ex = ds1[0]
check("target = a1 + a2", ex.target == pt.ELEMENTS[ex.elem_a] + pt.ELEMENTS[ex.elem_b])

msgs = pt.build_messages(ex, "dots", 10)
check("message layout: sys + 5 shot pairs + final user",
      len(msgs) == 12 and msgs[0]["role"] == "system"
      and msgs[-1]["role"] == "user")
check("few-shot contains filler",
      all(". . ." in m["content"] for m in msgs if m["role"] == "user"))
check("few-shot answers are the sums",
      [m["content"] for m in msgs if m["role"] == "assistant"]
      == [str(pt.ELEMENTS[a] + pt.ELEMENTS[b]) for a, b in pt.FEWSHOT_PAIRS])
check("system prompt names k and filler type",
      "10 filler tokens" in msgs[0]["content"] and "dots" in msgs[0]["content"])
msgs0 = pt.build_messages(ex, "dots", 0)
check("k=0: no Filler line anywhere",
      all("Filler:" not in m["content"] for m in msgs0)
      and "filler" not in msgs0[0]["content"])

check("parse_answer", pt.parse_answer(" Answer: 112\n") == 112
      and pt.parse_answer("nope") is None)

print("== span location ==")
tok = FakeTok()
text = tok.apply_chat_template(msgs, tokenize=False)
filler = pt.make_filler("dots", 10)
c0, c1 = pt.final_filler_char_span(text, filler, ex.question)
check("char span is exactly the filler", text[c0:c1] == filler)
check("span is in the FINAL user turn (after all few-shot fillers)",
      text.find(filler) < c0 and text.find(ex.question) < c0)
enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
neg = pt.span_to_negative_positions(enc["offset_mapping"], c0, c1)
check("10 filler tokens located", len(neg) == 10)
check("negative and contiguous",
      all(n < 0 for n in neg) and neg == list(range(neg[0], neg[0] + 10)))
words = [w for w, _, _ in tok._words(text)]
check("negative indices point at dots (BOS-proof)",
      all(words[len(words) + n] == "." for n in neg))
positions, n_fill = m04.locate_positions(tok, text, ex, "dots", 10)
check("locate_positions: filler + post tail",
      n_fill == 10 and len(positions) > n_fill and positions[-1] == -1)
check("post tail capped", len(positions) - n_fill <= m04.POST_TAIL_MAX)

print("== numeric utils ==")
# make sure integer words exist in the fake vocab
for v in range(0, 301):
    tok._id(str(v))
nids = pt.numeric_token_ids(tok)
check("numeric ids cover 0..300", len(nids) == 301
      and all(nids[tok.vocab[str(v)]] == v for v in (0, 42, 300)))
V = len(tok.vocab)
row = np.zeros(V, dtype=np.float32)
row[tok.vocab["77"]] = 5.0
row[tok.vocab["35"]] = 4.0
row[tok.vocab["."]] = 9.0  # non-numeric token on top overall
check("top numeric ignores non-numeric argmax",
      pt.top_numeric_value(row, nids) == 77)
check("value_rank: '.' then '77'", pt.value_rank(row, tok, 77) == 1
      and pt.value_rank(row, tok, 35) == 2)
check("value_rank sentinel for unspellable", pt.value_rank(row, tok, 10**7) == 10**9)

print("== stats ==")
check("binomial se", abs(pt.binomial_se(0.5, 100) - 0.05) < 1e-12)
check("mcnemar b=c is 1", pt.mcnemar_exact(3, 3) == 1.0)
p = pt.mcnemar_exact(8, 1)
check("mcnemar 8 vs 1 exact", abs(p - 2 * (1 + 9) / 512) < 1e-12)
check("mcnemar no flips", pt.mcnemar_exact(0, 0) == 1.0)
check("mcnemar strong effect is small", pt.mcnemar_exact(135, 40) < 1e-4)

# --------------------------------------------------------------------------- #
print("== fake-lens end-to-end (04 readout -> 05 analysis) ==")
LAYERS = [30, 40, 50]


def make_apply_fn(ex, n_fill, jlens_bonus):
    """Planted pattern: A1 decodes at filler pos 1 from layer 40, A2 at pos 6
    from layer 40, sum at last post position at layer 50. J-lens additionally
    decodes the sum IN the filler (pos 7) at layer 40 when jlens_bonus."""
    def apply_fn(text, positions, use_j):
        out = {}
        V = len(tok.vocab)
        for L in LAYERS:
            arr = np.random.default_rng(L).normal(0, 0.1, (len(positions), V)) \
                    .astype(np.float32)
            # deterministic non-target numeric winner so chance argmax hits
            # can't fake a decode (targets are never 0)
            arr[:, tok.vocab["0"]] = 5.0
            for pi, pos in enumerate(positions):
                gpos = pos_index[pos]  # 0-based index in the full read set
                if gpos == 1 and L >= 40:
                    arr[pi, tok.vocab[str(ex.a1)]] = 10.0
                if gpos == 6 and L >= 40:
                    arr[pi, tok.vocab[str(ex.a2)]] = 10.0
                if gpos == len(positions_full) - 1 and L >= 50:
                    arr[pi, tok.vocab[str(ex.target)]] = 10.0
                if use_j and jlens_bonus and gpos == 7 and L >= 40:
                    arr[pi, tok.vocab[str(ex.target)]] = 10.0
            out[L] = arr
        return out
    return apply_fn


with tempfile.TemporaryDirectory() as tmp:
    all_rows, answers = [], []
    for i, ex in enumerate(ds1[:6]):
        text = m04.render_chat(tok, ex, "dots", 10)
        positions_full, n_fill = m04.locate_positions(tok, text, ex, "dots", 10)
        pos_index = {p: i2 for i2, p in enumerate(positions_full)}
        correct = i % 3 != 0  # 4 correct, 2 wrong
        apply_fn = make_apply_fn(ex, n_fill, jlens_bonus=True)
        rows = m04.readout_example(ex, text, positions_full, n_fill, correct,
                                   apply_fn, tok, nids, pos_chunk=4)
        all_rows += rows
        answers.append(dict(idx=ex.idx, a1=ex.a1, a2=ex.a2, target=ex.target,
                            correct=correct))
    rdf = pd.DataFrame(all_rows)
    check("row count = ex * lens * layers * positions",
          len(rdf) == 6 * 2 * len(LAYERS) * len(positions_full))
    check("pos_type split", (rdf[rdf.pos < 10].pos_type == "filler").all()
          and (rdf[rdf.pos >= 10].pos_type == "post").all())
    a1_hit = rdf[(rdf.pos == 1) & (rdf.layer == 40) & (rdf.lens == "logit")]
    merged_a1 = a1_hit.merge(pd.DataFrame(answers)[["idx", "a1"]], on="idx")
    check("planted A1 recovered at (pos1, L40)",
          (merged_a1.top_num == merged_a1.a1).all()
          and (merged_a1.rank_A1 == 0).all())
    check("no A1 before layer 40",
          (rdf[(rdf.pos == 1) & (rdf.layer == 30)].rank_A1 > 0).all())

    readout_csv = os.path.join(tmp, "lens_readout_mock_dots-10.csv")
    ans_csv = os.path.join(tmp, "answers_mock_dots-10.csv")
    rdf.to_csv(readout_csv, index=False)
    pd.DataFrame(answers).to_csv(ans_csv, index=False)

    df = m05.load(readout_csv)
    n_filler = int(df[df.pos_type == "filler"].pos.max()) + 1
    check("05 load + n_filler", n_filler == 10 and "match_sum" in df)

    for lens in ("jlens", "logit"):
        m05.fig3(df, lens, n_filler, "mock", tmp)
    m05.compare_lenses(df, n_filler, "mock", tmp)
    made = os.listdir(tmp)
    check("plots written", "fig3_mock_jlens.png" in made
          and "fig3_mock_logit.png" in made
          and "jlens_vs_logit_mock.png" in made)

    s = m05.algorithm_summary(df, n_filler)
    jj, ll = s["jlens"], s["logit"]
    check("A1/A2 decoded in filler for both lenses",
          jj["A1_decoded_in_filler_frac"] == 1.0
          and ll["A2_decoded_in_filler_frac"] == 1.0)
    check("position specialization A1 < A2",
          jj["A1_mean_position"] < jj["A2_mean_position"])
    check("planted J-lens advantage: sum in filler only under jlens",
          jj["sum_decoded_in_filler_frac"] == 1.0
          and ll["sum_decoded_in_filler_frac"] == 0.0)
    check("sum in post tail for both", jj["sum_decoded_in_post_frac"] == 1.0
          and ll["sum_decoded_in_post_frac"] == 1.0)
    check("parallel retrieval flagged",
          jj["parallel_A1_A2_same_layer_frac"] == 1.0)
    check("retrieval before composition (logit)",
          ll["A1_first_layer_median"] < 50 <= (ll["sum_first_layer_median"] or 99)
          or ll["sum_first_layer_median"] is None)
    check("wrong-example block present", "wrong_examples_logit" in s)
    m05.print_report(s)

print(f"\nALL {len(PASS)} CHECKS PASSED")
