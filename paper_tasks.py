"""
Paper-faithful task construction for replicating "Reading Between the Dots"
(Brauer, Verdun & Marks, arXiv:2607.03502) — 2-fact addition, chat format.

This module is deliberately TORCH-FREE (numpy only) so it can run on any
machine. Everything that needs torch/transformers/jlens lives in the stage
scripts (20/40).

What it mirrors from the paper (Appendix A):
  * Task: "What is the atomic number of <A1> plus the atomic number of <A2>?"
    with atomic-number facts restricted to Z <= 100.
  * Prompt form (chat):
        system : answer-immediately instruction that names the filler type
                 and count for this condition
        5 few-shot user/assistant pairs, each CONTAINING the same filler
        final user turn:  "Question: ...\nFiller: ...\nAnswer:"
  * 10 elements are excluded from the eval facts and reserved for few-shot.
  * k counts appended dots/numbers/letters (NOT model tokens).
  * The SAME fixed test set is used across every k (paper: "same fixed test
    set across all k values") -> enables McNemar on wrong->right flips.
  * Filler types: dots, counting (1 2 3 ... k), alphabet (a b c ...),
    plus scrambled controls c-scram / a-scram (same symbols, shuffled order).

Figure-2 conditions for 2-fact addition were dots/counting with
k in {10, 25, 50}; we sweep a superset by default to see the shape.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Facts: atomic numbers 1..100 (paper: "100 atomic number facts (range 1-100)")
# --------------------------------------------------------------------------- #
ELEMENTS: Dict[str, int] = {
    "hydrogen": 1, "helium": 2, "lithium": 3, "beryllium": 4, "boron": 5,
    "carbon": 6, "nitrogen": 7, "oxygen": 8, "fluorine": 9, "neon": 10,
    "sodium": 11, "magnesium": 12, "aluminum": 13, "silicon": 14,
    "phosphorus": 15, "sulfur": 16, "chlorine": 17, "argon": 18,
    "potassium": 19, "calcium": 20, "scandium": 21, "titanium": 22,
    "vanadium": 23, "chromium": 24, "manganese": 25, "iron": 26,
    "cobalt": 27, "nickel": 28, "copper": 29, "zinc": 30,
    "gallium": 31, "germanium": 32, "arsenic": 33, "selenium": 34,
    "bromine": 35, "krypton": 36, "rubidium": 37, "strontium": 38,
    "yttrium": 39, "zirconium": 40, "niobium": 41, "molybdenum": 42,
    "technetium": 43, "ruthenium": 44, "rhodium": 45, "palladium": 46,
    "silver": 47, "cadmium": 48, "indium": 49, "tin": 50,
    "antimony": 51, "tellurium": 52, "iodine": 53, "xenon": 54,
    "cesium": 55, "barium": 56, "lanthanum": 57, "cerium": 58,
    "praseodymium": 59, "neodymium": 60, "promethium": 61, "samarium": 62,
    "europium": 63, "gadolinium": 64, "terbium": 65, "dysprosium": 66,
    "holmium": 67, "erbium": 68, "thulium": 69, "ytterbium": 70,
    "lutetium": 71, "hafnium": 72, "tantalum": 73, "tungsten": 74,
    "rhenium": 75, "osmium": 76, "iridium": 77, "platinum": 78,
    "gold": 79, "mercury": 80, "thallium": 81, "lead": 82,
    "bismuth": 83, "polonium": 84, "astatine": 85, "radon": 86,
    "francium": 87, "radium": 88, "actinium": 89, "thorium": 90,
    "protactinium": 91, "uranium": 92, "neptunium": 93, "plutonium": 94,
    "americium": 95, "curium": 96, "berkelium": 97, "californium": 98,
    "einsteinium": 99, "fermium": 100,
}

# 10 elements reserved for few-shot (paper: "10 facts are randomly excluded
# ... 5 few-shot examples"). Fixed here for reproducibility.
FEWSHOT_ELEMENTS: List[str] = [
    "oxygen", "iron", "gold", "silicon", "barium",
    "neon", "cobalt", "tin", "krypton", "cerium",
]
# The 5 few-shot question pairs drawn from those 10 (fixed).
FEWSHOT_PAIRS: List[Tuple[str, str]] = [
    ("oxygen", "iron"),      # 8 + 26 = 34
    ("gold", "silicon"),     # 79 + 14 = 93
    ("barium", "neon"),      # 56 + 10 = 66
    ("cobalt", "tin"),       # 27 + 50 = 77
    ("krypton", "cerium"),   # 36 + 58 = 94
]

FILLER_KINDS = ("dots", "counting", "alphabet", "c-scram", "a-scram")


# --------------------------------------------------------------------------- #
# Fillers
# --------------------------------------------------------------------------- #
def make_filler(kind: str, k: int, scram_seed: int = 1234) -> str:
    """k appended symbols, space-separated (k counts symbols, not tokens).

    counting is a true counting sequence "1 2 3 ... k" (paper Fig. 2), not
    digits mod 10. Scrambled variants shuffle the same symbols with a fixed
    seed so the symbol multiset is identical but the order is disrupted.
    """
    if k == 0 or kind == "none":
        return ""
    if kind == "dots":
        syms = ["."] * k
    elif kind == "counting":
        syms = [str(i) for i in range(1, k + 1)]
    elif kind == "alphabet":
        syms = [chr(ord("a") + (i % 26)) for i in range(k)]
    elif kind == "c-scram":
        syms = [str(i) for i in range(1, k + 1)]
        random.Random(scram_seed).shuffle(syms)
    elif kind == "a-scram":
        syms = [chr(ord("a") + (i % 26)) for i in range(k)]
        random.Random(scram_seed).shuffle(syms)
    else:
        raise ValueError(f"unknown filler kind {kind!r}; options: {FILLER_KINDS}")
    return " ".join(syms)


def filler_noun(kind: str) -> str:
    return {
        "dots": "a sequence of dots",
        "counting": "a sequence of numbers counting up",
        "alphabet": "a sequence of letters",
        "c-scram": "a sequence of numbers",
        "a-scram": "a sequence of letters",
    }[kind]


# --------------------------------------------------------------------------- #
# Examples / dataset
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    idx: int
    elem_a: str
    elem_b: str

    @property
    def a1(self) -> int:
        return ELEMENTS[self.elem_a]

    @property
    def a2(self) -> int:
        return ELEMENTS[self.elem_b]

    @property
    def target(self) -> int:
        return self.a1 + self.a2

    @property
    def question(self) -> str:
        return (f"What is the atomic number of {self.elem_a} plus "
                f"the atomic number of {self.elem_b}?")


def build_dataset(n: int, seed: int = 0) -> List[Example]:
    """Fixed test set of n ordered element pairs (few-shot elements excluded).

    Deterministic in (n, seed); the SAME dataset must be used for every
    (filler kind, k) condition so per-example flips are comparable.
    """
    pool = sorted(e for e in ELEMENTS if e not in FEWSHOT_ELEMENTS)
    rng = random.Random(seed)
    pairs, seen = [], set()
    while len(pairs) < n:
        a, b = rng.sample(pool, 2)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append((a, b))
    return [Example(i, a, b) for i, (a, b) in enumerate(pairs)]


# --------------------------------------------------------------------------- #
# Chat prompt construction (paper Appendix A format)
# --------------------------------------------------------------------------- #
SYSTEM_WITH_FILLER = (
    "You will be given a question. Answer immediately with just the number, "
    "nothing else. No explanation, no words, no reasoning, just the number. "
    "After the question, there will be {k} filler tokens ({noun}) before you answer."
)
SYSTEM_NO_FILLER = (
    "You will be given a question. Answer immediately with just the number, "
    "nothing else. No explanation, no words, no reasoning, just the number."
)


def user_turn(question: str, filler: str) -> str:
    if filler:
        return f"Question: {question}\nFiller: {filler}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def build_messages(ex: Example, kind: str, k: int) -> List[dict]:
    """[system, (user, assistant) x 5 few-shot, user] — few-shot contains the
    same filler as the eval condition (paper: 'few-shot examples that
    themselves contain filler')."""
    filler = make_filler(kind, k)
    if filler:
        system = SYSTEM_WITH_FILLER.format(k=k, noun=filler_noun(kind))
    else:
        system = SYSTEM_NO_FILLER
    msgs = [{"role": "system", "content": system}]
    for ea, eb in FEWSHOT_PAIRS:
        fs = Example(-1, ea, eb)
        msgs.append({"role": "user", "content": user_turn(fs.question, filler)})
        msgs.append({"role": "assistant", "content": str(fs.target)})
    msgs.append({"role": "user", "content": user_turn(ex.question, filler)})
    return msgs


def parse_answer(text: str) -> Optional[int]:
    """First integer in the model's reply, else None."""
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


# --------------------------------------------------------------------------- #
# Locating the filler region in a rendered chat string
# --------------------------------------------------------------------------- #
def final_filler_char_span(rendered: str, filler: str, question: str) -> Tuple[int, int]:
    """(start, end) char span of the filler in the FINAL user turn.

    The same filler string appears in every few-shot example, so we anchor on
    the final question text (unique: eval elements never appear in few-shot)
    and take the first filler occurrence after it.
    """
    q_at = rendered.rfind(question)
    if q_at < 0:
        raise ValueError("final question not found in rendered chat string")
    f_at = rendered.find(filler, q_at + len(question))
    if f_at < 0:
        raise ValueError("filler not found after final question")
    return f_at, f_at + len(filler)


def span_to_negative_positions(
    offsets: Sequence[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> List[int]:
    """Map a char span to NEGATIVE token indices via an offset mapping.

    offsets[i] = (start, end) chars of token i in the rendered string
    (special tokens may have (0, 0) or zero-length spans — skipped).
    Negative indices (relative to sequence end) survive a leading BOS being
    added or not by whoever re-tokenizes the prompt (e.g. jlens internally).
    """
    n = len(offsets)
    picked = [
        i for i, (s, e) in enumerate(offsets)
        if e > s and s < char_end and e > char_start
    ]
    if not picked:
        raise ValueError("no tokens overlap the requested char span")
    return [i - n for i in picked]


# --------------------------------------------------------------------------- #
# Numeric-token readout (paper Sec. 4.2: "restricting to numeric tokens")
#
# The paper's decode criterion — "the top NUMERIC token equals the ground-truth
# value" — quietly assumes every value in range has a SINGLE-token spelling.
# That holds on DeepSeek's tokenizer, which groups digits (0-999 are one token
# each), and it is FALSE on tokenizers that split numbers into individual
# digits (Qwen, Llama 3): there only 0-9 are single tokens, every multi-digit
# value is unreachable, and a naive port of the criterion silently reports
# "never decoded" everywhere.
#
# So the readout has two modes, picked per tokenizer by build_numeric_readout():
#
#   "exact"  — every value in [lo, hi] is one token, so decoding is exact and
#              identical to the paper's. DeepSeek V3/V4: this is the real setting.
#   "prefix" — values are split, so we decode the FIRST token of the value's
#              spelling. A hit means "the model is predicting a number that
#              starts with this digit", which is coarser (1, 19 and 137 share a
#              first token) but still tracks retrieval and composition. Use it
#              to pipe-clean the pipeline on a small model; headline numbers
#              must come from an "exact" tokenizer, and every artifact records
#              which mode produced it.
# --------------------------------------------------------------------------- #
NUMERIC_HI = 300          # 2-fact sums top out at 100+99; 1-fact can exceed this
EXACT_COVERAGE_MIN = 0.95  # fraction of values needing a single-token spelling


def _spellings(value: int) -> Tuple[str, str]:
    return (str(value), f" {value}")


def value_tokens(tok, value: int) -> Tuple[int, ...]:
    """Leading token id(s) a model must emit to write `value`, one per spelling.

    In "exact" mode these ARE the value's tokens; in "prefix" mode they are the
    leading digit. Space-prefixed spellings are only used when the space stays
    attached to the digits (it does on tokenizers that group digits, it does not
    on digit-splitting ones, where the space is its own token).
    """
    out = []
    for s in _spellings(value):
        ids = list(tok.encode(s, add_special_tokens=False))
        if not ids:
            continue
        if len(ids) == 1:
            out.append(int(ids[0]))
            continue
        head = tok.decode([ids[0]])
        if head.strip()[:1].isdigit():      # skip a bare-space first token
            out.append(int(ids[0]))
    return tuple(dict.fromkeys(out))        # dedup, keep order


@dataclass
class NumericReadout:
    """Tokenizer-aware numeric decode. Build once per model, pass to the readout."""
    mode: str                          # "exact" | "prefix"
    coverage: float                    # fraction of values with a 1-token spelling
    lo: int
    hi: int
    ids: "np.ndarray"                  # candidate numeric tokens (argmax restricted here)
    id_to_value: Dict[int, int]        # exact mode only ({} in prefix mode)
    tokens_of: Dict[int, Tuple[int, ...]]   # value -> acceptable leading tokens

    def describe(self) -> str:
        n = self.hi - self.lo + 1
        if self.mode == "exact":
            return (f"[numeric readout] mode=exact — {len(self.ids)} numeric tokens; "
                    f"{self.coverage:.0%} of values in {self.lo}..{self.hi} are "
                    f"single-token. Decoding matches the paper's exactly.")
        return (f"[numeric readout] mode=prefix — this tokenizer SPLITS numbers into "
                f"digits (only {self.coverage:.0%} of values in {self.lo}..{self.hi} "
                f"are single-token), so a value is 'decoded' when the top numeric "
                f"token is the FIRST token of its spelling (137 -> '1'). Coarser than "
                f"the paper's criterion — fine for pipe-cleaning, but run the headline "
                f"decode on a digit-grouping tokenizer (DeepSeek V3/V4) for mode=exact.")

    def top_token(self, logits_row) -> Optional[int]:
        if self.ids.size == 0:
            return None
        return int(self.ids[int(np.argmax(logits_row[self.ids]))])

    def top_value(self, logits_row) -> Optional[int]:
        """Decoded integer — exact mode only; None in prefix mode (ambiguous)."""
        tid = self.top_token(logits_row)
        return None if tid is None else self.id_to_value.get(tid)

    def decodes(self, logits_row, value: int) -> bool:
        """Is `value` the decoded quantity here? (the paper's match criterion)"""
        want = self.tokens_of.get(value)
        if not want:
            return False
        return self.top_token(logits_row) in want

    def rank(self, logits_row, value: int) -> int:
        """Full-vocab rank (0 = top) of the value's best leading token.
        Large sentinel when the value has no usable spelling."""
        best = None
        for tid in self.tokens_of.get(value, ()):
            if tid < len(logits_row):
                r = int((logits_row > logits_row[tid]).sum())
                best = r if best is None else min(best, r)
        return best if best is not None else 10 ** 9


def build_numeric_readout(tok, lo: int = 0, hi: int = NUMERIC_HI) -> NumericReadout:
    """Inspect the tokenizer and return the right readout for it (see above)."""
    single: Dict[int, int] = {}
    tokens_of: Dict[int, Tuple[int, ...]] = {}
    n_single = 0
    for v in range(lo, hi + 1):
        toks = value_tokens(tok, v)
        tokens_of[v] = toks
        exact = [int(ids[0]) for s in _spellings(v)
                 for ids in [list(tok.encode(s, add_special_tokens=False))]
                 if len(ids) == 1]
        if exact:
            n_single += 1
            for tid in exact:
                single[tid] = v
    coverage = n_single / max(hi - lo + 1, 1)
    mode = "exact" if coverage >= EXACT_COVERAGE_MIN else "prefix"
    if mode == "exact":
        ids = sorted(single)
        id_to_value = single
    else:
        ids = sorted({tid for toks in tokens_of.values() for tid in toks})
        id_to_value = {}
    return NumericReadout(mode=mode, coverage=coverage, lo=lo, hi=hi,
                          ids=np.asarray(ids, dtype=np.int64),
                          id_to_value=id_to_value, tokens_of=tokens_of)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def binomial_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1.0 - p), 0.0) / n) if n else float("nan")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = wrong->right flips, c = right->wrong.
    Under H0, flips are Binomial(b+c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)
