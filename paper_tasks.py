"""
Paper-faithful task construction for replicating "Reading Between the Dots"
(Brauer, Verdun & Marks, arXiv:2607.03502) — 2-fact addition, chat format.

This module is deliberately TORCH-FREE (numpy only) so the mock tests and the
API-based accuracy sweep can run on any machine. Everything that needs
torch/transformers/jlens lives in the runner scripts (03/04).

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
# Numeric-token utilities (paper: "restricting to numeric tokens")
# --------------------------------------------------------------------------- #
def numeric_token_ids(tok, lo: int = 0, hi: int = 300) -> Dict[int, int]:
    """{token_id: integer value} for single tokens that spell an integer in
    [lo, hi], trying both 'N' and ' N' spellings. If both spellings exist as
    distinct single tokens, both ids are included (mapped to the same value).
    """
    out: Dict[int, int] = {}
    for v in range(lo, hi + 1):
        for s in (str(v), f" {v}"):
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                out[int(ids[0])] = v
    return out


def top_numeric_value(logits_row, numeric_ids: Dict[int, int]) -> Optional[int]:
    """Integer value of the highest-scoring numeric token (numpy row)."""
    import numpy as np
    ids = np.fromiter(numeric_ids.keys(), dtype=np.int64)
    if ids.size == 0:
        return None
    best = ids[int(np.argmax(logits_row[ids]))]
    return numeric_ids[int(best)]


def value_rank(logits_row, tok, value: int) -> int:
    """Full-vocab rank (0 = top) of the best single-token spelling of value.
    Large sentinel if no single-token spelling exists."""
    import numpy as np
    best = None
    for s in (str(value), f" {value}"):
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1 and ids[0] < len(logits_row):
            r = int((logits_row > logits_row[ids[0]]).sum())
            best = r if best is None else min(best, r)
    return best if best is not None else 10 ** 9


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
