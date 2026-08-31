"""
Prompt builders for the filler-token tasks, mirroring
"Reading Between the Dots" (Brauer, Verdun & Marks, arXiv:2607.03502).

Prompt shape:   [question] [filler] Answer:
The model must (1) retrieve two facts A1, A2 and (2) compose them (A1 + A2),
with only meaningless filler tokens between question and answer slot.

Each builder returns a FillerPrompt with the prompt split into three parts
(pre / filler / post) so the caller can tokenize incrementally and recover the
exact token indices of the filler region.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


# Atomic numbers most 4B+ models know reliably — keeps the "fact retrieval" real
# while staying controllable.
ATOMIC_NUMBER = {
    "hydrogen": 1, "helium": 2, "lithium": 3, "carbon": 6, "nitrogen": 7,
    "oxygen": 8, "fluorine": 9, "neon": 10, "sodium": 11, "magnesium": 12,
    "aluminum": 13, "silicon": 14, "phosphorus": 15, "sulfur": 16,
    "chlorine": 17, "potassium": 19, "calcium": 20, "iron": 26, "copper": 29,
    "zinc": 30, "silver": 47, "gold": 79,
}


def make_filler(kind: str, k: int) -> str:
    """Build a k-token filler string. Content is meaningless by design;
    the paper shows dots/counting/alphabet all give similar uplift."""
    if kind == "dots":
        return " ".join(["."] * k)
    if kind == "counting":
        return " ".join(str(i % 10) for i in range(1, k + 1))
    if kind == "alphabet":
        return " ".join(chr(ord("a") + (i % 26)) for i in range(k))
    if kind == "none":
        return ""
    raise ValueError(f"unknown filler kind {kind!r} (use dots|counting|alphabet|none)")


@dataclass
class FillerPrompt:
    pre: str          # everything up to and including the space before the filler
    filler: str       # the filler tokens
    post: str         # " Answer:" tail
    a1: int           # first intermediate (retrieved fact)
    a2: int           # second intermediate
    target: int       # A1 + A2, the correct answer
    meta: dict

    @property
    def text(self) -> str:
        return f"{self.pre}{self.filler}{self.post}"


def two_fact_addition(elem_a: str, elem_b: str, filler_kind: str = "dots",
                      k: int = 10) -> FillerPrompt:
    a1, a2 = ATOMIC_NUMBER[elem_a], ATOMIC_NUMBER[elem_b]
    question = (f"What is the atomic number of {elem_a} plus "
                f"the atomic number of {elem_b}?")
    pre = question + " "
    filler = make_filler(filler_kind, k)
    post = " Answer:"
    return FillerPrompt(
        pre=pre, filler=filler, post=post,
        a1=a1, a2=a2, target=a1 + a2,
        meta=dict(task="2fact_add", elem_a=elem_a, elem_b=elem_b,
                  filler_kind=filler_kind, k=k),
    )


# A handful of pairs whose atomic numbers stay two-digit (single-token friendly).
DEFAULT_PAIRS: List[Tuple[str, str]] = [
    ("oxygen", "sodium"),     # 8 + 11 = 19
    ("carbon", "iron"),       # 6 + 26 = 32
    ("copper", "calcium"),    # 29 + 20 = 49
    ("silver", "zinc"),       # 47 + 30 = 77
    ("magnesium", "silver"),  # 12 + 47 = 59
]


def default_prompts(filler_kind: str = "dots", k: int = 10) -> List[FillerPrompt]:
    return [two_fact_addition(a, b, filler_kind, k) for a, b in DEFAULT_PAIRS]
