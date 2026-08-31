# %% [markdown]
# # Figure 2 dataset builder — 1-fact & 2-fact addition, dot filler
#
# Prepares the datasets for replicating Figure 2 of *Reading Between the Dots*
# (Brauer, Verdun & Marks, arXiv:2607.03502) on **DeepSeek V4 Flash**, dots only,
# k ∈ {0, 5, 10, 25, 50, 100}.
#
# Pipeline (paper Appendix A):
#   1. load facts from Ryan Greenblatt's compose_facts repo
#      (vendored in data/compose_facts/), with the paper's range restrictions;
#   2. knowledge check: ask each fact as a standalone question, 4 trials —
#      a fact passes if answered correctly ≥ 3/4 times (75%);
#   3. filter out facts that fail;
#   4. hold out few-shot facts (1-fact: 5 facts; 2-fact: 10 elements → 5 pairs,
#      as in the paper — a 2-fact few-shot example consumes 2 elements);
#   5. build the fixed test sets + rendered chat prompts for every k and save.
#
# Run cells top-to-bottom in VS Code (# %% = one Jupyter cell).
# The knowledge check makes API calls; it caches to results/knowledge_check.jsonl
# and is resumable — rerunning skips finished trials. Set USE_MOCK = True for a
# free dry run of the whole pipeline.

# %% Config
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SEED = 0                      # keep fixed — 07 reads the datasets built here
KS = [0, 5, 10, 25, 50, 100]  # dot-filler lengths (0 = no-filler baseline)
N_1FACT = 300                 # test examples per condition (paper: 800)
N_2FACT = 300                 # test examples per condition (paper: 1500)

# knowledge check (paper: "answered correctly at least 3/4 times")
KNOWLEDGE_TRIALS = 4
KNOWLEDGE_MIN_CORRECT = 3          # 3/4 = the paper's 75% criterion
# The paper's released eval code is temperature=0 everywhere; the 3/4 criterion
# is still meaningful because big-MoE API serving is not bit-deterministic
# (batching / expert routing). Bump to 0.6 if you want a harsher check.
KNOWLEDGE_TEMPERATURE = 0.0

# OpenRouter (OpenAI-compatible). Key: export OPENROUTER_API_KEY=sk-or-...
API_MODEL = "deepseek/deepseek-v4-flash"
BASE_URL = "https://openrouter.ai/api/v1"
WORKERS = 8
USE_MOCK = False                   # True = no API calls; simulated replies

DATA_DIR = "data/compose_facts"
OUT_DIR = "data"                   # datasets land here
RESULTS_DIR = "results"            # knowledge-check cache lands here
os.makedirs(RESULTS_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load facts (paper Appendix A ranges)
#
# The paper used: 125 age facts (age at death, 20–97), 118 atomic-number facts
# (1–118), 76 static facts (6–600). The repo has grown slightly since, so counts
# can differ by a few facts; the range restrictions below match the paper.
# The 2-fact task uses only atomic-number facts with Z ≤ 100.
#
# Each fact gets a `noun` — the noun phrase spliced into "What is {noun} plus X?"
# — using the same conversion as compose_facts' create_compositional_dataset.py.

# %%
def _load(name: str) -> list[dict]:
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def _static_noun(question: str) -> str:
    """'What is the number of moons of Mars?' -> 'the number of moons of Mars'."""
    q = question.rstrip("?")
    for prefix in ("What is ", "How many "):
        if q.startswith(prefix):
            q = q[len(prefix):]
            return q if prefix == "What is " else f"the number of {q}"
    return q


facts_1fact: list[dict] = []
for f in _load("age_facts.json"):                       # range 20–97 in the repo
    facts_1fact.append(dict(
        kind="age", key=f["name"], answer=f["answer"],
        standalone=f["question"],
        noun=f"the age at which {f['name']} died"))
for f in _load("atomic_facts.json"):
    if f["answer"] <= 118:                              # paper: range 1–118
        facts_1fact.append(dict(
            kind="atomic", key=f["name"].lower(), answer=f["answer"],
            standalone=f["question"],
            noun=f"the atomic number of {f['name'].lower()}"))
for f in _load("static_facts.json"):
    if 6 <= f["answer"] <= 600:                         # paper: range 6–600
        facts_1fact.append(dict(
            kind="static", key=f["question"], answer=f["answer"],
            standalone=f["question"],
            noun=_static_noun(f["question"])))

# 2-fact pool: atomic numbers Z <= 100 (subset of the atomic facts above)
facts_2fact = [f for f in facts_1fact if f["kind"] == "atomic" and f["answer"] <= 100]

print(f"1-fact pool: {len(facts_1fact)} facts "
      f"({sum(f['kind'] == 'age' for f in facts_1fact)} age, "
      f"{sum(f['kind'] == 'atomic' for f in facts_1fact)} atomic, "
      f"{sum(f['kind'] == 'static' for f in facts_1fact)} static)")
print(f"2-fact pool: {len(facts_2fact)} elements")

# %% [markdown]
# ## 2. API client (or mock)

# %%
ANSWER_FORMAT = (
    "You will be given a question. Answer immediately using the format "
    "'Answer: [ANSWER]' where [ANSWER] is just the number, nothing else. "
    "No explanation, no words, no reasoning, just the number."
)


def parse_answer(text: str) -> int | None:
    """First integer in the reply ('Answer: 138' -> 138)."""
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


class MockClient:
    """Deterministic stand-in: answers correctly with p=0.85 per (question, salt),
    so the knowledge filter has something to do. No network, no key."""

    def query(self, messages, temperature, salt=""):
        question = messages[-1]["content"]
        h = int(hashlib.md5((question + salt).encode()).hexdigest(), 16)
        # ground truth is recoverable from the question via the fact tables
        truth = _MOCK_TRUTH.get(question)
        if truth is None or h % 100 < 15:
            return f"Answer: {(h % 200) + 1}"
        return f"Answer: {truth}"


_MOCK_TRUTH = {f"Question: {f['standalone']}\nAnswer:": f["answer"]
               for f in facts_1fact}


def make_client():
    if USE_MOCK:
        return MockClient()
    from openai import OpenAI  # pip install openai
    key = (os.environ.get("OPENROUTER_API_KEY")
           or os.environ.get("DEEPSEEK_API_KEY")
           or os.environ.get("OPENAI_API_KEY"))
    assert key, "Set OPENROUTER_API_KEY in your environment."
    return OpenAI(api_key=key, base_url=BASE_URL)


def query(client, messages, temperature=0.0, salt="", max_retries=5) -> str:
    if isinstance(client, MockClient):
        return client.query(messages, temperature, salt)
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=API_MODEL, messages=messages,
                temperature=temperature, max_tokens=16)
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return ""


client = make_client()

# %% [markdown]
# ## 3. Knowledge check — 4 standalone trials per fact
#
# Each fact is asked on its own (no filler, no few-shot), `KNOWLEDGE_TRIALS`
# times at temperature 1. Cached one line per trial in
# `results/knowledge_check.jsonl`; rerun this cell to resume after an interruption.

# %%
KNOWLEDGE_CACHE = os.path.join(RESULTS_DIR, "knowledge_check.jsonl")


def knowledge_messages(fact: dict) -> list[dict]:
    return [{"role": "system", "content": ANSWER_FORMAT},
            {"role": "user", "content": f"Question: {fact['standalone']}\nAnswer:"}]


def run_knowledge_check(facts: list[dict]) -> dict[str, list[bool]]:
    done: dict[tuple[str, int], bool] = {}
    if os.path.exists(KNOWLEDGE_CACHE):
        with open(KNOWLEDGE_CACHE) as f:
            for line in f:
                r = json.loads(line)
                done[(r["key"], r["trial"])] = r["correct"]

    todo = [(fact, t) for fact in facts for t in range(KNOWLEDGE_TRIALS)
            if (fact["key"], t) not in done]
    print(f"knowledge check: {len(done)} trials cached, {len(todo)} to run")
    lock = threading.Lock()

    def work(item):
        fact, trial = item
        reply = query(client, knowledge_messages(fact),
                      temperature=KNOWLEDGE_TEMPERATURE, salt=str(trial))
        correct = parse_answer(reply) == fact["answer"]
        rec = dict(key=fact["key"], trial=trial, answer=fact["answer"],
                   reply=reply.strip()[:64], correct=correct)
        with lock:
            with open(KNOWLEDGE_CACHE, "a") as f:
                f.write(json.dumps(rec) + "\n")
        done[(fact["key"], trial)] = correct

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, fut in enumerate(as_completed([pool.submit(work, it) for it in todo])):
            fut.result()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(todo)}")

    out: dict[str, list[bool]] = {}
    for fact in facts:
        out[fact["key"]] = [done[(fact["key"], t)] for t in range(KNOWLEDGE_TRIALS)]
    return out


# every unique fact (the 2-fact pool is a subset of the 1-fact pool)
trials = run_knowledge_check(facts_1fact)

# %% [markdown]
# ## 4. Filter out facts the model doesn't know (≥ 3/4 correct to pass)

# %%
def passes(fact: dict) -> bool:
    return sum(trials[fact["key"]]) >= KNOWLEDGE_MIN_CORRECT


known_1fact = [f for f in facts_1fact if passes(f)]
known_2fact = [f for f in facts_2fact if passes(f)]

for name, pool, known in [("1-fact", facts_1fact, known_1fact),
                          ("2-fact", facts_2fact, known_2fact)]:
    print(f"{name}: {len(known)}/{len(pool)} facts pass the 75% knowledge check")
for kind in ("age", "atomic", "static"):
    sub = [f for f in facts_1fact if f["kind"] == kind]
    print(f"  {kind}: {sum(map(passes, sub))}/{len(sub)} pass")

# %% [markdown]
# ## 5. Hold out few-shot facts
#
# 1-fact: 5 random facts are removed and become the 5 few-shot examples.
# 2-fact: 10 random elements are removed and paired into 5 few-shot examples
# (paper Appendix A; each 2-fact example needs two elements).

# %%
rng = random.Random(SEED)

fs_1fact = rng.sample(known_1fact, 5)
eval_1fact = [f for f in known_1fact if f["key"] not in {g["key"] for g in fs_1fact}]

fs_elements = rng.sample(known_2fact, 10)
fs_2fact_pairs = [(fs_elements[2 * i], fs_elements[2 * i + 1]) for i in range(5)]
eval_2fact = [f for f in known_2fact
              if f["key"] not in {g["key"] for g in fs_elements}]

print("1-fact few-shot:", [f["key"] for f in fs_1fact])
print("2-fact few-shot pairs:",
      [(a["key"], b["key"]) for a, b in fs_2fact_pairs])
print(f"eval pools: {len(eval_1fact)} facts (1-fact), {len(eval_2fact)} elements (2-fact)")

# %% [markdown]
# ## 6. Fixed test sets
#
# The SAME test set is reused for every k (paper: "same fixed test set across
# all k values") so per-example wrong→right flips are comparable (McNemar).
#
# 1-fact: (fact, random two-digit addend X) — facts may repeat with different X.
# 2-fact: ordered pairs of distinct elements.

# %%
def sample_1fact_examples(pool, n, rng):
    out, seen = [], set()
    while len(out) < n:
        fact = rng.choice(pool)
        x = rng.randint(10, 99)
        if (fact["key"], x) in seen:
            continue
        seen.add((fact["key"], x))
        out.append(dict(
            idx=len(out), key=fact["key"], x=x,
            question=f"What is {fact['noun']} plus {x}?",
            target=fact["answer"] + x))
    return out


def sample_2fact_examples(pool, n, rng):
    out, seen = [], set()
    while len(out) < n:
        a, b = rng.sample(pool, 2)
        if (a["key"], b["key"]) in seen:
            continue
        seen.add((a["key"], b["key"]))
        out.append(dict(
            idx=len(out), key=f"{a['key']}+{b['key']}",
            question=(f"What is the atomic number of {a['key']} plus "
                      f"the atomic number of {b['key']}?"),
            target=a["answer"] + b["answer"]))
    return out


test_1fact = sample_1fact_examples(eval_1fact, N_1FACT, random.Random(SEED + 1))
test_2fact = sample_2fact_examples(eval_2fact, N_2FACT, random.Random(SEED + 2))
print(test_1fact[0]["question"], "->", test_1fact[0]["target"])
print(test_2fact[0]["question"], "->", test_2fact[0]["target"])

# %% [markdown]
# ## 7. Prompts (paper Appendix A) and dataset files
#
# System prompt (the filler sentence is dropped at k=0, and k is spliced in
# otherwise — the paper varies it "to reflect the correct filler type and length"):
#
# > You will be given a question. Answer immediately using the format 'Answer:
# > [ANSWER]' where [ANSWER] is just the number, nothing else. No explanation, no
# > words, no reasoning, just the number. After the question, there will be {k}
# > filler tokens (a sequence of dots) before you answer.
#
# Final user turn: `Question: {question}\nFiller: {k dots}\nAnswer:` — and the 5
# few-shot pairs carry the same filler as the eval condition.
#
# Output: `data/fig2_1fact.jsonl` and `data/fig2_2fact.jsonl`, one record per
# (example, k) with the fully rendered chat messages — 07 just replays them.

# %%
SYSTEM_FILLER = (
    ANSWER_FORMAT +
    " After the question, there will be {k} filler tokens "
    "(a sequence of dots) before you answer."
)


def dots(k: int) -> str:
    return " ".join(["."] * k)


def user_turn(question: str, k: int) -> str:
    if k == 0:
        return f"Question: {question}\nAnswer:"
    return f"Question: {question}\nFiller: {dots(k)}\nAnswer:"


def build_messages(question: str, k: int, fewshot: list[tuple[str, int]]) -> list[dict]:
    system = ANSWER_FORMAT if k == 0 else SYSTEM_FILLER.format(k=k)
    msgs = [{"role": "system", "content": system}]
    for fs_q, fs_target in fewshot:
        msgs.append({"role": "user", "content": user_turn(fs_q, k)})
        # the user turn already ends with "Answer:", so the assistant few-shot
        # reply is the bare number (paper Appendix A example)
        msgs.append({"role": "assistant", "content": str(fs_target)})
    msgs.append({"role": "user", "content": user_turn(question, k)})
    return msgs


fewshot_1fact = [(f"What is {f['noun']} plus {x}?", f["answer"] + x)
                 for f, x in zip(fs_1fact,
                                 random.Random(SEED + 3).sample(range(10, 100), 5))]
fewshot_2fact = [((f"What is the atomic number of {a['key']} plus "
                   f"the atomic number of {b['key']}?"), a["answer"] + b["answer"])
                 for a, b in fs_2fact_pairs]


def write_dataset(path, task, examples, fewshot):
    with open(path, "w") as f:
        for k in KS:
            for ex in examples:
                rec = dict(task=task, k=k, idx=ex["idx"], key=ex["key"],
                           question=ex["question"], target=ex["target"],
                           messages=build_messages(ex["question"], k, fewshot))
                f.write(json.dumps(rec) + "\n")
    print(f"wrote {path}  ({len(KS)} k-values x {len(examples)} examples)")


write_dataset(os.path.join(OUT_DIR, "fig2_1fact.jsonl"), "1fact",
              test_1fact, fewshot_1fact)
write_dataset(os.path.join(OUT_DIR, "fig2_2fact.jsonl"), "2fact",
              test_2fact, fewshot_2fact)

# metadata sidecar: everything needed to audit / rebuild the datasets
meta = dict(
    seed=SEED, ks=KS, n_1fact=N_1FACT, n_2fact=N_2FACT,
    api_model=API_MODEL, use_mock=USE_MOCK,
    knowledge_trials=KNOWLEDGE_TRIALS, knowledge_min_correct=KNOWLEDGE_MIN_CORRECT,
    knowledge_temperature=KNOWLEDGE_TEMPERATURE,
    n_facts_prefilter=dict(fact1=len(facts_1fact), fact2=len(facts_2fact)),
    n_facts_known=dict(fact1=len(known_1fact), fact2=len(known_2fact)),
    fewshot_1fact=fewshot_1fact, fewshot_2fact=fewshot_2fact,
    fewshot_1fact_keys=[f["key"] for f in fs_1fact],
    fewshot_2fact_elements=[f["key"] for f in fs_elements],
)
with open(os.path.join(OUT_DIR, "fig2_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("wrote", os.path.join(OUT_DIR, "fig2_meta.json"))

# %% [markdown]
# ## 8. Eyeball one rendered prompt per task

# %%
def show(messages):
    for m in messages:
        print(f"### {m['role']}\n{m['content']}\n")


show(build_messages(test_1fact[0]["question"], 10, fewshot_1fact)[:4])
print("=" * 70)
show(build_messages(test_2fact[0]["question"], 10, fewshot_2fact)[:2])
