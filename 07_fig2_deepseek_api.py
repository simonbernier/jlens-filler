# %% [markdown]
# # Figure 2 replication — DeepSeek V4 Flash via API, dot filler
#
# Runs the datasets built by `06_build_fig2_dataset.py`
# (`data/fig2_1fact.jsonl`, `data/fig2_2fact.jsonl`) against the API and
# reproduces Figure 2 of *Reading Between the Dots* (arXiv:2607.03502):
# accuracy vs number of dot filler tokens, one panel per task, with the
# no-filler (k=0) baseline as a dashed line, ±1 binomial-SE error bars, and
# exact McNemar tests of each k against baseline.
#
# Resumable: replies are appended to `results/fig2_raw.jsonl`; rerunning the
# sweep cell skips everything already answered. Set USE_MOCK = True for a free
# end-to-end dry run (simulated replies whose accuracy grows with k).
#
# Run cells top-to-bottom in VS Code (# %% = one Jupyter cell).

# %% Config
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenRouter (OpenAI-compatible). Key: export OPENROUTER_API_KEY=sk-or-...
API_MODEL = "deepseek/deepseek-v4-flash"
BASE_URL = "https://openrouter.ai/api/v1"
WORKERS = 8

# temperature=0 (greedy), matching the paper's released eval code (their vLLM
# SamplingParams AND API calls both use temperature=0). One greedy sample per
# example gives a deterministic per-example outcome, which is what the fixed
# test set + McNemar flip counts are built on. V4's serving default of 0.6 is
# a chat default, not a methodological choice — sampling at 0.6 would require
# several samples per example to estimate per-example accuracy (m x cost) and
# would add noise to the k=0 vs k pairing.
TEMPERATURE = 0.0

# Pin one OpenRouter provider so every call (and every k) hits the same serving
# stack — with ~17 providers behind this model id, routing drift is a bigger
# consistency risk than temperature. Example:
#   PROVIDER = {"order": ["deepseek"], "allow_fallbacks": False}
PROVIDER: dict | None = None

USE_MOCK = False                   # True = no API calls; simulated replies

DATASETS = {"1fact": "data/fig2_1fact.jsonl",
            "2fact": "data/fig2_2fact.jsonl"}
RAW_PATH = "results/fig2_raw.jsonl"
os.makedirs("results", exist_ok=True)

# %% [markdown]
# ## 1. Load the datasets from 06

# %%
records: list[dict] = []
for task, path in DATASETS.items():
    assert os.path.exists(path), f"{path} missing — run 06_build_fig2_dataset.py first"
    with open(path) as f:
        records += [json.loads(line) for line in f]

meta = json.load(open("data/fig2_meta.json"))
KS = meta["ks"]
print(f"{len(records)} (example, k) records; ks={KS}; "
      f"n per condition: 1fact={meta['n_1fact']}, 2fact={meta['n_2fact']}")
if meta.get("use_mock"):
    print("WARNING: datasets were built with USE_MOCK=True — rebuild 06 with the "
          "real API before a real sweep (knowledge filtering was simulated).")

# %% [markdown]
# ## 2. API client (or mock)

# %%
def parse_answer(text: str) -> int | None:
    """First integer in the reply ('Answer: 138' -> 138)."""
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


class MockClient:
    """Deterministic fake: per-example correctness probability rises with k,
    so the plot shows a Figure-2-like shape without spending API money."""

    def query(self, rec) -> str:
        h = int(hashlib.md5(f"{rec['task']}|{rec['idx']}|{rec['k']}".encode())
                .hexdigest(), 16)
        base = 0.55 if rec["task"] == "1fact" else 0.21
        uplift = (0.18 if rec["task"] == "1fact" else 0.05)
        p = base + uplift * min(rec["k"], 50) / 50
        return f"Answer: {rec['target'] if (h % 1000) / 1000 < p else rec['target'] + 1 + h % 7}"


def make_client():
    if USE_MOCK:
        return MockClient()
    from openai import OpenAI  # pip install openai
    key = (os.environ.get("OPENROUTER_API_KEY")
           or os.environ.get("DEEPSEEK_API_KEY")
           or os.environ.get("OPENAI_API_KEY"))
    assert key, "Set OPENROUTER_API_KEY in your environment."
    return OpenAI(api_key=key, base_url=BASE_URL)


def query(client, rec, max_retries=5) -> str:
    if isinstance(client, MockClient):
        return client.query(rec)
    delay = 2.0
    for attempt in range(max_retries):
        try:
            extra = {"extra_body": {"provider": PROVIDER}} if PROVIDER else {}
            resp = client.chat.completions.create(
                model=API_MODEL, messages=rec["messages"],
                temperature=TEMPERATURE, max_tokens=16, **extra)
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return ""


client = make_client()

# %% [markdown]
# ## 3. Sweep (resumable)
#
# One line per (task, k, example) in `results/fig2_raw.jsonl`. Interrupt freely;
# rerun this cell to continue where it stopped.

# %%
def load_done(path: str) -> dict:
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[(r["task"], r["k"], r["idx"])] = r
    return done


done = load_done(RAW_PATH)
todo = [r for r in records if (r["task"], r["k"], r["idx"]) not in done]
print(f"{len(done)} cached, {len(todo)} API calls to make")
lock = threading.Lock()


def work(rec):
    reply = query(client, rec)
    pred = parse_answer(reply)
    out = dict(task=rec["task"], k=rec["k"], idx=rec["idx"], key=rec["key"],
               target=rec["target"], reply=reply.strip()[:64],
               pred=pred, correct=bool(pred == rec["target"]))
    with lock:
        with open(RAW_PATH, "a") as f:
            f.write(json.dumps(out) + "\n")
    return out


t0, n_done = time.time(), 0
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for fut in as_completed([pool.submit(work, r) for r in todo]):
        fut.result()
        n_done += 1
        if n_done % 200 == 0:
            rate = n_done / (time.time() - t0)
            eta = (len(todo) - n_done) / max(rate, 1e-9) / 60
            print(f"  {n_done}/{len(todo)}  ({rate:.1f} req/s, ~{eta:.0f} min left)")
print("sweep complete")

# %% [markdown]
# ## 4. Summary table — accuracy, SE, McNemar vs baseline
#
# McNemar uses the paper's design: the SAME fixed test set at every k, so each
# example's k=0 vs k>0 outcomes pair up. b = wrong→right flips, c = right→wrong.

# %%
import math

import pandas as pd


def binomial_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (flips ~ Binomial(b+c, 1/2) under H0)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


df = pd.DataFrame(load_done(RAW_PATH).values())
summary = []
for task, tdf in df.groupby("task"):
    base = tdf[tdf.k == 0].set_index("idx")["correct"]
    for k, g in tdf.groupby("k"):
        acc = g["correct"].mean()
        rec = dict(task=task, k=int(k), n=len(g),
                   accuracy=acc, se=binomial_se(acc, len(g)))
        if k != 0 and len(base):
            m = g.set_index("idx")["correct"].reindex(base.index).dropna()
            b = int(((~base.loc[m.index]) & m).sum())
            c = int((base.loc[m.index] & (~m)).sum())
            rec.update(flips_wrong_to_right=b, flips_right_to_wrong=c,
                       mcnemar_p=mcnemar_exact(b, c))
        summary.append(rec)

sdf = pd.DataFrame(summary).sort_values(["task", "k"])
sdf.to_csv("results/fig2_summary.csv", index=False)
print(sdf.to_string(index=False))
print("\nwrote results/fig2_summary.csv")

# %% [markdown]
# ## 5. Figure 2 — accuracy vs k, one panel per task

# %%
import matplotlib.pyplot as plt

TASK_TITLES = {"1fact": "1-fact addition", "2fact": "2-fact addition"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, task in zip(axes, ["1fact", "2fact"]):
    g = sdf[sdf.task == task].sort_values("k")
    ks = [k for k in g.k if k != 0]
    xpos = {k: i for i, k in enumerate(ks)}

    base = g[g.k == 0]
    if len(base):
        b = base.iloc[0]
        ax.axhline(b.accuracy * 100, color="gray", ls="--", lw=1,
                   label=f"no filler ({b.accuracy * 100:.1f}%)")
        ax.fill_between([-0.4, len(ks) - 0.6],
                        (b.accuracy - b.se) * 100, (b.accuracy + b.se) * 100,
                        color="gray", alpha=0.15, lw=0)

    gk = g[g.k != 0]
    ax.errorbar([xpos[k] for k in gk.k], gk.accuracy * 100, yerr=gk.se * 100,
                marker="o", ms=4, capsize=3, color="tab:blue", label="dots")
    # asterisk over k values with a significant McNemar flip test
    for _, row in gk.iterrows():
        if row.get("mcnemar_p", 1.0) < 0.05:
            ax.annotate("*", (xpos[row.k], (row.accuracy + row.se) * 100 + 1),
                        ha="center", color="tab:blue")

    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.10)   # headroom for the asterisks
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(-0.4, len(ks) - 0.6)
    ax.set_xlabel("Number of Filler [k]")
    ax.set_ylabel("Accuracy (%)")
    n = int(g.n.max())
    ax.set_title(f"{TASK_TITLES[task]} — {API_MODEL} (n={n}/condition)")
    ax.legend(frameon=False, loc="lower right")

fig.suptitle("Filler-token uplift, dot filler (Fig. 2 replication)", y=1.02)
fig.tight_layout()
fig.savefig("results/fig2_accuracy_vs_k.png", dpi=150, bbox_inches="tight")
print("wrote results/fig2_accuracy_vs_k.png")
plt.show()
