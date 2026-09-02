"""
Stages 0-3 on the dev model in one go, each step logged to
results/pipeline_log.txt (console output is mirrored there).

    python run_qwen_pipeline.py            # or double-click run_qwen_pipeline.bat

Steps: smoke test -> 20 (both lenses, fresh greedy answers) -> 21 -> 30 ->
tokenizer-only preflight of the DeepSeek prompt. Stops at the first failure.
"""
import datetime
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)
LOG = "results/pipeline_log.txt"

env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8",
           PYTHONUNBUFFERED="1", TQDM_DISABLE="1")
try:                                   # HF token / API key from .env, as VS Code does
    from dotenv import dotenv_values
    env.update({k: v for k, v in dotenv_values(".env").items() if v})
except ImportError:
    pass

TAG = "dev_dots-10"
STEPS = [
    ("smoke test", ["00_smoke_test.py"]),
    ("readout, both lenses", ["20_lens_readout.py", "--model", "dev", "--n", "300",
                              "--k", "10", "--lens", "both", "--regen-answers"]),
    ("figure 3 heatmaps", ["21_analyze_readout.py", "--tag", TAG]),
    ("J-lens vs logit-lens", ["30_compare_lenses.py", "--tag", TAG]),
    ("deepseek tokenizer preflight", ["00_smoke_test.py", "--model", "deepseek",
                                      "--tokenizer-only"]),
]

with open(LOG, "a", encoding="utf-8") as log:
    def say(line: str):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"\n===== pipeline start {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
        f"({sys.executable}) =====")
    for name, argv in STEPS:
        say(f"\n----- {name}: python {' '.join(argv)} -----")
        proc = subprocess.Popen([sys.executable, "-u"] + argv, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            say(line.rstrip("\n"))
        proc.wait()
        say(f"----- {name}: exit code {proc.returncode} -----")
        if proc.returncode:
            say("stopping at the first failure")
            break
    say(f"===== pipeline end {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====")
