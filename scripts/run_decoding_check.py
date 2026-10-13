"""
run_sampling.py
===============
Decoding sensitivity check for the reviewer question "greedy decoding is not
ideal for reasoning models".

Re-runs every thinking-on condition (5 scaffolds) on the full ContractNLI dev
split with Qwen3's recommended thinking-mode sampling (temperature 0.6,
top-p 0.95, top-k 20), with three seeds, for Qwen3-8B, 4B and 1.7B. The three
seeds run in parallel against one vLLM server per model.

Results: results/matrix-think-s<seed>-<condition>/<model>/records_dev_n1037_seed0.jsonl
Safe to re-run: finished instances are skipped.
"""

import os
import sys
import json
import time
import subprocess
import urllib.request

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
LOGS = os.path.join(WORK, "logs")
MAX_LEN = {"Qwen/Qwen3-8B": 40960, "Qwen/Qwen3-4B": 32768,
           "Qwen/Qwen3-1.7B": 32768}
# Default: Qwen3-8B only. Add sizes with SAMPLING_MODELS="Qwen/Qwen3-8B Qwen/Qwen3-4B".
MODELS = [(m, MAX_LEN[m]) for m in
          os.environ.get("SAMPLING_MODELS", "Qwen/Qwen3-8B").split()]
SEEDS = [1, 2, 3]
# Default: zero-shot only. More with SAMPLING_CONDITIONS="zeroshot cot".
CONDITIONS = os.environ.get("SAMPLING_CONDITIONS", "zeroshot").split()
SAMPLING = {"CONTRACTNLI_TEMPERATURE": "0.6", "CONTRACTNLI_TOP_P": "0.95",
            "CONTRACTNLI_TOP_K": "20"}
BASE = "http://localhost:8000/v1"


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def up(model):
    try:
        with urllib.request.urlopen(f"{BASE}/models", timeout=5) as r:
            return model in [m["id"] for m in json.load(r)["data"]]
    except Exception:
        return False


def serve(model, max_len):
    fh = open(os.path.join(LOGS, f"vllm_{model.split('/')[-1]}.log"), "a")
    p = subprocess.Popen(
        ["vllm", "serve", model, "--served-model-name", model,
         "--dtype", "bfloat16", "--max-model-len", str(max_len),
         "--gpu-memory-utilization", "0.90", "--enable-prefix-caching",
         "--port", "8000"], stdout=fh, stderr=subprocess.STDOUT)
    log(f"starting {model} (pid {p.pid})")
    t0 = time.time()
    while not up(model):
        if p.poll() is not None:
            sys.exit(f"vLLM exited - see logs/vllm_{model.split('/')[-1]}.log")
        time.sleep(10)
    log(f"{model} ready after {time.time() - t0:.0f}s")
    return p


def cmd(model, n, seed, extra=()):
    return [sys.executable, "-u", "src/run_matrix.py", "--models", model,
            "--base-url", BASE, "--api-key", "EMPTY", "--thinking",
            "--n", str(n), "--concurrency", "16", *extra]


def full_cmd(model, seed):
    return cmd(model, 1037, seed, ["--conditions", *CONDITIONS])


def env_for(seed):
    return dict(os.environ,
                CONTRACTNLI_SAMPLE_SEED=str(seed),
                CONTRACTNLI_RUN_TAG=f"-s{seed}",
                CONTRACTNLI_TIMEOUT="900", CONTRACTNLI_MAX_TOKENS="6144",
                CONTRACTNLI_MAX_RETRIES="2", **SAMPLING)


def smoke(model):
    """Two seeds on 6 claims must not give identical outputs everywhere,
    otherwise sampling is not actually on."""
    usage = {}
    for seed in (101, 102):
        subprocess.run(cmd(model, 6, seed, ["--seed", "99", "--conditions",
                                            "zeroshot"]),
                       cwd=WORK, env=env_for(seed), check=True,
                       stdout=subprocess.DEVNULL)
        p = os.path.join(WORK, "results", f"matrix-think-s{seed}-zeroshot",
                         model.split("/")[-1].lower(),
                         "records_dev_n6_seed99.jsonl")
        rs = [json.loads(l) for l in open(p) if l.strip()]
        if any(r["error"] for r in rs):
            sys.exit(f"smoke errors: {[r['error'] for r in rs if r['error']][:2]}")
        usage[seed] = [(r["usage"] or {}).get("completion") for r in rs]
    log(f"  smoke completion tokens seed101={usage[101]} seed102={usage[102]}")
    if usage[101] == usage[102]:
        sys.exit("smoke FAILED: two seeds gave identical outputs - sampling is off")
    log("  smoke passed: sampling is on")


def main():
    os.makedirs(LOGS, exist_ok=True)
    t0 = time.time()
    for model, max_len in MODELS:
        srv = None if up(model) else serve(model, max_len)
        smoke(model)
        procs = []
        for seed in SEEDS:
            name = f"{model.split('/')[-1].lower()}_s{seed}"
            fh = open(os.path.join(LOGS, f"sampling_{name}.log"), "a")
            procs.append((name, subprocess.Popen(full_cmd(model, seed),
                                                 cwd=WORK, env=env_for(seed),
                                                 stdout=fh, stderr=subprocess.STDOUT)))
            log(f"started {name}")
        while any(p.poll() is None for _, p in procs):
            time.sleep(120)
            slug = model.split("/")[-1].lower()
            done = 0
            for seed in SEEDS:
                for c in CONDITIONS:
                    f = os.path.join(WORK, "results", f"matrix-think-s{seed}-{c}",
                                     slug, "records_dev_n1037_seed0.jsonl")
                    if os.path.exists(f):
                        done += sum(1 for _ in open(f))
            log(f"{model}: {done} / {len(SEEDS) * len(CONDITIONS) * 1037} records | {(time.time() - t0) / 3600:.1f} h")
        for name, p in procs:
            log(f"finished {name} (exit {p.returncode})")
        if srv:
            srv.terminate()
            srv.wait(timeout=120)
            time.sleep(10)
    log(f"SAMPLING DONE in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
