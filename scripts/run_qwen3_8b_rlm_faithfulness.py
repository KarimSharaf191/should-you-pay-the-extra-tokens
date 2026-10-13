"""
run_8b.py
=========
Phase 1 of the Colab run: everything that needs Qwen3-8B, self-hosted with vLLM.

  0. smoke tests  thinking really off; RLM + faithfulness run end to end on
                  1-4 instances of every benchmark (catches data-path bugs
                  before hours are spent)
  1. baseline     zeroshot on the full ContractNLI dev split, on THIS server,
                  as the full-context reference for faithfulness study 1
  2. in parallel  RLM on ContractNLI dev (1,037), CoverBench (733),
                  FinDVer testmini (700), each split into K processes;
                  faithfulness on ContractNLI (studies 1-4), FinDVer
                  (1-4, oracle form) and CoverBench (3-4)

The model is served as `qwen3-8b-vllm`, so results land in their own folders
and never mix with the earlier OpenRouter `qwen3-8b` runs.

Safe to re-run: every job resumes, so finished instances are skipped.
"""

import os
import sys
import glob
import json
import time
import subprocess
import urllib.request

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
LOGS = os.path.join(WORK, "logs")
PORT = 8000
SERVED = "qwen3-8b-vllm"
BASE = f"http://localhost:{PORT}/v1"
N = {"contractnli": 1037, "coverbench": 733, "findver": 700}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def gpu_mem_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    name, mib = out.splitlines()[0].split(", ")
    log(f"GPU: {name}, {int(mib) / 1024:.0f} GB")
    return int(mib) / 1024


def up():
    try:
        with urllib.request.urlopen(f"{BASE}/models", timeout=5) as r:
            return SERVED in [m["id"] for m in json.load(r)["data"]]
    except Exception:
        return False


def serve():
    if up():
        log("reusing the running Qwen3-8B server")
        return None
    fh = open(os.path.join(LOGS, "vllm_8b.log"), "a")
    p = subprocess.Popen(
        ["vllm", "serve", "Qwen/Qwen3-8B", "--served-model-name", SERVED,
         "--dtype", "bfloat16", "--max-model-len", "40960",
         "--gpu-memory-utilization", "0.90", "--enable-prefix-caching",
         "--port", str(PORT)],
        stdout=fh, stderr=subprocess.STDOUT)
    log(f"starting vLLM Qwen3-8B (pid {p.pid}); first start downloads ~16 GB")
    t0 = time.time()
    while time.time() - t0 < 2400:
        if p.poll() is not None:
            os.system(f"grep -iE 'error|memory' {LOGS}/vllm_8b.log | tail -n 15")
            sys.exit("vLLM exited - see logs/vllm_8b.log")
        if up():
            log(f"server ready after {time.time() - t0:.0f}s")
            return p
        time.sleep(10)
    sys.exit("vLLM did not come up in 40 min")


def py(*args):
    return [sys.executable, "-u", *args]


def run(cmd, name):
    """Run to completion, output to logs/<name>.log; stop on failure."""
    with open(os.path.join(LOGS, f"{name}.log"), "a") as fh:
        rc = subprocess.run(cmd, cwd=WORK, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        os.system(f"tail -n 25 {LOGS}/{name}.log")
        sys.exit(f"{name} failed (exit {rc}) - see logs/{name}.log")


def start(cmd, name):
    fh = open(os.path.join(LOGS, f"{name}.log"), "a")
    return name, subprocess.Popen(cmd, cwd=WORK, stdout=fh, stderr=subprocess.STDOUT)


def rows(pattern):
    n = 0
    for f in glob.glob(os.path.join(WORK, pattern)):
        with open(f, encoding="utf-8") as fh:
            n += sum(1 for line in fh if line.strip())
    return n


def read(pattern):
    out = []
    for f in glob.glob(os.path.join(WORK, pattern)):
        with open(f, encoding="utf-8") as fh:
            out += [json.loads(line) for line in fh if line.strip()]
    return out


def smoke():
    log("smoke 1/3: thinking really off on this server")
    run(py("src/run_matrix.py", "--models", SERVED, "--base-url", BASE,
           "--api-key", "EMPTY", "--conditions", "zeroshot", "--n", "6",
           "--seed", "99", "--concurrency", "6"), "smoke_matrix")
    recs = read(f"results/matrix-zeroshot/{SERVED}/records_dev_n6_seed99.jsonl")
    errs = [r["error"] for r in recs if r["error"]]
    think = [(r["usage"] or {}).get("reasoning", 0) for r in recs if not r["error"]]
    log(f"  errors={len(errs)} reasoning={think}")
    if errs or any(t > 2 for t in think):
        sys.exit(f"smoke FAILED: errors {errs[:2]} / reasoning {think}")

    log("smoke 2/3: RLM end to end, 1 instance per benchmark")
    for b in N:
        run(py("src/rlm_multi.py", "--benchmark", b, "--n", "1", "--seed", "99",
               "--backend", "vllm", "--base-url", BASE, "--model", SERVED),
            f"smoke_rlm_{b}")
        r = read(f"results/rlm-multi/{b}/rlm_nothink_n1_seed99.jsonl")
        bad = [x.get("error") for x in r if x.get("error") or not x.get("parsed_ok")]
        log(f"  {b:11} records={len(r)} calls={[x.get('n_calls') for x in r]} "
            f"think_blocks={[x.get('any_think_block') for x in r]} problems={bad}")
        if not r or any(x.get("error") for x in r):
            sys.exit(f"smoke FAILED: RLM {b}: {bad}")
        if any(x.get("any_think_block") for x in r):
            sys.exit(f"smoke FAILED: RLM {b} produced <think> blocks - thinking is on")

    log("smoke 3/3: faithfulness end to end, 4 instances")
    run(py("src/faithfulness_study.py", "--n", "4", "--seed", "99",
           "--studies", "1,2,3,4", "--model", SERVED, "--concurrency", "4"),
        "smoke_faith_contractnli")
    run(py("src/faithfulness_multi.py", "--benchmark", "findver", "--n", "4",
           "--seed", "99", "--studies", "1,2,3,4", "--model", SERVED,
           "--concurrency", "4"), "smoke_faith_findver")
    run(py("src/faithfulness_multi.py", "--benchmark", "coverbench", "--n", "4",
           "--seed", "99", "--studies", "3,4", "--model", SERVED,
           "--concurrency", "4"), "smoke_faith_coverbench")
    for pat in ("results/faithfulness/study*_n4.jsonl",
                "results/faithfulness-multi/findver/study*_n4.jsonl",
                "results/faithfulness-multi/coverbench/study*_n4.jsonl"):
        k = rows(pat)
        log(f"  {pat:52} rows={k}")
        if k == 0:
            sys.exit(f"smoke FAILED: nothing written for {pat}")
    log("smoke tests passed")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--shards", type=int, default=None,
                    help="RLM processes per benchmark (default: by GPU memory)")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    os.environ.update({
        "LLM_BASE_URL": BASE,
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY") or "EMPTY",
        "FAITH_BASELINE_SLUG": SERVED,
    })
    os.environ.setdefault("FINDVER_DIR", os.path.join(WORK, "data", "findver"))
    os.environ.setdefault("COVERBENCH_FILE",
                          os.path.join(WORK, "data", "coverbench", "coverbench.json"))
    os.environ.setdefault("CONTRACTNLI_TIMEOUT", "900")
    # Output caps only stop a runaway greedy loop; normal outputs are far shorter.
    os.environ.setdefault("CONTRACTNLI_MAX_TOKENS", "4096")
    os.environ.setdefault("RLM_MAX_TOKENS", "4096")

    mem = gpu_mem_gb()
    k = args.shards or (12 if mem > 60 else 6)
    t0 = time.time()
    server = serve()
    if not args.skip_smoke:
        smoke()

    log("step 1: full-context zeroshot baseline on ContractNLI dev (1,037)")
    run(py("src/run_matrix.py", "--models", SERVED, "--base-url", BASE,
           "--api-key", "EMPTY", "--conditions", "zeroshot", "--n", "1037",
           "--concurrency", "32"), "baseline_zeroshot")

    log(f"step 2: RLM ({k} processes per benchmark) + faithfulness, in parallel")
    jobs = []
    for b, n in N.items():
        for i in range(k):
            jobs.append(start(py("src/rlm_multi.py", "--benchmark", b, "--n", str(n),
                                 "--backend", "vllm", "--base-url", BASE,
                                 "--model", SERVED, "--shard", str(i),
                                 "--num-shards", str(k)), f"rlm_{b}_{i}of{k}"))
    jobs.append(start(py("src/faithfulness_study.py", "--n", "1037",
                         "--studies", "1,2,3,4", "--model", SERVED,
                         "--concurrency", "8"), "faith_contractnli"))
    jobs.append(start(py("src/faithfulness_multi.py", "--benchmark", "findver",
                         "--n", "700", "--studies", "1,2,3,4", "--model", SERVED,
                         "--concurrency", "8"), "faith_findver"))
    jobs.append(start(py("src/faithfulness_multi.py", "--benchmark", "coverbench",
                         "--n", "733", "--studies", "3,4", "--model", SERVED,
                         "--concurrency", "8"), "faith_coverbench"))

    while any(p.poll() is None for _, p in jobs):
        time.sleep(300)
        rlm = {b: len({r["uid"] for r in read(f"results/rlm-multi/{b}/rlm_nothink_n{n}_seed0*.jsonl")
                       if not r.get("error")})
               for b, n in N.items()}
        log("RLM " + "  ".join(f"{b} {rlm[b]}/{n}" for b, n in N.items())
            + f" | faithfulness rows: cnli {rows('results/faithfulness/study*_n1037.jsonl')}"
            f" findver {rows('results/faithfulness-multi/findver/study*_n700.jsonl')}"
            f" coverbench {rows('results/faithfulness-multi/coverbench/study*_n733.jsonl')}"
            f" | {(time.time() - t0) / 3600:.1f} h")
    failed = [name for name, p in jobs if p.returncode]
    for name in failed:
        log(f"FAILED: {name} (exit {jobs[[j[0] for j in jobs].index(name)][1].returncode})"
            f" - see logs/{name}.log")

    # Instances that errored (timeouts etc.) are retried once, in one pass.
    errs = {b: sum(1 for r in read(f"results/rlm-multi/{b}/rlm_nothink_n{n}_seed0*.jsonl")
                   if r.get("error")) for b, n in N.items()}
    log(f"RLM error rows (retried automatically on re-run): {errs}")

    if server:
        server.terminate()
        server.wait(timeout=120)
        log("stopped the Qwen3-8B server (frees the GPU for phase 2)")
    log(f"phase 1 done in {(time.time() - t0) / 3600:.1f} h"
        + (f" - {len(failed)} job(s) failed, re-run to resume" if failed else ""))


if __name__ == "__main__":
    main()
