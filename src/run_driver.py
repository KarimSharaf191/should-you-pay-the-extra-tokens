"""
run_driver.py
=============
Serves Qwen3-4B and Qwen3-1.7B with vLLM and runs the 5 scaffolds
(zeroshot, dspy, cot, rag, agentic_rag), thinking off and on, on the
150-instance stratified ContractNLI dev sample (seed 0) pinned by
sample_manifest_dev_n150_seed0.csv -- the same instances as the Qwen3-8B
n=150 runs.

  2 GPUs (Kaggle "GPU T4 x2"): 4B on GPU 0, 1.7B on GPU 1, all 4 jobs at once.
  1 GPU                      : one model at a time (1.7B first).

Starts with a 6-instance smoke test on Qwen3-1.7B that checks thinking is
really off (about 0 reasoning tokens) and really on.

Safe to re-run: run_matrix.py resumes, so finished instances are skipped.
"""

import os
import sys
import json
import time
import subprocess
import urllib.request

WORK = os.environ.get("CONTRACTNLI_DATA_DIR",
                      os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(WORK, "logs")
MANIFEST = "sample_manifest_dev_n150_seed0.csv"
N = "150"


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [line.split(", ") for line in out.strip().splitlines()]


# On a 15 GB T4, Qwen3-4B's fp16 weights (~8 GB) leave too little KV cache for
# a 32K context and vLLM refuses to start. 16K still fits the longest contract
# (~11K tokens) plus a long thinking trace.
MAX_LEN = {"Qwen/Qwen3-4B": 16384, "Qwen/Qwen3-1.7B": 32768}


def serve(model, port, gpu, frac, dtype):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    fh = open(os.path.join(LOGS, f"vllm_{port}.log"), "a")
    p = subprocess.Popen(
        ["vllm", "serve", model, "--served-model-name", model,
         "--dtype", dtype, "--max-model-len", str(MAX_LEN[model]),
         "--gpu-memory-utilization", str(frac), "--enable-prefix-caching",
         "--port", str(port)],
        env=env, stdout=fh, stderr=subprocess.STDOUT)
    log(f"serving {model} on GPU {gpu}, port {port} (pid {p.pid})")
    return p


def wait_health(port, proc, timeout=2400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            # The root cause sits well above the final traceback.
            os.system(f"grep -iE 'error|memory|exceed' {LOGS}/vllm_{port}.log "
                      f"| grep -v '^\\s*File' | tail -n 15")
            sys.exit(f"vLLM on port {port} exited - see logs/vllm_{port}.log")
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
            log(f"port {port} ready after {time.time() - t0:.0f}s")
            return
        except Exception:
            time.sleep(10)
    sys.exit(f"vLLM on port {port} did not come up in {timeout}s")


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(10)


def matrix_cmd(model, port, extra):
    return ([sys.executable, "-u", "run_matrix.py", "--models", model,
             "--base-url", f"http://localhost:{port}/v1", "--api-key", "EMPTY"]
            + extra)


def smoke_test(model, port):
    log("smoke test: is thinking really off / on?")
    for think in ([], ["--thinking"]):
        subprocess.run(matrix_cmd(model, port, ["--n", "6", "--seed", "99",
                                                "--conditions", "zeroshot", "dspy",
                                                "--concurrency", "6"] + think),
                       cwd=WORK, check=True, stdout=subprocess.DEVNULL)
    slug = model.split("/")[-1].lower()
    for prefix, want_think in (("matrix", False), ("matrix-think", True)):
        for cond in ("zeroshot", "dspy"):
            path = os.path.join(WORK, "results", f"{prefix}-{cond}", slug,
                                "records_dev_n6_seed99.jsonl")
            recs = [json.loads(line) for line in open(path)]
            errs = [r["error"] for r in recs if r["error"]]
            think = [(r["usage"] or {}).get("reasoning", 0)
                     for r in recs if not r["error"]]
            log(f"  {prefix:13} {cond:9} errors={len(errs)} reasoning={think}")
            if errs:
                sys.exit(f"smoke test errors: {errs[:2]}")
            # An empty <think></think> block still counts as 1-2 tokens.
            ok = (all(t > 20 for t in think) if want_think
                  else all(t <= 2 for t in think))
            if not ok:
                sys.exit(f"smoke test FAILED: thinking not "
                         f"{'on' if want_think else 'off'} for {cond}")
    log("smoke test passed")


def run_jobs(pairs):
    """pairs: (model, port). Runs thinking off + on for each, all in parallel."""
    procs = []
    for model, port in pairs:
        for think in (False, True):
            name = f"{model.split('/')[-1].lower()}_{'think' if think else 'nothink'}"
            fh = open(os.path.join(LOGS, f"run_{name}.log"), "a")
            cmd = matrix_cmd(model, port, ["--n", N, "--manifest", MANIFEST,
                                           "--concurrency", "16"]
                             + (["--thinking"] if think else []))
            procs.append((name, subprocess.Popen(cmd, cwd=WORK, stdout=fh,
                                                 stderr=subprocess.STDOUT)))
            log(f"started {name}")
    slugs = [m.split("/")[-1].lower() for m, _ in pairs]
    while any(p.poll() is None for _, p in procs):
        time.sleep(120)
        done = sum(sum(1 for _ in open(f)) for f in _record_files()
                   if any(f"{os.sep}{s}{os.sep}" in f for s in slugs))
        log(f"progress: {done} / {10 * len(pairs) * int(N)} records")
    for name, p in procs:
        log(f"finished {name} (exit {p.returncode})")


def _record_files():
    import glob
    return glob.glob(os.path.join(WORK, "results", "matrix*-*", "qwen3-*",
                                  f"records_dev_n{N}_seed0.jsonl"))


def already_up(port, model):
    """True if a server on `port` is already serving `model`."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models",
                                    timeout=5) as r:
            return model in [m["id"] for m in json.load(r)["data"]]
    except Exception:
        return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(MAX_LEN),
                    default=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"],
                    help="which models to run (default: both)")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    os.environ.setdefault("CONTRACTNLI_TIMEOUT", "900")
    g = gpus()
    for row in g:
        log(f"GPU: {row}")
    # Turing (T4, compute 7.5) and older have no bf16.
    dtype = "bfloat16" if float(g[0][2]) >= 8.0 else "float16"
    log(f"dtype: {dtype} | models: {args.models}")

    t0 = time.time()
    # One model per GPU when there are enough GPUs; otherwise one at a time.
    together = len(g) >= len(args.models)
    batches = ([list(enumerate(args.models))] if together
               else [[(0, m)] for m in args.models])
    smoke_done = args.skip_smoke
    for batch in batches:
        servers = []
        for gpu, model in batch:
            port = 8000 + gpu
            # Reuse a server left running by an earlier attempt in this session.
            proc = None if already_up(port, model) else serve(model, port, gpu, 0.90, dtype)
            servers.append((model, port, proc))
        for model, port, proc in servers:
            if proc:
                wait_health(port, proc)
            else:
                log(f"reusing server already running on port {port}")
        if not smoke_done:
            smoke_test(servers[0][0], servers[0][1])
            smoke_done = True
        run_jobs([(m, p) for m, p, _ in servers])
        for _, _, proc in servers:
            if proc:
                stop(proc)
    log(f"ALL DONE in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
