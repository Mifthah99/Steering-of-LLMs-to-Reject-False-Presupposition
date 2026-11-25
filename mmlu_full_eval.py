"""
MMLU eval for activation-steered models (and optionally the unsteered baseline).

Running with NO arguments evaluates, for both llama and qwen:
  - an unsteered baseline (same HF backend + same task, just no hook)
  - every chosen steering config
so baseline and steered differ ONLY in the steering vector.

NOTE: the earlier vLLM baseline used the log-likelihood `mmlu` task, which
is NOT comparable to `mmlu_generative` (and the last-token hook would have
no effect on log-likelihood scoring anyway).

Each eval runs in its OWN subprocess so GPU memory is fully released
between runs (vLLM / HF transformers don't always clean up in-process).

Usage:
    python mmlu_full_eval.py                     # baseline + steered for llama + qwen
    python mmlu_full_eval.py --skip_baseline     # steered runs only
    python mmlu_full_eval.py --force             # ignore cached result files
    python mmlu_full_eval.py --limit 20          # quick smoke test (20 docs/subtask)
    python mmlu_full_eval.py --gpus 0,2          # choose which GPUs to use (parallel, one job each)
    python mmlu_full_eval.py --worker ...        # internal: single run (called by orchestrator)
"""
import os
import sys
import json
import argparse
import subprocess
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Uses the token from the environment if set; otherwise relies on a cached
# `huggingface-cli login`. Avoid hardcoding tokens in scripts.
HF_TOKEN =  os.environ.get("HF_TOKEN")


RESULTS_DIR = Path("mmlu_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK = "mmlu_generative"   # generate_until variant, so the hook acts during decoding
NUM_FEWSHOT = 5

# ==========================================================================
# CHOSEN STEERING CONFIGS (from the layer/multiplier sweep)
# ==========================================================================
MODEL_CONFIGS = {
    "llama": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "size_b": 8,
        "vector_dir": "vectors/llama",
        "configs": [
            # {"layer": 11, "multiplier": 2.3},
            {"layer": 13, "multiplier": 6.5}, # for the two-layer config
        ],
    },
    "qwen": {
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "size_b": 14,
        "vector_dir": "vectors/qwen",
        "configs": [
            {"layer": 19, "multiplier": 3.0},
        ],
    },
}

def extract_acc(results: dict):
    metrics = results.get("results", {}).get(TASK, {})
    return metrics.get("exact_match,get_response", metrics.get("exact_match"))


def hf_login_if_token():
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)


def limit_suffix(limit):
    return f"_limit{limit}" if limit else ""


def baseline_path(tag, limit=None):
    return RESULTS_DIR / f"mmlu_{tag}_baseline{limit_suffix(limit)}.json"


def steered_path(tag, layer, mult, limit=None):
    return RESULTS_DIR / f"mmlu_{tag}_steered_L{layer}_M{mult}{limit_suffix(limit)}.json"


# ==========================================================================
# WORKERS: exactly one eval run each; write JSON; print accuracy last.
# ==========================================================================
def worker_hf(model_tag, model_name, vector_dir=None, layer_idx=None, multiplier=None, limit=None):
    """One HF eval run. With layer_idx=None it is the unsteered baseline;
    otherwise the steering hook is attached. Identical code path otherwise."""
    import torch
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_login_if_token()
    steered = layer_idx is not None
    desc = f"steered L{layer_idx} M{multiplier}" if steered else "baseline (no hook)"
    print(f"[{model_tag}] HF {desc} starting... (limit={limit})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()

    handle = None
    if steered:
        found_files = list(Path(vector_dir).glob(f"*layer_{layer_idx}_*.pt"))
        if not found_files:
            raise FileNotFoundError(f"Missing vector for layer {layer_idx} in {vector_dir}")
        if len(found_files) > 1:
            raise FileNotFoundError(f"Multiple vector files for layer {layer_idx}: {found_files}")
        vector = torch.load(found_files[0], map_location="cpu", weights_only=True)

        layer_module = model.model.layers[layer_idx]
        target_dev = next(layer_module.parameters()).device
        v = vector.view(1, 1, -1) if vector.dim() == 1 else vector
        v = (v * multiplier).to(device=target_dev, dtype=torch.bfloat16)

        # Same intervention as the generation pipeline: last prompt token during
        # prefill, every new token during decoding.
        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output
            if hidden_states.shape[1] > 1:
                hidden_states[:, -1:, :] += v
            else:
                hidden_states += v
            return (hidden_states,) + output[1:] if is_tuple else hidden_states

        handle = layer_module.register_forward_hook(hook)

    try:
        # NOTE: lm-eval warns "`pretrained` model kwarg is not of type `str`"
        # when given a loaded model. Harmless for a single-process run.
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=[TASK],
            num_fewshot=NUM_FEWSHOT,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            limit=limit,
        )
    finally:
        if handle is not None:
            handle.remove()

    out_path = (steered_path(model_tag, layer_idx, multiplier, limit) if steered
                else baseline_path(model_tag, limit))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    acc = extract_acc(results)
    print(f"[{model_tag}] Saved -> {out_path}", flush=True)
    print(f"RESULT_ACC={acc}", flush=True)   # parsed by orchestrator -- keep as last line


# ==========================================================================
# ORCHESTRATOR
# ==========================================================================
def parse_acc_from_output(stdout):
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("RESULT_ACC="):
            val = line.split("=", 1)[1].strip()
            return None if val == "None" else float(val)
    return None


LOG_DIR = RESULTS_DIR / "logs"


def load_cached_acc(path):
    with open(path) as f:
        return extract_acc(json.load(f))


def save_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def run_job_on_gpu(job, gpu_pool, limit_args):
    """Take a free GPU from the pool, run one worker pinned to it, give it back."""
    gpu = gpu_pool.get()
    log_path = LOG_DIR / f"{job['name']}.log"
    cmd = [sys.executable, os.path.abspath(sys.argv[0])] + job["args"] + limit_args
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",      # GPU ids match nvidia-smi
        CUDA_VISIBLE_DEVICES=str(gpu),
    )
    t0 = time.time()
    print(f"[GPU {gpu}] START {job['name']}   (tail -f {log_path})", flush=True)
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        mins = (time.time() - t0) / 60
        if proc.returncode != 0:
            print(f"[GPU {gpu}] FAILED {job['name']} after {mins:.1f} min "
                  f"(exit {proc.returncode}) -- see {log_path}", flush=True)
            return None, "failed"
        acc = parse_acc_from_output(log_path.read_text())
        print(f"[GPU {gpu}] DONE  {job['name']} in {mins:.1f} min -> acc={acc}", flush=True)
        return acc, "measured"
    finally:
        gpu_pool.put(gpu)


def print_table(rows, limit):
    print("\n" + "=" * 72)
    print("FINAL AGGREGATE MMLU SCORES (5-shot, generative)" + (f"  [limit={limit}]" if limit else ""))
    print("=" * 72)
    print(f"{'model':>6} | {'condition':<22} | {'acc':>8} | {'delta vs base':>13} | source")
    print("-" * 72)
    base_by_model = {r["model"]: r["acc"] for r in rows if r["condition"] == "baseline"}
    for row in rows:
        label = "baseline" if row["condition"] == "baseline" else f"steered L{row['layer']} M{row['multiplier']}"
        acc_str = f"{row['acc'] * 100:.2f}%" if row["acc"] is not None else "N/A"
        base = base_by_model.get(row["model"])
        if row["condition"] == "steered" and row["acc"] is not None and base is not None:
            delta_str = f"{(row['acc'] - base) * 100:+.2f} pp"
        else:
            delta_str = "-"
        print(f"{row['model']:>6} | {label:<22} | {acc_str:>8} | {delta_str:>13} | {row['source']}")
    print("=" * 72)


def orchestrate(skip_baseline=False, force=False, limit=None, gpus=(0,)):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / f"summary_all_scores{limit_suffix(limit)}.json"
    limit_args = ["--limit", str(limit)] if limit else []

    rows, jobs = [], []
    for tag, cfg in MODEL_CONFIGS.items():
        entries = [] if skip_baseline else [("baseline", None, None)]
        entries += [("steered", c["layer"], c["multiplier"]) for c in cfg["configs"]]

        for cond, layer, mult in entries:
            row = {"model": tag, "condition": cond, "layer": layer,
                   "multiplier": mult, "acc": None, "source": "pending"}
            rows.append(row)

            path = baseline_path(tag, limit) if cond == "baseline" else steered_path(tag, layer, mult, limit)
            if path.exists() and not force:
                row["acc"], row["source"] = load_cached_acc(path), "cached"
                print(f"[{tag}] Using cached {path}")
                continue

            args = ["--worker", cond, "--model_tag", tag, "--model_name", cfg["model_name"]]
            name = f"{tag}_{cond}"
            if cond == "steered":
                args += ["--vector_dir", cfg["vector_dir"], "--layer", str(layer), "--multiplier", str(mult)]
                name += f"_L{layer}_M{mult}"
            name += limit_suffix(limit)
            jobs.append({"row": row, "args": args, "name": name, "size": cfg.get("size_b", 0)})

    # Biggest models first, so the slowest runs don't end up last on their own.
    jobs.sort(key=lambda j: -j["size"])
    print(f"\n{len(jobs)} job(s) to run on GPU(s) {list(gpus)}; logs in {LOG_DIR}/\n", flush=True)

    gpu_pool = queue.Queue()
    for g in gpus:
        gpu_pool.put(g)
    lock = threading.Lock()

    def task(job):
        acc, src = run_job_on_gpu(job, gpu_pool, limit_args)
        with lock:
            job["row"]["acc"], job["row"]["source"] = acc, src
            save_summary(rows, summary_path)

    t0 = time.time()
    if jobs:
        with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
            list(ex.map(task, jobs))

    save_summary(rows, summary_path)
    print_table(rows, limit)
    print(f"Total wall time: {(time.time() - t0) / 60:.1f} min")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["baseline", "steered"], default=None,
                        help="Internal: run a single eval. Omit to run all steered configs.")
    parser.add_argument("--model_tag")
    parser.add_argument("--model_name")
    parser.add_argument("--vector_dir")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--multiplier", type=float)
    parser.add_argument("--limit", type=int, default=None,
                        help="Docs per subtask, for quick testing. Omit for the full eval.")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip the unsteered HF baseline runs.")
    parser.add_argument("--gpus", default="1,2,0",
                        help="Comma-separated GPU ids (nvidia-smi numbering) to run jobs on in parallel. "
                             "Order = preference (faster cards first).")
    parser.add_argument("--force", action="store_true",
                        help="Rerun even if a result JSON already exists.")
    args = parser.parse_args()

    if args.worker == "baseline":
        worker_hf(args.model_tag, args.model_name, limit=args.limit)
    elif args.worker == "steered":
        worker_hf(args.model_tag, args.model_name, args.vector_dir,
                  args.layer, args.multiplier, args.limit)
    else:
        gpu_list = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
        orchestrate(skip_baseline=args.skip_baseline, force=args.force,
                    limit=args.limit, gpus=gpu_list)
