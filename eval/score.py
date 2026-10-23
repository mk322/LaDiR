"""Score a generation directory written by eval/generate.py (or any JSONL with task_id, prompt, raw[, sample_idx]).

    python -m eval.score results/LaDiR-Code-8B [--benchmarks humaneval,mbpp] [--workers 16]

  humaneval / mbpp   EvalPlus HumanEval+ / MBPP+ evaluator (base and plus tests; evalplus' own ground truth,
                     time limits and output comparison), run in a pool with a scrubbed environment.
                     Metric: pass@1 for base and plus (plus = base AND plus tests); mean over samples, then tasks.
  lcb_v6             eval/lcb.py (official lcb_runner semantics, every public + private test, sandboxed).
                     Metric: test pass rate = passed / total tests per problem; mean over samples, then problems;
                     also broken down by difficulty and platform.
Benchmarks whose file does not cover every task are scored on the tasks present and flagged `complete: false`.

Writes <dir>/scores/<bench>.results.jsonl (per sample) and merges <dir>/summary.json; prints a table.
Files are recognised by name: humaneval.jsonl, mbpp.jsonl, lcb_v6.jsonl.
"""

import argparse
import json
import multiprocessing
import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

from . import lcb, postprocess, sandbox
from .benchmarks import by_task_id, canonical_name, load_benchmark

# evalplus applies wall-clock time limits (relative to ground-truth time); keep its pool small so CPU contention
# cannot turn slow-but-correct solutions into timeouts.
EVALPLUS_WORKERS = 8


# ----------------------------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------------------------
def summarize(per_task: Dict[str, List[float]], total_tasks: int, metric: str) -> dict:
    """per_task: task_id -> per-sample scores (0/1 pass flags or test pass rates). metric = mean over samples, then
    over tasks."""
    out = {"num_tasks": len(per_task), "total_tasks": total_tasks, "complete": len(per_task) == total_tasks}
    if per_task:
        ns = [len(v) for v in per_task.values()]
        out["samples_per_task"] = ns[0] if min(ns) == max(ns) else [min(ns), max(ns)]
        out[metric] = float(np.mean([np.mean(v) for v in per_task.values()]))
    return out


# ----------------------------------------------------------------------------------------------------------------
# loading generations
# ----------------------------------------------------------------------------------------------------------------
def load_generations(path: str) -> List[dict]:
    rows, seen, counter = [], set(), defaultdict(int)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "sample_idx" not in r:
                r["sample_idx"] = counter[r["task_id"]]
            counter[r["task_id"]] += 1
            key = (r["task_id"], r["sample_idx"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def _attach(rows, problems):
    """Fill grading-relevant fields from the benchmark when a file lacks them; drop unknown tasks."""
    out = []
    for r in rows:
        p = problems.get(r["task_id"])
        if p is None:
            print(f"warning: unknown task {r['task_id']} skipped", file=sys.stderr)
            continue
        for k in ("entry_point", "fn_name", "testtype", "starter_code", "platform", "difficulty"):
            if k in p and r.get(k) is None:
                r[k] = p[k]
        out.append(r)
    return out


# ----------------------------------------------------------------------------------------------------------------
# graders
# ----------------------------------------------------------------------------------------------------------------
def _scrubbed_worker_init():
    """evalplus executes candidates in multiprocessing children of these workers: give them no secrets and a scratch
    cwd."""
    keep = sandbox.clean_env(tempfile.mkdtemp(prefix="ladir_evalplus_"))
    keep.update({k: v for k, v in os.environ.items() if k.startswith("EVALPLUS_")})   # e.g. EVALPLUS_MAX_MEMORY_BYTES
    os.environ.clear()
    os.environ.update(keep)
    os.chdir(keep["HOME"])


def _evalplus_check(args):
    from evalplus.evaluate import check_correctness
    return check_correctness(*args)


def grade_evalplus(bench: str, rows: List[dict], workers: int) -> List[dict]:
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.eval import PASS
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth
    from evalplus.config import DEFAULT_GT_TIME_LIMIT_FACTOR, DEFAULT_MIN_TIME_LIMIT

    if bench == "humaneval":
        problems, h, nn = get_human_eval_plus(), get_human_eval_plus_hash(), []
    else:
        problems, h, nn = get_mbpp_plus(), get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS
    expected = get_groundtruth(problems, h, nn)
    jobs = []
    for i, r in enumerate(rows):
        r["solution"] = postprocess.build_solution(bench, r, r["raw"])
        jobs.append((bench, i, problems[r["task_id"]], r["solution"], expected[r["task_id"]], False, True,
                     None, DEFAULT_MIN_TIME_LIMIT, DEFAULT_GT_TIME_LIMIT_FACTOR))
    # spawn (not fork): evalplus' RLIMIT_AS would otherwise count the address space inherited from this process
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_scrubbed_worker_init) as ex:
        results = list(ex.map(_evalplus_check, jobs, chunksize=1))
    out = []
    for r, res in zip(rows, results):
        base, plus = res["base"][0], res["plus"][0]
        out.append({"task_id": r["task_id"], "sample_idx": r["sample_idx"], "passed": base == PASS,
                    "plus_passed": base == PASS and plus == PASS, "base_status": base, "plus_status": plus,
                    "solution": r["solution"]})
    return out


def _grade_lcb(args):
    r, problem = args
    info = lcb.grade(r["solution"], problem)
    return {"task_id": r["task_id"], "sample_idx": r["sample_idx"], "solution": r["solution"],
            "platform": problem.get("platform"), "difficulty": problem.get("difficulty"), **info}


def grade_lcb(rows: List[dict], problems: dict, workers: int) -> List[dict]:
    for r in rows:
        r["solution"] = postprocess.build_solution("lcb_v6", r, r["raw"])
    return sandbox.parallel_map(_grade_lcb, [(r, problems[r["task_id"]]) for r in rows], workers)


# ----------------------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------------------
def score_file(path: str, workers: int = None, lcb_path: str = None, evalplus_workers: int = EVALPLUS_WORKERS) -> dict:
    name = os.path.basename(path)[:-len(".jsonl")]
    bench = canonical_name(name)
    workers = workers or max(1, min(32, (os.cpu_count() or 2) // 2))
    problems = by_task_id(load_benchmark(bench, lcb_path=lcb_path))
    rows = _attach(load_generations(path), problems)
    if bench in ("humaneval", "mbpp"):
        results = grade_evalplus(bench, rows, evalplus_workers or workers)
    else:
        results = grade_lcb(rows, problems, workers)

    def per_task(key, subset=None):
        d = defaultdict(list)
        for x in results:
            if subset is None or subset(x):
                d[x["task_id"]].append(float(x[key]))
        return d

    summary = {"benchmark": bench, "file": os.path.abspath(path)}
    if bench in ("humaneval", "mbpp"):
        summary.update(summarize(per_task("passed"), len(problems), "pass@1"))
        summary["plus"] = summarize(per_task("plus_passed"), len(problems), "pass@1")
    else:
        summary.update(summarize(per_task("pass_rate"), len(problems), "test_pass_rate"))
        for field in ("difficulty", "platform"):
            groups = sorted({p.get(field) for p in problems.values()})
            summary[f"by_{field}"] = {
                g: summarize(per_task("pass_rate", lambda x, g=g: x.get(field) == g),
                             sum(p.get(field) == g for p in problems.values()), "test_pass_rate") for g in groups}
    out_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "scores")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + ".results.jsonl"), "w") as f:
        for x in sorted(results, key=lambda x: (str(x["task_id"]), x["sample_idx"])):
            f.write(json.dumps(x, default=str) + "\n")
    return summary


def _pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def format_table(summaries: Dict[str, dict]) -> str:
    lines = [f"{'benchmark':<28} {'tasks':>9}  metric"]
    for name, s in summaries.items():
        tasks = f"{s['num_tasks']}/{s['total_tasks']}" + ("" if s.get("complete") else "*")
        if "plus" in s:
            metric = f"pass@1 {_pct(s.get('pass@1'))}  plus pass@1 {_pct(s['plus'].get('pass@1'))}"
        else:
            metric = f"test pass rate {_pct(s.get('test_pass_rate'))}"
        lines.append(f"{name:<28} {tasks:>9}  {metric}")
        for field in ("by_difficulty", "by_platform"):
            for g, sub in s.get(field, {}).items():
                t = f"{sub['num_tasks']}/{sub['total_tasks']}"
                lines.append(f"  {field[3:]}={g:<16} {t:>11}  {_pct(sub.get('test_pass_rate'))}")
    if any(not s.get("complete") for s in summaries.values()):
        lines.append("* = not every task of the benchmark is present in the generation file")
    return "\n".join(lines)


def score_dir(gen_dir: str, benchmarks: List[str] = None, workers: int = None, lcb_path: str = None,
              evalplus_workers: int = EVALPLUS_WORKERS) -> dict:
    files = sorted(f for f in os.listdir(gen_dir) if f.endswith(".jsonl"))
    picked = []
    for f in files:
        name = f[:-len(".jsonl")]
        try:
            b = canonical_name(name)
        except ValueError:
            continue
        if benchmarks and b not in benchmarks and name not in benchmarks:
            continue
        picked.append(f)
    summaries = {}
    for f in picked:
        print(f"scoring {f} ...", flush=True)
        summaries[f[:-len(".jsonl")]] = score_file(os.path.join(gen_dir, f), workers, lcb_path, evalplus_workers)
    summary_path = os.path.join(gen_dir, "summary.json")
    merged = {}
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            merged = json.load(fh)
    merged.update(summaries)
    meta = {}
    for name in summaries:
        mp = os.path.join(gen_dir, name + ".meta.json")
        if os.path.exists(mp):
            with open(mp) as fh:
                meta[name] = json.load(fh)
    if meta:
        merged["_generation_meta"] = {**merged.get("_generation_meta", {}), **meta}
    merged["_scoring"] = {"sandbox_network_isolated": sandbox.network_isolated(),
                          "sandbox_pid_isolated": sandbox.pid_isolated()}
    with open(summary_path, "w") as fh:
        json.dump(merged, fh, indent=2)
    print(format_table(summaries))
    print(f"wrote {summary_path}")
    return summaries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gen_dir")
    ap.add_argument("--benchmarks", default=None, help="comma list (default: every recognised *.jsonl)")
    ap.add_argument("--workers", type=int, default=None, help="sandbox pool size (LCB)")
    ap.add_argument("--evalplus_workers", type=int, default=EVALPLUS_WORKERS, help="evalplus pool size")
    ap.add_argument("--lcb_path", default=None)
    a = ap.parse_args(argv)
    bl = [canonical_name(b) for b in a.benchmarks.split(",")] if a.benchmarks else None
    score_dir(a.gen_dir, bl, a.workers, a.lcb_path, a.evalplus_workers)


if __name__ == "__main__":
    main()
