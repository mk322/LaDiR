"""Benchmark loaders with one uniform record format and the prompt template of each benchmark.

Every loader returns a list of dicts with at least
    task_id   unique id within the benchmark
    prompt    raw text fed to the model (NO chat template: that is how LaDiR was trained)
plus grading info. Fields listed in GEN_FIELDS[bench] are light metadata that generate.py copies into each output
row; heavy grading data (tests) is looked up again at scoring time via `load_benchmark(...)` and the task_id.

Benchmarks and their prompt templates:
  humaneval  EvalPlus HumanEval+ (164)   humaneval_prompt: "Complete the following Python function `f`:" + the
                                         EvalPlus stub in a ```python block
  mbpp       EvalPlus MBPP+ (378)        function_task_prompt(task text, entry point, example asserts)
  lcb_v6     LiveCodeBench v6 (175)      lcb_prompt: stdin_task_prompt(question) (competitive-programming header),
                                         or class_task_prompt(question, starter code) for LeetCode problems
HumanEval+/MBPP+ records keep the EvalPlus prompt as `code_prefix` (valid Python, prepended at scoring time:
imports, helper functions, completion-style bodies). LCB records carry all public AND private tests.
"""

import base64
import functools
import json
import os
import pickle
import zlib
from typing import List

# LiveCodeBench v6 problems (175, jsonl). Empty -> downloaded once from the HF dataset (eval/lcb_v6.jsonl).
LCB_V6_PATH = os.environ.get("LADIR_LCB_V6_PATH", "")
LCB_V6_HF = ("mk322/ladir-eval", "lcb_v6.jsonl")


def lcb_v6_path(path: str = "") -> str:
    path = path or LCB_V6_PATH
    if path:
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(LCB_V6_HF[0], LCB_V6_HF[1], repo_type="dataset")


BENCHMARKS = ("humaneval", "mbpp", "lcb_v6")
ALIASES = {"lcb": "lcb_v6", "livecodebench": "lcb_v6", "humaneval_plus": "humaneval", "mbpp_plus": "mbpp"}

GEN_FIELDS = {
    "humaneval": ("entry_point", "code_prefix"),
    "mbpp": ("entry_point", "code_prefix"),
    "lcb_v6": ("platform", "difficulty", "testtype", "fn_name", "starter_code"),
}


def canonical_name(bench: str) -> str:
    """'lcb' -> 'lcb_v6', 'mbpp_plus' -> 'mbpp', ..."""
    base = ALIASES.get(bench, bench)
    if base not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {bench!r}; known: {BENCHMARKS}")
    return base


def open_fence(prompt: str) -> int:
    """1 if the prompt ends inside an open ``` block (the model then continues the code directly)."""
    return prompt.count("```") % 2


def load_benchmark(name: str, lcb_path: str = None) -> List[dict]:
    bench = canonical_name(name)
    if bench == "lcb_v6":
        return lcb_records(lcb_path or LCB_V6_PATH)
    return _evalplus(bench)


def by_task_id(records: List[dict]) -> dict:
    return {r["task_id"]: r for r in records}


# ----------------------------------------------------------------------------------------------------------------
# prompt templates
# ----------------------------------------------------------------------------------------------------------------
# The competitive-programming header of the SFT data.
SFT_CP_HEADER = ("Now, a programming problem is given to you, and you are asked to implement it in python. You need to "
                 "provide the complete code and the problem-solving approach. The problem to be solved is below:\n")


def humaneval_prompt(stub: str, entry_point: str) -> str:
    """HumanEval: the EvalPlus stub (signature + docstring) as a fenced block to complete."""
    return f"Complete the following Python function `{entry_point}`:\n\n```python\n{stub.rstrip()}\n```"


def mbpp_prompt(evalplus_prompt: str, entry_point: str) -> str:
    """MBPP+: the EvalPlus prompt ('\"\"\"\\n<text>\\n<assert ...>\\n\"\"\"') as a function task."""
    body = evalplus_prompt.strip().strip('"').strip()
    lines = [l for l in body.split("\n") if l.strip()]
    tests = [l for l in lines if l.lstrip().startswith("assert")]
    text = " ".join(l.strip() for l in lines if not l.lstrip().startswith("assert"))
    return function_task_prompt(text, entry_point, tests)


def function_task_prompt(text: str, entry_point: str, examples=()) -> str:
    """Function task (MBPP):
        <task text> Write it as a Python function named `<entry_point>`.\n\nExample:\n```python\n<asserts>\n```"""
    out = f"{text} Write it as a Python function named `{entry_point}`."
    if examples:
        out += "\n\nExample:\n```python\n" + "\n".join(examples) + "\n```"
    return out


def stdin_task_prompt(question: str) -> str:
    """Stdin (competitive-programming) task: SFT_CP_HEADER + question."""
    return f"{SFT_CP_HEADER}{question.rstrip()}"


def class_task_prompt(question: str, starter_code: str) -> str:
    """LeetCode-style task with a starter class:
        <question>\n\nComplete the following Python class:\n\n```python\n<starter code, stripped>\n```"""
    return f"{question.rstrip()}\n\nComplete the following Python class:\n\n```python\n{(starter_code or '').strip()}\n```"


def lcb_prompt(question: str, starter_code: str) -> str:
    sc = (starter_code or "").strip()
    return class_task_prompt(question, sc) if sc else stdin_task_prompt(question)


# ----------------------------------------------------------------------------------------------------------------
# EvalPlus (HumanEval+, MBPP+)
# ----------------------------------------------------------------------------------------------------------------
def _evalplus(which):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    problems = get_human_eval_plus() if which == "humaneval" else get_mbpp_plus()
    build = humaneval_prompt if which == "humaneval" else mbpp_prompt
    return [{"task_id": k, "prompt": build(p["prompt"], p["entry_point"]),
             "code_prefix": p["prompt"], "entry_point": p["entry_point"],
             "canonical_solution": p["canonical_solution"]} for k, p in problems.items()]


# ----------------------------------------------------------------------------------------------------------------
# LiveCodeBench
# ----------------------------------------------------------------------------------------------------------------
def decode_lcb_tests(field) -> list:
    """public_test_cases is plain JSON; private_test_cases is base64(zlib(pickle(json str))) (as in lcb_runner)."""
    if not field:
        return []
    if isinstance(field, list):
        return field
    try:
        return json.loads(field)
    except (json.JSONDecodeError, TypeError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(field.encode("utf-8")))))


@functools.lru_cache(maxsize=4)
def _lcb_raw(path=""):
    with open(lcb_v6_path(path)) as f:
        return tuple(json.loads(l) for l in f if l.strip())


def lcb_records(path: str = LCB_V6_PATH) -> List[dict]:
    out = []
    for r in _lcb_raw(path):
        meta = json.loads(r.get("metadata") or "{}")
        fn_name = meta.get("func_name")
        rec = {
            "task_id": r["question_id"],
            "prompt": lcb_prompt(r["question_content"], r.get("starter_code", "")),
            "platform": r.get("platform"),
            "difficulty": r.get("difficulty"),
            "contest_date": r.get("contest_date"),
            "starter_code": r.get("starter_code", ""),
            "fn_name": fn_name,
            "testtype": "functional" if fn_name else "stdin",
        }
        public = decode_lcb_tests(r.get("public_test_cases"))
        tests = public + decode_lcb_tests(r.get("private_test_cases"))
        rec["num_public"] = len(public)
        rec["tests"] = [{"input": t["input"], "output": t["output"], "testtype": t.get("testtype")} for t in tests]
        out.append(rec)
    return out
