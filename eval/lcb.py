"""LiveCodeBench grading with the semantics of the official lcb_runner (lcb_runner/evaluation/testing_util.py).

Metric: test pass rate = passed tests / all public + private tests of the problem (every test is run).

  * functional (LeetCode): `import_string` + code; the method is looked up on Solution() if "class Solution" is in
    the code, otherwise on the module; inputs are one JSON value per line -> method(*args); a tuple result is
    converted to list; pass iff `prediction == json.loads(expected)`
  * stdin (AtCoder/Codeforces): `import_string` + code, stdout compared line by line after strip(): exact match per
    line, else every whitespace token parsed as Decimal and compared; SystemExit is not an error
  * per-test timeout 6 s
Every program runs in its own sandboxed subprocess (eval/sandbox.py); functional results are reported after a random
sentinel line, so exiting early cannot fake a pass.
"""

import json
from decimal import Decimal, InvalidOperation
from typing import List

from . import sandbox

TIMEOUT_S = 6
STARTUP_GRACE_S = 2.0    # added to the hard-kill timeout; the verdict uses the program's own execution time
MEMORY_MB = 65536

# verbatim from lcb_runner/evaluation/testing_util.py
IMPORT_STRING = ("from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
                 "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
                 "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
                 "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
                 "from builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\n"
                 "import collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\n"
                 "import statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\n"
                 "import json\nsys.setrecursionlimit(50000)\n")


# ----------------------------------------------------------------------------------------------------------------
# stdin
# ----------------------------------------------------------------------------------------------------------------
def _stripped_lines(val: str) -> List[str]:
    return [l.strip() for l in val.strip().split("\n")]


def _decimals(line: str):
    try:
        return True, [Decimal(e) for e in line.split()]
    except (InvalidOperation, ValueError):
        return False, []


def stdout_matches(prediction: str, expected: str) -> bool:
    """Official grade_stdio comparison."""
    pl, gl = _stripped_lines(prediction), _stripped_lines(expected)
    if len(pl) != len(gl):
        return False
    for p, g in zip(pl, gl):
        if p == g:
            continue
        ok1, dp = _decimals(p)
        ok2, dg = _decimals(g)
        if not (ok1 and ok2) or dp != dg:
            return False
    return True


# The candidate runs via exec() in a fresh __main__-like namespace so that a SystemExit (allowed, as in the official
# runner) can be told apart from a crash; the sentinel goes to stderr so stdout is exactly the program's output.
_STDIN_RUNNER = '''import sys
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(50000)       # as lcb_runner's compute_code_generation_metrics
_src = open("solution.py").read()
_ns = {"__name__": "__main__", "__builtins__": __builtins__, "__file__": "solution.py"}
import time as _time
_t0 = _time.perf_counter()
try:
    exec(compile(_src, "solution.py", "exec"), _ns)
except SystemExit:
    pass
_elapsed = _time.perf_counter() - _t0
sys.stdout.flush()
sys.stderr.write("\\nLADIR_ELAPSED=%%.6f\\n" %% _elapsed)
sys.stderr.write("\\n" + %r + "\\n")
sys.stderr.flush()
'''


def _elapsed(stderr: str) -> float:
    for line in reversed(stderr.splitlines()):
        if line.startswith("LADIR_ELAPSED="):
            return float(line.split("=", 1)[1])
    return 0.0


def run_stdin_test(code: str, test_input: str, expected: str, timeout: float = TIMEOUT_S) -> dict:
    sentinel = sandbox.new_sentinel()
    res = sandbox.run_python(_STDIN_RUNNER % sentinel, stdin=test_input, timeout=timeout + STARTUP_GRACE_S,
                             files={"solution.py": IMPORT_STRING + "\n" + code}, memory_mb=MEMORY_MB)
    if res.timed_out or _elapsed(res.stderr) > timeout:
        return {"status": -3, "error": "Time Limit Exceeded"}
    if res.returncode != 0 or sentinel not in res.stderr:
        return {"status": -4, "error": "Runtime Error", "stderr": res.stderr[-1000:]}
    if not stdout_matches(res.stdout, expected):
        return {"status": -2, "error": "Wrong Answer", "output": res.stdout[:300]}
    return {"status": True}


def run_stdin_tests(code: str, tests: List[dict], timeout: float = TIMEOUT_S) -> List[int]:
    """1/0 per test; one sandboxed process per test. A syntax error fails every test, so the rest are skipped."""
    res = []
    for t in tests:
        r = run_stdin_test(code, t["input"], t["output"], timeout)
        res.append(int(r["status"] is True))
        if not res[-1] and r.get("error") == "Runtime Error" and "SyntaxError" in r.get("stderr", ""):
            return res + [0] * (len(tests) - len(res))
    return res


# ----------------------------------------------------------------------------------------------------------------
# functional (LeetCode)
# ----------------------------------------------------------------------------------------------------------------
# One interpreter runs every test with a per-test alarm; the 1/0 list is reported after a random sentinel line.
_FUNCTIONAL_RUNNER = r'''import io, json, signal, sys, types
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(50000)
_SENTINEL = %r
_FN = %r
_T = %r
_report = sys.stdout
sys.stdout = io.StringIO()          # the candidate's prints are not part of the result channel
class _Timeout(BaseException):
    pass
def _alarm(signum, frame):
    raise _Timeout()
signal.signal(signal.SIGALRM, _alarm)
_tests = json.load(open("tests.json"))
_src = open("solution.py").read()
_mod = types.ModuleType("tmp_sol")
_res = []
try:
    signal.setitimer(signal.ITIMER_REAL, _T)
    exec(compile(_src, "solution.py", "exec"), _mod.__dict__)
    _obj = _mod.Solution() if "class Solution" in _src else _mod
    _method = getattr(_obj, _FN)
    signal.setitimer(signal.ITIMER_REAL, 0)
except BaseException:
    signal.setitimer(signal.ITIMER_REAL, 0)
    _method = None
for _t in _tests:
    if _method is None:
        _res.append(0)
        continue
    try:
        _args = [json.loads(_l) for _l in _t["input"].split("\n")]
        _expected = json.loads(_t["output"])
        signal.setitimer(signal.ITIMER_REAL, _T)
        _pred = _method(*_args)
        signal.setitimer(signal.ITIMER_REAL, 0)
        if isinstance(_pred, tuple):
            _pred = list(_pred)
        _res.append(int(_pred == _expected))
    except BaseException:           # includes timeouts and SystemExit raised inside the method
        signal.setitimer(signal.ITIMER_REAL, 0)
        _res.append(0)
_report.write("\n" + _SENTINEL + json.dumps(_res) + "\n")
_report.flush()
'''


def run_functional_tests(code: str, fn_name: str, tests: List[dict], timeout: float = TIMEOUT_S,
                         memory_mb: int = MEMORY_MB) -> List[int]:
    """1/0 per test, all tests in one sandboxed interpreter; a crash / global timeout / early exit fails them all."""
    n = len(tests)
    sentinel = sandbox.new_sentinel()
    runner = _FUNCTIONAL_RUNNER % (sentinel, fn_name, float(timeout))
    wall = (timeout + 1) * max(n, 1) + 5          # official global timeout: (timeout + 1) * n_tests + 5
    res = sandbox.run_python(runner, timeout=wall, files={"solution.py": IMPORT_STRING + "\n\n" + code,
                                                          "tests.json": json.dumps(tests)}, memory_mb=memory_mb)
    if not res.timed_out and res.returncode == 0:
        for line in reversed(res.stdout.split("\n")):
            if line.startswith(sentinel):
                out = json.loads(line[len(sentinel):])
                return out if len(out) == n else [0] * n
    return [0] * n


# ----------------------------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------------------------
def grade(code: str, problem: dict, timeout: float = TIMEOUT_S) -> dict:
    """problem: a benchmarks.lcb_records() record. -> {pass_rate, num_passed, num_tests, results}."""
    tests = problem["tests"]
    if problem.get("fn_name"):
        res = run_functional_tests(code, problem["fn_name"], tests, timeout)
    else:
        res = run_stdin_tests(code, tests, timeout)
    n = len(tests)
    return {"pass_rate": sum(res) / max(n, 1), "num_passed": int(sum(res)), "num_tests": n, "results": res}
