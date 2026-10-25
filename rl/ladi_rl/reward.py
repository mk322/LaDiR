"""Unit-test reward for code answers, run in the eval sandbox.

tests: {"kind": "assert", "setup", "tests": [str]} | {"kind": "functional", "fn_name", "tests": [{"input", "output"}]}
     | {"kind": "stdin", "tests": [{"input", "output"}]}
extra.solution_style: "prepend_prompt" | "code_block" | "io"
"""

from __future__ import annotations

import json
from typing import Optional

PER_TEST_TIMEOUT = 4.0
MAX_STDIN_TESTS = 12


def _sandbox():
    from eval import sandbox
    return sandbox


def _lcb():
    from eval import lcb
    return lcb


def build_program(raw: str, prompt: str, extra: dict) -> str:
    from eval import postprocess
    from eval.benchmarks import open_fence
    of = bool(open_fence(prompt))
    style = extra.get("solution_style", "code_block")
    if style == "prepend_prompt":
        return postprocess.evalplus_solution(prompt, raw, extra.get("entry_point") or "", of,
                                             code_prefix=extra.get("code_prefix"))
    if style == "io":
        return postprocess.lcb_solution(raw, extra.get("testtype") or "stdin", extra.get("starter_code", ""), of)
    code = postprocess.first_code_block(raw, of)
    ep = extra.get("entry_point")
    cleaned = postprocess.strip_toplevel_tests(code)
    if cleaned is not None and (not ep or postprocess.defines(cleaned, ep)):
        return cleaned
    return code


_ASSERT_HARNESS = r'''import io, json, signal, sys
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(50000)
_SENTINEL = %r
_T = %r
_report = sys.stdout
sys.stdout = io.StringIO()
class _Timeout(BaseException):
    pass
def _alarm(signum, frame):
    raise _Timeout()
signal.signal(signal.SIGALRM, _alarm)
_spec = json.load(open("tests.json"))
_ns = {"__name__": "solution"}
_res = []
try:
    signal.setitimer(signal.ITIMER_REAL, _T)
    exec(compile(open("solution.py").read(), "solution.py", "exec"), _ns)
    exec(compile(_spec.get("setup") or "", "setup.py", "exec"), _ns)
    signal.setitimer(signal.ITIMER_REAL, 0)
    _ok = True
except BaseException:
    signal.setitimer(signal.ITIMER_REAL, 0)
    _ok = False
for _t in _spec["tests"]:
    if not _ok:
        _res.append(0)
        continue
    try:
        signal.setitimer(signal.ITIMER_REAL, _T)
        exec(compile(_t, "test.py", "exec"), _ns)
        signal.setitimer(signal.ITIMER_REAL, 0)
        _res.append(1)
    except BaseException:
        signal.setitimer(signal.ITIMER_REAL, 0)
        _res.append(0)
_report.write("\n" + _SENTINEL + json.dumps(_res) + "\n")
_report.flush()
'''


def _run_assert_tests(program: str, spec: dict, n_tests: int, timeout: float) -> list:
    sb = _sandbox()
    sentinel = sb.new_sentinel()
    res = sb.run_python(_ASSERT_HARNESS % (sentinel, float(timeout)), timeout=(timeout + 1) * max(n_tests, 1) + 5,
                        files={"solution.py": program, "tests.json": json.dumps(spec)})
    for line in reversed(res.stdout.split("\n")):
        if line.startswith(sentinel):
            if res.timed_out or res.returncode != 0:
                break
            out = json.loads(line[len(sentinel):])
            if len(out) == n_tests:
                return out
            break
    return [0] * n_tests


def pass_rate(raw: str, prompt: str, tests, extra: Optional[dict] = None, timeout: Optional[float] = None,
              max_stdin_tests: int = MAX_STDIN_TESTS, program: Optional[str] = None, io_timeout: Optional[float] = None):
    """-> (fraction of tests passed, info). Sandbox infrastructure errors score 0 with info["infra_error"]."""
    from eval import sandbox as _sandbox
    try:
        return _pass_rate(raw, prompt, tests, extra, timeout, max_stdin_tests, program, io_timeout)
    except _sandbox.SandboxInfraError as e:
        return 0.0, {"num_passed": 0, "num_tests": 0, "infra_error": True, "error": str(e)[:300]}


def _pass_rate(raw: str, prompt: str, tests, extra: Optional[dict] = None, timeout: Optional[float] = None,
               max_stdin_tests: int = MAX_STDIN_TESTS, program: Optional[str] = None,
               io_timeout: Optional[float] = None):
    spec = json.loads(tests) if isinstance(tests, str) else dict(tests)
    extra = json.loads(extra) if isinstance(extra, str) else dict(extra or {})
    if program is None:
        try:
            program = build_program(raw, prompt, extra)
        except Exception as e:
            return 0.0, {"num_passed": 0, "num_tests": len(spec.get("tests", [])), "error": f"build_program: {e!r}"[:300]}
    kind = spec["kind"]
    if timeout is None:
        timeout = PER_TEST_TIMEOUT if kind == "assert" else float(io_timeout or _lcb().TIMEOUT_S)
    if kind == "assert":
        n = len(spec["tests"])
        res = _run_assert_tests(program, spec, n, timeout)
    elif kind == "functional":
        n = len(spec["tests"])
        res = _lcb().run_functional_tests(program, spec["fn_name"], spec["tests"], timeout,
                                          memory_mb=_sandbox().DEFAULT_MEMORY_MB)
    elif kind == "stdin":
        tests_l = spec["tests"][:max_stdin_tests] if max_stdin_tests else spec["tests"]
        n = len(tests_l)
        res = _lcb().run_stdin_tests(program, tests_l, timeout)
    else:
        raise ValueError(f"unknown test kind {kind!r}")
    rate = float(sum(res)) / max(n, 1)
    return rate, {"num_tests": n, "num_passed": int(sum(res)), "all_passed": bool(n and sum(res) == n)}


def all_pass_reward(result):
    """1.0 if every test passed, else 0.0."""
    _, info = result
    return float(bool(info.get("all_passed"))), info
