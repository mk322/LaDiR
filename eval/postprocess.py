"""Turn raw model outputs into programs.

Rules (no function renaming anywhere):
  * Code = the FIRST fenced block of the output (the models stop after it). Indentation is kept; only leading blank
    lines and trailing whitespace are removed. An unclosed block (hit max_new_tokens) runs to the end of the text.
  * If the prompt itself ends inside an open fence, the output *is* code up to the first fence.
  * Only if the output starts with parseable code followed by a bare ``` (completion style that never opened a
    fence, so that ``` is a closing fence) is the leading code used; prose before the first fence is always dropped.
  * HumanEval / MBPP+ (EvalPlus): the prompt is always prepended. It carries imports and helper functions (e.g.
    HumanEval/38); a completion-style body continues the prompt's signature, and a full redefinition of the entry
    point simply overrides the prompt's docstring-only stub.
  * LCB: a full program (stdin) or a `class Solution` (functional); a bare indented method gets the starter class
    header.
"""

import ast
import re
import textwrap

_OPEN = re.compile(r"```[ \t]*([A-Za-z0-9_+\-.]*)[ \t]*\n")
_CODE_HINT = re.compile(r"^\s*(def |class |return\b|import |from \S+ import |print\(|for |while |if |\w+\s*=)", re.M)


def _looks_like_code(text: str) -> bool:
    if not _CODE_HINT.search(text):
        return False
    for candidate in (text, textwrap.dedent(text), "def _f():\n" + text):
        try:
            ast.parse(candidate)
            return True
        except Exception:   # SyntaxError, and CPython SystemError/RecursionError on pathological input
            pass
    return False


def _tidy(code: str) -> str:
    lines = code.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).rstrip() + "\n"


def first_code_block(raw: str, prompt_open_fence: bool = False) -> str:
    if prompt_open_fence:
        return _tidy(raw.split("```")[0])
    m = _OPEN.search(raw)
    if m is None:
        # no opening fence: either pure code, or code followed by a bare closing fence
        return _tidy(raw.split("```")[0])
    pre = raw[:m.start()]
    if not m.group(1) and pre.strip() and _looks_like_code(pre):
        # a bare ``` after code is a *closing* fence (completion-style output that never opened one); a fence with a
        # language tag always opens a block, so prose or not, its content is the code
        return _tidy(pre)
    body = raw[m.end():]
    end = body.find("```")
    return _tidy(body if end < 0 else body[:end])


def defines(code: str, name: str) -> bool:
    return re.search(rf"^\s*def\s+{re.escape(name)}\s*\(", code, re.M) is not None


# ----------------------------------------------------------------------------------------------------------------
# per-benchmark solution builders
# ----------------------------------------------------------------------------------------------------------------
def evalplus_solution(prompt: str, raw: str, entry_point: str, prompt_open_fence: bool = False,
                      code_prefix: str = None, sanitize: bool = True) -> str:
    """HumanEval+/MBPP+ solution: the EvalPlus prompt (code_prefix; defaults to the fed prompt) is always prepended
    (see module docstring). sanitize=True then drops the model's own top-level test code (strip_toplevel_tests)."""
    prefix = code_prefix if code_prefix is not None else prompt
    code = first_code_block(raw, prompt_open_fence)
    full = prefix + "\n\n" + code if defines(code, entry_point) else prefix + code
    if sanitize:
        cleaned = strip_toplevel_tests(full)
        if cleaned is not None and defines(cleaned, entry_point):
            return cleaned
    return full


_KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_main_guard(node) -> bool:
    try:
        return isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
    except Exception:
        return False


def strip_toplevel_tests(code: str):
    """Drop the model's own top-level test code (asserts, prints/calls, `if __name__ == "__main__":`, loops) and keep
    imports, definitions and side-effect-free assignments. Returns None if the code does not parse."""
    try:
        tree = ast.parse(code)
    except Exception:   # SyntaxError, and CPython SystemError/RecursionError/MemoryError on pathological model output
        return None
    lines = code.split("\n")
    keep = []
    for node in tree.body:
        ok = isinstance(node, _KEEP) or (
            isinstance(node, (ast.Assign, ast.AnnAssign)) and not any(isinstance(n, ast.Call) for n in ast.walk(node)))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):   # docstrings / string literals
            ok = True
        if ok and not _is_main_guard(node):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
            keep.append("\n".join(lines[start - 1:node.end_lineno]))
    return "\n\n".join(keep) + "\n"


def lcb_solution(raw: str, testtype: str = "stdin", starter_code: str = "", prompt_open_fence: bool = False) -> str:
    code = first_code_block(raw, prompt_open_fence)
    if testtype == "functional" and "class Solution" not in code and code[:1] in (" ", "\t"):
        # a bare (indented) method body: put it back under the starter's class header
        header = next((l for l in (starter_code or "").split("\n") if l.startswith("class ")), "class Solution:")
        code = header + "\n" + code
    return code


def build_solution(bench: str, rec: dict, raw: str) -> str:
    """rec: a row from the generation file (needs prompt and the GEN_FIELDS of the benchmark)."""
    from .benchmarks import canonical_name, open_fence
    b = canonical_name(bench)
    of = bool(open_fence(rec["prompt"]))
    if b in ("humaneval", "mbpp"):
        return evalplus_solution(rec["prompt"], raw, rec["entry_point"], of, code_prefix=rec.get("code_prefix"))
    if b == "lcb_v6":
        return lcb_solution(raw, rec.get("testtype") or "stdin", rec.get("starter_code", ""), of)
    raise ValueError(bench)
