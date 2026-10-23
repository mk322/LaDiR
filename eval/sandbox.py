"""Subprocess-per-program execution of untrusted (model-generated) Python.

Every program runs in
  * a fresh temporary directory (cwd, HOME and TMPDIR point there; removed afterwards),
  * a fresh interpreter (`python -I`), stdin = /dev/null unless a test input is given (then a file),
  * its own session / process group, killed with SIGKILL on timeout and again after it exits (stray children),
  * rlimits: address space, CPU time, file size (stdout/stderr go to size-limited files, not pipes), nproc,
  * a scrubbed environment (only PATH/HOME/TMPDIR/LANG/...; no tokens, keys or proxies),
  * when unprivileged user namespaces are available (`unshare -rn --pid --fork --mount-proc`): no network, and a
   private PID namespace, so the program cannot see or signal any other process (including the scorer) and every
   process it starts dies with it, even after a double fork / setsid.

This is a guard against accidents (sys.exit, infinite loops, fork bombs, stray files, env leaks), not a security
boundary against code written to escape it; run evaluations on a machine without secrets on disk.

Graders report results after a per-run random sentinel (new_sentinel()), so a program that exits early
(sys.exit(0), os._exit(0)) or prints a guessed marker cannot fake a pass.
"""

import os
import resource
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

DEFAULT_MEMORY_MB = 4096
DEFAULT_FSIZE_MB = 256
OUTPUT_READ_LIMIT = 64 * 1024 * 1024
NPROC_MARGIN = 4096          # tasks the program may add on top of the user's current count
INFRA_ERROR_MARKERS = ("fork failed", "Resource temporarily unavailable", "Cannot allocate memory: unshare")
INFRA_RETRIES = 6


class SandboxInfraError(RuntimeError):
    """The sandbox itself failed to start the program (not a property of the program): never score it as a fail."""


@dataclass
class ExecResult:
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    duration: float
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------------------------
# environment / limits
# ----------------------------------------------------------------------------------------------------------------
_UNSHARE = None


_NS_FLAGS = (["-rn", "--pid", "--fork", "--kill-child", "--mount-proc"],   # no network, no view of other processes
             ["-rn"])                                                     # no network only


def _unshare_prefix() -> List[str]:
    """unshare command prefix for the strongest unprivileged namespace setup that works here, else []."""
    global _UNSHARE
    if _UNSHARE is None:
        _UNSHARE = []
        exe = shutil.which("unshare")
        if exe and os.environ.get("LADIR_SANDBOX_NO_UNSHARE") != "1":
            for flags in _NS_FLAGS:
                try:
                    r = subprocess.run([exe] + flags + ["--", "/bin/true"], stdin=subprocess.DEVNULL,
                                       capture_output=True, timeout=10)
                except Exception:
                    continue
                if r.returncode == 0:
                    _UNSHARE = [exe] + flags + ["--"]
                    break
    return _UNSHARE


def network_isolated() -> bool:
    return bool(_unshare_prefix())


def pid_isolated() -> bool:
    return "--pid" in _unshare_prefix()


def _user_task_count() -> int:
    """Number of tasks (threads count toward RLIMIT_NPROC on Linux) owned by the current uid."""
    uid, n = os.getuid(), 0
    try:
        for pid in os.listdir("/proc"):
            if pid.isdigit():
                try:
                    if os.stat(f"/proc/{pid}").st_uid == uid:
                        n += len(os.listdir(f"/proc/{pid}/task"))
                except OSError:
                    pass
    except OSError:
        return 0
    return n


_NPROC_CACHE = [0.0, 0]


def _nproc_limit() -> int:
    """RLIMIT_NPROC is per *user*, so the limit must sit above what this user already runs; refreshed every 30 s."""
    now = time.time()
    if now - _NPROC_CACHE[0] > 30:
        _NPROC_CACHE[:] = [now, _user_task_count()]
    return _NPROC_CACHE[1] + NPROC_MARGIN


def clean_env(workdir: str) -> dict:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": workdir,
        "TMPDIR": workdir,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def _limit_values(memory_mb, cpu_s, fsize_mb, nproc):
    lim = {"as": (memory_mb * 1024 * 1024,) * 2, "fsize": (fsize_mb * 1024 * 1024,) * 2, "core": (0, 0)}
    cpu = max(1, int(cpu_s + 0.999))
    lim["cpu"] = (cpu, cpu + 1)
    if nproc:
        lim["nproc"] = (nproc, nproc)
    return lim


_RES = {"as": resource.RLIMIT_AS, "fsize": resource.RLIMIT_FSIZE, "core": resource.RLIMIT_CORE,
        "cpu": resource.RLIMIT_CPU, "nproc": resource.RLIMIT_NPROC}


def _limit_prefix(memory_mb, cpu_s, fsize_mb, nproc):
    """(command prefix, preexec_fn). Uses util-linux `prlimit` (no Python code between fork and exec, which is the
    thread-safe option since we drive subprocesses from a thread pool); falls back to a preexec_fn."""
    lim = _limit_values(memory_mb, cpu_s, fsize_mb, nproc)
    exe = shutil.which("prlimit")
    if exe:
        return [exe] + [f"--{k}={s}:{h}" for k, (s, h) in lim.items()] + ["--"], None

    def apply():
        for k, v in lim.items():
            resource.setrlimit(_RES[k], v)
    return [], apply


def _killpg(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _read(path, limit=OUTPUT_READ_LIMIT):
    try:
        with open(path, "rb") as f:
            return f.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


# ----------------------------------------------------------------------------------------------------------------
# running programs
# ----------------------------------------------------------------------------------------------------------------
def run_python(code: str, stdin: Optional[str] = None, timeout: float = 10.0, files: dict = None,
               memory_mb: int = DEFAULT_MEMORY_MB, fsize_mb: int = DEFAULT_FSIZE_MB, limit_nproc: bool = True,
               python: str = None, script_name: str = "main.py") -> ExecResult:
    """Run `code` as a script in a fresh sandboxed interpreter. `files`: extra {name: text} written to the work dir.

    If the sandbox cannot start the program (fork/namespace failure under load), retry with backoff and finally raise
    SandboxInfraError rather than returning a result that would be scored as the program failing."""
    for attempt in range(INFRA_RETRIES):
        res = _run_python_once(code, stdin, timeout, files, memory_mb, fsize_mb, limit_nproc, python, script_name)
        if res.returncode == 0 or res.timed_out or not any(m in res.stderr for m in INFRA_ERROR_MARKERS):
            return res
        _NPROC_CACHE[0] = 0.0                         # refresh the user's task count before retrying
        time.sleep(0.5 * (attempt + 1))
    raise SandboxInfraError(f"sandbox failed to start a program {INFRA_RETRIES} times: {res.stderr[-300:]!r}")


def _run_python_once(code, stdin, timeout, files, memory_mb, fsize_mb, limit_nproc, python, script_name) -> ExecResult:
    workdir = tempfile.mkdtemp(prefix="ladir_sbx_")
    try:
        with open(os.path.join(workdir, script_name), "w") as f:
            f.write(code)
        for name, text in (files or {}).items():
            with open(os.path.join(workdir, name), "w") as f:
                f.write(text)
        out_p, err_p = os.path.join(workdir, ".stdout"), os.path.join(workdir, ".stderr")
        if stdin is not None:
            in_p = os.path.join(workdir, ".stdin")
            with open(in_p, "w") as f:
                f.write(stdin)
            stdin_f = open(in_p, "rb")
        else:
            stdin_f = subprocess.DEVNULL
        nproc = _nproc_limit() if limit_nproc else 0
        limits, preexec = _limit_prefix(memory_mb, timeout + 1, fsize_mb, nproc)
        cmd = limits + _unshare_prefix() + [python or sys.executable, "-I", script_name]
        t0 = time.time()
        timed_out = False
        with open(out_p, "wb") as out_f, open(err_p, "wb") as err_f:
            proc = subprocess.Popen(cmd, cwd=workdir, env=clean_env(workdir), stdin=stdin_f, stdout=out_f,
                                    stderr=err_f, start_new_session=True, preexec_fn=preexec)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _killpg(proc.pid)
                proc.wait()
            finally:
                _killpg(proc.pid)    # children left behind in the group
                if stdin_f is not subprocess.DEVNULL:
                    stdin_f.close()
        dur = time.time() - t0
        return ExecResult(proc.returncode, _read(out_p), _read(err_p, 1 << 16), timed_out, dur)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def new_sentinel() -> str:
    return "__LADIR_PASS_" + secrets.token_hex(16) + "__"


def parallel_map(fn: Callable, items: Iterable, workers: int = None) -> list:
    """Map fn over items in a pool; each call spawns its own sandboxed subprocess, so threads suffice to drive them
    (the GIL is idle while waiting). Order is preserved."""
    items = list(items)
    workers = workers or max(1, min(32, (os.cpu_count() or 2) // 2))
    if workers == 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))
