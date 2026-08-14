"""Every flag a batch script passes must exist in the script it calls.

A misspelled flag is not caught until the job reaches the front of the GPU
queue, and then it dies in seconds having waited hours. `--folds` for
`--n-folds` was written and would have done exactly that. argparse already
knows the answer; this asks it before submission rather than after.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLURM_DIR = os.path.join(REPO, "rockfish", "slurm")

#: `python rockfish/foo.py ... --flag ...`, continued over backslash-newlines.
#: The continuation alternative must come first: `[^\n]` would otherwise consume
#: the backslash and the match would stop at the end of the first line, so a
#: multi-line invocation — which is every real one — would contribute no flags
#: and the whole check would pass vacuously.
INVOCATION = re.compile(
    r"^[ \t]*(?:srun[^\n]*?\s)?python3?\s+(rockfish/[\w/]+\.py)((?:\\\n|[^\n])*)",
    re.MULTILINE,
)


def sbatch_files():
    if not os.path.isdir(SLURM_DIR):
        return []
    return sorted(os.path.join(SLURM_DIR, f) for f in os.listdir(SLURM_DIR)
                  if f.endswith(".sbatch"))


def invocations():
    """(sbatch path, script path, literal long flags) for each python call."""
    out = []
    for path in sbatch_files():
        text = open(path).read()
        for script, tail in INVOCATION.findall(text):
            flags = sorted(set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", tail)))
            if flags:
                out.append((path, script, flags))
    return out


def declared_flags(script_rel):
    """Long options the script's parser accepts, read without importing it.

    Importing would pull in torch and the rest of the training stack, which is
    both slow and unavailable on a machine that only wants to lint. The parser
    is built from literal add_argument calls, so the AST has everything.
    """
    path = os.path.join(REPO, script_rel)
    tree = ast.parse(open(path).read())
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("--"):
                    flags.add(arg.value)
    return flags


@pytest.mark.parametrize(
    "sbatch,script,flags",
    invocations(),
    ids=[f"{os.path.basename(s)}->{os.path.basename(c)}"
         for s, c, _ in invocations()] or None,
)
def test_every_flag_is_accepted_by_the_script(sbatch, script, flags):
    declared = declared_flags(script)
    if not declared:
        pytest.skip(f"{script} builds its parser dynamically")
    unknown = [f for f in flags if f not in declared]
    assert not unknown, (
        f"{os.path.basename(sbatch)} passes {unknown} to {script}, which does "
        f"not define them. The job would be accepted by Slurm, wait in the "
        f"queue, then exit 2 on argparse. Accepted: {sorted(declared)}"
    )


def test_there_is_something_to_check():
    """Guard the regex: a refactor that stops matching would pass vacuously."""
    found = invocations()
    assert found, "no python invocations found in rockfish/slurm/*.sbatch"


@pytest.mark.parametrize("path", sbatch_files(),
                         ids=[os.path.basename(p) for p in sbatch_files()] or None)
def test_sbatch_is_valid_bash(path):
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, f"{os.path.basename(path)}: {r.stderr.strip()}"


@pytest.mark.parametrize("path", sbatch_files(),
                         ids=[os.path.basename(p) for p in sbatch_files()] or None)
def test_gpu_jobs_respect_max_mem_per_cpu(path):
    """ica100 caps memory at 4000M per CPU; over it, sbatch rejects at submit."""
    text = open(path).read()
    if "--partition=ica100" not in text:
        pytest.skip("not an ica100 job")
    mem = re.search(r"#SBATCH --mem=(\d+)M", text)
    cpus = re.search(r"#SBATCH --cpus-per-task=(\d+)", text)
    if not (mem and cpus):
        pytest.skip("memory or cpus not declared literally")
    limit = int(cpus.group(1)) * 4000
    assert int(mem.group(1)) <= limit, (
        f"{os.path.basename(path)}: --mem={mem.group(1)}M exceeds "
        f"{cpus.group(1)} CPUs x 4000M = {limit}M (MaxMemPerCPU on ica100)")



@pytest.mark.parametrize("path", sbatch_files(),
                         ids=[os.path.basename(p) for p in sbatch_files()] or None)
def test_gpu_qos_uses_the_gpu_account(path):
    """qos_gpu is only granted to the *_gpu account association.

    Submitting qos_gpu under the CPU account fails with "Invalid qos
    specification" — at submit time, so it is cheap, but it is also invisible
    until you try, and it silently blocked an evaluation that was otherwise
    ready to run.
    """
    text = open(path).read()
    if "--qos=qos_gpu" not in text:
        pytest.skip("not a GPU-QOS job")
    acct = re.search(r"#SBATCH --account=(\S+)", text)
    assert acct, f"{os.path.basename(path)}: qos_gpu but no --account declared"
    assert acct.group(1).endswith("_gpu"), (
        f"{os.path.basename(path)}: --qos=qos_gpu with --account="
        f"{acct.group(1)}. The qos_gpu association is on the _gpu account; "
        f"Slurm rejects this at submit with 'Invalid qos specification'.")


@pytest.mark.parametrize("path", sbatch_files(),
                         ids=[os.path.basename(p) for p in sbatch_files()] or None)
def test_gpu_jobs_request_a_gpu(path):
    text = open(path).read()
    if "--partition=ica100" not in text:
        pytest.skip("not a GPU partition")
    assert "--gres=gpu:" in text, (
        f"{os.path.basename(path)}: on ica100 without --gres=gpu:N. The job "
        f"runs, sees no CUDA device, and falls back to CPU — which for this "
        f"model means a run that never finishes rather than one that fails.")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
