"""An audit of the tests themselves.

This suite is the only thing standing between a plausible wrong number and a
published one, and it has already failed at that job in ways worth naming:

- a parametrised test collected an **empty** parameter set and reported success
  while checking nothing (the sbatch flag audit, whose regex silently matched
  no invocations);
- a hand-maintained list of files to check missed a sixth consumer, then a
  seventh (the evidence-sentinel audit);
- a test asserted a guard held while nothing in the code could have made it
  fail (the detach guard needed a companion asserting gradients *do* flow).

Each looked like a passing test. So these check the tests: that every test
asserts something, that no parametrisation is empty, that skips are deliberate
and explained, and that the modules carrying published claims are covered at
all.
"""

from __future__ import annotations

import ast
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")

#: Modules whose behaviour appears in a published number. A change here that no
#: test notices is a change to a claim that no test notices.
CLAIM_BEARING = [
    "colab/caid3_official.py",
    "colab/lite_head.py",
    "colab/biological_utility.py",
    "rockfish/train_multitask.py",
    "rockfish/eval_caid3_official.py",
]


def _test_files():
    return sorted(os.path.join(TESTS, f) for f in os.listdir(TESTS)
                  if f.startswith("test_") and f.endswith(".py"))


def parsed():
    for path in _test_files():
        try:
            yield path, ast.parse(open(path).read())
        except SyntaxError as exc:  # pragma: no cover - would fail collection
            pytest.fail(f"{path}: {exc}")


def _test_functions():
    """(path, node) for every test function, including those in classes."""
    out = []
    for path, tree in parsed():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                    node.name.startswith("test_"):
                out.append((path, node))
    return out


def _has_assertion(node) -> bool:
    """An assert, a pytest.raises/warns block, or a pytest.fail call."""
    for n in ast.walk(node):
        if isinstance(n, ast.Assert):
            return True
        if isinstance(n, ast.Call):
            f = n.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name in {"raises", "warns", "fail", "approx", "deprecated_call",
                        "assert_allclose", "assert_array_equal",
                        "assert_frame_equal", "assert_series_equal"}:
                return True
        if isinstance(n, ast.With):
            for item in n.items:
                if isinstance(item.context_expr, ast.Call):
                    f = item.context_expr.func
                    if (getattr(f, "attr", None) or "") in {"raises", "warns"}:
                        return True
    return False


class TestEveryTestChecksSomething:
    def test_no_test_function_is_assertion_free(self):
        """A test that asserts nothing passes forever and protects nothing."""
        empty = []
        for path, node in _test_functions():
            if _has_assertion(node):
                continue
            # A test may delegate its assertions to a helper it calls.
            calls = {getattr(n.func, "id", None) or getattr(n.func, "attr", "")
                     for n in ast.walk(node) if isinstance(n, ast.Call)}
            if any(c and (c.startswith("_") or c.startswith("check")
                          or c.startswith("assert")) for c in calls):
                continue
            empty.append(f"{os.path.basename(path)}::{node.name}")
        assert not empty, (
            "these test functions contain no assertion and cannot fail:\n  "
            + "\n  ".join(sorted(empty)))

    def test_there_are_tests_to_audit(self):
        """Guard the audit: a collector that finds nothing passes vacuously,
        which is the exact failure this file exists to catch."""
        fns = _test_functions()
        assert len(fns) > 200, f"only {len(fns)} test functions discovered"
        assert len(_test_files()) > 10


class TestNoParametrisationIsEmpty:
    """An empty parameter set makes pytest report success having run nothing.

    tests/test_sbatch_flags_exist.py did this: its regex matched no invocations,
    so the flag check collected zero cases and passed while checking nothing.
    """

    def test_no_parametrize_receives_a_literal_empty_list(self):
        offenders = []
        for path, tree in parsed():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if (getattr(node.func, "attr", "") != "parametrize"):
                    continue
                if len(node.args) < 2:
                    continue
                arg = node.args[1]
                if isinstance(arg, (ast.List, ast.Tuple)) and not arg.elts:
                    offenders.append(f"{os.path.basename(path)}:{node.lineno}")
        assert not offenders, offenders

    def test_dynamic_parametrisations_are_guarded(self):
        """A parametrisation built from a function call can collect empty at
        runtime, so the file must also assert the collection is non-empty."""
        needs_guard = {
            "test_sbatch_flags_exist.py": "test_there_is_something_to_check",
            "test_test_suite_integrity.py": "test_there_are_tests_to_audit",
        }
        for fname, guard in needs_guard.items():
            path = os.path.join(TESTS, fname)
            if not os.path.isfile(path):
                continue
            assert guard in open(path).read(), (
                f"{fname} parametrises from a function and must assert the "
                f"collection is non-empty in {guard}")


class TestSkipsAreDeliberate:
    def test_every_skip_carries_a_reason(self):
        """A skip without a reason is indistinguishable from a test quietly
        switched off."""
        bad = []
        for path, tree in parsed():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", "")
                if name not in {"skip", "skipif", "xfail"}:
                    continue
                has_reason = any(k.arg == "reason" for k in node.keywords)
                # pytest.skip("msg") takes the reason positionally.
                if name == "skip" and node.args:
                    has_reason = True
                if not has_reason:
                    bad.append(f"{os.path.basename(path)}:{node.lineno} {name}")
        assert not bad, bad

    def test_no_test_is_unconditionally_skipped(self):
        """skipif(True) and a bare skip decorator disable a test permanently."""
        bad = []
        for path, tree in parsed():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if getattr(node.func, "attr", "") != "skipif":
                    continue
                if node.args and isinstance(node.args[0], ast.Constant) \
                        and node.args[0].value is True:
                    bad.append(f"{os.path.basename(path)}:{node.lineno}")
        assert not bad, bad


class TestClaimBearingModulesAreCovered:
    """Every module behind a published number must be exercised somewhere."""

    @pytest.mark.parametrize("module", CLAIM_BEARING)
    def test_module_is_imported_by_some_test(self, module):
        assert os.path.isfile(os.path.join(REPO, module)), module
        dotted = module.replace("/", ".")[:-3]
        haystack = "".join(open(p).read() for p in _test_files())
        assert dotted in haystack or module in haystack, (
            f"{module} carries published behaviour and no test imports it")

    def test_the_claim_list_is_not_empty(self):
        assert len(CLAIM_BEARING) >= 5


class TestGuardsHaveCompanions:
    """A guard asserting "X does not happen" can pass because nothing happens.

    The detach guard asserts binding's gradient never reaches the disorder
    read-out. That would hold just as well if the conditioned read-out received
    no gradient either — if the whole path were dead. It needs a companion
    asserting the intended effect *does* occur, and so does any guard shaped
    like it.
    """

    def test_the_detach_guard_has_its_companion(self):
        path = os.path.join(TESTS, "test_disorder_conditioned_binding.py")
        if not os.path.isfile(path):
            pytest.skip("conditioned-binding tests not present")
        src = open(path).read()
        assert "test_binding_loss_never_reaches_the_disorder_readout" in src
        assert "test_the_conditioned_readout_does_receive_gradient" in src, (
            "the negative guard needs a positive companion, or it passes when "
            "the path is simply dead")

    def test_the_sentinel_audit_enumerates_rather_than_lists(self):
        path = os.path.join(TESTS, "test_partial_evidence_alignment.py")
        if not os.path.isfile(path):
            pytest.skip("partial-evidence tests not present")
        src = open(path).read()
        assert "TestTheRealPipelineSurvivesPartialEvidence" in src, (
            "the syntactic audit missed the sixth and seventh consumers; the "
            "suite must also run the real pipeline")
