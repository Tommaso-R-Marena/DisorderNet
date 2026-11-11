# Contributing

## The one rule

**A number in the paper is a number in a test.** `tests/test_disordernet_package.py`
asserts the published capacities, the 798-comparison count, the CAID3 escape from
8 to 51, and the calibration invariance the protocol rests on. If a change moves
one of those, CI fails and the paper is wrong somewhere — fix the paper or fix
the change, but do not adjust the test to match.

## Setup

```bash
pip install -e ".[dev]"
pytest tests/test_disordernet_package.py --doctest-modules disordernet
ruff check disordernet && ruff format --check disordernet
```

## Adding a result

If you add a function implementing a theorem, name the theorem in the docstring
and add a test that reproduces a number from the paper or from the Lean. The
package exists so that the arithmetic is checkable against something stronger
than prose; a function whose docstring cites nothing does not belong in it.

## The formal development

`lean/` is deposited as built, pinned to a toolchain. CI checks that no `sorry`
appears and that the pin has not drifted. Changing it means rebuilding against
that toolchain and re-running `#print axioms` on anything the paper cites.
