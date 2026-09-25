# Changelog

## 1.0.0

First release of the library behind the paper.

- `disordernet.capacity` — the capacity bound `⌈1/2ε⌉` and its `n`-dependent
  form, the imbalance factor `κ`, the pairwise bound `κ·ε²/(1−ε)²` with no
  balance hypothesis, and the counting converse.
- `disordernet.protocol` — the six-step pairwise scoring protocol: eligibility
  gate, per-target AUC, unweighted mean, Holm-corrected separation, and a
  capacity beyond which no rank is printed.
- `disordernet.noise` — both annotation error rates from repeat determinations,
  counted exactly rather than estimated.
- `disordernet` CLI: `capacity`, `noise`, `rank`, `table`.
- Gradio Space, Docker images, and CI that fails if a published number moves or
  a `sorry` appears in the Lean development.

Every function names the Lean theorem it implements; the development is in
`lean/`, pinned to `leanprover/lean4:v4.28.0` and mathlib4 `rev = v4.28.0`.
