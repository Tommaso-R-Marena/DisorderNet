# The formal development

Lean 4, `sorry`-free, on the standard axioms (`propext`, `Classical.choice`,
`Quot.sound`). 403 files under `RequestProject/`, plus the prose companions the
manuscript cites (`CAPACITY_CONTEXT.md`, `DISCORDANT_CONTEXT.md`,
`DEPENDENT_BH_CONTEXT.md`, `PROOF_GAPS_CLOSED.md`, and the rest).

Produced with Aristotle (Harmonic) from statements specified by the author; see
`../AUTHORSHIP.md`. A kernel-checked proof is correct independently of how it
was written, which is why the toolchain is pinned here rather than described.

## Building

```bash
cd lean && lake update && lake build
```

Toolchain and dependency are pinned: `lean-toolchain` is
`leanprover/lean4:v4.28.0` and `lakefile.toml` requires mathlib4 at
`rev = "v4.28.0"`. The axiom claim is only meaningful against those, so check
against them.

## Checking a claim from the paper

```lean
import RequestProject.DiscordantImbalance
#print axioms IDR.nuPair_le_imbalanced
```

The statements the manuscript cites, and where they live:

| claim in the paper | file |
|---|---|
| capacity `k ≤ ⌈1/2ε⌉`, three score models, attained | `BenchmarkCapacity.lean` |
| the constructive converse | `BenchmarkCapacity.lean` (`unresolvable_pair`) |
| `AUC_pooled` decomposition and its invariance | `AUCInvariance.lean` |
| inversions are certificates | `AUCInversion.lean` |
| `discordant = 2·d·u`, exact comparable-pair count | `DiscordantPairs.lean` |
| `ν_pair ≤ κ·ε²/(1−ε)²`, no balance hypothesis | `DiscordantImbalance.lean` |
| Benjamini–Yekutieli under arbitrary dependence | `DependentBH.lean` |
| the harmonic factor is attained | `DependentBHSharp.lean` |
| region-level screening, `fdp_lift` | `RegionScreen.lean` |
| conformal p-values are superuniform, and compose | `ConformalPValueBH.lean` |
| conformal risk control validity | `Calibration.lean` |
| NP-hardness of the recalibration optimum | `Complexity/` |
| per-target AUC invariance | `AUCTargetMean.lean` |
| separated family vs chain | `SeparationVsChain.lean` |
| capacity from an estimated rate | `CapacityConfidence.lean` |
| counting converse, 798 of 6,786 | `UnresolvableCount.lean` |

`PROOF_GAPS_CLOSED.md` maps each statement the manuscript once used ahead of its
proof to the theorem that now closes it.
