# Per-target analysis — power, and the targets nobody solves

Both analyses use CAID3 only, which is genuinely held out for `mt_windowed`:
the leak filter removed all 319 targets plus homologues at identity ≥ 0.40.

Exploratory, and labelled so. Neither test is in `PREREGISTRATION.md`. Every
comparison run is reported, not a chosen subset.

## Per-target win/loss — a more powerful question than pooled AUC

A pooled AUC per benchmark gives five numbers, three of them from 31 to 52
targets that cannot resolve a winner. But the suite has 658 targets, and each
supports its own AUC. A Wilcoxon signed-rank over paired per-target differences
asks whether we beat an opponent on more targets and by more.

| benchmark | targets | wins | losses | median Δ | p | opponent |
|---|---:|---:|---:|---:|---:|---|
| Disorder-PDB | 233 | **137** | 77 | +0.0023 | **0.00003** | PUNCH2 |
| Disorder-PDB | 233 | 142 | 72 | +0.0038 | **0.00000** | AlphaFold-rsa |
| Disorder-NOX | 155 | 80 | 72 | +0.0015 | 0.684 | ESMDisPred-2PDB |
| Disorder-NOX | 178 | 96 | 75 | +0.0061 | **0.00065** | AlphaFold-rsa |
| Binding | 48 | 27 | 21 | +0.0173 | 0.345 | DisoFLAG-PB |
| Binding | 49 | 34 | 15 | +0.1091 | **0.00129** | AlphaFold-rsa |
| Binding-IDR | 42 | 21 | 21 | −0.0075 | 0.866 | bindEmbed21IDR |
| Binding-IDR | 42 | 32 | 10 | +0.3670 | **0.00072** | AlphaFold-rsa |
| Linker | 27 | 14 | 13 | +0.0011 | 0.581 | IPA-AF2-Linker |
| Linker | 31 | 26 | 5 | +0.0765 | **0.00005** | AlphaFold-rsa |

**This is a different claim from the pooled one and must be written as such.**
Pooled AUC is what CAID reports and what a leaderboard position means. The
signed-rank test asks on how many targets we win. A method can win on more
targets and still lose on pooled AUC. Here both point the same way on
Disorder-PDB — pooled 0.9595 against 0.9552, and 137 target wins against 77 —
but the pooled bootstrap gives p=0.20 for that same comparison, because it
resamples whole proteins for one statistic while the signed-rank test uses 233
paired observations directly.

Two readings worth separating:

- Against **AlphaFold-rsa** we win on every benchmark, all five significant.
  The model has clearly learned something beyond the structural feature.
- Against the **published leaders** only Disorder-PDB separates. NOX, Binding,
  Binding-IDR and Linker are all coin-flips on this test, which agrees with the
  pooled result and with the power analysis.

Binding-IDR is the interesting one: 21 wins to 21 losses against a leader that
beats us on pooled AUC by 0.14. Per-target discrimination is level; the pooled
gap is **cross-protein calibration**, not an inability to rank residues within
a protein. That is a different defect from the one diagnosed earlier and points
at a different fix.

## The targets the field cannot solve

Every entrant's predictions are published, so difficulty needs no assumption:
it is the median per-target AUC across all 117 methods, computed without our
own predictions, so there is no circularity.

| benchmark | band | n | field median | ours | entrants above us |
|---|---|---:|---:|---:|---:|
| Disorder-PDB | **hardest quartile** | 58 | 0.6514 | **0.8480** | **0** |
| Disorder-PDB | easiest quartile | 58 | 0.9836 | 0.9980 | 0 |
| Disorder-NOX | **hardest quartile** | 44 | 0.4978 | **0.6504** | **1** |
| Disorder-NOX | easiest quartile | 44 | 0.9641 | 0.9828 | 9 |
| Binding | hardest quartile | 12 | 0.3410 | 0.5054 | 16 |
| Binding | easiest quartile | 12 | 0.9120 | 0.9475 | 2 |

**No entrant of 117 beats us on the hardest quarter of Disorder-PDB**, where
the field's median target is at 0.6514 and we are at 0.8480. On Disorder-NOX's
hardest quarter the field sits at 0.4978 — chance — and one method beats us.

This is what a pooled score cannot show. A model that is merely well calibrated
on easy targets and one that solves hard ones post the same headline number.
Note also the Disorder-NOX rows: nine entrants beat us on the *easiest*
quartile while one beats us on the hardest, so our advantage is concentrated
exactly where the benchmark is difficult.

Binding is the exception in the informative direction — 16 entrants beat us on
its hardest quartile, and the field median there is 0.3410, well below chance,
meaning those twelve targets are anti-predicted by most methods. That is a
property of the targets, not a ranking of skill.
