/-
# Part LIX  Benchmarks with unsound annotations

Part XLV assumes the annotation is sound and only incomplete.  A disorder benchmark cannot make
that assumption: a residue is called disordered because it is missing from a map, and Part
XXXVIII enumerated what else produces a missing residue.  `RequestProject.LabelNoise` removes the
soundness assumption entirely -- errors run in both directions and nothing is assumed about their
direction -- and prices what remains.

`IDR.label_noise_laws` bundles five statements:

1. *The measured score brackets the true score*, in both directions, by the annotation error
   count `noise` and by nothing else.
2. *A perfect predictor scores exactly the noise*: a model that reproduces the truth residue by
   residue is recorded as making `noise T L` mistakes.
3. *A margin of more than twice the noise certifies the ranking.*  This is the usable statement.
4. *A single spurious label inverts a ranking*: two explicit instances where the better model
   measures worse.
5. *The factor two is sharp*: at a margin of exactly twice the noise the benchmark records a tie,
   so the certificate cannot be weakened.

The reading: a disorder benchmark measures the annotation, not the protein, and the gap between
them is bounded by the annotation error rate alone.  A leaderboard whose margins are smaller than
that rate reports a ranking the data do not contain.
-/
import Mathlib
import RequestProject.LabelNoise

set_option autoImplicit false

namespace IDR

open IDR.LabelNoise

/-- **The benchmark laws under two-sided annotation error.**

1. the measured error count brackets the true one, both ways, by the noise;
2. a perfect predictor is recorded as making exactly `noise` mistakes;
3. a margin exceeding twice the noise certifies the ranking;
4. explicit ranking inversions, one of them from a single spurious label;
5. the factor two is sharp: at margin exactly twice the noise the benchmark ties. -/
theorem label_noise_laws :
    (∀ (α : Type) (_ : Fintype α) (_ : DecidableEq α) (T L P : Finset α),
        errors L P ≤ errors T P + noise T L ∧ errors T P ≤ errors L P + noise T L) ∧
    (∀ (α : Type) (_ : Fintype α) (_ : DecidableEq α) (T L : Finset α),
        errors L T = noise T L) ∧
    (∀ (α : Type) (_ : Fintype α) (_ : DecidableEq α) (T L P Q : Finset α),
        errors T P + 2 * noise T L < errors T Q → errors L P < errors L Q) ∧
    ((noise truth4 annot4 = 2 ∧
        errors truth4 truth4 < errors truth4 annot4 ∧
        errors annot4 annot4 < errors annot4 truth4) ∧
      (noise (∅ : Finset (Fin 2)) {0} = 1 ∧
        errors (∅ : Finset (Fin 2)) ∅ < errors (∅ : Finset (Fin 2)) {0} ∧
        errors ({0} : Finset (Fin 2)) {0} < errors ({0} : Finset (Fin 2)) ∅)) ∧
    (errors (∅ : Finset (Fin 3)) ∅ + 2 * noise (∅ : Finset (Fin 3)) {0}
        = errors (∅ : Finset (Fin 3)) {0, 1} ∧
      errors ({0} : Finset (Fin 3)) ∅ = errors ({0} : Finset (Fin 3)) {0, 1}) := by
  refine ⟨fun α _ _ T L P => ⟨errors_le_add_noise T L P, errors_le_add_noise' T L P⟩,
    fun α _ _ T L => perfect_predictor_penalised T L,
    fun α _ _ T L P Q h => ranking_certified h,
    ⟨ranking_inversion_false_positive, ranking_inversion_single_label⟩,
    ranking_certificate_sharp⟩

end IDR
