/-
# Part LIX.1  Benchmarks with unsound annotations: label noise in both directions

Part XLV assumes the annotation is *sound*: every residue labelled disordered really is, and only
the coverage is partial.  That is the charitable assumption, and it is the one a disorder
benchmark cannot make.  Regions are annotated disordered because they are missing from a
crystallographic map, and Part XXXVIII showed exactly how many other things produce a missing
region; regions are annotated ordered because a structure was solved under conditions that need
not be the cellular ones.  Errors run in both directions.  This file removes the soundness
assumption and prices what is left.

The setting is deliberately minimal: residues form a finite type, the truth `T`, the annotation
`L` and a prediction `P` are subsets, `errors R P` counts the residues on which `P` disagrees with
the reference `R` (the symmetric difference), and `noise T L` counts the residues the annotation
gets wrong.  Nothing is assumed about the direction of the errors.

* `errors_le_add_noise`, `errors_le_add_noise'` -- **the measured score is within the noise of the
  true score, in both directions.**  This is the whole positive content of a benchmark with
  unsound labels: a measured error count certifies the true error count to within `noise`, and no
  better.
* `perfect_predictor_penalised` -- **a perfect predictor scores exactly the noise.**  A model that
  reproduces the truth residue for residue is measured to make `noise T L` mistakes -- not zero.
  Under the sound-annotation assumption of Part XLV those mistakes are all of one kind and can be
  argued away; with false positives present they cannot.
* `ranking_certified` -- **a margin of more than twice the noise certifies the ranking.**  If `P`
  beats `Q` on the truth by more than `2·noise`, then `P` beats `Q` on the benchmark.  This is the
  usable statement: quote a margin, compare it with an estimate of the annotation error rate, and
  the comparison is either certified or it is not.
* `ranking_inversion_false_positive`, `ranking_inversion_single_label` -- **and a single
  spurious label is enough to invert a ranking.**  Two explicit instances in which `P` is
  strictly better than `Q` on the truth and strictly worse on the benchmark: a four-residue one
  with a false negative and a false positive, and a two-residue one with a single false
  positive.
* `ranking_certificate_sharp` -- **the factor two is sharp.**  An explicit instance with margin
  exactly `2·noise` on which the benchmark records a tie.  So `> 2·noise` cannot be weakened to
  `≥ 2·noise`, and the certificate is the best one available.

The reading: a disorder benchmark measures the annotation, not the protein, and the difference is
bounded by the annotation error rate and by nothing else.  A leaderboard whose margins are
smaller than that rate reports a ranking that the data do not contain.
-/
import Mathlib

set_option autoImplicit false

namespace IDR
namespace LabelNoise

open Finset

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The number of residues on which a prediction `P` disagrees with a reference set `R`. -/
def errors (R P : Finset α) : ℕ := (symmDiff R P).card

/-- The number of residues the annotation `L` gets wrong: both missed disorder and spurious
disorder. -/
def noise (T L : Finset α) : ℕ := (symmDiff T L).card

lemma noise_comm (T L : Finset α) : noise T L = noise L T := by
  unfold noise
  rw [symmDiff_comm]

/-- **The measured score is within the noise of the true score.** -/
theorem errors_le_add_noise (T L P : Finset α) : errors L P ≤ errors T P + noise T L := by
  unfold errors noise
  have hsub : symmDiff L P ≤ symmDiff L T ⊔ symmDiff T P := symmDiff_triangle L T P
  calc (symmDiff L P).card ≤ (symmDiff L T ⊔ symmDiff T P).card := Finset.card_le_card hsub
    _ ≤ (symmDiff L T).card + (symmDiff T P).card := Finset.card_union_le _ _
    _ = (symmDiff T P).card + (symmDiff T L).card := by rw [symmDiff_comm L T]; ring

/-- The other direction of the same bound. -/
theorem errors_le_add_noise' (T L P : Finset α) : errors T P ≤ errors L P + noise T L := by
  have := errors_le_add_noise L T P
  rwa [noise_comm L T] at this

/-- **A perfect predictor is measured to make exactly `noise` mistakes.**  With sound annotations
those mistakes are all of one kind; with false positives they are not. -/
theorem perfect_predictor_penalised (T L : Finset α) : errors L T = noise T L := by
  unfold errors noise
  rw [symmDiff_comm]

/-- **A margin of more than twice the noise certifies the ranking.** -/
theorem ranking_certified {T L P Q : Finset α}
    (h : errors T P + 2 * noise T L < errors T Q) : errors L P < errors L Q := by
  have h1 : errors L P ≤ errors T P + noise T L := errors_le_add_noise T L P
  have h2 : errors T Q ≤ errors L Q + noise T L := errors_le_add_noise' T L Q
  omega

/-! ### Sharpness -/

/-- The truth: residues `0` and `1` are disordered. -/
def truth4 : Finset (Fin 4) := {0, 1}

/-- The annotation: one false negative (`1`) and one false positive (`2`). -/
def annot4 : Finset (Fin 4) := {0, 2}

/-- **A ranking inversion caused by annotation error.**  `P` is exactly right and `Q` is exactly
the annotation; the benchmark reverses them, on two mislabelled residues out of four. -/
theorem ranking_inversion_false_positive :
    noise truth4 annot4 = 2 ∧
      errors truth4 truth4 < errors truth4 annot4 ∧
      errors annot4 annot4 < errors annot4 truth4 := by
  refine ⟨by decide, by decide, by decide⟩

/-- A single spurious label already inverts a ranking. -/
theorem ranking_inversion_single_label :
    noise (∅ : Finset (Fin 2)) {0} = 1 ∧
      errors (∅ : Finset (Fin 2)) ∅ < errors (∅ : Finset (Fin 2)) {0} ∧
      errors ({0} : Finset (Fin 2)) {0} < errors ({0} : Finset (Fin 2)) ∅ := by
  refine ⟨by decide, by decide, by decide⟩

/-- **The factor two is sharp.**  With a margin of exactly `2·noise` the benchmark records a
tie, so the certificate `> 2·noise` cannot be weakened to `≥ 2·noise`. -/
theorem ranking_certificate_sharp :
    errors (∅ : Finset (Fin 3)) ∅ + 2 * noise (∅ : Finset (Fin 3)) {0}
        = errors (∅ : Finset (Fin 3)) {0, 1} ∧
      errors ({0} : Finset (Fin 3)) ∅ = errors ({0} : Finset (Fin 3)) {0, 1} := by
  refine ⟨by decide, by decide⟩

end LabelNoise
end IDR
