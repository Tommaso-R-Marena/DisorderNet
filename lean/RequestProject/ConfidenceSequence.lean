/-
# Part XCII.4  From a test to an estimate: an anytime-valid confidence sequence

A refutation is a poor summary of an experiment.  What the design of Part XC is really trying to
learn is *how much* population the model under test omits, and the sequential machinery answers
that question directly: run the wealth test against every candidate value of the omitted
population at once, and report the candidates that have not been refuted.  That set is a
*confidence sequence* — a confidence interval valid at every time simultaneously, so it may be
watched, plotted as the molecules accumulate, and stopped on, without correction.

* `Excluded` — a candidate value is excluded once the wealth of the bet against it has crossed
  `1/α`.
* `crossed_append`, `excluded_of_prefix` — **the sequence is nested**: a candidate excluded on a
  prefix of the data stays excluded on all of it.  The reported set only shrinks, so a plot of it
  against the molecule count is monotone and needs no re-reading of the earlier points.
* `confSeq_coverage`, `confSeq_coverage_all_horizons` — **coverage.** If the true read rate is the
  candidate's, the probability that the candidate is *ever* excluded is at most `α`, at every
  horizon at once.
* `population_confSeq_coverage` — the same statement in the units of the problem: candidate
  omitted populations `t`, read through a probe of sensitivity `se` and specificity `sp`, so that
  the reported set is a confidence sequence for the omitted population itself.
-/
import Mathlib
import RequestProject.Sequential

set_option autoImplicit false

namespace IDR
namespace Seq

open Finset

/-- Extending a run cannot undo a crossing. -/
lemma crossed_append {q₁ q₀ c : ℝ} :
    ∀ (l m : List Bool) (w : ℝ), Crossed q₁ q₀ c w l → Crossed q₁ q₀ c w (l ++ m) := by
  intro l
  induction l with
  | nil =>
      intro m w h
      cases m with
      | nil => exact h
      | cons b t => exact Or.inl h
  | cons b t ih =>
      intro m w h
      rcases h with h | h
      · exact Or.inl h
      · exact Or.inr (ih m _ h)

/-- The candidate read rate `qc` is excluded by the data `l` at level `α`: the wealth of the bet
of `q₁` against `qc` has crossed `1/α` at some point of the run. -/
def Excluded (q₁ qc α : ℝ) (l : List Bool) : Prop := Crossed q₁ qc (1 / α) 1 l

/-- **The confidence sequence is nested**: once a candidate is excluded it stays excluded. -/
lemma excluded_of_prefix {q₁ qc α : ℝ} (l m : List Bool) (h : Excluded q₁ qc α l) :
    Excluded q₁ qc α (l ++ m) :=
  crossed_append l m 1 h

open Classical in
/-- The indicator of exclusion, for summing against the read-out law. -/
noncomputable def exclInd (q₁ qc α : ℝ) (l : List Bool) : ℝ :=
  if Excluded q₁ qc α l then 1 else 0

lemma exclInd_eq_crossInd (q₁ qc α : ℝ) (l : List Bool) :
    exclInd q₁ qc α l = crossInd q₁ qc (1 / α) 1 l := rfl

/-- **Coverage.**  If the data really are generated at the candidate rate, the probability that
the candidate is ever excluded during a run of `n` molecules is at most `α`. -/
theorem confSeq_coverage {q₁ qc α : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1) (h00 : 0 < qc)
    (h01 : qc < 1) (hα : 0 < α) (n : ℕ) :
    ∑ l ∈ words n, probL qc l * exclInd q₁ qc α l ≤ α := by
  simpa [exclInd_eq_crossInd] using anytime_valid h10 h11 h00 h01 hα n

/-- Coverage holds at every horizon at once, which is what makes the sequence watchable. -/
theorem confSeq_coverage_all_horizons {q₁ qc α : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1) (h00 : 0 < qc)
    (h01 : qc < 1) (hα : 0 < α) :
    ∀ n : ℕ, ∑ l ∈ words n, probL qc l * exclInd q₁ qc α l ≤ α :=
  fun n => confSeq_coverage h10 h11 h00 h01 hα n

/-- **A confidence sequence for the omitted population.**  Candidate omitted populations `t` are
read through a probe of sensitivity `se` and specificity `sp`, so a candidate predicts the rate
`readRate t se sp`; the reported set is those candidates not yet excluded.  If the true omitted
population is `t`, the probability that `t` is ever dropped from the reported set is at most `α`,
at every horizon. -/
theorem population_confSeq_coverage {t se sp q₁ α : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1)
    (h00 : 0 < Noisy.readRate t se sp) (h01 : Noisy.readRate t se sp < 1) (hα : 0 < α) :
    ∀ n : ℕ, ∑ l ∈ words n,
        probL (Noisy.readRate t se sp) l * exclInd q₁ (Noisy.readRate t se sp) α l ≤ α :=
  fun n => confSeq_coverage h10 h11 h00 h01 hα n

end Seq
end IDR
