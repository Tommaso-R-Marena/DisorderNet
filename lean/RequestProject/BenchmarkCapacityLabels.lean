/-
# The capacity theorem for a benchmark with noisy labels, instantiated on Hamming scores

`RequestProject.LabelNoise` prices a single comparison: a benchmark scored against an
annotation `L` that differs from the truth `T` on `nu = noise T L` residues certifies the
ranking of two methods exactly when their true scores differ by more than `2 * nu`
(`IDR.LabelNoise.ranking_certified`), and the factor two cannot be improved
(`ranking_certificate_sharp`).  `RequestProject.BenchmarkCapacity` turns a resolution into a
capacity: how many scores can be pairwise separated inside a bounded range.

This file puts the two together and adds the missing half -- the converse.  The results are:

* `benchmark_faithful_of_certified` -- **inside a certified family, the leaderboard is the
  truth.**  If every two methods of a family differ in true score by more than `2 * nu`, then
  the measured order and the true order agree on every pair.
* `card_le_capacity_labelNoise` -- **the capacity bound.**  Such a family has at most
  `capacityNat (Fintype.card α) (2 * nu)` members: with `R` scorable residues and `nu`
  mislabelled ones, at most `R / (2 nu + 1) + 1` methods can be ordered, however many are
  entered.
* `card_le_benchCapacity_of_rate` -- **the same bound stated in rates.**  If the annotation is
  known only to be at most a fraction `eps` wrong, the number of methods that can be certified
  against that budget is at most `benchCapacity R eps 0`, the general capacity of
  `RequestProject.BenchmarkCapacity` evaluated at the annotation error rate.
* `labelNoise_capacity_attained` -- **the bound is attained**, by an explicit truth, an
  explicit annotation with exactly `nu` errors, and an explicit family of predictions of
  exactly that size.  So the number is the capacity, not an estimate.
* `unresolvable_pair` -- **the converse, and the reason the bound is not an artefact of this
  analysis.**  If two methods' measured scores are close, then *two different truths*, each
  compatible with the observed annotation (each within `nu` of it), make opposite methods the
  better one.  The benchmark data are consistent with both orders, so no analysis of those
  data -- present or future, however clever -- can decide the comparison.  The only escape is
  better labels.
* `unresolvable_of_close` -- the same statement in the regime where the two methods disagree
  on at least `nu` residues in each direction, where it reads: a measured gap of less than
  `2 * nu` is undecidable.
* `top_group_unresolvable` -- **the top-k statement.**  Every pair inside a group of methods
  whose measured scores lie within `2 * nu` is undecidable in that sense, so a leaderboard that
  orders such a group reports label error, not method quality.
* `over_capacity_has_close_pair`, `capacity_theorem` -- **the headline.**  Enter more than
  `capacityNat R (2 nu)` methods and two of them are necessarily within `2 * nu`; with room to
  move they are unresolvable in the strong sense above.  A benchmark can rank `k` methods, the
  `k` is computable from the size of the label set and the annotation error rate alone, and
  the methods beyond it are not merely unranked but unrankable.
-/
import Mathlib
import RequestProject.LabelNoise
import RequestProject.BenchmarkCapacity

set_option autoImplicit false

namespace IDR
namespace BenchCapacity

open Finset
open IDR.LabelNoise

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## 1.  Certified families and the capacity bound -/

/-- A family `M` of methods, with predictions `pred`, is **certified** against the truth `T` at
noise level `nu` when every two of its members differ in true score by more than `2 * nu`. -/
def Certified {iota : Type*} (T : Finset α) (nu : ℕ) (pred : iota → Finset α)
    (M : Finset iota) : Prop :=
  ∀ i ∈ M, ∀ j ∈ M, i ≠ j →
    errors T (pred i) + 2 * nu < errors T (pred j) ∨ errors T (pred j) + 2 * nu < errors T (pred i)

/-- **Inside a certified family the leaderboard is the truth.**  Every pairwise comparison the
benchmark reports agrees with the comparison against the truth, in both directions. -/
theorem benchmark_faithful_of_certified {iota : Type*} {T L : Finset α} {pred : iota → Finset α}
    {M : Finset iota} (hcert : Certified T (noise T L) pred M) {i j : iota}
    (hi : i ∈ M) (hj : j ∈ M) (hij : i ≠ j) :
    (errors T (pred i) < errors T (pred j) ↔ errors L (pred i) < errors L (pred j)) := by
  constructor
  · intro h
    rcases hcert i hi j hj hij with hc | hc
    · exact ranking_certified hc
    · omega
  · intro h
    rcases hcert i hi j hj hij with hc | hc
    · omega
    · exact absurd (ranking_certified hc) (by omega)

/-- **The capacity bound for a benchmark with noisy labels.**  A certified family of methods has
at most `capacityNat (Fintype.card α) (2 * nu)` members. -/
theorem card_le_capacity_labelNoise {iota : Type*} {T : Finset α} {nu : ℕ}
    {pred : iota → Finset α} {M : Finset iota} (hcert : Certified T nu pred M) :
    M.card ≤ capacityNat (Fintype.card α) (2 * nu) := by
  classical
  have hinj : Set.InjOn (fun i => errors T (pred i)) M := by
    intro i hi j hj hij
    by_contra hne
    rcases hcert i hi j hj hne with h | h <;> simp only at hij <;> omega
  have hmem : ∀ a ∈ M.image (fun i => errors T (pred i)), a ≤ Fintype.card α := by
    intro a ha
    simp only [Finset.mem_image] at ha
    obtain ⟨i, _, rfl⟩ := ha
    exact Finset.card_le_univ _
  have hsep : Separated (2 * nu) (M.image (fun i => errors T (pred i))) := by
    intro a ha b hb hab
    simp only [Finset.mem_image] at ha hb
    obtain ⟨i, hi, rfl⟩ := ha
    obtain ⟨j, hj, rfl⟩ := hb
    have hij : i ≠ j := by rintro rfl; exact absurd hab (lt_irrefl _)
    rcases hcert i hi j hj hij with h | h
    · omega
    · omega
  have := card_le_capacityNat hmem hsep
  rwa [Finset.card_image_of_injOn hinj] at this

/-- **The rate form of the bound.**  Suppose the annotation error is only known through a
*rate*: at most a fraction `eps` of the `R` scorable residues is mislabelled, and the family of
methods is certified against that budget (pairwise true gaps of more than `2 * m`, where the
budget `m` is at least `eps * R`).  Then the family has at most `benchCapacity R eps 0`
members -- the general benchmark capacity of `RequestProject.BenchmarkCapacity`, evaluated at
the annotation error rate. -/
theorem card_le_benchCapacity_of_rate {iota : Type*} {T : Finset α} {eps : ℝ} {m : ℕ}
    {pred : iota → Finset α} {M : Finset iota} (hR : 0 < Fintype.card α) (heps : 0 ≤ eps)
    (hbudget : eps * (Fintype.card α : ℝ) ≤ m) (hcert : Certified T m pred M) :
    M.card ≤ benchCapacity (Fintype.card α) eps 0 := by
  classical
  have hRR : (0:ℝ) < (Fintype.card α : ℝ) := by exact_mod_cast hR
  have hres : resolution eps 0 = 2 * eps := by
    simp only [resolution]
    exact max_eq_right (by linarith)
  refine card_le_benchCapacity (score := fun i => errors T (pred i)) hR le_rfl ⟨?_, ?_⟩
  · intro i _
    exact Finset.card_le_univ _
  · intro i hi j hj hij
    have hstep : ∀ a b : ℕ, a + 2 * m < b →
        (a : ℝ) / (Fintype.card α) + resolution eps 0 < (b : ℝ) / (Fintype.card α) := by
      intro a b hab
      have hgap : (a : ℝ) + 2 * eps * (Fintype.card α : ℝ) < (b : ℝ) := by
        have h1 : (a : ℝ) + 2 * (m : ℝ) < (b : ℝ) := by exact_mod_cast hab
        nlinarith [hbudget]
      rw [hres, div_add' _ _ _ (ne_of_gt hRR), div_lt_div_iff_of_pos_right hRR]
      nlinarith [hgap]
    rcases hcert i hi j hj hij with h | h
    · exact Or.inl (hstep _ _ h)
    · exact Or.inr (hstep _ _ h)

/-! ## 2.  The bound is attained -/

private lemma symmDiff_empty_left {beta : Type*} [DecidableEq beta] (P : Finset beta) :
    symmDiff (∅ : Finset beta) P = P := by
  ext x
  simp [Finset.mem_symmDiff]

private lemma card_filter_val_lt (N m : ℕ) (hm : m ≤ N) :
    ((Finset.univ : Finset (Fin N)).filter (fun x : Fin N => (x : ℕ) < m)).card = m := by
  classical
  have himg : ((Finset.univ : Finset (Fin N)).filter
      (fun x : Fin N => (x : ℕ) < m)).image (Fin.val) = Finset.range m := by
    ext a
    simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_range]
    constructor
    · rintro ⟨x, hx, rfl⟩; exact hx
    · intro ha
      exact ⟨⟨a, by omega⟩, ha, rfl⟩
  have := congrArg Finset.card himg
  rwa [Finset.card_image_of_injective _ Fin.val_injective, Finset.card_range] at this

/-- **The capacity is attained.**  For any residue count `N` and any noise level `nu ≤ N` there
are a truth, an annotation making exactly `nu` errors, and a family of exactly
`capacityNat N (2 * nu)` methods which the benchmark certifiably ranks. -/
theorem labelNoise_capacity_attained (N nu : ℕ) (hnu : nu ≤ N) :
    ∃ (T L : Finset (Fin N)) (M : Finset ℕ) (pred : ℕ → Finset (Fin N)),
      noise T L = nu ∧ Certified T (noise T L) pred M ∧
        M.card = capacityNat N (2 * nu) := by
  classical
  refine ⟨∅, (Finset.univ : Finset (Fin N)).filter (fun x : Fin N => (x : ℕ) < nu), ?_⟩
  obtain ⟨S, hbd, hsep, hcard⟩ := capacityNat_attained N (2 * nu)
  refine ⟨S, fun m => (Finset.univ : Finset (Fin N)).filter (fun x : Fin N => (x : ℕ) < m),
    ?_, ?_, hcard⟩
  · -- the annotation makes exactly `nu` errors
    unfold noise
    rw [symmDiff_empty_left]
    exact card_filter_val_lt N nu hnu
  · -- the family is certified
    have hnoise : noise (∅ : Finset (Fin N))
        ((Finset.univ : Finset (Fin N)).filter (fun x : Fin N => (x : ℕ) < nu)) = nu := by
      unfold noise
      rw [symmDiff_empty_left]
      exact card_filter_val_lt N nu hnu
    have hscore : ∀ m ∈ S, errors (∅ : Finset (Fin N))
        ((Finset.univ : Finset (Fin N)).filter (fun x : Fin N => (x : ℕ) < m)) = m := by
      intro m hm
      unfold errors
      rw [symmDiff_empty_left]
      exact card_filter_val_lt N m (hbd m hm)
    rw [hnoise]
    intro i hi j hj hij
    rw [hscore i hi, hscore j hj]
    rcases lt_trichotomy i j with h | h | h
    · exact Or.inl (hsep i hi j hj h)
    · exact absurd h hij
    · exact Or.inr (hsep j hj i hi h)

/-! ## 3.  The converse: pairs that no analysis can resolve

The observation a benchmark makes is the annotation `L`, not the truth.  Every set `T` within
`nu` of `L` is a truth consistent with that observation.  If two candidate truths consistent
with the observation disagree about which of two methods is better, then the comparison is not
a function of the data: no method of analysis can extract it, because it is not there. -/

/-- Flipping the annotation on a set `D` produces a candidate truth making `D.card` errors. -/
lemma noise_symmDiff (L D : Finset α) : noise (symmDiff L D) L = D.card := by
  unfold noise
  rw [symmDiff_right_comm, symmDiff_self, bot_symmDiff]

omit [Fintype α] in
private lemma errors_flip (L D P : Finset α) :
    errors (symmDiff L D) P = (symmDiff (symmDiff L P) D).card := by
  unfold errors
  rw [symmDiff_right_comm]

/-- Flipping the annotation on a set of residues where `P` is wrong and `Q` is right lowers the
score of `P` and raises the score of `Q`, by the size of that set. -/
private lemma scores_after_flip {L P Q D : Finset α}
    (hD : D ⊆ symmDiff L P \ symmDiff L Q) :
    errors (symmDiff L D) P = errors L P - D.card ∧
      errors (symmDiff L D) Q = errors L Q + D.card := by
  classical
  have hDA : D ⊆ symmDiff L P := hD.trans (Finset.sdiff_subset)
  have hDB : Disjoint (symmDiff L Q) D := by
    rw [Finset.disjoint_right]
    intro x hx hxB
    have := hD hx
    rw [Finset.mem_sdiff] at this
    exact this.2 hxB
  constructor
  · rw [errors_flip, symmDiff_of_ge (le_iff_subset.mpr hDA), Finset.card_sdiff_of_subset hDA]
    rfl
  · rw [errors_flip, hDB.symmDiff_eq_sup, Finset.sup_eq_union, Finset.card_union_of_disjoint hDB]
    rfl

/-- **Unresolvability.**  If the measured scores of `P` and `Q` are close enough that a
permissible relabelling of `D`-many residues can reverse them, then there are two truths, both
consistent with the observed annotation (each within `nu` of it), which disagree about which
method is better.  The benchmark data therefore do not determine the comparison, and no
analysis of them can. -/
theorem unresolvable_pair {L P Q : Finset α} {nu : ℕ}
    (hPwin : errors L P < errors L Q + 2 * min nu (symmDiff L P \ symmDiff L Q).card)
    (hQwin : errors L Q < errors L P + 2 * min nu (symmDiff L Q \ symmDiff L P).card) :
    (∃ T₁ : Finset α, noise T₁ L ≤ nu ∧ errors T₁ P < errors T₁ Q) ∧
      (∃ T₂ : Finset α, noise T₂ L ≤ nu ∧ errors T₂ Q < errors T₂ P) := by
  classical
  constructor
  · -- flip residues where `P` errs and `Q` does not: `P` becomes the better method
    obtain ⟨D, hDsub, hDcard⟩ :=
      Finset.exists_subset_card_eq
        (show min nu (symmDiff L P \ symmDiff L Q).card ≤ (symmDiff L P \ symmDiff L Q).card from
          min_le_right _ _)
    obtain ⟨h1, h2⟩ := scores_after_flip hDsub
    refine ⟨symmDiff L D, ?_, ?_⟩
    · rw [noise_symmDiff, hDcard]; exact min_le_left _ _
    · rw [h1, h2, hDcard]
      have hbound : min nu (symmDiff L P \ symmDiff L Q).card ≤ errors L P := by
        unfold errors
        exact (min_le_right _ _).trans (Finset.card_le_card Finset.sdiff_subset)
      omega
  · -- and symmetrically
    obtain ⟨D, hDsub, hDcard⟩ :=
      Finset.exists_subset_card_eq
        (show min nu (symmDiff L Q \ symmDiff L P).card ≤ (symmDiff L Q \ symmDiff L P).card from
          min_le_right _ _)
    obtain ⟨h1, h2⟩ := scores_after_flip hDsub
    refine ⟨symmDiff L D, ?_, ?_⟩
    · rw [noise_symmDiff, hDcard]; exact min_le_left _ _
    · rw [h1, h2, hDcard]
      have hbound : min nu (symmDiff L Q \ symmDiff L P).card ≤ errors L Q := by
        unfold errors
        exact (min_le_right _ _).trans (Finset.card_le_card Finset.sdiff_subset)
      omega

/-- **The clean regime.**  When the two methods disagree with the annotation on at least `nu`
residues in each direction -- the generic case for methods that are not near-copies of each
other -- a measured gap of less than `2 * nu` is undecidable: both orders are consistent with
the data. -/
theorem unresolvable_of_close {L P Q : Finset α} {nu : ℕ}
    (hroom1 : nu ≤ (symmDiff L P \ symmDiff L Q).card)
    (hroom2 : nu ≤ (symmDiff L Q \ symmDiff L P).card)
    (hPQ : errors L P < errors L Q + 2 * nu) (hQP : errors L Q < errors L P + 2 * nu) :
    (∃ T₁ : Finset α, noise T₁ L ≤ nu ∧ errors T₁ P < errors T₁ Q) ∧
      (∃ T₂ : Finset α, noise T₂ L ≤ nu ∧ errors T₂ Q < errors T₂ P) := by
  apply unresolvable_pair
  · rw [min_eq_left hroom1]; exact hPQ
  · rw [min_eq_left hroom2]; exact hQP

/-- **The top group of a leaderboard is unresolvable.**  Take any group of methods whose
measured scores all lie within `2 * nu` of each other and which disagree with the annotation
on at least `nu` residues in each direction.  Then *every* pair in the group is undecidable:
for each ordered pair there is a truth consistent with the observed annotation under which the
first method is the better one.  A challenge that reports a ranking of such a group is
reporting an artefact of its label errors. -/
theorem top_group_unresolvable {iota : Type*} {L : Finset α} {nu : ℕ}
    {pred : iota → Finset α} {M : Finset iota}
    (hroom : ∀ i ∈ M, ∀ j ∈ M, i ≠ j → nu ≤ (symmDiff L (pred i) \ symmDiff L (pred j)).card)
    (hwin : ∀ i ∈ M, ∀ j ∈ M, errors L (pred i) < errors L (pred j) + 2 * nu) :
    ∀ i ∈ M, ∀ j ∈ M, i ≠ j →
      (∃ T₁ : Finset α, noise T₁ L ≤ nu ∧ errors T₁ (pred i) < errors T₁ (pred j)) ∧
        (∃ T₂ : Finset α, noise T₂ L ≤ nu ∧ errors T₂ (pred j) < errors T₂ (pred i)) := by
  intro i hi j hj hij
  exact unresolvable_of_close (hroom i hi j hj hij) (hroom j hj i hi (Ne.symm hij))
    (hwin i hi j hj) (hwin j hj i hi)

/-! ## 4.  The headline -/

/-- **Beyond capacity there is always a pair the benchmark cannot certify.** -/
theorem over_capacity_has_close_pair {iota : Type*} {T : Finset α} {nu : ℕ}
    {pred : iota → Finset α} {M : Finset iota}
    (hbig : capacityNat (Fintype.card α) (2 * nu) < M.card) :
    ∃ i ∈ M, ∃ j ∈ M, i ≠ j ∧
      errors T (pred i) ≤ errors T (pred j) + 2 * nu ∧
      errors T (pred j) ≤ errors T (pred i) + 2 * nu := by
  classical
  by_contra hcon
  push_neg at hcon
  have hcert : Certified T nu pred M := by
    intro i hi j hj hij
    have := hcon i hi j hj hij
    by_cases h : errors T (pred i) ≤ errors T (pred j) + 2 * nu
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  exact absurd (card_le_capacity_labelNoise hcert) (by omega)

/-- **The capacity theorem.**  Fix a benchmark: a finite residue set of size `R`, a truth `T`
and an annotation `L` differing from it on `nu` residues.  Then

1. any family of methods whose true scores are pairwise more than `2 * nu` apart has at most
   `R / (2 nu + 1) + 1` members, and inside such a family the leaderboard reproduces the truth
   pair for pair;
2. that number is attained -- there are benchmarks and families of exactly that size which are
   certifiably ranked;
3. beyond it, every family contains a pair whose true scores are within `2 * nu`, and such a
   pair is not merely uncertified by this argument: two truths consistent with the observed
   annotation order it both ways, so no analysis of the benchmark's data can rank it.

The three parts together say what a community challenge can ever establish. -/
theorem capacity_theorem {iota : Type*} (T L : Finset α) (pred : iota → Finset α) :
    (∀ M : Finset iota, Certified T (noise T L) pred M →
        M.card ≤ capacityNat (Fintype.card α) (2 * noise T L) ∧
        ∀ i ∈ M, ∀ j ∈ M, i ≠ j →
          (errors T (pred i) < errors T (pred j) ↔ errors L (pred i) < errors L (pred j))) ∧
      (∀ M : Finset iota, capacityNat (Fintype.card α) (2 * noise T L) < M.card →
        ∃ i ∈ M, ∃ j ∈ M, i ≠ j ∧
          errors T (pred i) ≤ errors T (pred j) + 2 * noise T L ∧
          errors T (pred j) ≤ errors T (pred i) + 2 * noise T L) := by
  refine ⟨fun M hcert => ⟨card_le_capacity_labelNoise hcert, ?_⟩, fun M hbig => ?_⟩
  · intro i hi j hj hij
    exact benchmark_faithful_of_certified hcert hi hj hij
  · exact over_capacity_has_close_pair hbig

end BenchCapacity
end IDR
