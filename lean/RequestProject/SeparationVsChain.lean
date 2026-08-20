/-
# Separated pairs versus separated chains: what the capacity theorem does and does not bound

The capacity theorem bounds the size of a family whose scores are **pairwise** separated — a
*chain* of mutually distinguishable methods.  It is tempting to read it as a bound on the number of
*separations*: on how many comparisons the benchmark can certify.  It is not, and the difference is
quadratic.

* `pairSeparated_card_le_capacity` — the statement the theorem does make, for an arbitrary score
  function: a family that is pairwise separated at resolution `g`, with scores in `{0, …, N}`, has
  at most `capacityNat N g` members.  (`RequestProject.BenchmarkCapacityLabels` states the same for
  Hamming scores against a truth.)
* `many_separations_short_chain` — **and the statement it does not make.**  On `2k` methods split
  into two well-separated clusters of `k`, the benchmark certifies `2k²` ordered comparisons — as
  many as one likes — while *every* pairwise separated subfamily has at most `2` members.  Counting
  certified comparisons therefore says nothing about how many methods can be ranked.

So a leaderboard may legitimately report a great many certified comparisons and still be unable to
order more than two methods; and conversely the capacity number bounds the chain, not the
comparisons.  The two counts are `k` and `k²` apart.
-/
import Mathlib
import RequestProject.BenchmarkCapacityLabels

set_option autoImplicit false

namespace IDR
namespace SepChain

open Finset
open IDR.BenchCapacity

variable {iota : Type*} [DecidableEq iota]

/-- A family of methods is **pairwise separated** at resolution `g` when any two of its members
differ in score by more than `g` — the certification condition, for an arbitrary score. -/
def PairSeparated (sc : iota → ℕ) (g : ℕ) (M : Finset iota) : Prop :=
  ∀ i ∈ M, ∀ j ∈ M, i ≠ j → sc i + g < sc j ∨ sc j + g < sc i

/-- The **certified comparisons** of a family: the ordered pairs the benchmark can separate. -/
def sepPairs (sc : iota → ℕ) (g : ℕ) (M : Finset iota) : Finset (iota × iota) :=
  (M ×ˢ M).filter (fun q => sc q.1 + g < sc q.2 ∨ sc q.2 + g < sc q.1)

/-- **What the capacity theorem bounds: the chain.**  A pairwise separated family of methods with
scores in `{0, …, N}` has at most `capacityNat N g` members. -/
theorem pairSeparated_card_le_capacity {sc : iota → ℕ} {g N : ℕ} {M : Finset iota}
    (hbd : ∀ i ∈ M, sc i ≤ N) (hsep : PairSeparated sc g M) :
    M.card ≤ capacityNat N g := by
  classical
  have hinj : Set.InjOn sc M := by
    intro i hi j hj hij
    by_contra hne
    rcases hsep i hi j hj hne with h | h <;> omega
  have hsepset : Separated g (M.image sc) := by
    intro a ha b hb hab
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.mp ha
    obtain ⟨j, hj, rfl⟩ := Finset.mem_image.mp hb
    have hij : i ≠ j := by rintro rfl; omega
    rcases hsep i hi j hj hij with h | h
    · exact h
    · omega
  have hmem : ∀ a ∈ M.image sc, a ≤ N := by
    intro a ha
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.mp ha
    exact hbd i hi
  calc M.card = (M.image sc).card := (Finset.card_image_of_injOn hinj).symm
    _ ≤ capacityNat N g := card_le_capacityNat hmem hsepset

/-! ## The instance: many separations, a chain of two -/

/-- Two clusters of `k` methods, separated by more than `g`: the first `k` score `0`, the rest
score `g + 1`. -/
def cluster (k g : ℕ) : ℕ → ℕ := fun i => if i < k then 0 else g + 1

lemma range_low (k g : ℕ) :
    (Finset.range (2 * k)).filter (fun i => cluster k g i = 0) = Finset.range k := by
  ext i
  simp only [Finset.mem_filter, Finset.mem_range, cluster]
  constructor
  · rintro ⟨h1, h2⟩
    by_contra h
    rw [if_neg (by omega)] at h2
    omega
  · intro h
    exact ⟨by omega, by rw [if_pos h]⟩

lemma range_high (k g : ℕ) :
    (Finset.range (2 * k)).filter (fun i => ¬ cluster k g i = 0) = Finset.Ico k (2 * k) := by
  ext i
  simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico, cluster]
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨?_, h1⟩
    by_contra h
    rw [if_pos (by omega)] at h2
    exact h2 rfl
  · rintro ⟨h1, h2⟩
    exact ⟨h2, by rw [if_neg (by omega)]; omega⟩

/-- The certified comparisons of the two-cluster instance are exactly the cross-cluster pairs. -/
lemma sepPairs_cluster (k g : ℕ) :
    sepPairs (cluster k g) g (Finset.range (2 * k))
      = (Finset.range k ×ˢ Finset.Ico k (2 * k)) ∪ (Finset.Ico k (2 * k) ×ˢ Finset.range k) := by
  ext q
  simp only [sepPairs, Finset.mem_filter, Finset.mem_product, Finset.mem_union, Finset.mem_range,
    Finset.mem_Ico, cluster]
  constructor
  · rintro ⟨⟨h1, h2⟩, h3⟩
    by_cases hq1 : q.1 < k <;> by_cases hq2 : q.2 < k
    · rw [if_pos hq1, if_pos hq2] at h3; omega
    · exact Or.inl ⟨hq1, by omega, h2⟩
    · exact Or.inr ⟨⟨by omega, h1⟩, hq2⟩
    · rw [if_neg hq1, if_neg hq2] at h3; omega
  · rintro (⟨h1, h2, h3⟩ | ⟨⟨h1, h2⟩, h3⟩)
    · exact ⟨⟨by omega, h3⟩, by rw [if_pos h1, if_neg (by omega)]; omega⟩
    · exact ⟨⟨h2, by omega⟩, by rw [if_neg (by omega), if_pos h3]; omega⟩

/-- **Many separations, a chain of two.**  The two-cluster instance certifies `2k²` ordered
comparisons — arbitrarily many — while every pairwise separated subfamily has at most two members.
The number of certified comparisons is therefore no guide at all to how many methods can be
ranked. -/
theorem many_separations_short_chain (k g : ℕ) :
    (sepPairs (cluster k g) g (Finset.range (2 * k))).card = 2 * k * k ∧
      ∀ S ⊆ Finset.range (2 * k), PairSeparated (cluster k g) g S → S.card ≤ 2 := by
  classical
  constructor
  · rw [sepPairs_cluster]
    have hdisj : Disjoint (Finset.range k ×ˢ Finset.Ico k (2 * k))
        (Finset.Ico k (2 * k) ×ˢ Finset.range k) := by
      rw [Finset.disjoint_left]
      intro q hq hq'
      have h1 := (Finset.mem_product.mp hq).1
      have h2 := (Finset.mem_product.mp hq').1
      rw [Finset.mem_range] at h1
      rw [Finset.mem_Ico] at h2
      omega
    rw [Finset.card_union_of_disjoint hdisj, Finset.card_product, Finset.card_product,
      Finset.card_range, Nat.card_Ico]
    have : 2 * k - k = k := by omega
    rw [this]
    ring
  · intro S _ hsep
    have hinj : Set.InjOn (fun i => decide (cluster k g i = 0)) S := by
      intro i hi j hj hij
      by_contra hne
      have h : cluster k g i = cluster k g j := by
        by_cases h0 : cluster k g i = 0 <;> by_cases h1 : cluster k g j = 0
        · rw [h0, h1]
        · simp [h0, h1] at hij
        · simp [h0, h1] at hij
        · unfold cluster at h0 h1 ⊢
          split_ifs at h0 h1 ⊢ <;> omega
      rcases hsep i hi j hj hne with hs | hs <;> omega
    calc S.card ≤ (Finset.univ : Finset Bool).card :=
          Finset.card_le_card_of_injOn _ (fun i _ => by simp [em]) hinj
      _ = 2 := by decide

end SepChain
end IDR
