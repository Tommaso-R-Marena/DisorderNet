/-
# Maximum matchings in a threshold bipartite graph, and how to compute one

The crossed graph of `AUCCrossedMatching.lean` splits, over unordered pairs of proteins, into
bipartite graphs of a very special kind: a comparison with gap `u` on one side is joined to a
comparison with gap `v` on the other exactly when `u + v ≥ 0`.  Such a *threshold* graph has a
maximum matching computable by sorting, and this file proves it.

Fix `u : Fin p → ℝ` and `v : Fin q → ℝ`, **sorted in decreasing order**.

* `IsMatch`, `matchSizes`, `maxMatch` — matchings of size `m` are pairs of injections
  `f : Fin m → Fin p`, `g : Fin m → Fin q` with `u (f t) + v (g t) ≥ 0`; `maxMatch` is the largest
  size.
* `GreedyFeasible u v m` — the greedy test: pair the `m` largest `u`'s with the `m` largest `v`'s
  *in reverse*, `u t` with `v (m−1−t)`, and check every sum is non-negative.
* `greedy_mem_matchSizes` — the test is *sufficient*: if it passes, the reversed pairing is a
  matching of size `m`.
* `greedy_of_mem_matchSizes` — and *necessary*: every matching of size `m` forces it.
* `le_maxMatch_iff_greedyFeasible` — hence **`maxMatch` is exactly the largest `m` passing the
  greedy test**, and since the test is monotone in `m` (`greedyFeasible_of_succ`) it is found by a
  single scan over `m` after an `O(n log n)` sort — no general matching algorithm needed.
-/
import Mathlib

set_option autoImplicit false

namespace IDR.ThresholdMatching

open Finset

variable {p q : ℕ}

/-- A matching of size `m` in the threshold graph `u i + v j ≥ 0`. -/
def IsMatch (u : Fin p → ℝ) (v : Fin q → ℝ) (m : ℕ) (f : Fin m → Fin p) (g : Fin m → Fin q) :
    Prop :=
  Function.Injective f ∧ Function.Injective g ∧ ∀ t : Fin m, 0 ≤ u (f t) + v (g t)

/-- The sizes of matchings of the threshold graph. -/
def matchSizes (u : Fin p → ℝ) (v : Fin q → ℝ) : Set ℕ :=
  {m | ∃ f : Fin m → Fin p, ∃ g : Fin m → Fin q, IsMatch u v m f g}

theorem matchSizes_zero (u : Fin p → ℝ) (v : Fin q → ℝ) : 0 ∈ matchSizes u v :=
  ⟨Fin.elim0, Fin.elim0, fun t => t.elim0, fun t => t.elim0, fun t => t.elim0⟩

theorem le_of_mem_matchSizes {u : Fin p → ℝ} {v : Fin q → ℝ} {m : ℕ} (h : m ∈ matchSizes u v) :
    m ≤ p ∧ m ≤ q := by
  obtain ⟨f, g, hf, hg, -⟩ := h
  constructor
  · simpa using Fintype.card_le_of_injective f hf
  · simpa using Fintype.card_le_of_injective g hg

theorem matchSizes_bddAbove (u : Fin p → ℝ) (v : Fin q → ℝ) : BddAbove (matchSizes u v) :=
  ⟨p, fun _ hm => (le_of_mem_matchSizes hm).1⟩

/-- The size of a largest matching of the threshold graph. -/
noncomputable def maxMatch (u : Fin p → ℝ) (v : Fin q → ℝ) : ℕ := sSup (matchSizes u v)

theorem maxMatch_mem (u : Fin p → ℝ) (v : Fin q → ℝ) : maxMatch u v ∈ matchSizes u v :=
  Nat.sSup_mem ⟨0, matchSizes_zero u v⟩ (matchSizes_bddAbove u v)

theorem le_maxMatch {u : Fin p → ℝ} {v : Fin q → ℝ} {m : ℕ} (h : m ∈ matchSizes u v) :
    m ≤ maxMatch u v :=
  le_csSup (matchSizes_bddAbove u v) h

/-- Matching sizes are downward closed: drop the last matched pair. -/
theorem matchSizes_mono {u : Fin p → ℝ} {v : Fin q → ℝ} {m n : ℕ} (hmn : m ≤ n)
    (h : n ∈ matchSizes u v) : m ∈ matchSizes u v := by
  obtain ⟨f, g, hf, hg, hthr⟩ := h
  refine ⟨fun t => f (Fin.castLE hmn t), fun t => g (Fin.castLE hmn t), ?_, ?_, ?_⟩
  · exact fun a b hab => Fin.castLE_injective hmn (hf hab)
  · exact fun a b hab => Fin.castLE_injective hmn (hg hab)
  · exact fun t => hthr _

/-- The greedy test: the `m` largest entries of `u` matched in reverse against the `m` largest
entries of `v`. -/
def GreedyFeasible (u : Fin p → ℝ) (v : Fin q → ℝ) (m : ℕ) (hp : m ≤ p) (hq : m ≤ q) : Prop :=
  ∀ t : Fin m, 0 ≤ u (Fin.castLE hp t) + v (Fin.castLE hq t.rev)

/-- **The greedy test is sufficient**: the reversed pairing of the top `m` entries is a matching. -/
theorem greedy_mem_matchSizes (u : Fin p → ℝ) (v : Fin q → ℝ) (m : ℕ) (hp : m ≤ p) (hq : m ≤ q)
    (h : GreedyFeasible u v m hp hq) : m ∈ matchSizes u v := by
  refine ⟨fun t => Fin.castLE hp t, fun t => Fin.castLE hq t.rev, Fin.castLE_injective hp, ?_, h⟩
  intro a b hab
  have := Fin.castLE_injective hq hab
  exact Fin.rev_injective this

/-- A finite set of positions of size `c` reaches at least position `c − 1`. -/
private theorem exists_ge_card_sub_one {n : ℕ} (S : Finset (Fin n)) (hS : S.Nonempty) :
    ∃ x ∈ S, S.card - 1 ≤ (x : ℕ) := by
  by_contra hcon
  push_neg at hcon
  have hsub : S.image (Fin.val) ⊆ Finset.range (S.card - 1) := by
    intro y hy
    rw [Finset.mem_image] at hy
    obtain ⟨x, hx, rfl⟩ := hy
    exact Finset.mem_range.mpr (hcon x hx)
  have h1 : S.card ≤ S.card - 1 := by
    calc S.card = (S.image (Fin.val)).card :=
          (Finset.card_image_of_injective S Fin.val_injective).symm
      _ ≤ (Finset.range (S.card - 1)).card := Finset.card_le_card hsub
      _ = S.card - 1 := by simp
  have h2 : 0 < S.card := Finset.card_pos.mpr hS
  omega

/-- **The greedy test is necessary**: if the threshold graph has a matching of size `m` and both
score lists are sorted decreasingly, then the reversed pairing of the top `m` entries already
works.  So greedy is optimal. -/
theorem greedy_of_mem_matchSizes {u : Fin p → ℝ} {v : Fin q → ℝ} (hu : Antitone u)
    (hv : Antitone v) {m : ℕ} (h : m ∈ matchSizes u v) :
    GreedyFeasible u v m (le_of_mem_matchSizes h).1 (le_of_mem_matchSizes h).2 := by
  classical
  obtain ⟨hp, hq⟩ := le_of_mem_matchSizes h
  obtain ⟨f, g, hf, hg, hthr⟩ := h
  intro t
  -- the matched positions of `u` that are at least as far down the list as `t`
  set A : Finset (Fin m) := univ.filter (fun j => (t : ℕ) ≤ (f j : ℕ)) with hAdef
  have hBcard : (univ.filter (fun j : Fin m => (f j : ℕ) < (t : ℕ))).card ≤ (t : ℕ) := by
    have hmaps : Set.MapsTo (fun j : Fin m => (f j : ℕ))
        ↑(univ.filter (fun j : Fin m => (f j : ℕ) < (t : ℕ))) ↑(Finset.range (t : ℕ)) := by
      intro j hj
      simp only [Finset.coe_filter, Set.mem_setOf_eq] at hj
      simpa using hj.2
    have hinj : Set.InjOn (fun j : Fin m => (f j : ℕ))
        ↑(univ.filter (fun j : Fin m => (f j : ℕ) < (t : ℕ))) := by
      intro a _ b _ hab
      exact hf (Fin.val_injective hab)
    have := Finset.card_le_card_of_injOn (fun j : Fin m => (f j : ℕ)) hmaps hinj
    simpa using this
  have hAcard : m - (t : ℕ) ≤ A.card := by
    have hsplit := Finset.card_filter_add_card_filter_not
      (s := (univ : Finset (Fin m))) (p := fun j : Fin m => (t : ℕ) ≤ (f j : ℕ))
    have hneg : (univ.filter (fun j : Fin m => ¬ ((t : ℕ) ≤ (f j : ℕ)))).card
        = (univ.filter (fun j : Fin m => (f j : ℕ) < (t : ℕ))).card := by
      congr 1
      apply Finset.filter_congr
      intro j _
      simp
    rw [hneg] at hsplit
    simp only [Finset.card_univ, Fintype.card_fin] at hsplit
    rw [hAdef]
    omega
  -- so the matched partners on the `v` side reach at least position `m − 1 − t`
  have hAne : A.Nonempty := by
    rw [← Finset.card_pos]
    have := t.isLt
    omega
  set S : Finset (Fin q) := A.image g with hSdef
  have hScard : S.card = A.card := Finset.card_image_of_injective A hg
  have hSne : S.Nonempty := hAne.image g
  obtain ⟨x, hxS, hxge⟩ := exists_ge_card_sub_one S hSne
  rw [hSdef, Finset.mem_image] at hxS
  obtain ⟨j, hjA, rfl⟩ := hxS
  have hjt : (t : ℕ) ≤ (f j : ℕ) := by
    rw [hAdef, Finset.mem_filter] at hjA
    exact hjA.2
  have hrev : ((Fin.castLE hq t.rev : Fin q) : ℕ) ≤ ((g j : Fin q) : ℕ) := by
    have h1 : (t.rev : ℕ) = m - 1 - (t : ℕ) := by
      rw [Fin.val_rev]; omega
    have h2 : S.card - 1 ≤ ((g j : Fin q) : ℕ) := by rw [hScard] at hxge ⊢; omega
    have h3 : m - (t : ℕ) ≤ S.card := by rw [hScard]; exact hAcard
    simp only [Fin.val_castLE, h1]
    omega
  have hu' : u (f j) ≤ u (Fin.castLE hp t) := hu (Fin.le_def.mpr (by simpa using hjt))
  have hv' : v (g j) ≤ v (Fin.castLE hq t.rev) := hv (Fin.le_def.mpr hrev)
  have := hthr j
  linarith

/-- **Greedy computes the maximum matching.**  For decreasingly sorted lists, a matching of size
`m` exists exactly when the reversed pairing of the top `m` entries passes the threshold test. -/
theorem le_maxMatch_iff_greedyFeasible {u : Fin p → ℝ} {v : Fin q → ℝ} (hu : Antitone u)
    (hv : Antitone v) (m : ℕ) :
    m ≤ maxMatch u v ↔ ∃ hp : m ≤ p, ∃ hq : m ≤ q, GreedyFeasible u v m hp hq := by
  constructor
  · intro hm
    have hmem : m ∈ matchSizes u v := matchSizes_mono hm (maxMatch_mem u v)
    exact ⟨(le_of_mem_matchSizes hmem).1, (le_of_mem_matchSizes hmem).2,
      greedy_of_mem_matchSizes hu hv hmem⟩
  · rintro ⟨hp, hq, h⟩
    exact le_maxMatch (greedy_mem_matchSizes u v m hp hq h)

/-- The greedy test is monotone in `m`, so the largest `m` passing it is found by one scan. -/
theorem greedyFeasible_of_succ {u : Fin p → ℝ} {v : Fin q → ℝ} (hu : Antitone u) (hv : Antitone v)
    (m : ℕ) (hp : m + 1 ≤ p) (hq : m + 1 ≤ q) (h : GreedyFeasible u v (m + 1) hp hq) :
    GreedyFeasible u v m (by omega) (by omega) := by
  have hmem : m + 1 ∈ matchSizes u v := greedy_mem_matchSizes u v (m + 1) hp hq h
  exact greedy_of_mem_matchSizes hu hv (matchSizes_mono (Nat.le_succ m) hmem)

end IDR.ThresholdMatching
