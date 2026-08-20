/-
# Part IV.5  How many latent states?  Coupling is nonnegative rank

`RequestProject.EnergyModels` shows that a factorised (one-body, mean-field, two-tower)
model cannot carry correlation between two segments of a disordered region, and that every
ensemble is a *mixture* of factorised ones -- so a latent variable is what carries
correlations.  That leaves the quantitative question: **how many latent states does a given
coupling need?**

This file answers it.  The population table of a two-segment ensemble is a nonnegative
matrix, a mixture of `k` product ensembles is exactly a decomposition of that matrix into
`k` nonnegative rank-one terms, and the number of latent states needed is therefore the
*nonnegative rank* of the population table.

* `MixtureOfProducts k M` -- `M` is a mixture of `k` factorised ensembles: `k` latent
  states, each decoupling the two segments.  `k = 1` is the mean-field/two-tower model.
* `latent_dim_lower_bound` -- **the rank bound.**  If the target has `m` mutually exclusive
  coupled substates -- segment A is in state `i` exactly when segment B is in state `i` --
  then every mixture-of-products model needs `k ≥ m` latent states.  Correlation is not a
  small correction to be added to a factorised model; its cost in latent capacity is the
  number of coupled substates.
* `diagEns_mixture` -- and `m` states suffice, so the bound is exact.
* `latent_rank_theorem` -- the two halves together, for the canonical `m`-substate coupled
  ensemble: it is a mixture of `m` products and of no fewer.  For `m = 2` this recovers
  (and for general `m` strengthens) the earlier statement that no factorised model captures
  a coupled pair.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.ModelNature
import RequestProject.EnergyModels
import RequestProject.Metric

namespace IDR

open Finset
open scoped Classical

variable {X Y : Type*}

/-- `M` is a mixture of `k` product ensembles: a latent-variable model with `k` states, each
of which decouples the two segments.  `k = 1` is exactly a factorised (mean-field,
two-tower) model. -/
def MixtureOfProducts (k : ℕ) (M : Ens (X × Y)) : Prop :=
  ∃ (a : Fin k → ℝ) (u : Fin k → X → ℝ) (v : Fin k → Y → ℝ),
    (∀ c, 0 ≤ a c) ∧ (∀ c x, 0 ≤ u c x) ∧ (∀ c y, 0 ≤ v c y) ∧
      ∀ x y, M.prob (x, y) = ∑ c, a c * (u c x * v c y)

/-- **The latent-capacity cost of coupling.**  Suppose the target ensemble couples the two
segments through `m` mutually exclusive substates: conformation `xs i` of the first segment
occurs exactly with conformation `ys i` of the second.  Then any model built as a mixture
of product ensembles needs at least `m` latent states.  Correlation between parts of a
disordered region is paid for in latent capacity, one state per coupled substate. -/
theorem latent_dim_lower_bound {m k : ℕ} {M : Ens (X × Y)} (h : MixtureOfProducts k M)
    {xs : Fin m → X} {ys : Fin m → Y}
    (hdiag : ∀ i, 0 < M.prob (xs i, ys i))
    (hoff : ∀ i j, i ≠ j → M.prob (xs i, ys j) = 0) : m ≤ k := by
  classical
  obtain ⟨a, u, v, ha, hu, hv, hM⟩ := h
  -- every populated coupled substate is carried by some latent state
  have hpick : ∀ i : Fin m, ∃ c : Fin k, 0 < a c * (u c (xs i) * v c (ys i)) := by
    intro i
    by_contra hcon
    push_neg at hcon
    have hzero : ∑ c, a c * (u c (xs i) * v c (ys i)) = 0 := by
      refine le_antisymm ?_ ?_
      · exact Finset.sum_nonpos fun c _ => hcon c
      · exact Finset.sum_nonneg fun c _ =>
          mul_nonneg (ha c) (mul_nonneg (hu c _) (hv c _))
    have := hdiag i
    rw [hM (xs i) (ys i), hzero] at this
    exact lt_irrefl 0 this
  choose phi hphi using hpick
  have hinj : Function.Injective phi := by
    intro i j hij
    by_contra hne
    -- the latent state `phi i` populates `xs i` on the left and `ys j` on the right,
    -- hence the forbidden combination `(xs i, ys j)`
    have hi := hphi i
    have hj := hphi j
    rw [← hij] at hj
    have hapos : 0 < a (phi i) := by
      rcases eq_or_lt_of_le (ha (phi i)) with h0 | hpos
      · rw [← h0] at hi; simp at hi
      · exact hpos
    have hui : 0 < u (phi i) (xs i) := by
      rcases eq_or_lt_of_le (hu (phi i) (xs i)) with h0 | hpos
      · rw [← h0] at hi; simp at hi
      · exact hpos
    have hvj : 0 < v (phi i) (ys j) := by
      rcases eq_or_lt_of_le (hv (phi i) (ys j)) with h0 | hpos
      · rw [← h0] at hj; simp at hj
      · exact hpos
    have hterm : 0 < a (phi i) * (u (phi i) (xs i) * v (phi i) (ys j)) := by positivity
    have hsum : 0 < ∑ c, a c * (u c (xs i) * v c (ys j)) :=
      Finset.sum_pos' (fun c _ => mul_nonneg (ha c) (mul_nonneg (hu c _) (hv c _)))
        ⟨phi i, Finset.mem_univ _, hterm⟩
    rw [← hM (xs i) (ys j), hoff i j hne] at hsum
    exact lt_irrefl 0 hsum
  simpa using Fintype.card_le_of_injective phi hinj

/-! ## The bound is attained -/

/-- The canonical coupled ensemble: `m` mutually exclusive substates, the two segments
perfectly correlated. -/
noncomputable def diagEns {m : ℕ} (hm : 0 < m) (xs : Fin m → X) (ys : Fin m → Y) :
    Ens (X × Y) :=
  unif hm (fun i => (xs i, ys i))

lemma diagEns_prob_diag {m : ℕ} (hm : 0 < m) {xs : Fin m → X} {ys : Fin m → Y}
    (hx : Function.Injective xs) (i : Fin m) :
    (diagEns hm xs ys).prob (xs i, ys i) = 1 / m := by
  have hinj : Function.Injective (fun i => (xs i, ys i)) := by
    intro i j hij
    exact hx (congrArg Prod.fst hij)
  exact unif_prob_eq hm hinj i

lemma diagEns_prob_off {m : ℕ} (hm : 0 < m) {xs : Fin m → X} {ys : Fin m → Y}
    (hx : Function.Injective xs) (hy : Function.Injective ys) {i j : Fin m} (hij : i ≠ j) :
    (diagEns hm xs ys).prob (xs i, ys j) = 0 := by
  refine unif_prob_not_mem hm ?_
  intro l hl
  have h1 : xs l = xs i := congrArg Prod.fst hl
  have h2 : ys l = ys j := congrArg Prod.snd hl
  exact hij ((hx h1).symm.trans (hy h2))

/-- `m` latent states suffice for the `m`-substate coupled ensemble: one latent state per
substate, each of which decouples the two segments. -/
theorem diagEns_mixture {m : ℕ} (hm : 0 < m) {xs : Fin m → X} {ys : Fin m → Y}
    (hx : Function.Injective xs) :
    MixtureOfProducts m (diagEns hm xs ys) := by
  classical
  refine ⟨fun _ => 1 / m, fun c x => if x = xs c then 1 else 0,
    fun c y => if y = ys c then 1 else 0, fun c => by positivity,
    fun c x => by dsimp only; split <;> norm_num,
    fun c y => by dsimp only; split <;> norm_num, ?_⟩
  intro x y
  by_cases hx' : ∃ i, x = xs i
  · obtain ⟨i, rfl⟩ := hx'
    by_cases hy' : y = ys i
    · subst hy'
      rw [diagEns_prob_diag hm hx i]
      rw [Finset.sum_eq_single i]
      · simp
      · intro c _ hc
        have : xs i ≠ xs c := fun hcon => hc (hx hcon).symm
        simp [this]
      · intro hcon; exact absurd (Finset.mem_univ i) hcon
    · -- an unpopulated pair: both sides vanish
      have hne : ∀ c : Fin m, ¬ (xs i = xs c ∧ y = ys c) := by
        rintro c ⟨h1, h2⟩
        exact hy' (by rw [h2, hx h1])
      have hsum : ∑ c, (1 / (m : ℝ)) * ((if xs i = xs c then (1 : ℝ) else 0)
          * (if y = ys c then (1 : ℝ) else 0)) = 0 := by
        refine Finset.sum_eq_zero fun c _ => ?_
        rcases (not_and_or.1 (hne c)) with h | h <;> simp [h]
      rw [hsum]
      refine unif_prob_not_mem hm ?_
      intro l hl
      have h1 : xs l = xs i := congrArg Prod.fst hl
      have h2 : ys l = y := congrArg Prod.snd hl
      exact hy' (by rw [← h2, hx h1])
  · push_neg at hx'
    have hsum : ∑ c, (1 / (m : ℝ)) * ((if x = xs c then (1 : ℝ) else 0)
        * (if y = ys c then (1 : ℝ) else 0)) = 0 := by
      refine Finset.sum_eq_zero fun c _ => ?_
      simp [hx' c]
    rw [hsum]
    refine unif_prob_not_mem hm ?_
    intro l hl
    exact hx' l (congrArg Prod.fst hl).symm

/-- **The latent-rank theorem for coupled segments.**  The `m`-substate coupled ensemble is
a mixture of `m` product ensembles and of no fewer: the latent capacity a model must devote
to inter-segment coupling is exactly the number of coupled substates.  In particular
(`k = 1`) no factorised model represents it at all, and a model with a small latent space
cannot represent a disordered region whose parts are coupled through many substates. -/
theorem latent_rank_theorem {m : ℕ} (hm : 0 < m)
    {xs : Fin m → X} {ys : Fin m → Y} (hx : Function.Injective xs)
    (hy : Function.Injective ys) :
    MixtureOfProducts m (diagEns hm xs ys) ∧
      ∀ k : ℕ, MixtureOfProducts k (diagEns hm xs ys) → m ≤ k := by
  refine ⟨diagEns_mixture hm hx, fun k hk => ?_⟩
  refine latent_dim_lower_bound hk (xs := xs) (ys := ys) (fun i => ?_) (fun i j hij => ?_)
  · rw [diagEns_prob_diag hm hx i]
    have : (0 : ℝ) < m := by exact_mod_cast hm
    positivity
  · exact diagEns_prob_off hm hx hy hij

end IDR
