/-
# Part IV.2  Rate--distortion: how many structures, in the *geometric* metric

`RequestProject.Metric` sizes a model by the population-space `ℓ¹` error, and
`RequestProject.Transport` shows that `ℓ¹` is blind to structural similarity: a model whose
structures are all within `0.1 Å` of the truth can still be maximally wrong in `ℓ¹`.  That
leaves the real design question open -- *how many structures does a model need in order to
be geometrically close to the target?*  This file answers it.

The setting is a structural dissimilarity `c` that is a pseudometric (`StructDist`: RMSD,
a contact-map distance, ...) and a target ensemble that is **`Δ`-separated**: it populates
`m` conformations that are pairwise at least `Δ` apart -- distinct conformational
substates, not a cloud of near-copies.

* `card_near_le` -- the covering step: each structure the model carries can be within
  `Δ/2` of at most one substate, because the target's substates are `Δ` apart.
* `transportCost_ge_of_card_le` -- **the distortion lower bound.**  A model of at most `k`
  structures has transport cost at least `(Δ/2)·(m-k)/m` against the target.  Unlike the
  `ℓ¹` bounds this is a statement about *geometry*: it cannot be cheated by moving the
  predicted structures slightly.
* `quantization_capacity` -- **the rate--distortion law.**  Achieving transport distortion
  `D` forces `k ≥ m·(1 - 2D/Δ)`: to halve the distortion you must (essentially) double the
  number of structures, until `D` reaches `Δ/2`, the radius at which the substates become
  indistinguishable.
* `bit_rate_lower_bound` -- in information units: a model that resolves the substates at
  distortion `D` must store at least `log m + log (1 - 2D/Δ)` nats.  Combined with
  `RequestProject.Chain`, where `m = 2^N`, this is a bit rate **linear in the length of the
  disordered region**.
* `discreteDist_structDist`, `discreteDist_separated` -- the degenerate dissimilarity that
  recovers the earlier counting bounds as a special case, so the two families of laws are
  one family.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.ModelNature
import RequestProject.Metric
import RequestProject.Transport

namespace IDR

open Finset
open scoped Classical

variable {X : Type*}

/-- A structural dissimilarity that behaves like a distance: nonnegative, symmetric and
subadditive.  RMSD after optimal superposition, a contact-map distance or any metric on
conformation space qualifies. -/
structure StructDist (c : X → X → ℝ) : Prop where
  nonneg : ∀ x y, 0 ≤ c x y
  symm : ∀ x y, c x y = c y x
  triangle : ∀ x y z, c x z ≤ c x y + c y z

/-- The target populates `m` conformational substates that are pairwise at least `Delta`
apart: genuinely different structures, not a cloud of near-copies. -/
def Separated (c : X → X → ℝ) {m : ℕ} (g : Fin m → X) (Delta : ℝ) : Prop :=
  ∀ l l' : Fin m, l ≠ l' → Delta ≤ c (g l) (g l')

/-- **The covering step.**  Each structure a model carries lies within `Δ/2` of at most one
of the target's substates, so a model of `k` structures can cover at most `k` substates. -/
theorem card_near_le {m : ℕ} {c : X → X → ℝ} (hc : StructDist c) {g : Fin m → X}
    {Delta : ℝ} (hsep : Separated c g Delta) (M : Ens X) :
    (Finset.univ.filter
        (fun l : Fin m => ∃ i : Fin M.card, c (M.pt i) (g l) < Delta / 2)).card ≤ M.card := by
  classical
  set Near : Finset (Fin m) :=
    Finset.univ.filter (fun l : Fin m => ∃ i : Fin M.card, c (M.pt i) (g l) < Delta / 2)
    with hNear
  have hchoice : ∀ l ∈ Near, ∃ i : Fin M.card, c (M.pt i) (g l) < Delta / 2 := by
    intro l hl
    exact (Finset.mem_filter.1 hl).2
  rcases Finset.eq_empty_or_nonempty Near with h | ⟨l₀, hl₀⟩
  · simp [h]
  obtain ⟨i₀, -⟩ := hchoice l₀ hl₀
  haveI : Nonempty (Fin M.card) := ⟨i₀⟩
  choose! phi hphi using hchoice
  have hinj : Set.InjOn phi Near := by
    intro a ha b hb hab
    by_contra hne
    have h1 : c (M.pt (phi a)) (g a) < Delta / 2 := hphi a ha
    have h2 : c (M.pt (phi b)) (g b) < Delta / 2 := hphi b hb
    rw [hab] at h1
    have h3 : Delta ≤ c (g a) (g b) := hsep a b hne
    have h4 : c (g a) (g b) ≤ c (g a) (M.pt (phi b)) + c (M.pt (phi b)) (g b) :=
      hc.triangle _ _ _
    rw [hc.symm (g a) (M.pt (phi b))] at h4
    linarith
  have := Finset.card_le_card_of_injOn (t := (Finset.univ : Finset (Fin M.card))) phi
    (fun x _ => by simp) hinj
  simpa using this

/-- **The geometric capacity bound.**  Against a `Δ`-separated target populating `m`
substates uniformly, every model of at most `k` structures pays transport cost at least
`(Δ/2)·(m-k)/m`.  Because the cost is measured by the structural dissimilarity itself, the
bound survives the objection that `ℓ¹` ignores geometry. -/
theorem transportCost_ge_of_card_le {m k : ℕ} (hm : 0 < m) {c : X → X → ℝ}
    (hc : StructDist c) {g : Fin m → X} {Delta : ℝ} (hDelta : 0 ≤ Delta)
    (hsep : Separated c g Delta) {M : Ens X} (hM : M.card ≤ k) :
    Delta / 2 * (((m : ℝ) - k) / m) ≤ transportCost c M (unif hm g) := by
  classical
  set Near : Finset (Fin m) :=
    Finset.univ.filter (fun l : Fin m => ∃ i : Fin M.card, c (M.pt i) (g l) < Delta / 2)
    with hNear
  set Far : Finset (Fin m) :=
    Finset.univ.filter (fun l : Fin m => ¬ ∃ i : Fin M.card, c (M.pt i) (g l) < Delta / 2)
    with hFar
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hNearcard : Near.card ≤ k := le_trans (card_near_le hc hsep M) hM
  have hFarcard : (m : ℝ) - k ≤ (Far.card : ℝ) := by
    have hcards : Near.card + Far.card = m := by
      rw [hNear, hFar, Finset.card_filter_add_card_filter_not]
      simp
    have : (m : ℕ) ≤ Far.card + k := by omega
    have := (Nat.cast_le (α := ℝ)).2 this
    push_cast at this
    linarith
  refine le_csInf (transportCost_set_nonempty M (unif hm g) c) ?_
  rintro r ⟨gam, ⟨hnn, -, hcol⟩, rfl⟩
  have hfar_far : ∀ l ∈ Far, ∀ i : Fin M.card, Delta / 2 ≤ c (M.pt i) (g l) := by
    intro l hl i
    have h := (Finset.mem_filter.1 hl).2
    push_neg at h
    exact h i
  have hcost : Delta / 2 * ((Far.card : ℝ) / m) ≤ planCost M (unif hm g) c gam := by
    have hstep : ∑ i, ∑ l ∈ Far, gam i l * (Delta / 2)
        ≤ ∑ i, ∑ l, gam i l * c (M.pt i) ((unif hm g).pt l) := by
      refine Finset.sum_le_sum fun i _ => ?_
      calc ∑ l ∈ Far, gam i l * (Delta / 2)
          ≤ ∑ l ∈ Far, gam i l * c (M.pt i) (g l) := by
            refine Finset.sum_le_sum fun l hl => ?_
            exact mul_le_mul_of_nonneg_left (hfar_far l hl i) (hnn i l)
        _ ≤ ∑ l, gam i l * c (M.pt i) ((unif hm g).pt l) := by
            refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ Far) ?_
            intro l _ _
            exact mul_nonneg (hnn i l) (hc.nonneg _ _)
    have hswap : ∑ i, ∑ l ∈ Far, gam i l * (Delta / 2)
        = Delta / 2 * ∑ l ∈ Far, ∑ i, gam i l := by
      rw [Finset.sum_comm, Finset.mul_sum]
      refine Finset.sum_congr rfl fun l _ => ?_
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => by ring
    have hcolsum : ∑ l ∈ Far, ∑ i, gam i l = (Far.card : ℝ) / m := by
      have : ∀ l ∈ Far, ∑ i, gam i l = 1 / (m : ℝ) := by
        intro l _
        rw [hcol l]
        rfl
      rw [Finset.sum_congr rfl this, Finset.sum_const, nsmul_eq_mul]
      field_simp
    rw [hswap, hcolsum] at hstep
    exact hstep
  refine le_trans ?_ hcost
  have hmono : ((m : ℝ) - k) / m ≤ (Far.card : ℝ) / m := by
    gcongr
  exact mul_le_mul_of_nonneg_left hmono (by linarith)

/-- **The rate--distortion law for disorder models.**  To reach transport distortion `D`
against a `Δ`-separated target with `m` substates, a model must carry at least
`m·(1 - 2D/Δ)` structures.  Distortion is bought with capacity at a fixed exchange rate,
and no capacity at all suffices below `D = Δ/2`, where the substates cease to be
resolved. -/
theorem quantization_capacity {m k : ℕ} (hm : 0 < m) {c : X → X → ℝ} (hc : StructDist c)
    {g : Fin m → X} {Delta D : ℝ} (hDelta : 0 < Delta) (hsep : Separated c g Delta)
    {M : Ens X} (hM : M.card ≤ k) (hD : transportCost c M (unif hm g) ≤ D) :
    (m : ℝ) * (1 - 2 * D / Delta) ≤ k := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have h1 := transportCost_ge_of_card_le hm hc (le_of_lt hDelta) hsep hM
  have h2 : Delta / 2 * (((m : ℝ) - k) / m) ≤ D := le_trans h1 hD
  have h3 : ((m : ℝ) - k) / m ≤ 2 * D / Delta := by
    rw [le_div_iff₀ hDelta]
    nlinarith [h2]
  have h4 : (m : ℝ) - k ≤ 2 * D / Delta * m := by
    rw [div_le_iff₀ hm'] at h3
    linarith
  nlinarith [h4]

/-- **The bit rate of a disorder model.**  Storing enough to reach distortion `D` costs at
least `log m + log (1 - 2D/Δ)` nats.  For a region whose substates are exponentially many
in its length this rate is *linear in the length*: an ensemble model of a disordered region
is intrinsically a large object. -/
theorem bit_rate_lower_bound {m k : ℕ} (hm : 0 < m) {c : X → X → ℝ} (hc : StructDist c)
    {g : Fin m → X} {Delta D : ℝ} (hDelta : 0 < Delta) (hsep : Separated c g Delta)
    {M : Ens X} (hM : M.card ≤ k) (hD : transportCost c M (unif hm g) ≤ D)
    (hpos : 0 < 1 - 2 * D / Delta) :
    Real.log m + Real.log (1 - 2 * D / Delta) ≤ Real.log k := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hk := quantization_capacity hm hc hDelta hsep hM hD
  have hprod : (0 : ℝ) < (m : ℝ) * (1 - 2 * D / Delta) := by positivity
  calc Real.log m + Real.log (1 - 2 * D / Delta)
      = Real.log ((m : ℝ) * (1 - 2 * D / Delta)) := by
        rw [Real.log_mul (ne_of_gt hm') (ne_of_gt hpos)]
    _ ≤ Real.log k := Real.log_le_log hprod hk

/-! ## Achievability: a covering of the populated region is a model -/

/-- **The upper half of the rate--distortion law.**  A covering of the populated region by
`k` structures -- an `eps`-net `g`, with `r` assigning each conformation a nearby net point
-- *is* a model of `k` structures within transport distortion `eps` of the target, obtained
by collapsing each conformation onto its representative.  Together with
`quantization_capacity` this pins the number of structures a model of a disordered region
needs: it is the covering number of the populated region at the resolution demanded, up to
the factor two between the two bounds. -/
theorem exists_model_of_covering {k : ℕ} {c : X → X → ℝ} (hc : StructDist c) (E : Ens X)
    (g : Fin k → X) (r : X → Fin k) {eps : ℝ}
    (hcov : ∀ j : Fin E.card, c (g (r (E.pt j))) (E.pt j) ≤ eps) :
    ∃ M : Ens X, M.card = k ∧ (∀ i, ∃ b : Fin k, M.pt i = g b) ∧
      transportCost c M E ≤ eps := by
  classical
  have hwnn : ∀ b : Fin k, 0 ≤ ∑ j ∈ Finset.univ.filter (fun j => r (E.pt j) = b), E.w j :=
    fun b => Finset.sum_nonneg fun j _ => E.w_nonneg j
  have hwsum : ∑ b : Fin k, ∑ j ∈ Finset.univ.filter (fun j => r (E.pt j) = b), E.w j = 1 :=
    (Finset.sum_fiberwise Finset.univ (fun j => r (E.pt j)) E.w).trans E.w_sum
  refine ⟨⟨k, g, fun b => ∑ j ∈ Finset.univ.filter (fun j => r (E.pt j) = b), E.w j,
    hwnn, hwsum⟩, rfl, fun i => ⟨i, rfl⟩, ?_⟩
  have hcoupling : IsCoupling
      (⟨k, g, fun b => ∑ j ∈ Finset.univ.filter (fun j => r (E.pt j) = b), E.w j,
        hwnn, hwsum⟩ : Ens X) E (fun b j => if r (E.pt j) = b then E.w j else 0) := by
    refine ⟨fun b j => ?_, fun b => ?_, fun j => ?_⟩
    · by_cases h : r (E.pt j) = b <;> simp [h, E.w_nonneg j]
    · show ∑ j, (if r (E.pt j) = b then E.w j else 0)
        = ∑ j ∈ Finset.univ.filter (fun j => r (E.pt j) = b), E.w j
      rw [Finset.sum_filter]
    · simp
  refine le_trans (transportCost_le_of_coupling hc.nonneg hcoupling) ?_
  have hswap : planCost
      (⟨k, g, fun b => ∑ j ∈ Finset.univ.filter (fun j => r (E.pt j) = b), E.w j,
        hwnn, hwsum⟩ : Ens X) E c (fun b j => if r (E.pt j) = b then E.w j else 0)
      = ∑ j, E.w j * c (g (r (E.pt j))) (E.pt j) := by
    rw [planCost, Finset.sum_comm]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [Finset.sum_eq_single (r (E.pt j))]
    · simp
    · intro b _ hb
      simp [Ne.symm hb]
    · intro hcon
      exact absurd (Finset.mem_univ _) hcon
  rw [hswap]
  calc ∑ j, E.w j * c (g (r (E.pt j))) (E.pt j)
      ≤ ∑ j, E.w j * eps :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hcov j) (E.w_nonneg j)
    _ = eps := by rw [← Finset.sum_mul, E.w_sum, one_mul]

/-! ## The counting bounds are the special case of the discrete dissimilarity -/

/-- The crudest structural dissimilarity: distinct conformations are at distance one. -/
noncomputable def discreteDist (X : Type*) : X → X → ℝ :=
  fun x y => if x = y then 0 else 1

theorem discreteDist_structDist : StructDist (discreteDist X) := by
  classical
  refine ⟨fun x y => ?_, fun x y => ?_, fun x y z => ?_⟩
  · simp only [discreteDist]; split <;> norm_num
  · simp only [discreteDist]
    by_cases h : x = y
    · simp [h]
    · simp [h, Ne.symm h]
  · have h1 : (0 : ℝ) ≤ (if x = y then 0 else 1) := by
      rcases eq_or_ne x y with h | h <;> simp [h]
    have h2 : (0 : ℝ) ≤ (if y = z then 0 else 1) := by
      rcases eq_or_ne y z with h | h <;> simp [h]
    by_cases hxz : x = z
    · simp only [discreteDist, if_pos hxz]
      linarith
    · simp only [discreteDist, if_neg hxz]
      by_cases hxy : x = y
      · subst hxy
        simp [hxz]
      · simp only [if_neg hxy]
        simp only [if_neg hxy] at h1
        linarith

theorem discreteDist_separated {m : ℕ} {g : Fin m → X} (hg : Function.Injective g) :
    Separated (discreteDist X) g 1 := by
  intro l l' hll
  simp only [discreteDist]
  rw [if_neg (fun h => hll (hg h))]

end IDR
