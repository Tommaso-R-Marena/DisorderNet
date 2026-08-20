/-
# Part CXLII  How many salt conditions? A finite titration ladder identifies a finite ensemble

Part CXLI showed that the object a model of a charged disordered region must predict is a
*distribution* of internal distances per sequence separation, and that a *complete* titration —
agreement of the curves at every ionic strength on a half-line — determines that distribution,
support and all.  A complete titration is not an experiment.  A real titration is a finite ladder
of salt conditions, typically prepared by serial dilution, and the design question is the one an
experimenter actually asks: **how many conditions do I need, and where do I put them?**

This part answers it, in both directions, for the ensemble kernel
`measCurve T w κ = ∑_{t ∈ T} w t · e^{−κt}/t` of Part CXLI.

* **Positive.**  `ladder_identifies_distribution` — an arithmetic ladder of `n` ionic strengths
  `κ_j = κ₀ + j·h` (`h > 0`, any `κ₀`, any spacing) determines every weight of a distance
  distribution supported on at most `n` distances.  Nothing about the ladder matters except that
  the conditions are equally spaced and there are enough of them; the reachable range of the
  titration is irrelevant.  `ensemble_ladder_design` states the practical corollary: to separate
  two ensembles of at most `m` conformers each, `2m` equally spaced conditions always suffice.
  The mechanism is that at equally spaced conditions the screening factors become *powers*,
  `e^{−κ_j t} = e^{−κ₀t}·(e^{−ht})^j`, so the design matrix is a Vandermonde matrix in the
  distinct nodes `e^{−ht}`, and Lagrange interpolation inverts it exactly.

* **Negative, and sharp.**  `finite_conditions_insufficient` — for *any* finite list of `n` ionic
  strengths, chosen however one likes, and any `n+2` prescribed distinct positive distances, there
  are two different strictly positive probability distributions on those distances whose readings
  agree at every one of the `n` conditions.  So no finite titration identifies an ensemble of
  unbounded complexity, and the count in the positive theorem cannot be improved by more than a
  constant: `n` conditions never resolve `n+2` conformers.  The mechanism is dimensional — `n`
  readings plus the normalisation are `n+1` linear constraints on `n+2` weights — and the
  perturbation is taken small enough to keep every weight positive, so the two ensembles are
  physically legitimate, not formal.

* **Exact identification is not resolution.**  `two_distance_resolution_horizon` — for two
  distances `r₁ ≠ r₂` and any tolerance `δ > 0` there are two strictly positive weightings whose
  readings at both ladder conditions agree to within `δ`, while their weights at `r₁` differ by
  exactly `δ·r₁·e^{κ₀r₁}/|e^{−hr₁} − e^{−hr₂}|`.  The amplification factor is the reciprocal of
  the gap between the Vandermonde nodes, and `resolution_horizon_unbounded` shows it exceeds any
  prescribed bound once the two distances are close enough.  So the ladder inverts exactly in the
  noiseless limit and arbitrarily badly at any fixed precision: the counting theorem above is a
  statement about *identifiability*, not about *resolvability*, and a model claiming two conformers
  at nearby distances must quote the measurement precision to be falsifiable at all.

* `titration_ladder_law` collects the three.

Design consequence.  A titration used to calibrate an ensemble model should report the number of
salt conditions together with the number of conformers the model claims to resolve; the first must
be at least the second, whatever the ionic-strength range, and if the conditions are equally spaced
that necessary count is also sufficient.  It should also report the precision of the readings
against the node gaps `|e^{−hr} − e^{−hr'}|` of the distances it claims to separate, since that
ratio, not the number of conditions, is what bounds the error on the recovered weights.
-/
import Mathlib
import RequestProject.DistanceEnsemble

set_option autoImplicit false

namespace IDR
namespace TitrationLadder

open Finset Polynomial

/-! ## 1. Vandermonde independence: powers with distinct nodes -/

/-- **Distinct nodes, finitely many powers.**  If a finite family of amplitudes indexed by a finite
set of reals annihilates the first `n` powers of an injective node function, and there are at most
`n` indices, then every amplitude vanishes.  This is the invertibility of a Vandermonde matrix,
proved by evaluating against Lagrange basis polynomials. -/
theorem power_amplitudes_zero {U : Finset ℝ} {B x : ℝ → ℝ} {n : ℕ}
    (hinj : Set.InjOn x U) (hcard : U.card ≤ n)
    (h : ∀ j < n, ∑ t ∈ U, B t * (x t) ^ j = 0) :
    ∀ t ∈ U, B t = 0 := by
  classical
  intro t0 ht0
  have hpoly : ∀ P : ℝ[X], P.natDegree < n → ∑ t ∈ U, B t * P.eval (x t) = 0 := by
    intro P hP
    have e1 : ∀ t ∈ U, B t * P.eval (x t)
        = ∑ j ∈ Finset.range n, P.coeff j * (B t * (x t) ^ j) := by
      intro t _
      rw [eval_eq_sum_range' hP, Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [Finset.sum_congr rfl e1, Finset.sum_comm]
    refine Finset.sum_eq_zero fun j hj => ?_
    rw [← Finset.mul_sum, h j (Finset.mem_range.1 hj), mul_zero]
  have hcard1 : 1 ≤ U.card := Finset.card_pos.2 ⟨t0, ht0⟩
  have hdeg : (Lagrange.basis U x t0).natDegree < n := by
    rw [Lagrange.natDegree_basis hinj ht0]; omega
  have hsum := hpoly _ hdeg
  rw [Finset.sum_eq_single t0] at hsum
  · rwa [Lagrange.eval_basis_self hinj ht0, mul_one] at hsum
  · intro b hb hne
    rw [Lagrange.eval_basis_of_ne (Ne.symm hne) hb, mul_zero]
  · intro hc; exact absurd ht0 hc

/-- The `j`-th condition of an arithmetic titration ladder with base `kappa0` and spacing
`hstep`. -/
def ladderPoint (kappa0 hstep : ℝ) (j : ℕ) : ℝ := kappa0 + j * hstep

/-- **Finitely many equally spaced conditions kill finitely many exponentials.**  If a sum of at
most `n` real exponentials, with distinct rates, vanishes at the `n` points of an arithmetic
ladder, every amplitude vanishes.  Equally spaced conditions turn the screening factors into powers
of `e^{−h t}`, and those nodes are distinct because the distances are. -/
theorem ladder_amplitudes_zero {U : Finset ℝ} {A : ℝ → ℝ} {kappa0 hstep : ℝ} {n : ℕ}
    (hstep0 : 0 < hstep) (hcard : U.card ≤ n)
    (h : ∀ j < n, ∑ t ∈ U, A t * Real.exp (-(ladderPoint kappa0 hstep j * t)) = 0) :
    ∀ t ∈ U, A t = 0 := by
  classical
  set B : ℝ → ℝ := fun t => A t * Real.exp (-(kappa0 * t)) with hB
  set x : ℝ → ℝ := fun t => Real.exp (-(hstep * t)) with hx
  have hinj : Set.InjOn x U := by
    intro a _ b _ hab
    have h1 : -(hstep * a) = -(hstep * b) := Real.exp_injective (by simpa [hx] using hab)
    have h2 : hstep * a = hstep * b := by linarith
    exact mul_left_cancel₀ (ne_of_gt hstep0) h2
  have hkey : ∀ j < n, ∑ t ∈ U, B t * (x t) ^ j = 0 := by
    intro j hj
    rw [← h j hj]
    refine Finset.sum_congr rfl fun t _ => ?_
    have hxj : (x t) ^ j = Real.exp (-(j * hstep * t)) := by
      rw [hx, ← Real.exp_nat_mul]; ring_nf
    rw [hB, hxj, mul_assoc, ← Real.exp_add, ladderPoint]
    ring_nf
  intro t ht
  have h0 := power_amplitudes_zero hinj hcard hkey t ht
  rcases mul_eq_zero.1 h0 with h1 | h2
  · exact h1
  · exact absurd h2 (Real.exp_ne_zero _)

/-! ## 2. A finite ladder identifies a finite ensemble -/

/-- **The design theorem.**  An arithmetic ladder of `n` ionic strengths determines the internal
distance distribution of an ensemble realising at most `n` distances in total: if two finitely
supported distributions of positive distances, with at most `n` distances between them, give the
same reading at each of the `n` equally spaced conditions, they have the same weight at every
distance.  No half-line of conditions and no limit is needed — the titration is finite. -/
theorem ladder_identifies_distribution {T T' : Finset ℝ} {w w' : ℝ → ℝ} {kappa0 hstep : ℝ} {n : ℕ}
    (hstep0 : 0 < hstep) (hT : ∀ t ∈ T, 0 < t) (hT' : ∀ t ∈ T', 0 < t)
    (hcard : (T ∪ T').card ≤ n)
    (h : ∀ j < n, DistanceEnsemble.measCurve T w (ladderPoint kappa0 hstep j)
      = DistanceEnsemble.measCurve T' w' (ladderPoint kappa0 hstep j)) :
    ∀ t : ℝ, DistanceEnsemble.ext T w t = DistanceEnsemble.ext T' w' t := by
  classical
  set U : Finset ℝ := T ∪ T' with hU
  have hUpos : ∀ t ∈ U, 0 < t := by
    intro t htU
    rcases Finset.mem_union.1 htU with ht | ht
    · exact hT t ht
    · exact hT' t ht
  have hcurve : ∀ (S : Finset ℝ) (v : ℝ → ℝ), S ⊆ U → ∀ kappa,
      DistanceEnsemble.measCurve S v kappa
        = ∑ t ∈ U, DistanceEnsemble.ext S v t * (Real.exp (-(kappa * t)) / t) := by
    intro S v hSU kappa
    rw [DistanceEnsemble.measCurve]
    have hsub : ∑ t ∈ S, DistanceEnsemble.ext S v t * (Real.exp (-(kappa * t)) / t)
        = ∑ t ∈ U, DistanceEnsemble.ext S v t * (Real.exp (-(kappa * t)) / t) := by
      refine Finset.sum_subset hSU ?_
      intro y _ hy
      simp [DistanceEnsemble.ext, hy]
    rw [← hsub]
    exact Finset.sum_congr rfl fun t htS => by simp [DistanceEnsemble.ext, htS]
  have hamp : ∀ j < n,
      ∑ t ∈ U, ((DistanceEnsemble.ext T w t - DistanceEnsemble.ext T' w' t) / t)
        * Real.exp (-(ladderPoint kappa0 hstep j * t)) = 0 := by
    intro j hj
    have h1 := hcurve T w Finset.subset_union_left (ladderPoint kappa0 hstep j)
    have h2 := hcurve T' w' Finset.subset_union_right (ladderPoint kappa0 hstep j)
    have h3 := h j hj
    rw [h1, h2] at h3
    have h4 : ∑ t ∈ U,
        (DistanceEnsemble.ext T w t * (Real.exp (-(ladderPoint kappa0 hstep j * t)) / t)
          - DistanceEnsemble.ext T' w' t
              * (Real.exp (-(ladderPoint kappa0 hstep j * t)) / t)) = 0 := by
      rw [Finset.sum_sub_distrib, h3, sub_self]
    rw [← h4]
    exact Finset.sum_congr rfl fun t _ => by ring
  have hzero := ladder_amplitudes_zero hstep0 hcard hamp
  intro t
  by_cases htU : t ∈ U
  · have ht0 : t ≠ 0 := ne_of_gt (hUpos t htU)
    have hz := hzero t htU
    rcases div_eq_zero_iff.1 hz with h1 | h2
    · linarith [sub_eq_zero.1 h1]
    · exact absurd h2 ht0
  · have h1 : t ∉ T := fun hc => htU (Finset.mem_union_left _ hc)
    have h2 : t ∉ T' := fun hc => htU (Finset.mem_union_right _ hc)
    simp [DistanceEnsemble.ext, h1, h2]

/-- **The experimental design rule.**  Two ensembles of at most `m` conformers each are separated
by an arithmetic ladder of `2m` ionic strengths: either they have the same distance distribution,
or one of the `2m` readings differs.  Any spacing works, and the range of the ladder is
irrelevant. -/
theorem ensemble_ladder_design {T T' : Finset ℝ} {w w' : ℝ → ℝ} {kappa0 hstep : ℝ} {m : ℕ}
    (hstep0 : 0 < hstep) (hT : ∀ t ∈ T, 0 < t) (hT' : ∀ t ∈ T', 0 < t)
    (hm : T.card ≤ m) (hm' : T'.card ≤ m)
    (hne : ∃ t : ℝ, DistanceEnsemble.ext T w t ≠ DistanceEnsemble.ext T' w' t) :
    ∃ j < 2 * m, DistanceEnsemble.measCurve T w (ladderPoint kappa0 hstep j)
      ≠ DistanceEnsemble.measCurve T' w' (ladderPoint kappa0 hstep j) := by
  classical
  by_contra hcon
  push_neg at hcon
  have hcard : (T ∪ T').card ≤ 2 * m := by
    have := Finset.card_union_le T T'
    omega
  obtain ⟨t, ht⟩ := hne
  exact ht (ladder_identifies_distribution hstep0 hT hT' hcard hcon t)

/-! ## 3. No finite titration identifies an ensemble of unbounded complexity -/

/-- **The dimensional obstruction.**  Fix any finite list of `n` ionic strengths, chosen however
one likes — equally spaced or not, high salt or low — and any set of at least `n+2` candidate
distances.  Then there are two *different* strictly positive probability distributions on those
distances whose readings agree at every one of the `n` conditions.  A finite titration therefore
never identifies an ensemble whose number of conformers exceeds its number of conditions by two,
and the count in `ladder_identifies_distribution` is optimal up to that additive constant.

The two ensembles are genuine: every weight is strictly positive and both weight vectors sum to
`1`, because the perturbation is taken in the kernel of the `n` readings *and* of the total mass,
and is scaled small enough to preserve positivity. -/
theorem finite_conditions_insufficient {T : Finset ℝ} {n : ℕ}
    (hcard : n + 2 ≤ T.card) (kappa : Fin n → ℝ) :
    ∃ w w' : ℝ → ℝ,
      (∀ t ∈ T, 0 < w t) ∧ (∀ t ∈ T, 0 < w' t) ∧
      ∑ t ∈ T, w t = 1 ∧ ∑ t ∈ T, w' t = 1 ∧
      (∃ t ∈ T, w t ≠ w' t) ∧
      ∀ j : Fin n, DistanceEnsemble.measCurve T w (kappa j)
        = DistanceEnsemble.measCurve T w' (kappa j) := by
  classical
  set N : ℕ := T.card with hN
  have hNne : (N : ℝ) ≠ 0 := by
    have : 0 < N := by omega
    positivity
  have hNpos : 0 < (N : ℝ) := by
    have : 0 < N := by omega
    exact_mod_cast this
  set K : Fin n → {x // x ∈ T} → ℝ := fun j t => Real.exp (-(kappa j * (t:ℝ))) / (t:ℝ) with hK
  -- the `n` readings together with the total mass: `n+1` linear functionals on `N ≥ n+2` weights
  set Phi : ({x // x ∈ T} → ℝ) →ₗ[ℝ] ((Fin n → ℝ) × ℝ) :=
    { toFun := fun v => (fun j => ∑ t, v t * K j t, ∑ t, v t)
      map_add' := by
        intro u v
        simp [Prod.ext_iff, funext_iff, add_mul, Finset.sum_add_distrib]
      map_smul' := by
        intro c v
        simp [Prod.ext_iff, funext_iff, mul_assoc, Finset.mul_sum] } with hPhi
  have hnotinj : ¬ Function.Injective Phi := by
    intro hinj
    have h1 := LinearMap.finrank_le_finrank_of_injective hinj
    rw [Module.finrank_fintype_fun_eq_card, Module.finrank_prod,
      Module.finrank_fintype_fun_eq_card, Module.finrank_self] at h1
    simp [Fintype.card_coe] at h1
    omega
  obtain ⟨u, v, huv, hne⟩ := Function.not_injective_iff.1 hnotinj
  set z : {x // x ∈ T} → ℝ := u - v with hz
  have hz0 : z ≠ 0 := sub_ne_zero.2 hne
  have hPhiz : Phi z = 0 := by rw [hz, map_sub, huv, sub_self]
  have hzread : ∀ j : Fin n, ∑ t, z t * K j t = 0 := by
    intro j
    have h1 := congrArg Prod.fst hPhiz
    simpa [hPhi, funext_iff] using congrFun h1 j
  have hzmass : ∑ t, z t = 0 := by
    have h1 := congrArg Prod.snd hPhiz
    simpa [hPhi] using h1
  set C : ℝ := ∑ t, |z t| with hC
  have hCnn : 0 ≤ C := Finset.sum_nonneg fun t _ => abs_nonneg _
  have hCbound : ∀ t, |z t| ≤ C :=
    fun t => Finset.single_le_sum (f := fun t => |z t|) (fun i _ => abs_nonneg _)
      (Finset.mem_univ t)
  set s : ℝ := 1 / ((N : ℝ) * (1 + C)) with hs
  have hspos : 0 < s := div_pos one_pos (mul_pos hNpos (by linarith))
  have hsbound : ∀ t, |s * z t| < 1 / (N : ℝ) := by
    intro t
    rw [abs_mul, abs_of_pos hspos]
    have h1 : s * |z t| ≤ s * C := mul_le_mul_of_nonneg_left (hCbound t) (le_of_lt hspos)
    have h2 : 1 / (N : ℝ) - s * C = 1 / ((N : ℝ) * (1 + C)) := by
      rw [hs]; field_simp; ring
    have h3 : (0:ℝ) < 1 / ((N : ℝ) * (1 + C)) := div_pos one_pos (mul_pos hNpos (by linarith))
    linarith
  set w0 : ℝ → ℝ := fun _ => 1 / (N : ℝ) with hw0
  set w1 : ℝ → ℝ := fun x => if h : x ∈ T then 1 / (N : ℝ) + s * z ⟨x, h⟩ else 0 with hw1
  have hw1val : ∀ t : {x // x ∈ T}, w1 (t : ℝ) = 1 / (N : ℝ) + s * z t := by
    intro t; simp [hw1, t.2]
  have hsum0 : ∑ x ∈ T, w0 x = 1 := by
    rw [hw0]
    simp only [Finset.sum_const, nsmul_eq_mul, ← hN]
    exact mul_one_div_cancel hNne
  have hsum1 : ∑ x ∈ T, w1 x = 1 := by
    rw [← Finset.sum_coe_sort T w1, Finset.sum_congr rfl (fun t _ => hw1val t),
      Finset.sum_add_distrib, ← Finset.mul_sum, hzmass, mul_zero, add_zero]
    simp only [Finset.sum_const, nsmul_eq_mul, Finset.card_univ, Fintype.card_coe, ← hN]
    exact mul_one_div_cancel hNne
  refine ⟨w0, w1, ?_, ?_, hsum0, hsum1, ?_, ?_⟩
  · intro t _
    rw [hw0]
    positivity
  · intro t ht
    have h1 := hw1val ⟨t, ht⟩
    have h2 := abs_lt.1 (hsbound ⟨t, ht⟩)
    have h3 : (0:ℝ) < 1 / (N:ℝ) := by positivity
    rw [h1]
    linarith [h2.1]
  · obtain ⟨t, htz⟩ := Function.ne_iff.1 hz0
    refine ⟨(t : ℝ), t.2, ?_⟩
    rw [hw1val t, hw0]
    have hsz : s * z t ≠ 0 := mul_ne_zero (ne_of_gt hspos) (by simpa using htz)
    intro hc
    apply hsz
    simp only at hc
    linarith [hc]
  · intro j
    rw [DistanceEnsemble.measCurve, DistanceEnsemble.measCurve,
      ← Finset.sum_coe_sort T (fun x => w0 x * (Real.exp (-(kappa j * x)) / x)),
      ← Finset.sum_coe_sort T (fun x => w1 x * (Real.exp (-(kappa j * x)) / x))]
    have e : ∀ t : {x // x ∈ T},
        w1 (t:ℝ) * (Real.exp (-(kappa j * (t:ℝ))) / (t:ℝ))
          = w0 (t:ℝ) * (Real.exp (-(kappa j * (t:ℝ))) / (t:ℝ)) + s * (z t * K j t) := by
      intro t
      rw [hw1val t, hw0, hK]
      ring
    rw [Finset.sum_congr rfl (fun t _ => e t), Finset.sum_add_distrib]
    have h2 : ∑ x, s * (z x * K j x) = 0 := by rw [← Finset.mul_sum, hzread j, mul_zero]
    rw [h2, add_zero]

/-! ## 4. Identification is not resolution: the conditioning of the ladder -/

/-- **The resolution horizon of a two-condition ladder.**  Two internal distances `r₁ ≠ r₂` read at
the two conditions `κ₀`, `κ₀+h`: for every tolerance `δ > 0` there are two strictly positive
weightings whose readings differ by at most `δ` at both conditions — they agree exactly at `κ₀` —
while their weights at `r₁` differ by exactly

  `δ · r₁ · e^{κ₀r₁} / |e^{−h r₁} − e^{−h r₂}|`.

The Vandermonde inversion of §1 is exact, but its amplification factor is the reciprocal of the gap
between the nodes `e^{−h r}`; at any finite precision the recoverable weights are only as sharp as
that gap allows. -/
theorem two_distance_resolution_horizon {r1 r2 kappa0 hstep delta : ℝ}
    (hr1 : 0 < r1) (hr2 : 0 < r2) (hne : r1 ≠ r2) (hdelta : 0 < delta) (hstep0 : 0 < hstep) :
    ∃ w w' : ℝ → ℝ,
      (∀ t ∈ ({r1, r2} : Finset ℝ), 0 < w t) ∧ (∀ t ∈ ({r1, r2} : Finset ℝ), 0 < w' t) ∧
      (∀ j < 2, |DistanceEnsemble.measCurve {r1, r2} w (ladderPoint kappa0 hstep j)
          - DistanceEnsemble.measCurve {r1, r2} w' (ladderPoint kappa0 hstep j)| ≤ delta) ∧
      |w r1 - w' r1| = delta * (r1 * Real.exp (kappa0 * r1))
          / |Real.exp (-(hstep * r1)) - Real.exp (-(hstep * r2))| := by
  classical
  set x1 : ℝ := Real.exp (-(hstep * r1)) with hx1
  set x2 : ℝ := Real.exp (-(hstep * r2)) with hx2
  have hxne : x1 - x2 ≠ 0 := by
    rw [sub_ne_zero, hx1, hx2]
    intro hc
    have h1 := Real.exp_injective hc
    have h2 : hstep * r1 = hstep * r2 := by linarith
    exact hne (mul_left_cancel₀ (ne_of_gt hstep0) h2)
  set A : ℝ := delta / |x1 - x2| with hA
  have hApos : 0 < A := div_pos hdelta (abs_pos.2 hxne)
  set d1 : ℝ := r1 * Real.exp (kappa0 * r1) * A with hd1
  set d2 : ℝ := -(r2 * Real.exp (kappa0 * r2) * A) with hd2
  set W : ℝ → ℝ := fun x => if x = r1 then 1 + |d1| else if x = r2 then 1 + |d2| else 0 with hW
  set W' : ℝ → ℝ :=
    fun x => if x = r1 then 1 + |d1| + d1 else if x = r2 then 1 + |d2| + d2 else 0 with hW'
  have hW1 : W r1 = 1 + |d1| := by simp [hW]
  have hW2 : W r2 = 1 + |d2| := by simp [hW, Ne.symm hne]
  have hW1' : W' r1 = 1 + |d1| + d1 := by simp [hW']
  have hW2' : W' r2 = 1 + |d2| + d2 := by simp [hW', Ne.symm hne]
  have hpow : ∀ (r : ℝ) (j : ℕ), Real.exp (-(ladderPoint kappa0 hstep j * r))
      = Real.exp (-(kappa0 * r)) * Real.exp (-(hstep * r)) ^ j := by
    intro r j
    rw [← Real.exp_nat_mul, ← Real.exp_add, ladderPoint]
    ring_nf
  have hread : ∀ j : ℕ,
      DistanceEnsemble.measCurve {r1, r2} W (ladderPoint kappa0 hstep j)
        - DistanceEnsemble.measCurve {r1, r2} W' (ladderPoint kappa0 hstep j)
        = -(A * (x1 ^ j - x2 ^ j)) := by
    intro j
    rw [DistanceEnsemble.measCurve, DistanceEnsemble.measCurve,
      Finset.sum_pair hne, Finset.sum_pair hne, hW1, hW2, hW1', hW2',
      hpow r1 j, hpow r2 j, ← hx1, ← hx2, hd1, hd2, Real.exp_neg, Real.exp_neg]
    have hE1 : Real.exp (kappa0 * r1) ≠ 0 := Real.exp_ne_zero _
    have hE2 : Real.exp (kappa0 * r2) ≠ 0 := Real.exp_ne_zero _
    field_simp
    ring
  have habs : |A * (x1 - x2)| = delta := by
    rw [hA, abs_mul, abs_div, abs_abs, abs_of_pos hdelta,
      div_mul_cancel₀ _ (abs_ne_zero.2 hxne)]
  refine ⟨W, W', ?_, ?_, ?_, ?_⟩
  · intro t ht
    simp only [Finset.mem_insert, Finset.mem_singleton] at ht
    rcases ht with h | h
    · rw [h, hW1]; positivity
    · rw [h, hW2]; positivity
  · intro t ht
    simp only [Finset.mem_insert, Finset.mem_singleton] at ht
    rcases ht with h | h
    · rw [h, hW1']; linarith [neg_abs_le d1]
    · rw [h, hW2']; linarith [neg_abs_le d2]
  · intro j hj
    rw [hread j]
    interval_cases j
    · simp; positivity
    · rw [pow_one, pow_one, abs_neg, habs]
  · rw [hW1, hW1']
    have hsub : 1 + |d1| - (1 + |d1| + d1) = -d1 := by ring
    rw [hsub, abs_neg, hd1, hA, abs_mul,
      abs_of_pos (show (0:ℝ) < r1 * Real.exp (kappa0 * r1) by positivity),
      abs_of_pos (show (0:ℝ) < delta / |x1 - x2| by positivity)]
    field_simp

/-- **The amplification is unbounded.**  For any prescribed bound `M` there is a second distance
`r₂` — as close to `r₁` as needed — at which the amplification factor of
`two_distance_resolution_horizon` exceeds `M`.  Two conformers at nearby distances are therefore
not resolved by any titration ladder at any fixed precision, however many conditions it has: the
node gap `|e^{−h r₁} − e^{−h r₂}|`, not the number of conditions, is what limits resolution. -/
theorem resolution_horizon_unbounded {r1 kappa0 hstep delta : ℝ} (M : ℝ)
    (hr1 : 0 < r1) (hstep0 : 0 < hstep) (hdelta : 0 < delta) :
    ∃ r2 : ℝ, 0 < r2 ∧ r1 ≠ r2 ∧
      M ≤ delta * (r1 * Real.exp (kappa0 * r1))
        / |Real.exp (-(hstep * r1)) - Real.exp (-(hstep * r2))| := by
  set c : ℝ := delta * (r1 * Real.exp (kappa0 * r1)) with hc
  have hcpos : 0 < c := by rw [hc]; positivity
  set B : ℝ := Real.exp (-(hstep * r1)) * hstep with hB
  have hBpos : 0 < B := by rw [hB]; positivity
  have key : ∀ t : ℝ, 0 < t →
      |Real.exp (-(hstep * r1)) - Real.exp (-(hstep * (r1 + t)))| ≤ B * t := by
    intro t ht
    have h1 : Real.exp (-(hstep * (r1 + t)))
        = Real.exp (-(hstep * r1)) * Real.exp (-(hstep * t)) := by
      rw [← Real.exp_add]; ring_nf
    have h2 : 1 - Real.exp (-(hstep * t)) ≤ hstep * t := by
      have := Real.add_one_le_exp (-(hstep * t))
      linarith
    have h3 : 0 ≤ 1 - Real.exp (-(hstep * t)) := by
      have : Real.exp (-(hstep * t)) ≤ 1 := Real.exp_le_one_iff.2 (by nlinarith)
      linarith
    rw [h1, abs_of_nonneg (by nlinarith [Real.exp_pos (-(hstep * r1))]), hB]
    nlinarith [Real.exp_pos (-(hstep * r1))]
  rcases le_or_gt M 0 with hM | hM
  · refine ⟨r1 + 1, by linarith, by linarith, ?_⟩
    have : 0 ≤ c / |Real.exp (-(hstep * r1)) - Real.exp (-(hstep * (r1 + 1)))| := by positivity
    linarith
  · have htpos : 0 < c / (M * B) := div_pos hcpos (mul_pos hM hBpos)
    refine ⟨r1 + c / (M * B), by linarith, by linarith, ?_⟩
    have h1 := key (c / (M * B)) htpos
    have h2 : B * (c / (M * B)) = c / M := by field_simp
    rw [h2] at h1
    have hxpos : 0 < |Real.exp (-(hstep * r1)) - Real.exp (-(hstep * (r1 + c / (M * B))))| := by
      rw [abs_pos, sub_ne_zero]
      intro hceq
      have h4 := Real.exp_injective hceq
      have h3 : hstep * r1 = hstep * (r1 + c / (M * B)) := by linarith
      have := mul_left_cancel₀ (ne_of_gt hstep0) h3
      linarith
    rw [le_div_iff₀ hxpos]
    calc M * |Real.exp (-(hstep * r1)) - Real.exp (-(hstep * (r1 + c / (M * B))))|
        ≤ M * (c / M) := mul_le_mul_of_nonneg_left h1 (le_of_lt hM)
      _ = c := by field_simp

/-! ## 5. The law -/

/-- **The titration-ladder law.**  Three statements about a finite salt titration of an ensemble of
internal distances, holding together:

1. *Sufficiency.*  An arithmetic ladder of `2m` ionic strengths separates any two ensembles of at
   most `m` distances each — if their distance distributions differ at all, one of the `2m`
   readings differs.
2. *Necessity.*  For any `n` conditions whatsoever and any `n+2` candidate distances, two distinct
   strictly positive probability distributions on those distances give identical readings at every
   condition.  Conformer count must not outrun condition count.
3. *Conditioning.*  Even where identification holds, the weights are recovered with an
   amplification factor set by the gap between the Vandermonde nodes `e^{−hr}`, and that factor is
   unbounded as two distances approach each other.  Identifiability is not resolvability, and a
   claim to resolve two conformers is meaningful only relative to a stated precision. -/
theorem titration_ladder_law {kappa0 hstep : ℝ} (hstep0 : 0 < hstep) :
    (∀ (T T' : Finset ℝ) (w w' : ℝ → ℝ) (m : ℕ), (∀ t ∈ T, 0 < t) → (∀ t ∈ T', 0 < t) →
        T.card ≤ m → T'.card ≤ m →
        (∃ t : ℝ, DistanceEnsemble.ext T w t ≠ DistanceEnsemble.ext T' w' t) →
        ∃ j < 2 * m, DistanceEnsemble.measCurve T w (ladderPoint kappa0 hstep j)
          ≠ DistanceEnsemble.measCurve T' w' (ladderPoint kappa0 hstep j)) ∧
    (∀ (T : Finset ℝ) (n : ℕ) (kappa : Fin n → ℝ), n + 2 ≤ T.card →
        ∃ w w' : ℝ → ℝ,
          (∀ t ∈ T, 0 < w t) ∧ (∀ t ∈ T, 0 < w' t) ∧
          ∑ t ∈ T, w t = 1 ∧ ∑ t ∈ T, w' t = 1 ∧
          (∃ t ∈ T, w t ≠ w' t) ∧
          ∀ j : Fin n, DistanceEnsemble.measCurve T w (kappa j)
            = DistanceEnsemble.measCurve T w' (kappa j)) ∧
    (∀ (r1 delta M : ℝ), 0 < r1 → 0 < delta →
        ∃ r2 : ℝ, 0 < r2 ∧ r1 ≠ r2 ∧
          M ≤ delta * (r1 * Real.exp (kappa0 * r1))
            / |Real.exp (-(hstep * r1)) - Real.exp (-(hstep * r2))|) := by
  refine ⟨?_, ?_, ?_⟩
  · intro T T' w w' m hT hT' hm hm' hne
    exact ensemble_ladder_design hstep0 hT hT' hm hm' hne
  · intro T n kappa hcard
    exact finite_conditions_insufficient hcard kappa
  · intro r1 delta M hr1 hdelta
    exact resolution_horizon_unbounded M hr1 hstep0 hdelta


end TitrationLadder
end IDR
