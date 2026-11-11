/-
# Part CXLIII  The ensemble responds to the salt: what a titration can still measure

Parts CXLI–CXLII treated the internal-distance distribution of a disordered region as a fixed
object and the ionic strength as a knob that only changes how that fixed object is *read*.  That is
the assumption every salt titration of a polyelectrolyte quietly violates.  Screening changes the
electrostatic free energy of each conformer, so the ensemble itself is reweighted as the salt is
raised: the weights are `w_κ`, not `w`.  This part asks what survives.

* **Nothing, if the response is unconstrained.**  `arbitrary_salt_response_unidentifiable` — a
  *single* internal distance, with a freely salt-dependent weight, reproduces an arbitrary
  titration curve exactly, and with strictly positive weights wherever the curve is positive.  So a
  titration of a region whose ensemble may respond arbitrarily to salt has no content at all: the
  identification theorems of Parts CXLI–CXLII are theorems about the *response model*, not about
  the kernel.  Any model calibrated on a titration must therefore declare how its ensemble
  responds to ionic strength, and that declaration is doing the identifying work.

* **A quantitative inverse.**  `ladder_weight_stability` — the exact Vandermonde inversion of Part
  CXLII, made quantitative.  If the readings of two distance distributions on an arithmetic ladder
  of `n` conditions agree to within `δ`, then at each distance `t` their weights agree to within

  `t · e^{κ₀t} · nodeAmp · δ`,

  where `nodeAmp` is the ℓ¹ norm of the coefficients of the Lagrange basis polynomial at the node
  `e^{−ht}` — the exact amplification constant of the inversion, and the quantitative form of the
  resolution horizon of Part CXLII.

* **Slow response is a bias, not a catastrophe.**  `drift_bias_bound` — suppose the ensemble does
  drift with salt, by at most `ε` in the weight of each distance across the ladder, and suppose a
  *static* distribution is fitted to the resulting data and reproduces it exactly.  Then the fitted
  weights differ from the reference weights by at most

  `t · e^{κ₀t} · nodeAmp · (ε · S)`,   `S = ∑_{s} e^{−κ₀s}/s`,

  the drift `ε` entering through the total kernel mass `S` at the lowest condition.  A
  salt-responsive ensemble analysed as a static one is therefore biased by an explicitly bounded
  amount, and the bound degrades exactly as the inversion is ill-conditioned.

`salt_response_law` collects the three.  Design consequence, closing Parts CXLI–CXLIII: a model of
a charged disordered region must supply a *response model* — how the conformational weights move
with ionic strength — because the titration identifies the ensemble only relative to it; and if the
response is asserted to be weak, the residual bias on the recovered weights is bounded by the
drift, amplified by the same node-gap constant that limits resolution.
-/
import Mathlib
import RequestProject.TitrationLadder

set_option autoImplicit false

namespace IDR
namespace SaltResponsive

open Finset Polynomial

/-! ## 1. The amplification constant of the ladder inversion -/

/-- The ℓ¹ norm of the coefficients of the Lagrange basis polynomial at the node of `t`: the exact
factor by which a reading error is amplified into a weight error when an `n`-condition arithmetic
ladder with spacing `hstep` is inverted on the distances `U`. -/
noncomputable def nodeAmp (U : Finset ℝ) (hstep : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  ∑ j ∈ Finset.range n,
    |(Lagrange.basis U (fun s => Real.exp (-(hstep * s))) t).coeff j|

/-- **Quantitative Vandermonde inversion.**  If a sum of at most `n` amplitudes against the powers
of an injective node function is bounded by `δ` at each of the first `n` powers, every amplitude is
bounded by `δ` times the ℓ¹ norm of the corresponding Lagrange basis coefficients. -/
theorem power_amplitudes_bound {U : Finset ℝ} {B x : ℝ → ℝ} {n : ℕ} {delta : ℝ}
    (hinj : Set.InjOn x U) (hcard : U.card ≤ n)
    (hd : ∀ j < n, |∑ t ∈ U, B t * (x t) ^ j| ≤ delta) :
    ∀ t0 ∈ U, |B t0| ≤ (∑ j ∈ Finset.range n, |(Lagrange.basis U x t0).coeff j|) * delta := by
  classical
  intro t0 ht0
  have hcard1 : 1 ≤ U.card := Finset.card_pos.2 ⟨t0, ht0⟩
  set P : ℝ[X] := Lagrange.basis U x t0 with hP
  have hdeg : P.natDegree < n := by
    rw [hP, Lagrange.natDegree_basis hinj ht0]; omega
  have hid : ∑ t ∈ U, B t * P.eval (x t)
      = ∑ j ∈ Finset.range n, P.coeff j * (∑ t ∈ U, B t * (x t) ^ j) := by
    have e1 : ∀ t ∈ U, B t * P.eval (x t)
        = ∑ j ∈ Finset.range n, P.coeff j * (B t * (x t) ^ j) := by
      intro t _
      rw [eval_eq_sum_range' hdeg, Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [Finset.sum_congr rfl e1, Finset.sum_comm]
    exact Finset.sum_congr rfl fun j _ => by rw [Finset.mul_sum]
  have hleft : ∑ t ∈ U, B t * P.eval (x t) = B t0 := by
    rw [Finset.sum_eq_single t0]
    · rw [hP, Lagrange.eval_basis_self hinj ht0, mul_one]
    · intro b hb hne
      rw [hP, Lagrange.eval_basis_of_ne (Ne.symm hne) hb, mul_zero]
    · intro hc; exact absurd ht0 hc
  rw [hleft] at hid
  rw [hid]
  calc |∑ j ∈ Finset.range n, P.coeff j * (∑ t ∈ U, B t * (x t) ^ j)|
      ≤ ∑ j ∈ Finset.range n, |P.coeff j * (∑ t ∈ U, B t * (x t) ^ j)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j ∈ Finset.range n, |P.coeff j| * delta := by
        refine Finset.sum_le_sum fun j hj => ?_
        rw [abs_mul]
        exact mul_le_mul_of_nonneg_left (hd j (Finset.mem_range.1 hj)) (abs_nonneg _)
    _ = (∑ j ∈ Finset.range n, |P.coeff j|) * delta := by rw [Finset.sum_mul]

/-- The difference of two titration readings, written as a single exponential sum over the union of
the two supports. -/
theorem curve_diff_expSum (T T' : Finset ℝ) (w w' : ℝ → ℝ) (kappa : ℝ) :
    DistanceEnsemble.measCurve T w kappa - DistanceEnsemble.measCurve T' w' kappa
      = ∑ t ∈ T ∪ T',
          ((DistanceEnsemble.ext T w t - DistanceEnsemble.ext T' w' t) / t)
            * Real.exp (-(kappa * t)) := by
  classical
  have hcurve : ∀ (S : Finset ℝ) (v : ℝ → ℝ), S ⊆ T ∪ T' →
      DistanceEnsemble.measCurve S v kappa
        = ∑ t ∈ T ∪ T', DistanceEnsemble.ext S v t * (Real.exp (-(kappa * t)) / t) := by
    intro S v hSU
    rw [DistanceEnsemble.measCurve]
    have hsub : ∑ t ∈ S, DistanceEnsemble.ext S v t * (Real.exp (-(kappa * t)) / t)
        = ∑ t ∈ T ∪ T', DistanceEnsemble.ext S v t * (Real.exp (-(kappa * t)) / t) := by
      refine Finset.sum_subset hSU ?_
      intro y _ hy
      simp [DistanceEnsemble.ext, hy]
    rw [← hsub]
    exact Finset.sum_congr rfl fun t htS => by simp [DistanceEnsemble.ext, htS]
  rw [hcurve T w Finset.subset_union_left, hcurve T' w' Finset.subset_union_right,
    ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun t _ => ?_
  rcases eq_or_ne t 0 with rfl | ht
  · simp
  · field_simp

/-- **The ladder inversion is stable, with an explicit constant.**  If the readings of two finitely
supported distance distributions agree to within `δ` at each of the `n` conditions of an arithmetic
ladder, then their weights agree at every distance `t`, to within `t·e^{κ₀t}·nodeAmp·δ`.  Taking
`δ = 0` recovers the exact identification theorem of Part CXLII; for `δ > 0` the constant is the
amplification factor whose divergence for nearby distances was exhibited there. -/
theorem ladder_weight_stability {T T' : Finset ℝ} {w w' : ℝ → ℝ} {kappa0 hstep delta : ℝ} {n : ℕ}
    (hstep0 : 0 < hstep) (hpos : ∀ t ∈ T ∪ T', 0 < t) (hcard : (T ∪ T').card ≤ n)
    (h : ∀ j < n, |DistanceEnsemble.measCurve T w (TitrationLadder.ladderPoint kappa0 hstep j)
      - DistanceEnsemble.measCurve T' w' (TitrationLadder.ladderPoint kappa0 hstep j)| ≤ delta) :
    ∀ t ∈ T ∪ T', |DistanceEnsemble.ext T w t - DistanceEnsemble.ext T' w' t|
      ≤ t * Real.exp (kappa0 * t) * nodeAmp (T ∪ T') hstep n t * delta := by
  classical
  set U : Finset ℝ := T ∪ T' with hU
  set A : ℝ → ℝ := fun t => (DistanceEnsemble.ext T w t - DistanceEnsemble.ext T' w' t) / t with hA
  set x : ℝ → ℝ := fun t => Real.exp (-(hstep * t)) with hx
  set B : ℝ → ℝ := fun t => A t * Real.exp (-(kappa0 * t)) with hB
  have hinj : Set.InjOn x U := by
    intro a _ b _ hab
    have h1 : -(hstep * a) = -(hstep * b) := Real.exp_injective (by simpa [hx] using hab)
    have h2 : hstep * a = hstep * b := by linarith
    exact mul_left_cancel₀ (ne_of_gt hstep0) h2
  have hpow : ∀ (t : ℝ) (j : ℕ),
      Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t))
        = Real.exp (-(kappa0 * t)) * (x t) ^ j := by
    intro t j
    rw [hx, ← Real.exp_nat_mul, ← Real.exp_add, TitrationLadder.ladderPoint]
    ring_nf
  have hd : ∀ j < n, |∑ t ∈ U, B t * (x t) ^ j| ≤ delta := by
    intro j hj
    have h1 := h j hj
    rw [curve_diff_expSum T T' w w'] at h1
    rw [← hU] at h1
    have h2 : ∑ t ∈ U, B t * (x t) ^ j
        = ∑ t ∈ U, A t
            * Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) := by
      refine Finset.sum_congr rfl fun t _ => ?_
      rw [hB, hpow t j]
      ring
    rw [h2]
    exact h1
  have hbound := power_amplitudes_bound hinj hcard hd
  intro t ht
  have h3 := hbound t ht
  have htpos : 0 < t := hpos t ht
  have hAeq : |A t| = |B t| * Real.exp (kappa0 * t) := by
    rw [hB, abs_mul, abs_of_pos (Real.exp_pos _)]
    rw [mul_assoc, ← Real.exp_add]
    simp
  have hnum : |DistanceEnsemble.ext T w t - DistanceEnsemble.ext T' w' t| = t * |A t| := by
    rw [hA, abs_div, abs_of_pos htpos]
    field_simp
  rw [hnum, hAeq]
  have hamp : nodeAmp U hstep n t
      = ∑ j ∈ Finset.range n, |(Lagrange.basis U x t).coeff j| := by
    rw [nodeAmp, hx]
  have hle : |B t| ≤ nodeAmp U hstep n t * delta := by rw [hamp]; exact h3
  have hexp : (0:ℝ) < Real.exp (kappa0 * t) := Real.exp_pos _
  calc t * (|B t| * Real.exp (kappa0 * t))
      ≤ t * ((nodeAmp U hstep n t * delta) * Real.exp (kappa0 * t)) := by
        have := mul_le_mul_of_nonneg_right hle (le_of_lt hexp)
        exact mul_le_mul_of_nonneg_left this (le_of_lt htpos)
    _ = t * Real.exp (kappa0 * t) * nodeAmp U hstep n t * delta := by ring

/-! ## 2. With an unconstrained response, a titration has no content -/

/-- **An arbitrary salt response makes the titration vacuous.**  Let the weight of a *single*
internal distance `R` be allowed to depend on the ionic strength.  Then for any target titration
curve `f` whatsoever there is a salt-dependent weight reproducing `f` exactly at every ionic
strength — and the weight is strictly positive wherever `f` is.  No feature of a titration curve
is evidence about the distance distribution unless the salt response of the ensemble is
constrained. -/
theorem arbitrary_salt_response_unidentifiable {R : ℝ} (hR : 0 < R) (f : ℝ → ℝ) :
    ∃ p : ℝ → ℝ, (∀ kappa, p kappa * (Real.exp (-(kappa * R)) / R) = f kappa)
      ∧ (∀ kappa, 0 < f kappa → 0 < p kappa) := by
  refine ⟨fun kappa => f kappa * R * Real.exp (kappa * R), fun kappa => ?_, fun kappa hf => ?_⟩
  · have hE : Real.exp (kappa * R) * Real.exp (-(kappa * R)) = 1 := by
      rw [← Real.exp_add]; simp
    have hrw : (f kappa * R * Real.exp (kappa * R)) * (Real.exp (-(kappa * R)) / R)
        = f kappa * (Real.exp (kappa * R) * Real.exp (-(kappa * R))) * (R / R) := by
      field_simp
    rw [hrw, hE, div_self (ne_of_gt hR), mul_one, mul_one]
  · have : (0:ℝ) < Real.exp (kappa * R) := Real.exp_pos _
    positivity

/-! ## 3. A slowly responding ensemble: the bias of a static fit -/

/-- **The bias of analysing a responsive ensemble as a static one.**  Suppose the true weights
`v j t` at the `j`-th ladder condition drift from a reference distribution `w` by at most `ε` at
each distance, and suppose a static distribution `w'` reproduces the resulting readings exactly at
every condition.  Then the fitted weights differ from the reference by at most
`t·e^{κ₀t}·nodeAmp·(ε·S)`, where `S = ∑_{s ∈ T} e^{−κ₀s}/s` is the total kernel mass at the lowest
condition.  Ignoring a weak salt response is therefore an explicitly bounded bias — bounded by the
drift, amplified by exactly the constant that governs the conditioning of the inversion. -/
theorem drift_bias_bound {T : Finset ℝ} {w w' : ℝ → ℝ} {v : ℕ → ℝ → ℝ}
    {kappa0 hstep eps : ℝ} {n : ℕ}
    (hstep0 : 0 < hstep) (hpos : ∀ t ∈ T, 0 < t) (hcard : T.card ≤ n)
    (heps : 0 ≤ eps) (hdrift : ∀ j < n, ∀ t ∈ T, |v j t - w t| ≤ eps)
    (hfit : ∀ j < n, DistanceEnsemble.measCurve T w' (TitrationLadder.ladderPoint kappa0 hstep j)
      = ∑ t ∈ T, v j t
          * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t)) :
    ∀ t ∈ T, |DistanceEnsemble.ext T w t - DistanceEnsemble.ext T w' t|
      ≤ t * Real.exp (kappa0 * t) * nodeAmp T hstep n t
          * (eps * ∑ s ∈ T, Real.exp (-(kappa0 * s)) / s) := by
  classical
  set S : ℝ := ∑ s ∈ T, Real.exp (-(kappa0 * s)) / s with hS
  have hkernel : ∀ (j : ℕ) (t : ℝ), t ∈ T →
      Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t
        ≤ Real.exp (-(kappa0 * t)) / t := by
    intro j t ht
    have htpos : 0 < t := hpos t ht
    have hkap : kappa0 * t ≤ TitrationLadder.ladderPoint kappa0 hstep j * t := by
      have hj : 0 ≤ (j : ℝ) * hstep := by positivity
      have : kappa0 ≤ TitrationLadder.ladderPoint kappa0 hstep j := by
        rw [TitrationLadder.ladderPoint]; linarith
      exact mul_le_mul_of_nonneg_right this (le_of_lt htpos)
    gcongr
  have hkernel0 : ∀ (j : ℕ) (t : ℝ), t ∈ T →
      0 ≤ Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t := by
    intro j t ht
    have htpos : 0 < t := hpos t ht
    positivity
  have hdisc : ∀ j < n,
      |DistanceEnsemble.measCurve T w (TitrationLadder.ladderPoint kappa0 hstep j)
        - DistanceEnsemble.measCurve T w' (TitrationLadder.ladderPoint kappa0 hstep j)|
      ≤ eps * S := by
    intro j hj
    rw [hfit j hj, DistanceEnsemble.measCurve, ← Finset.sum_sub_distrib]
    have hterm : ∀ t ∈ T,
        |w t * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t)
          - v j t * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t)|
        ≤ eps * (Real.exp (-(kappa0 * t)) / t) := by
      intro t ht
      have h1 : w t * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t)
          - v j t * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t)
          = (w t - v j t)
              * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t) := by ring
      rw [h1, abs_mul, abs_of_nonneg (hkernel0 j t ht)]
      have h2 : |w t - v j t| ≤ eps := by
        rw [abs_sub_comm]
        exact hdrift j hj t ht
      calc |w t - v j t| * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t)
          ≤ eps * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t) :=
            mul_le_mul_of_nonneg_right h2 (hkernel0 j t ht)
        _ ≤ eps * (Real.exp (-(kappa0 * t)) / t) :=
            mul_le_mul_of_nonneg_left (hkernel j t ht) heps
    calc |∑ t ∈ T, (w t * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t)
            - v j t * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t))|
        ≤ ∑ t ∈ T, |w t * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t)
            - v j t * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ t ∈ T, eps * (Real.exp (-(kappa0 * t)) / t) := Finset.sum_le_sum hterm
      _ = eps * S := by rw [hS, Finset.mul_sum]
  have hposU : ∀ t ∈ T ∪ T, 0 < t := by
    intro t ht
    rw [Finset.union_self] at ht
    exact hpos t ht
  have hcardU : (T ∪ T).card ≤ n := by rw [Finset.union_self]; exact hcard
  have hdiscU : ∀ j < n,
      |DistanceEnsemble.measCurve T w (TitrationLadder.ladderPoint kappa0 hstep j)
        - DistanceEnsemble.measCurve T w' (TitrationLadder.ladderPoint kappa0 hstep j)|
      ≤ eps * S := hdisc
  have hmain := ladder_weight_stability (T := T) (T' := T) (w := w) (w' := w')
    (delta := eps * S) hstep0 hposU hcardU hdiscU
  rw [Finset.union_self] at hmain
  intro t ht
  exact hmain t ht

/-! ## 4. The law -/

/-- **The salt-response law.**  A titration measures a distance distribution only relative to a
declared model of how the ensemble responds to ionic strength: with an unconstrained response even
a single distance fits any curve; with the response constrained, the inversion is stable with an
explicit amplification constant; and if the response is weak rather than absent, a static fit is
biased by at most the drift times that constant times the kernel mass. -/
theorem salt_response_law {kappa0 hstep : ℝ} (hstep0 : 0 < hstep) :
    (∀ (R : ℝ), 0 < R → ∀ f : ℝ → ℝ, ∃ p : ℝ → ℝ,
        (∀ kappa, p kappa * (Real.exp (-(kappa * R)) / R) = f kappa)
          ∧ (∀ kappa, 0 < f kappa → 0 < p kappa)) ∧
    (∀ (T T' : Finset ℝ) (w w' : ℝ → ℝ) (delta : ℝ) (n : ℕ),
        (∀ t ∈ T ∪ T', 0 < t) → (T ∪ T').card ≤ n →
        (∀ j < n, |DistanceEnsemble.measCurve T w (TitrationLadder.ladderPoint kappa0 hstep j)
            - DistanceEnsemble.measCurve T' w'
                (TitrationLadder.ladderPoint kappa0 hstep j)| ≤ delta) →
        ∀ t ∈ T ∪ T', |DistanceEnsemble.ext T w t - DistanceEnsemble.ext T' w' t|
          ≤ t * Real.exp (kappa0 * t) * nodeAmp (T ∪ T') hstep n t * delta) ∧
    (∀ (T : Finset ℝ) (w w' : ℝ → ℝ) (v : ℕ → ℝ → ℝ) (eps : ℝ) (n : ℕ),
        (∀ t ∈ T, 0 < t) → T.card ≤ n → 0 ≤ eps →
        (∀ j < n, ∀ t ∈ T, |v j t - w t| ≤ eps) →
        (∀ j < n, DistanceEnsemble.measCurve T w'
            (TitrationLadder.ladderPoint kappa0 hstep j)
          = ∑ t ∈ T, v j t
              * (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t)) →
        ∀ t ∈ T, |DistanceEnsemble.ext T w t - DistanceEnsemble.ext T w' t|
          ≤ t * Real.exp (kappa0 * t) * nodeAmp T hstep n t
              * (eps * ∑ s ∈ T, Real.exp (-(kappa0 * s)) / s)) := by
  refine ⟨?_, ?_, ?_⟩
  · intro R hR f
    exact arbitrary_salt_response_unidentifiable hR f
  · intro T T' w w' delta n hpos hcard h
    exact ladder_weight_stability hstep0 hpos hcard h
  · intro T w w' v eps n hpos hcard heps hdrift hfit
    exact drift_bias_bound hstep0 hpos hcard heps hdrift hfit

end SaltResponsive
end IDR
