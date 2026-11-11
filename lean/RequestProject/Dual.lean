/-
# Part LXXXI  Ensemble reweighting as convex duality: the certificate, and the diverging multiplier

`RequestProject.MaxEnt` establishes *what* maximum-entropy ensemble refinement computes: the tilted
ensemble `tilt q lam f` is the unique minimum-relative-entropy ensemble matching the data.  It says
nothing about *finding* the multipliers `lam`, nor about what happens when the restraints cannot be
matched at all.  Both are questions about the dual objective

`dual q f d lam = log Z(lam) - ⟨lam, d⟩`,

whose stationary points are the multipliers of the fit.  This file proves the four facts a
practitioner needs about it.

* `logPartition_convex`, `dual_convex` -- **the problem is convex.**  The log-partition function is
  convex in the multipliers (weighted arithmetic–geometric mean inequality applied termwise), and
  subtracting the linear data term keeps it so.  Ensemble refinement therefore has no spurious local
  minima: a descent method that stops has found the global answer, whatever the pool.
* `dual_gap` -- **the optimality gap is a relative entropy.**  If the tilt at `lam` matches the
  data, then for *every* other multiplier vector `mu`,
  `dual mu - dual lam = KL(tilt lam ‖ tilt mu)`.
  The duality gap is not merely nonnegative; it is exactly the information distance between the two
  reweighted ensembles.  This is a *certificate*: the shortfall of a candidate fit is measurable in
  the same units as the refinement objective itself.
* `dual_min_of_matches`, `tilt_eq_of_both_match` -- consequently a matching multiplier vector is a
  global minimiser, and any two matching multiplier vectors give the *same* reweighted ensemble --
  the multipliers may be non-unique (they are, whenever the restraints are linearly dependent on
  the pool), but the ensemble they produce is not.
* `dual_ge_of_feasible` -- **feasible data bound the objective below**, by `log c` where `c` is the
  smallest prior weight: the optimisation cannot run away when the measurements are consistent with
  the pool.
* `dual_unbounded_of_separated` -- **and infeasible data make it diverge.**  If some direction `u`
  in restraint space separates the data from every conformation of the pool by a margin `eps`, then
  `dual (t·u) ≤ -t·eps`: the objective goes to `-∞` and the multipliers diverge.  A refinement run
  whose multipliers blow up is therefore not a numerical failure to be damped away; it is a proof
  that the restraints are inconsistent with the conformational pool, and the correct response is to
  enlarge the pool or re-examine the data.
-/
import Mathlib
import RequestProject.MaxEnt

set_option autoImplicit false

namespace IDR

namespace MaxEnt

open Finset

variable {n r : ℕ}

/-- The dual (log-partition minus data) objective of ensemble refinement.  Its stationary points
are exactly the multiplier vectors whose tilted ensemble matches the data. -/
noncomputable def dual (q : Fin n → ℝ) (f : Fin r → Fin n → ℝ) (d lam : Fin r → ℝ) : ℝ :=
  Real.log (partition q lam f) - ∑ a, lam a * d a

/-! ### Convexity -/

/-- **The log-partition function is convex in the multipliers.** -/
theorem logPartition_convex {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hn : 0 < n)
    (f : Fin r → Fin n → ℝ) (lam mu : Fin r → ℝ) {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    Real.log (partition q (fun a => t * lam a + (1 - t) * mu a) f)
      ≤ t * Real.log (partition q lam f) + (1 - t) * Real.log (partition q mu f) := by
  set A := partition q lam f with hA
  set B := partition q mu f with hB
  have hApos : 0 < A := partition_pos hq hn lam f
  have hBpos : 0 < B := partition_pos hq hn mu f
  set L : Fin n → ℝ := fun j => ∑ a, lam a * f a j with hL
  set M : Fin n → ℝ := fun j => ∑ a, mu a * f a j with hM
  have ht1' : 0 ≤ 1 - t := by linarith
  -- the mixed exponential factorises
  have hmix : ∀ j, Real.exp (∑ a, (t * lam a + (1 - t) * mu a) * f a j)
      = Real.exp (L j) ^ t * Real.exp (M j) ^ (1 - t) := by
    intro j
    have hsum : ∑ a, (t * lam a + (1 - t) * mu a) * f a j = t * L j + (1 - t) * M j := by
      rw [hL, hM, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun a _ => by ring
    rw [hsum, Real.exp_add,
      Real.rpow_def_of_pos (Real.exp_pos (L j)), Real.rpow_def_of_pos (Real.exp_pos (M j)),
      Real.log_exp, Real.log_exp, mul_comm (L j) t, mul_comm (M j) (1 - t)]
  -- termwise weighted AM–GM
  have hterm : ∀ j : Fin n,
      q j * Real.exp (∑ a, (t * lam a + (1 - t) * mu a) * f a j)
        ≤ A ^ t * B ^ (1 - t) *
            (t / A * (q j * Real.exp (L j)) + (1 - t) / B * (q j * Real.exp (M j))) := by
    intro j
    have hgm : (Real.exp (L j) / A) ^ t * (Real.exp (M j) / B) ^ (1 - t)
        ≤ t * (Real.exp (L j) / A) + (1 - t) * (Real.exp (M j) / B) :=
      Real.geom_mean_le_arith_mean2_weighted ht0 ht1'
        (le_of_lt (div_pos (Real.exp_pos _) hApos))
        (le_of_lt (div_pos (Real.exp_pos _) hBpos)) (by ring)
    have hfac : Real.exp (L j) ^ t * Real.exp (M j) ^ (1 - t)
        = A ^ t * B ^ (1 - t) *
          ((Real.exp (L j) / A) ^ t * (Real.exp (M j) / B) ^ (1 - t)) := by
      rw [Real.div_rpow (Real.exp_pos _).le hApos.le, Real.div_rpow (Real.exp_pos _).le hBpos.le]
      field_simp
    rw [hmix j, hfac]
    have hq' : 0 ≤ q j := (hq j).le
    have hAB : 0 < A ^ t * B ^ (1 - t) :=
      mul_pos (Real.rpow_pos_of_pos hApos t) (Real.rpow_pos_of_pos hBpos (1 - t))
    calc q j * (A ^ t * B ^ (1 - t) *
            ((Real.exp (L j) / A) ^ t * (Real.exp (M j) / B) ^ (1 - t)))
        ≤ q j * (A ^ t * B ^ (1 - t) *
            (t * (Real.exp (L j) / A) + (1 - t) * (Real.exp (M j) / B))) := by
          exact mul_le_mul_of_nonneg_left
            (mul_le_mul_of_nonneg_left hgm hAB.le) hq'
      _ = A ^ t * B ^ (1 - t) *
            (t / A * (q j * Real.exp (L j)) + (1 - t) / B * (q j * Real.exp (M j))) := by
          field_simp
  -- summing gives the multiplicative bound on the partition function
  have hbound : partition q (fun a => t * lam a + (1 - t) * mu a) f ≤ A ^ t * B ^ (1 - t) := by
    have hsum : partition q (fun a => t * lam a + (1 - t) * mu a) f
        ≤ ∑ j, A ^ t * B ^ (1 - t) *
            (t / A * (q j * Real.exp (L j)) + (1 - t) / B * (q j * Real.exp (M j))) :=
      Finset.sum_le_sum fun j _ => hterm j
    have hcollapse : ∑ j, A ^ t * B ^ (1 - t) *
          (t / A * (q j * Real.exp (L j)) + (1 - t) / B * (q j * Real.exp (M j)))
        = A ^ t * B ^ (1 - t) := by
      rw [← Finset.mul_sum, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
      have hAsum : ∑ j, q j * Real.exp (L j) = A := rfl
      have hBsum : ∑ j, q j * Real.exp (M j) = B := rfl
      rw [hAsum, hBsum]
      field_simp
      ring
    rw [hcollapse] at hsum
    exact hsum
  have hpos : 0 < partition q (fun a => t * lam a + (1 - t) * mu a) f :=
    partition_pos hq hn _ f
  calc Real.log (partition q (fun a => t * lam a + (1 - t) * mu a) f)
      ≤ Real.log (A ^ t * B ^ (1 - t)) := Real.log_le_log hpos hbound
    _ = t * Real.log A + (1 - t) * Real.log B := by
        rw [Real.log_mul (ne_of_gt (Real.rpow_pos_of_pos hApos t))
          (ne_of_gt (Real.rpow_pos_of_pos hBpos (1 - t))),
          Real.log_rpow hApos, Real.log_rpow hBpos]

/-- **The refinement objective is convex**, so ensemble reweighting is a convex optimisation
problem with no spurious local minima. -/
theorem dual_convex {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hn : 0 < n)
    (f : Fin r → Fin n → ℝ) (d lam mu : Fin r → ℝ) {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    dual q f d (fun a => t * lam a + (1 - t) * mu a)
      ≤ t * dual q f d lam + (1 - t) * dual q f d mu := by
  have hlin : ∑ a, (t * lam a + (1 - t) * mu a) * d a
      = t * (∑ a, lam a * d a) + (1 - t) * (∑ a, mu a * d a) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun a _ => by ring
  have := logPartition_convex hq hn f lam mu ht0 ht1
  simp only [dual, hlin]
  linarith

/-! ### The duality gap is a relative entropy -/

/-- **The exact duality gap.**  If the tilted ensemble at `lam` reproduces the data, then the
excess of the dual objective at any other multiplier vector `mu` is exactly the relative entropy
between the two reweighted ensembles. -/
theorem dual_gap {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hn : 0 < n)
    {f : Fin r → Fin n → ℝ} {d lam : Fin r → ℝ}
    (hmatch : Matches (tilt q lam f) f d) (mu : Fin r → ℝ) :
    dual q f d mu - dual q f d lam = klDiv (tilt q lam f) (tilt q mu f) := by
  set A := partition q lam f with hA
  set B := partition q mu f with hB
  have hApos : 0 < A := partition_pos hq hn lam f
  have hBpos : 0 < B := partition_pos hq hn mu f
  set L : Fin n → ℝ := fun j => ∑ a, lam a * f a j with hL
  set M : Fin n → ℝ := fun j => ∑ a, mu a * f a j with hM
  have hlog : ∀ j, Real.log (tilt q lam f j / tilt q mu f j)
      = (L j - M j) + (Real.log B - Real.log A) := by
    intro j
    have hratio : tilt q lam f j / tilt q mu f j = Real.exp (L j - M j) * (B / A) := by
      have hqj : q j ≠ 0 := (hq j).ne'
      have h1 : tilt q lam f j = q j * Real.exp (L j) / A := rfl
      have h2 : tilt q mu f j = q j * Real.exp (M j) / B := rfl
      rw [h1, h2, Real.exp_sub]
      field_simp [hqj, hApos.ne', hBpos.ne']
    rw [hratio, Real.log_mul (ne_of_gt (Real.exp_pos _))
      (ne_of_gt (div_pos hBpos hApos)), Real.log_exp, Real.log_div (ne_of_gt hBpos)
      (ne_of_gt hApos)]
  have hts : ∑ j, tilt q lam f j = 1 := tilt_sum_one hq hn lam f
  have hkl : klDiv (tilt q lam f) (tilt q mu f)
      = (∑ j, tilt q lam f j * (L j - M j)) + (Real.log B - Real.log A) := by
    rw [klDiv]
    have : ∀ j, tilt q lam f j * Real.log (tilt q lam f j / tilt q mu f j)
        = tilt q lam f j * (L j - M j) + tilt q lam f j * (Real.log B - Real.log A) := by
      intro j; rw [hlog j]; ring
    rw [Finset.sum_congr rfl fun j _ => this j, Finset.sum_add_distrib, ← Finset.sum_mul, hts,
      one_mul]
  have hswap : ∑ j, tilt q lam f j * (L j - M j) = ∑ a, (lam a - mu a) * d a := by
    have hexpand : ∀ j, tilt q lam f j * (L j - M j)
        = ∑ a, (lam a - mu a) * (tilt q lam f j * f a j) := by
      intro j
      have hd : L j - M j = ∑ a, (lam a - mu a) * f a j := by
        simp only [hL, hM]
        rw [← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun a _ => by ring
      rw [hd, Finset.mul_sum]
      exact Finset.sum_congr rfl fun a _ => by ring
    rw [Finset.sum_congr rfl fun j _ => hexpand j, Finset.sum_comm]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [← Finset.mul_sum, hmatch a]
  rw [hkl, hswap, dual, dual, ← hA, ← hB]
  have : ∑ a, (lam a - mu a) * d a = (∑ a, lam a * d a) - ∑ a, mu a * d a := by
    rw [← Finset.sum_sub_distrib]; exact Finset.sum_congr rfl fun a _ => by ring
  rw [this]
  ring

/-- A matching multiplier vector is a **global** minimiser of the dual objective. -/
theorem dual_min_of_matches {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hn : 0 < n)
    {f : Fin r → Fin n → ℝ} {d lam : Fin r → ℝ}
    (hmatch : Matches (tilt q lam f) f d) (mu : Fin r → ℝ) :
    dual q f d lam ≤ dual q f d mu := by
  have hgap := dual_gap hq hn hmatch mu
  have hnn : 0 ≤ klDiv (tilt q lam f) (tilt q mu f) :=
    klDiv_nonneg (fun j => (tilt_pos hq hn lam f j).le) (tilt_sum_one hq hn lam f)
      (fun j => tilt_pos hq hn mu f j) (tilt_sum_one hq hn mu f)
  linarith

/-- **The fitted ensemble is unique even when the multipliers are not.**  Any two multiplier
vectors whose tilts match the data produce the same reweighted ensemble. -/
theorem tilt_eq_of_both_match {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hn : 0 < n)
    {f : Fin r → Fin n → ℝ} {d lam mu : Fin r → ℝ}
    (hlam : Matches (tilt q lam f) f d) (hmu : Matches (tilt q mu f) f d) :
    tilt q lam f = tilt q mu f := by
  have h₁ := dual_gap hq hn hlam mu
  have h₂ := dual_gap hq hn hmu lam
  have hnn₁ : 0 ≤ klDiv (tilt q lam f) (tilt q mu f) :=
    klDiv_nonneg (fun j => (tilt_pos hq hn lam f j).le) (tilt_sum_one hq hn lam f)
      (fun j => tilt_pos hq hn mu f j) (tilt_sum_one hq hn mu f)
  have hnn₂ : 0 ≤ klDiv (tilt q mu f) (tilt q lam f) :=
    klDiv_nonneg (fun j => (tilt_pos hq hn mu f j).le) (tilt_sum_one hq hn mu f)
      (fun j => tilt_pos hq hn lam f j) (tilt_sum_one hq hn lam f)
  have hzero : klDiv (tilt q lam f) (tilt q mu f) = 0 := by linarith
  exact (klDiv_eq_zero_iff (fun j => (tilt_pos hq hn lam f j).le) (tilt_sum_one hq hn lam f)
    (fun j => tilt_pos hq hn mu f j) (tilt_sum_one hq hn mu f)).1 hzero

/-! ### Feasibility, and the diverging multiplier -/

/-- **Feasible data bound the objective below.**  If some ensemble on the pool reproduces the data,
the dual objective never falls below the logarithm of the smallest prior weight. -/
theorem dual_ge_of_feasible {q : Fin n → ℝ} (hn : 0 < n) {c : ℝ} (hc : 0 < c)
    (hqc : ∀ j, c ≤ q j) {f : Fin r → Fin n → ℝ} {d : Fin r → ℝ}
    {p : Fin n → ℝ} (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1) (hmatch : Matches p f d)
    (lam : Fin r → ℝ) :
    Real.log c ≤ dual q f d lam := by
  have hne : (Finset.univ : Finset (Fin n)).Nonempty :=
    Finset.univ_nonempty_iff.2 (Fin.pos_iff_nonempty.1 hn)
  obtain ⟨j₀, -, hj₀⟩ :=
    Finset.exists_max_image (Finset.univ : Finset (Fin n)) (fun j => ∑ a, lam a * f a j) hne
  -- the data point cannot beat the best conformation in the direction `lam`
  have hdata : ∑ a, lam a * d a ≤ ∑ a, lam a * f a j₀ := by
    have hrw : ∑ a, lam a * d a = ∑ j, p j * (∑ a, lam a * f a j) := by
      have h1 : ∀ a, lam a * d a = ∑ j, lam a * (p j * f a j) := by
        intro a
        rw [← Finset.mul_sum, hmatch a]
      rw [Finset.sum_congr rfl fun a _ => h1 a, Finset.sum_comm]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun a _ => by ring
    rw [hrw]
    calc ∑ j, p j * (∑ a, lam a * f a j)
        ≤ ∑ j, p j * (∑ a, lam a * f a j₀) :=
          Finset.sum_le_sum fun j _ =>
            mul_le_mul_of_nonneg_left (hj₀ j (Finset.mem_univ j)) (hp j)
      _ = ∑ a, lam a * f a j₀ := by rw [← Finset.sum_mul, hps, one_mul]
  -- and the partition function is at least the single term at `j₀`
  have hZ : c * Real.exp (∑ a, lam a * f a j₀) ≤ partition q lam f := by
    refine le_trans ?_ (Finset.single_le_sum
      (f := fun j => q j * Real.exp (∑ a, lam a * f a j))
      (fun j _ => mul_nonneg (le_trans hc.le (hqc j)) (Real.exp_pos _).le) (Finset.mem_univ j₀))
    exact mul_le_mul_of_nonneg_right (hqc j₀) (Real.exp_pos _).le
  have hZpos : 0 < c * Real.exp (∑ a, lam a * f a j₀) := mul_pos hc (Real.exp_pos _)
  have hlog : Real.log c + (∑ a, lam a * f a j₀) ≤ Real.log (partition q lam f) := by
    have := Real.log_le_log hZpos hZ
    rwa [Real.log_mul (ne_of_gt hc) (ne_of_gt (Real.exp_pos _)), Real.log_exp] at this
  simp only [dual]
  linarith

/-- **Infeasible data drive the objective down without limit.**  If a direction `u` in restraint
space separates the data from every conformation of the pool by a margin `eps`, the dual objective
along that direction is at most `-t·eps`. -/
theorem dual_le_of_separated {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hqs : ∑ j, q j = 1)
    {f : Fin r → Fin n → ℝ} {d u : Fin r → ℝ} {eps : ℝ}
    (hsep : ∀ j, ∑ a, u a * f a j ≤ (∑ a, u a * d a) - eps) (t : ℝ) (ht : 0 ≤ t) :
    dual q f d (fun a => t * u a) ≤ -(t * eps) := by
  have hterm : ∀ j : Fin n, q j * Real.exp (∑ a, (t * u a) * f a j)
      ≤ q j * Real.exp (t * ((∑ a, u a * d a) - eps)) := by
    intro j
    refine mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 ?_) (hq j).le
    have hrw : ∑ a, (t * u a) * f a j = t * ∑ a, u a * f a j := by
      rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun a _ => by ring
    rw [hrw]
    exact mul_le_mul_of_nonneg_left (hsep j) ht
  have hZ : partition q (fun a => t * u a) f ≤ Real.exp (t * ((∑ a, u a * d a) - eps)) := by
    have := Finset.sum_le_sum fun j (_ : j ∈ (Finset.univ : Finset (Fin n))) => hterm j
    calc partition q (fun a => t * u a) f
        ≤ ∑ j, q j * Real.exp (t * ((∑ a, u a * d a) - eps)) := this
      _ = Real.exp (t * ((∑ a, u a * d a) - eps)) := by
          rw [← Finset.sum_mul, hqs, one_mul]
  have hn : 0 < n := by
    rcases Nat.eq_zero_or_pos n with rfl | h
    · simp at hqs
    · exact h
  have hpos : 0 < partition q (fun a => t * u a) f := partition_pos hq hn _ f
  have hlog : Real.log (partition q (fun a => t * u a) f)
      ≤ t * ((∑ a, u a * d a) - eps) := by
    have := Real.log_le_log hpos hZ
    rwa [Real.log_exp] at this
  have hlin : ∑ a, (t * u a) * d a = t * ∑ a, u a * d a := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun a _ => by ring
  simp only [dual, hlin]
  nlinarith [hlog]

/-- **A diverging multiplier is a proof of infeasibility.**  If the data are separated from the
pool by a positive margin in some direction of restraint space, the dual objective is unbounded
below, so the refinement has no optimum and the fitted multipliers run away. -/
theorem dual_unbounded_of_separated {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hqs : ∑ j, q j = 1)
    {f : Fin r → Fin n → ℝ} {d u : Fin r → ℝ} {eps : ℝ} (heps : 0 < eps)
    (hsep : ∀ j, ∑ a, u a * f a j ≤ (∑ a, u a * d a) - eps) (b : ℝ) :
    ∃ lam : Fin r → ℝ, dual q f d lam < b := by
  refine ⟨fun a => ((|b| + 1) / eps) * u a, ?_⟩
  have ht : 0 ≤ (|b| + 1) / eps := div_nonneg (by positivity) heps.le
  have hle := dual_le_of_separated hq hqs hsep ((|b| + 1) / eps) ht
  have hval : (|b| + 1) / eps * eps = |b| + 1 := div_mul_cancel₀ _ (ne_of_gt heps)
  have hb : -(|b| + 1) < b := by
    have := neg_abs_le b
    linarith
  rw [hval] at hle
  linarith

end MaxEnt

end IDR
