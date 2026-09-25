/-
# Part V.5  Pinsker's inequality: the training loss controls the operational error

Two currencies have been used in this development.  Models are *trained* in relative
entropy -- maximum likelihood (`RequestProject.DisorderedRegions`), maximum entropy
refinement (`RequestProject.MaxEnt`), variational free energy
(`RequestProject.FreeEnergy`) -- while models are *judged* in the population-space `ℓ¹`
distance, which is what the capacity, quantisation and sample-complexity laws speak in
(`RequestProject.Metric`, `RequestProject.Quantization`,
`RequestProject.SampleComplexity`).  Without a bridge between them, a small training loss
would carry no guarantee about any experimentally meaningful error.

This file builds the bridge, from scratch:

* `log_pointwise` -- the sharp pointwise inequality
  `(3/2)·(x-y)²/(x+2y) ≤ x·log(x/y) - x + y`, obtained by a two-level derivative argument
  (`Fbridge_nonneg`, whose derivative is `Hbridge`, whose derivative is nonnegative because
  `(t+2)³ ≥ 27t`).
* `pinsker` -- **Pinsker's inequality** `‖p - q‖₁² ≤ 2·KL(p‖q)`, and `ell1_le_sqrt_two_kl`.
* `ell1_le_of_kl_le` -- the design statement: driving the relative entropy below `eps²/2`
  guarantees `ℓ¹` accuracy `eps`, so the capacity and data lower bounds proved elsewhere
  apply verbatim to a KL-trained model.
* `ell1_le_of_freeEnergy_gap` -- the same for variational (force-field) training: an excess
  free energy `ΔF` bounds the operational error by `sqrt (2·β·ΔF)`.
-/
import Mathlib
import RequestProject.DisorderedRegions
import RequestProject.EnsembleCore
import RequestProject.Metric
import RequestProject.FreeEnergy

namespace IDR

open Finset
open scoped Classical

namespace Pinsker

/-! ## The scalar inequality -/

/-- The bridge function whose nonnegativity is the pointwise form of Pinsker's
inequality. -/
noncomputable def Fbridge (t : ℝ) : ℝ := t * Real.log t - t + 1 - (3/2) * (t-1)^2/(t+2)

/-- Its derivative. -/
noncomputable def Hbridge (t : ℝ) : ℝ := Real.log t - (3/2) * ((t-1)*(t+5))/(t+2)^2

lemma hasDerivAt_Fbridge {t : ℝ} (ht : 0 < t) : HasDerivAt Fbridge (Hbridge t) t := by
  have h2 : (0:ℝ) < t + 2 := by linarith
  have h1 : HasDerivAt (fun x : ℝ => x * Real.log x) (Real.log t + 1) t := by
    have h := (hasDerivAt_id t).mul (Real.hasDerivAt_log ht.ne')
    have he : 1 * Real.log t + id t * t⁻¹ = Real.log t + 1 := by
      simp [id]; field_simp
    rw [he] at h
    exact h
  have h3 : HasDerivAt (fun x : ℝ => (x-1)^2) (2*(t-1)) t := by
    have := ((hasDerivAt_id t).sub_const 1).pow 2
    simpa using this
  have h4 : HasDerivAt (fun x : ℝ => x + 2) 1 t := (hasDerivAt_id t).add_const 2
  have h5 : HasDerivAt (fun x : ℝ => (x-1)^2/(x+2))
      ((2*(t-1)*(t+2) - (t-1)^2*1)/(t+2)^2) t := h3.div h4 h2.ne'
  have h6 : HasDerivAt (fun x : ℝ => x * Real.log x - x + 1 - (3/2) * ((x-1)^2/(x+2)))
      (Real.log t + 1 - 1 - 3 / 2 * ((2*(t-1)*(t+2) - (t-1)^2*1)/(t+2)^2)) t :=
    ((h1.sub (hasDerivAt_id t)).add_const 1).sub (h5.const_mul (3/2 : ℝ))
  have heq : Real.log t + 1 - 1 - 3 / 2 * ((2*(t-1)*(t+2) - (t-1)^2*1)/(t+2)^2) = Hbridge t := by
    unfold Hbridge
    field_simp
    ring
  rw [heq] at h6
  have hfun : Fbridge = fun x : ℝ => x * Real.log x - x + 1 - (3/2) * ((x-1)^2/(x+2)) := by
    funext x; unfold Fbridge; ring
  rw [hfun]
  exact h6

lemma hasDerivAt_Hbridge {t : ℝ} (ht : 0 < t) :
    HasDerivAt Hbridge (1/t - 27/(t+2)^3) t := by
  have h2 : (0:ℝ) < t + 2 := by linarith
  have hlog := Real.hasDerivAt_log ht.ne'
  have hnum : HasDerivAt (fun x : ℝ => (x-1)*(x+5)) ((1:ℝ)*(t+5) + (t-1)*1) t :=
    ((hasDerivAt_id t).sub_const 1).mul ((hasDerivAt_id t).add_const 5)
  have hden : HasDerivAt (fun x : ℝ => (x+2)^2) (2*(t+2)) t := by
    have := ((hasDerivAt_id t).add_const 2).pow 2
    simpa using this
  have hdiv : HasDerivAt (fun x : ℝ => ((x-1)*(x+5))/((x+2)^2))
      (((1*(t+5) + (t-1)*1) * (t+2)^2 - ((t-1)*(t+5)) * (2*(t+2)))/((t+2)^2)^2) t :=
    hnum.div hden (by positivity)
  have h6 : HasDerivAt (fun x : ℝ => Real.log x - (3/2) * (((x-1)*(x+5))/((x+2)^2)))
      (t⁻¹ - 3/2 * (((1*(t+5) + (t-1)*1) * (t+2)^2
        - ((t-1)*(t+5)) * (2*(t+2)))/((t+2)^2)^2)) t :=
    hlog.sub (hdiv.const_mul (3/2 : ℝ))
  have heq : t⁻¹ - 3/2 * (((1*(t+5) + (t-1)*1) * (t+2)^2
      - ((t-1)*(t+5)) * (2*(t+2)))/((t+2)^2)^2) = 1/t - 27/(t+2)^3 := by
    field_simp
    ring
  rw [heq] at h6
  have hfun : Hbridge = fun x : ℝ => Real.log x - (3/2) * (((x-1)*(x+5))/((x+2)^2)) := by
    funext x; unfold Hbridge; ring
  rw [hfun]; exact h6

/-- `(t+2)³ ≥ 27t` -- the AM-GM step that makes `Hbridge` monotone. -/
lemma cube_ge {t : ℝ} (ht : 0 ≤ t) : 27 * t ≤ (t+2)^3 := by
  nlinarith [sq_nonneg (t-1), ht]

lemma Hbridge_deriv_nonneg {t : ℝ} (ht : 0 < t) : 0 ≤ 1/t - 27/(t+2)^3 := by
  have h2 : (0:ℝ) < t + 2 := by linarith
  have key := cube_ge ht.le
  rw [sub_nonneg, div_le_div_iff₀ (by positivity) ht]
  nlinarith [key]

lemma Hbridge_one : Hbridge 1 = 0 := by
  unfold Hbridge; norm_num

lemma Hbridge_monotone : MonotoneOn Hbridge (Set.Ioi (0:ℝ)) := by
  have hint : interior (Set.Ioi (0:ℝ)) = Set.Ioi (0:ℝ) := interior_Ioi
  refine monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ioi 0)
    (fun x hx => (hasDerivAt_Hbridge (Set.mem_Ioi.1 hx)).continuousAt.continuousWithinAt)
    (f' := fun x => 1/x - 27/(x+2)^3) ?_ ?_
  · intro x hx
    rw [hint] at hx
    exact ((hasDerivAt_Hbridge (Set.mem_Ioi.1 hx)).hasDerivWithinAt)
  · intro x hx
    rw [hint] at hx
    exact Hbridge_deriv_nonneg (Set.mem_Ioi.1 hx)

lemma Hbridge_nonneg_of_one_le {t : ℝ} (ht : 1 ≤ t) : 0 ≤ Hbridge t := by
  have := Hbridge_monotone (Set.mem_Ioi.2 (by norm_num : (0:ℝ) < 1))
    (Set.mem_Ioi.2 (by linarith : (0:ℝ) < t)) ht
  rwa [Hbridge_one] at this

lemma Hbridge_nonpos_of_le_one {t : ℝ} (ht0 : 0 < t) (ht : t ≤ 1) : Hbridge t ≤ 0 := by
  have := Hbridge_monotone (Set.mem_Ioi.2 ht0)
    (Set.mem_Ioi.2 (by norm_num : (0:ℝ) < 1)) ht
  rwa [Hbridge_one] at this

lemma Fbridge_one : Fbridge 1 = 0 := by
  unfold Fbridge; norm_num

lemma Fbridge_nonneg_of_one_le {t : ℝ} (ht : 1 ≤ t) : 0 ≤ Fbridge t := by
  have hmono : MonotoneOn Fbridge (Set.Ici (1:ℝ)) := by
    have hint : interior (Set.Ici (1:ℝ)) = Set.Ioi (1:ℝ) := interior_Ici
    refine monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici 1)
      (fun x hx => (hasDerivAt_Fbridge
        (lt_of_lt_of_le zero_lt_one (Set.mem_Ici.1 hx))).continuousAt.continuousWithinAt)
      (f' := Hbridge) ?_ ?_
    · intro x hx
      rw [hint] at hx
      exact (hasDerivAt_Fbridge (lt_trans zero_lt_one (Set.mem_Ioi.1 hx))).hasDerivWithinAt
    · intro x hx
      rw [hint] at hx
      exact Hbridge_nonneg_of_one_le (le_of_lt (Set.mem_Ioi.1 hx))
  have := hmono (Set.mem_Ici.2 (le_refl (1:ℝ))) (Set.mem_Ici.2 ht) ht
  rwa [Fbridge_one] at this

lemma Fbridge_nonneg_of_le_one {t : ℝ} (ht0 : 0 < t) (ht : t ≤ 1) : 0 ≤ Fbridge t := by
  have hanti : AntitoneOn Fbridge (Set.Ioc (0:ℝ) 1) := by
    have hint : interior (Set.Ioc (0:ℝ) 1) = Set.Ioo (0:ℝ) 1 := interior_Ioc
    refine antitoneOn_of_hasDerivWithinAt_nonpos (convex_Ioc 0 1)
      (fun x hx => (hasDerivAt_Fbridge (Set.mem_Ioc.1 hx).1).continuousAt.continuousWithinAt)
      (f' := Hbridge) ?_ ?_
    · intro x hx
      rw [hint] at hx
      exact (hasDerivAt_Fbridge (Set.mem_Ioo.1 hx).1).hasDerivWithinAt
    · intro x hx
      rw [hint] at hx
      exact Hbridge_nonpos_of_le_one (Set.mem_Ioo.1 hx).1 (le_of_lt (Set.mem_Ioo.1 hx).2)
  have := hanti (Set.mem_Ioc.2 ⟨ht0, ht⟩) (Set.mem_Ioc.2 ⟨zero_lt_one, le_refl 1⟩) ht
  rwa [Fbridge_one] at this

lemma Fbridge_nonneg {t : ℝ} (ht : 0 < t) : 0 ≤ Fbridge t := by
  rcases le_total t 1 with h | h
  · exact Fbridge_nonneg_of_le_one ht h
  · exact Fbridge_nonneg_of_one_le h

/-- **The pointwise inequality behind Pinsker's inequality.**  For `x ≥ 0` and `y > 0`,
`(3/2)·(x-y)²/(x+2y) ≤ x·log (x/y) - x + y`. -/
theorem log_pointwise {x y : ℝ} (hx : 0 ≤ x) (hy : 0 < y) :
    (3/2) * (x - y)^2/(x + 2*y) ≤ x * Real.log (x / y) - x + y := by
  rcases eq_or_lt_of_le hx with h0 | hxpos
  · -- `x = 0`: the inequality reads `(3/4)·y ≤ y`
    rw [← h0]
    have hzero : (0:ℝ) * Real.log (0 / y) - 0 + y = y := by simp
    have hleft : (3:ℝ)/2 * ((0:ℝ) - y)^2/((0:ℝ) + 2*y) = (3/4) * y := by
      field_simp
      ring
    rw [hzero, hleft]
    linarith
  · set t : ℝ := x / y with htdef
    have htpos : 0 < t := div_pos hxpos hy
    have hxt : x = t * y := by rw [htdef]; field_simp
    have hF := Fbridge_nonneg htpos
    unfold Fbridge at hF
    have hden : (0:ℝ) < t + 2 := by linarith
    have hmul := mul_le_mul_of_nonneg_left hF hy.le
    -- multiply the scalar inequality by `y`
    have hgoal : (3/2) * (x - y)^2/(x + 2*y) ≤ x * Real.log (x/y) - x + y := by
      have hxy : x + 2*y = y * (t + 2) := by rw [hxt]; ring
      have hnum : (x - y)^2 = y^2 * (t-1)^2 := by rw [hxt]; ring
      rw [hxy, hnum, ← htdef]
      have hrw : (3:ℝ)/2 * (y^2 * (t-1)^2) / (y * (t+2)) = y * ((3/2) * (t-1)^2/(t+2)) := by
        field_simp
      rw [hrw, hxt]
      nlinarith [hmul, hy]
    exact hgoal

/-! ## Pinsker's inequality -/

section General

variable {ι : Type*} [Fintype ι]

/-- Relative entropy on an arbitrary finite index type (the conformation library, or a
sample space). -/
noncomputable def klG (p q : ι → ℝ) : ℝ := ∑ i, p i * Real.log (p i / q i)

/-- Gibbs' inequality on an arbitrary finite index type. -/
theorem klG_nonneg {p q : ι → ℝ} (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i)
    (hps : ∑ i, p i = 1) (hqs : ∑ i, q i = 1) : 0 ≤ klG p q := by
  have hle : ∑ i, (p i - q i) ≤ klG p q :=
    Finset.sum_le_sum fun i _ => klDiv_term_le (hp i) (hq i)
  rw [Finset.sum_sub_distrib, hps, hqs] at hle
  simpa using hle

/-- **Pinsker's inequality.**  The square of the population-space `ℓ¹` distance is at most
twice the relative entropy.  Relative entropy is therefore a legitimate surrogate for the
operational error of a disorder model. -/
theorem pinskerG {p q : ι → ℝ} (hp : ∀ j, 0 ≤ p j) (hq : ∀ j, 0 < q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) :
    (∑ j, |p j - q j|) ^ 2 ≤ 2 * klG p q := by
  -- termwise bound
  have hterm : ∀ j : ι,
      (3/2) * (p j - q j)^2/(p j + 2*q j) ≤ p j * Real.log (p j / q j) - p j + q j :=
    fun j => log_pointwise (hp j) (hq j)
  have hsum := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hterm j
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, hps, hqs] at hsum
  have hkl : ∑ j, (3/2) * (p j - q j)^2/(p j + 2*q j) ≤ klG p q := by
    simpa [klG] using hsum
  -- Cauchy--Schwarz
  have hposw : ∀ j : ι, 0 < p j + 2*q j := fun j => by
    have := hp j; have := hq j; linarith
  have hCS : (∑ j, Real.sqrt (p j + 2*q j) * (|p j - q j| / Real.sqrt (p j + 2*q j)))^2
      ≤ (∑ j, (Real.sqrt (p j + 2*q j))^2)
        * (∑ j, (|p j - q j| / Real.sqrt (p j + 2*q j))^2) :=
    Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  have h1 : ∀ j : ι,
      Real.sqrt (p j + 2*q j) * (|p j - q j| / Real.sqrt (p j + 2*q j)) = |p j - q j| := by
    intro j
    have hs : (0:ℝ) < Real.sqrt (p j + 2*q j) := Real.sqrt_pos.2 (hposw j)
    field_simp
  have h2 : ∑ j, (Real.sqrt (p j + 2*q j))^2 = 3 := by
    have : ∀ j : ι, (Real.sqrt (p j + 2*q j))^2 = p j + 2 * q j :=
      fun j => Real.sq_sqrt (hposw j).le
    rw [Finset.sum_congr rfl fun j (_ : j ∈ Finset.univ) => this j, Finset.sum_add_distrib,
      ← Finset.mul_sum, hps, hqs]
    norm_num
  have h3 : ∀ j : ι, (|p j - q j| / Real.sqrt (p j + 2*q j))^2
      = (p j - q j)^2 / (p j + 2*q j) := by
    intro j
    rw [div_pow, sq_abs, Real.sq_sqrt (hposw j).le]
  rw [Finset.sum_congr rfl fun j (_ : j ∈ Finset.univ) => h1 j, h2,
    Finset.sum_congr rfl fun j (_ : j ∈ Finset.univ) => h3 j] at hCS
  -- assemble
  have hfrac : (3:ℝ)/2 * ∑ j, (p j - q j)^2 / (p j + 2*q j) ≤ klG p q := by
    have hrw : (3:ℝ)/2 * ∑ j, (p j - q j)^2 / (p j + 2*q j)
        = ∑ j, (3/2) * (p j - q j)^2/(p j + 2*q j) := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [hrw]
    exact hkl
  have hnn : 0 ≤ ∑ j, (p j - q j)^2 / (p j + 2*q j) :=
    Finset.sum_nonneg fun j _ => div_nonneg (sq_nonneg _) (hposw j).le
  linarith [hCS, hfrac]

/-- The metric form: `‖p - q‖₁ ≤ sqrt (2·KL(p‖q))`. -/
theorem ell1_le_sqrt_two_klG {p q : ι → ℝ} (hp : ∀ j, 0 ≤ p j) (hq : ∀ j, 0 < q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) :
    ∑ j, |p j - q j| ≤ Real.sqrt (2 * klG p q) := by
  have hnn : 0 ≤ ∑ j, |p j - q j| := Finset.sum_nonneg fun j _ => abs_nonneg _
  have hkl : 0 ≤ klG p q := klG_nonneg hp hq hps hqs
  exact (Real.le_sqrt hnn (by linarith)).2 (pinskerG hp hq hps hqs)

end General

variable {m : ℕ}

/-- Pinsker's inequality on a conformation library, in terms of the `klDiv` of
`RequestProject.DisorderedRegions`. -/
theorem pinsker {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hq : ∀ j, 0 < q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) :
    (∑ j, |p j - q j|) ^ 2 ≤ 2 * klDiv p q :=
  pinskerG hp hq hps hqs

/-- The metric form on a conformation library. -/
theorem ell1_le_sqrt_two_kl {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hq : ∀ j, 0 < q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) :
    ∑ j, |p j - q j| ≤ Real.sqrt (2 * klDiv p q) :=
  ell1_le_sqrt_two_klG hp hq hps hqs

/-- **The design statement.**  A model trained to relative entropy at most `eps²/2` of the
truth is within `eps` in the operational `ℓ¹` metric -- so all the capacity, resolution and
data lower bounds of the earlier parts apply to it. -/
theorem ell1_le_of_kl_le {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hq : ∀ j, 0 < q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) {eps : ℝ} (heps : 0 ≤ eps)
    (h : klDiv p q ≤ eps^2/2) :
    ∑ j, |p j - q j| ≤ eps := by
  have hkl : 0 ≤ klDiv p q := klDiv_nonneg hp hps hq hqs
  have h1 := pinsker hp hq hps hqs
  have h2 : (∑ j, |p j - q j|)^2 ≤ eps^2 := by linarith
  have hnn : 0 ≤ ∑ j, |p j - q j| := Finset.sum_nonneg fun j _ => abs_nonneg _
  nlinarith [h2, hnn, heps]

/-- **Variational training is also controlled.**  A candidate ensemble whose excess
variational free energy over the Boltzmann ensemble is `dF` is within `sqrt (2·β·dF)` of the
truth in the operational metric. -/
theorem ell1_le_of_freeEnergy_gap {n : ℕ} (hn : 0 < n) {beta : ℝ} (hbeta : 0 < beta)
    {U p : Fin n → ℝ} (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1) :
    ∑ j, |p j - FreeEnergy.boltz beta U j|
      ≤ Real.sqrt (2 * beta *
          (FreeEnergy.freeEnergy beta U p - FreeEnergy.freeEnergy beta U
            (FreeEnergy.boltz beta U))) := by
  have hgap := FreeEnergy.freeEnergy_gap (U := U) hn hbeta hp hps
  have hb : ∀ j, 0 < FreeEnergy.boltz beta U j := fun j => FreeEnergy.boltz_pos hn beta U j
  have hbs : ∑ j, FreeEnergy.boltz beta U j = 1 := FreeEnergy.boltz_sum_one hn beta U
  have hkl : klDiv p (FreeEnergy.boltz beta U)
      = beta * (FreeEnergy.freeEnergy beta U p
          - FreeEnergy.freeEnergy beta U (FreeEnergy.boltz beta U)) := by
    rw [hgap]
    field_simp
  have h := ell1_le_sqrt_two_kl hp hb hps hbs
  rw [hkl] at h
  calc ∑ j, |p j - FreeEnergy.boltz beta U j|
      ≤ Real.sqrt (2 * (beta * (FreeEnergy.freeEnergy beta U p
          - FreeEnergy.freeEnergy beta U (FreeEnergy.boltz beta U)))) := h
    _ = Real.sqrt (2 * beta * (FreeEnergy.freeEnergy beta U p
          - FreeEnergy.freeEnergy beta U (FreeEnergy.boltz beta U))) := by
        rw [mul_assoc]

end Pinsker

end IDR
