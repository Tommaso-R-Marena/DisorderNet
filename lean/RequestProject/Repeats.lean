/-
# Part VI.2  Many frames: the chi-squared cost tensorises, and detection is its reciprocal

`RequestProject.Fisher` bounds what one conformation can say about the context.  A real
experiment, or a real simulation, sees `n` independent conformations, and the question is
how the discrimination limit improves with `n`.  This file answers it exactly.

* `sum_sq_div_prodP` and `chiSqG_prodP` -- **the chi-squared divergence tensorises
  multiplicatively**: `1 + chi²(p^{⊗n} ‖ q^{⊗n}) = (1 + chi²(p‖q))^n`.  This is the exact
  analogue, for the reweighting cost, of the additivity of relative entropy proved in
  `RequestProject.SharpBound`, and it is the reason the effective sample size of a
  reweighted trajectory of `n` frames decays *geometrically* rather than linearly.
* `context_discrimination_frames` -- consequently, for **any** readout of the whole `n`-frame
  data set (a per-frame average, an autocorrelation, a neural classifier, anything), the
  shift of the readout between two contexts is at most
  `sqrt (Var · ((1 + chi²)^n - 1))`.
* `frames_needed_to_discriminate` and `log_frames_needed` -- inverting it: separating two
  contexts by `delta` standard deviations of the readout requires
  `n ≥ log (1 + delta²/V) / log (1 + chi²)` frames, so the number of frames needed to *see*
  a change of context is the reciprocal of the chi-squared cost of *reweighting* between
  them.  Detection and reweighting are the same quantity read in two directions: the
  effective sample size of `RequestProject.Reweighting` is exactly what an experiment can
  resolve.
-/
import Mathlib
import RequestProject.Fisher
import RequestProject.Reweighting
import RequestProject.SampleComplexity
import RequestProject.Estimation

namespace IDR

open Finset
open scoped Classical

namespace Fisher

variable {m n : ℕ}

open Learn

/-- The second moment of the importance weight over `n` independent draws factorises. -/
lemma sum_sq_div_prodP {p q : Fin m → ℝ} (hq : ∀ j, 0 < q j) (hps : ∑ j, p j = 1)
    (hqs : ∑ j, q j = 1) :
    ∑ s : Fin n → Fin m, prodP p s ^ 2 / prodP q s = (1 + Reweight.chiSq p q) ^ n := by
  have hterm : ∀ s : Fin n → Fin m,
      prodP p s ^ 2 / prodP q s = ∏ i, (fun _ : Fin n => fun a => p a ^ 2 / q a) i (s i) := by
    intro s
    simp only [prodP, ← Finset.prod_pow, ← Finset.prod_div_distrib]
  rw [Finset.sum_congr rfl fun s (_ : s ∈ univ) => hterm s,
    Learn.sum_prod_coord (fun _ : Fin n => fun a => p a ^ 2 / q a)]
  simp only [Reweight.sum_sq_div hq hps hqs, Finset.prod_const, Finset.card_univ,
    Fintype.card_fin]

/-- **Multiplicative tensorisation of the chi-squared divergence.**  `n` independent frames
raise `1 + chi²` to the `n`-th power. -/
theorem chiSqG_prodP {p q : Fin m → ℝ} (hq : ∀ j, 0 < q j) (hps : ∑ j, p j = 1)
    (hqs : ∑ j, q j = 1) :
    1 + chiSqG (prodP (n := n) p) (prodP (n := n) q) = (1 + Reweight.chiSq p q) ^ n := by
  have hQpos : ∀ s : Fin n → Fin m, 0 < prodP q s := fun s => Finset.prod_pos fun i _ => hq _
  have hPs : ∑ s : Fin n → Fin m, prodP p s = 1 := prodP_sum_one hps
  have hQs : ∑ s : Fin n → Fin m, prodP q s = 1 := prodP_sum_one hqs
  -- `chi² = ∑ P²/Q - 1`
  have hkey : ∀ s : Fin n → Fin m,
      (prodP p s - prodP q s) ^ 2 / prodP q s
        = prodP p s ^ 2 / prodP q s - (2 * prodP p s - prodP q s) := by
    intro s
    have h := (hQpos s).ne'
    field_simp
    ring
  have hchi : chiSqG (prodP (n := n) p) (prodP (n := n) q)
      = (∑ s : Fin n → Fin m, prodP p s ^ 2 / prodP q s) - 1 := by
    simp only [chiSqG]
    rw [Finset.sum_congr rfl fun s (_ : s ∈ univ) => hkey s, Finset.sum_sub_distrib,
      Finset.sum_sub_distrib, ← Finset.mul_sum, hPs, hQs]
    ring
  rw [hchi, sum_sq_div_prodP hq hps hqs]
  ring

/-! ## What `n` frames can resolve -/

/-- **The discrimination limit for an `n`-frame experiment.**  For any readout `T` of the
whole data set, the shift of the readout between two contexts is controlled by the
chi-squared cost of reweighting between them, raised to the `n`-th power. -/
theorem context_discrimination_frames {p q : Fin m → ℝ} (T : (Fin n → Fin m) → ℝ)
    (hq : ∀ j, 0 < q j) (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) :
    |(∑ s, prodP p s * T s) - ∑ s, prodP q s * T s|
      ≤ Real.sqrt (varW (prodP (n := n) q) T * ((1 + Reweight.chiSq p q) ^ n - 1)) := by
  have hQpos : ∀ s : Fin n → Fin m, 0 < prodP q s := fun s => Finset.prod_pos fun i _ => hq _
  have hPs : ∑ s : Fin n → Fin m, prodP p s = 1 := prodP_sum_one hps
  have hQs : ∑ s : Fin n → Fin m, prodP q s = 1 := prodP_sum_one hqs
  have h := context_discrimination (T := T) hQpos hPs hQs
  have hchi : chiSqG (prodP (n := n) p) (prodP (n := n) q)
      = (1 + Reweight.chiSq p q) ^ n - 1 := by
    have := chiSqG_prodP (n := n) hq hps hqs
    linarith
  rwa [hchi] at h

/-- **How many frames are needed to see a change of context.**  If a readout of the data
must shift by `delta` between the two contexts while fluctuating by at most `V`, then the
reweighting cost must have grown to at least `1 + delta²/V` over the `n` frames. -/
theorem frames_needed_to_discriminate {p q : Fin m → ℝ} (T : (Fin n → Fin m) → ℝ)
    (hq : ∀ j, 0 < q j) (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1)
    {delta V : ℝ} (hdelta : 0 ≤ delta) (hV : 0 < V) (hvar : varW (prodP (n := n) q) T ≤ V)
    (hshift : delta ≤ |(∑ s, prodP p s * T s) - ∑ s, prodP q s * T s|) :
    1 + delta ^ 2 / V ≤ (1 + Reweight.chiSq p q) ^ n := by
  have hQpos : ∀ s : Fin n → Fin m, 0 < prodP q s := fun s => Finset.prod_pos fun i _ => hq _
  have hPs : ∑ s : Fin n → Fin m, prodP p s = 1 := prodP_sum_one hps
  have hQs : ∑ s : Fin n → Fin m, prodP q s = 1 := prodP_sum_one hqs
  have hcr := chapman_robbins (T := T) hQpos hPs hQs
  have hchi : chiSqG (prodP (n := n) p) (prodP (n := n) q)
      = (1 + Reweight.chiSq p q) ^ n - 1 := by
    have := chiSqG_prodP (n := n) hq hps hqs
    linarith
  rw [hchi] at hcr
  have hchinn : 0 ≤ (1 + Reweight.chiSq p q) ^ n - 1 := by
    have hcs := chiSqG_nonneg (p := prodP (n := n) p) (q := prodP (n := n) q) hQpos
    linarith [hchi ▸ hcs]
  have hsq : delta ^ 2 ≤ ((∑ s, prodP p s * T s) - ∑ s, prodP q s * T s) ^ 2 := by
    have h2 : delta ^ 2 ≤ |(∑ s, prodP p s * T s) - ∑ s, prodP q s * T s| ^ 2 := by
      nlinarith [hshift, hdelta]
    rwa [sq_abs] at h2
  have hstep : delta ^ 2 ≤ V * ((1 + Reweight.chiSq p q) ^ n - 1) := by
    have hvv : varW (prodP (n := n) q) T * ((1 + Reweight.chiSq p q) ^ n - 1)
        ≤ V * ((1 + Reweight.chiSq p q) ^ n - 1) := by
      exact mul_le_mul_of_nonneg_right hvar hchinn
    linarith
  have hfinal : delta ^ 2 / V ≤ (1 + Reweight.chiSq p q) ^ n - 1 := by
    rw [div_le_iff₀ hV]
    linarith
  linarith

/-- The same statement in the form a designer uses: the number of frames needed to detect a
change of context grows like the reciprocal of the logarithm of `1 + chi²`, i.e. like
`1/chi²` for nearby contexts.  The cost of *reweighting* between two contexts and the number
of frames needed to *tell them apart* are reciprocal. -/
theorem log_frames_needed {p q : Fin m → ℝ} (T : (Fin n → Fin m) → ℝ)
    (hq : ∀ j, 0 < q j) (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1)
    {delta V : ℝ} (hdelta : 0 ≤ delta) (hV : 0 < V) (hvar : varW (prodP (n := n) q) T ≤ V)
    (hshift : delta ≤ |(∑ s, prodP p s * T s) - ∑ s, prodP q s * T s|) :
    Real.log (1 + delta ^ 2 / V) ≤ n * Real.log (1 + Reweight.chiSq p q) := by
  have hpow := frames_needed_to_discriminate T hq hps hqs hdelta hV hvar hshift
  have hbase : 0 < 1 + Reweight.chiSq p q := Reweight.one_add_chiSq_pos hq
  have hlog := Real.log_le_log (by positivity) hpow
  rwa [Real.log_pow] at hlog

end Fisher

end IDR
