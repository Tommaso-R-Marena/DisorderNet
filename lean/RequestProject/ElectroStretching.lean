/-
# Part CXXIII  Electro-stretching: the field-induced extension of a disordered region

Parts CXX--CXXII computed the dipole moment, the field free energy and the polarization of a
disordered polyampholyte exactly.  This file computes the observable an experiment would
actually watch: the **extension of the chain** in the field.

Adding a second conjugate field to the exact solution and differentiating gives the closed
form (`extension_eq`)

    ⟨R⟩_u = - ∑_{i<N} b tanh(u b S_{i+1}),

with `S_k` the charge of the first `k` residues, exactly as in the polarization law.  Every
bond is stretched by a Langevin `tanh`, but *the field it feels is its own prefix charge*.
Two consequences, both exact and both about the same fixed charge composition:

* `extension_alt_le_half` -- for the perfectly mixed sequence `+-+-...` **half the bonds
  have zero prefix charge and therefore never respond at all**.  However strong the field,
  the mixed sequence cannot be extended beyond `t|b|`, half of its contour length `(2t-1)|b|`.
* `extension_blk_le` -- for the diblock `++...+--...-` *every* prefix charge is at least one,
  so the whole chain responds and the extension reaches `N b tanh(u b)`, i.e. the full
  contour length as the field grows.

`electro_stretching_contrast` states the two together.  A saturation extension that differs
by a factor of two between two sequences of identical amino acid composition is about as
sharp a statement of "patterning, not content" as a model can make -- and here it is a
theorem, not a simulation.
-/
import Mathlib
import RequestProject.PolarizationLaw

namespace IDR
namespace Charge

open Finset
open scoped Classical

variable {N : ℕ}

/-! ## 1. `tanh` is monotone -/

lemma tanh_eq_one_sub (x : ℝ) : Real.tanh x = 1 - 2 / (Real.exp (2 * x) + 1) := by
  have hx : Real.exp x ≠ 0 := ne_of_gt (Real.exp_pos x)
  have hden : Real.exp (2 * x) + 1 ≠ 0 := by positivity
  have h2 : Real.exp (2 * x) = Real.exp x * Real.exp x := by
    rw [← Real.exp_add]; ring_nf
  have hneg : Real.exp (-x) = 1 / Real.exp x := by
    rw [Real.exp_neg]; simp
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq, hneg, h2]
  field_simp
  ring

lemma tanh_mono {x y : ℝ} (h : x ≤ y) : Real.tanh x ≤ Real.tanh y := by
  rw [tanh_eq_one_sub, tanh_eq_one_sub]
  have hx : (0 : ℝ) < Real.exp (2 * x) + 1 := by positivity
  have hy : (0 : ℝ) < Real.exp (2 * y) + 1 := by positivity
  have hle : Real.exp (2 * x) + 1 ≤ Real.exp (2 * y) + 1 := by
    have := Real.exp_le_exp.2 (by linarith : 2 * x ≤ 2 * y)
    linarith
  have : 2 / (Real.exp (2 * y) + 1) ≤ 2 / (Real.exp (2 * x) + 1) :=
    div_le_div_of_nonneg_left (by norm_num) hx hle
  linarith

/-! ## 2. The two-field partition function -/

/-- The conformational sum of the exponential of any linear form in the bonds. -/
lemma sum_exp_linear (a : Fin N → ℝ) (b : ℝ) :
    ∑ s : Chain N, Real.exp (∑ i, a i * bondVec b s i) = ∏ i, (2 * Real.cosh (a i * b)) := by
  have hstep : ∀ s : Chain N, Real.exp (∑ i, a i * bondVec b s i)
      = ∏ i : Fin N, (fun (i : Fin N) (β : Bool) => Real.exp (a i * (if β then b else -b)))
          i (s i) := by
    intro s
    rw [Real.exp_sum]
    rfl
  rw [Finset.sum_congr rfl (fun s (_ : s ∈ (univ : Finset (Chain N))) => hstep s)]
  rw [sum_prod_bool (N := N)
    (fun (i : Fin N) (β : Bool) => Real.exp (a i * (if β then b else -b)))]
  refine Finset.prod_congr rfl fun i _ => ?_
  have hb1 : (if (true : Bool) then b else -b) = b := by simp
  have hb2 : (if (false : Bool) then b else -b) = -b := by simp
  rw [hb1, hb2, Real.cosh_eq]
  have h1 : a i * -b = -(a i * b) := by ring
  rw [h1]
  ring

/-- The partition function with both a dipole field `u` and a stretching field `f`. -/
noncomputable def Zgen (q : ℕ → ℝ) (b u f : ℝ) (N : ℕ) : ℝ :=
  ∑ s : Chain N, Real.exp (u * dipole (N := N) q b s + f * endToEnd b s)

lemma Zgen_zero (q : ℕ → ℝ) (b u : ℝ) : Zgen q b u 0 N = Zdip q b u N := by
  rw [Zgen, Zdip]
  exact Finset.sum_congr rfl fun s _ => by norm_num

/-- **The two-field solution.**  Bond `i` feels the stretching field shifted by `u` times
its own prefix charge. -/
theorem Zgen_prod (q : ℕ → ℝ) (b u f : ℝ) (hQ : pre q (N + 1) = 0) :
    Zgen q b u f N = ∏ i : Fin N, (2 * Real.cosh ((f - u * pre q ((i : ℕ) + 1)) * b)) := by
  have hlin : ∀ s : Chain N, u * dipole (N := N) q b s + f * endToEnd b s
      = ∑ i : Fin N, (f - u * pre q ((i : ℕ) + 1)) * bondVec b s i := by
    intro s
    rw [dipole_eq_linear q b s hQ, endToEnd, Finset.mul_sum, Finset.mul_sum,
      ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [Zgen, Finset.sum_congr rfl (fun s (_ : s ∈ (univ : Finset (Chain N))) => by rw [hlin s]),
    sum_exp_linear]

/-! ## 3. The extension -/

/-- The unnormalised mean extension. -/
noncomputable def Wext (q : ℕ → ℝ) (b u : ℝ) (N : ℕ) : ℝ :=
  ∑ s : Chain N, endToEnd b s * Real.exp (u * dipole (N := N) q b s)

/-- The mean extension of the chain in the field. -/
noncomputable def extension (q : ℕ → ℝ) (b u : ℝ) (N : ℕ) : ℝ :=
  Wext q b u N / Zdip q b u N

lemma hasDerivAt_Zgen (q : ℕ → ℝ) (b u : ℝ) :
    HasDerivAt (fun f => Zgen q b u f N) (Wext q b u N) 0 := by
  have hterm : ∀ s : Chain N,
      HasDerivAt (fun f => Real.exp (u * dipole (N := N) q b s + f * endToEnd b s))
        (endToEnd b s * Real.exp (u * dipole (N := N) q b s)) 0 := by
    intro s
    have h1 : HasDerivAt
        (fun f : ℝ => u * dipole (N := N) q b s + f * endToEnd b s) (endToEnd b s) 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (endToEnd b s)).const_add
        (u * dipole (N := N) q b s)
    simpa [mul_comm] using h1.exp
  have h := hasDerivAt_finsum (univ : Finset (Chain N))
    (fun s f => Real.exp (u * dipole (N := N) q b s + f * endToEnd b s))
    (fun s => endToEnd b s * Real.exp (u * dipole (N := N) q b s)) 0
    (fun s _ => hterm s)
  simpa [Zgen, Wext] using h

lemma hasDerivAt_log_two_cosh_aff (c d f : ℝ) :
    HasDerivAt (fun v => Real.log (2 * Real.cosh ((v - d) * c))) (c * Real.tanh ((f - d) * c)) f := by
  have h1 : HasDerivAt (fun v : ℝ => (v - d) * c) c f := by
    simpa using ((hasDerivAt_id f).sub_const d).mul_const c
  have h2 := ((Real.hasDerivAt_cosh ((f - d) * c)).comp f h1).const_mul (2 : ℝ)
  have h3 := h2.log (ne_of_gt (cosh_pos' ((f - d) * c)))
  have hval : 2 * (Real.sinh ((f - d) * c) * c) / (2 * Real.cosh ((f - d) * c))
      = c * Real.tanh ((f - d) * c) := by
    rw [Real.tanh_eq_sinh_div_cosh]
    field_simp
  have hcomp : (Real.cosh ∘ fun v : ℝ => (v - d) * c) = fun v : ℝ => Real.cosh ((v - d) * c) := rfl
  rw [hcomp] at h3
  rwa [hval] at h3

/-- **The exact force--extension law of a charged disordered region.**  Each bond is
stretched by a Langevin `tanh` in the field it feels, namely the applied field times its own
prefix charge. -/
theorem extension_eq (q : ℕ → ℝ) (b u : ℝ) (hQ : pre q (N + 1) = 0) :
    extension q b u N = -∑ i : Fin N, b * Real.tanh (u * pre q ((i : ℕ) + 1) * b) := by
  have hL : HasDerivAt (fun f => Real.log (Zgen q b u f N)) (extension q b u N) 0 := by
    have h := (hasDerivAt_Zgen (N := N) q b u).log
      (by rw [Zgen_zero]; exact ne_of_gt (Zdip_pos q b u))
    rw [Zgen_zero] at h
    rwa [extension]
  have hfun : (fun f => Real.log (Zgen q b u f N))
      = fun f => ∑ i : Fin N, Real.log (2 * Real.cosh ((f - u * pre q ((i : ℕ) + 1)) * b)) := by
    funext f
    rw [Zgen_prod q b u f hQ, Real.log_prod]
    intro i _
    exact ne_of_gt (cosh_pos' _)
  have hR : HasDerivAt
      (fun f => ∑ i : Fin N, Real.log (2 * Real.cosh ((f - u * pre q ((i : ℕ) + 1)) * b)))
      (∑ i : Fin N, b * Real.tanh ((0 - u * pre q ((i : ℕ) + 1)) * b)) 0 :=
    hasDerivAt_finsum (univ : Finset (Fin N))
      (fun i f => Real.log (2 * Real.cosh ((f - u * pre q ((i : ℕ) + 1)) * b)))
      (fun i => b * Real.tanh ((0 - u * pre q ((i : ℕ) + 1)) * b)) 0
      (fun i _ => hasDerivAt_log_two_cosh_aff b (u * pre q ((i : ℕ) + 1)) 0)
  rw [hfun] at hL
  have heq := hL.unique hR
  rw [heq, ← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  have : (0 - u * pre q ((i : ℕ) + 1)) * b = -(u * pre q ((i : ℕ) + 1) * b) := by ring
  rw [this, Real.tanh_neg]
  ring

/-- The extension never exceeds the contour length. -/
theorem abs_extension_le_contour (q : ℕ → ℝ) (b u : ℝ) (hQ : pre q (N + 1) = 0) :
    |extension q b u N| ≤ (N : ℝ) * |b| := by
  rw [extension_eq q b u hQ, abs_neg]
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
  have hterm : ∀ i : Fin N, |b * Real.tanh (u * pre q ((i : ℕ) + 1) * b)| ≤ |b| := by
    intro i
    rw [abs_mul]
    nlinarith [abs_nonneg b, abs_tanh_le_one (u * pre q ((i : ℕ) + 1) * b),
      abs_nonneg (Real.tanh (u * pre q ((i : ℕ) + 1) * b))]
  calc ∑ i : Fin N, |b * Real.tanh (u * pre q ((i : ℕ) + 1) * b)|
      ≤ ∑ _i : Fin N, |b| := Finset.sum_le_sum fun i _ => hterm i
    _ = (N : ℝ) * |b| := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-! ## 4. The two extremal patterns -/

lemma card_even_range (m : ℕ) : ((range m).filter (fun k => Even k)).card = (m + 1) / 2 := by
  have h := Finset.card_filter_add_card_filter_not (s := range m) (fun k => Even k)
  rw [card_odd_range m, Finset.card_range] at h
  omega

/-- **The perfectly mixed sequence can never be stretched past half its contour length.**
Half of its bonds sit at zero prefix charge and are invisible to the field. -/
theorem extension_alt_le_half {N t : ℕ} (hm : N + 1 = 2 * t) (b u : ℝ) :
    |extension (fun k => (alt k : ℝ)) b u N| ≤ (t : ℝ) * |b| := by
  have heven : Even (N + 1) := ⟨t, by omega⟩
  have hQ : pre (fun k => (alt k : ℝ)) (N + 1) = 0 := by
    rw [pre_cast, alt_neutral heven]; simp
  rw [extension_eq _ b u hQ, abs_neg]
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
  have hterm : ∀ i : Fin N,
      |b * Real.tanh (u * pre (fun k => (alt k : ℝ)) ((i : ℕ) + 1) * b)|
        ≤ if Even (i : ℕ) then |b| else 0 := by
    intro i
    rw [pre_cast, preZ_alt]
    by_cases h : Even ((i : ℕ) + 1)
    · have hi : ¬ Even (i : ℕ) := by
        rcases Nat.even_or_odd (i : ℕ) with he | ho
        · exact absurd h (by simpa [Nat.even_add_one] using he)
        · exact Nat.not_even_iff_odd.2 ho
      rw [if_pos h, if_neg hi]
      simp
    · have hi : Even (i : ℕ) := by
        rcases Nat.even_or_odd (i : ℕ) with he | ho
        · exact he
        · exact absurd (by simpa [Nat.even_add_one] using Nat.not_even_iff_odd.2 ho) h
      rw [if_neg h, if_pos hi, abs_mul]
      nlinarith [abs_nonneg b, abs_tanh_le_one (u * ((1 : ℤ) : ℝ) * b),
        abs_nonneg (Real.tanh (u * ((1 : ℤ) : ℝ) * b))]
  have hsum : ∑ i : Fin N, (if Even (i : ℕ) then |b| else 0) = (t : ℝ) * |b| := by
    rw [Fin.sum_univ_eq_sum_range (fun k => if Even k then |b| else 0) N]
    rw [Finset.sum_ite, Finset.sum_const, Finset.sum_const, card_even_range]
    have ht : (N + 1) / 2 = t := by omega
    rw [ht]
    simp [mul_comm]
  calc ∑ i : Fin N, |b * Real.tanh (u * pre (fun k => (alt k : ℝ)) ((i : ℕ) + 1) * b)|
      ≤ ∑ i : Fin N, (if Even (i : ℕ) then |b| else 0) := Finset.sum_le_sum fun i _ => hterm i
    _ = (t : ℝ) * |b| := hsum

/-- **The diblock stretches to its full contour length.**  Every prefix charge of the
diblock is at least one, so every bond responds, and the extension reaches
`N b tanh(u b)` -- the whole contour as the field grows. -/
theorem extension_blk_le {N t : ℕ} (hm : N + 1 = 2 * t) {b u : ℝ} (hb : 0 ≤ b) (hu : 0 ≤ u) :
    extension (fun k => (blk t k : ℝ)) b u N ≤ -((N : ℝ) * b * Real.tanh (u * b)) := by
  have hQ : pre (fun k => (blk t k : ℝ)) (N + 1) = 0 := by
    rw [pre_cast, hm, blk_neutral t]; simp
  rw [extension_eq _ b u hQ, neg_le_neg_iff]
  have hterm : ∀ i : Fin N,
      b * Real.tanh (u * b) ≤ b * Real.tanh (u * pre (fun k => (blk t k : ℝ)) ((i : ℕ) + 1) * b) := by
    intro i
    have hiN : (i : ℕ) < N := i.isLt
    have hle : (1 : ℝ) ≤ pre (fun k => (blk t k : ℝ)) ((i : ℕ) + 1) := by
      rw [pre_cast, preZ_blk t ((i : ℕ) + 1) (by omega)]
      have h1 : 1 ≤ min ((i : ℕ) + 1) (2 * t - ((i : ℕ) + 1)) := by omega
      exact_mod_cast h1
    have hmul : u * b ≤ u * pre (fun k => (blk t k : ℝ)) ((i : ℕ) + 1) * b := by
      have hub : 0 ≤ u * b := mul_nonneg hu hb
      nlinarith
    have := tanh_mono hmul
    nlinarith [Real.tanh_lt_one (u * b), Real.neg_one_lt_tanh (u * b)]
  calc (N : ℝ) * b * Real.tanh (u * b)
      = ∑ _i : Fin N, b * Real.tanh (u * b) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        ring
    _ ≤ ∑ i : Fin N, b * Real.tanh (u * pre (fun k => (blk t k : ℝ)) ((i : ℕ) + 1) * b) :=
        Finset.sum_le_sum fun i _ => hterm i

/-- **The contrast.**  At identical charge composition, the diblock is stretched by the
field to `N b tanh(u b)` -- its whole contour length in a strong field -- while the
perfectly mixed sequence of the same length can never exceed `t |b|`, half of it. -/
theorem electro_stretching_contrast {N t : ℕ} (hm : N + 1 = 2 * t) {b u : ℝ} (hb : 0 ≤ b)
    (hu : 0 ≤ u) :
    (N : ℝ) * b * Real.tanh (u * b) ≤ -extension (fun k => (blk t k : ℝ)) b u N
      ∧ |extension (fun k => (alt k : ℝ)) b u N| ≤ (t : ℝ) * |b| :=
  ⟨by linarith [extension_blk_le hm hb hu], extension_alt_le_half hm b u⟩

end Charge
end IDR
