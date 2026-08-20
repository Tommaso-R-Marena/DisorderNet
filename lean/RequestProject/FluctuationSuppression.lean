/-
# Part CXXV  How much conformational freedom the field can take away

Part CXXIII computed the mean extension of a charged disordered region in a field.  This file
computes its **fluctuation**, exactly, and finds the same patterning dichotomy one level
deeper.

Differentiating the two-field solution twice gives `extension_variance_eq`:

    (⟨R²⟩ - ⟨R⟩²)  =  b² ∑_{i<N} (1 - tanh²(u b S_{i+1})),

with `S_k` the charge of the first `k` residues.  Each bond contributes `b² sech²` of
fluctuation in the field it feels -- and the field it feels is its own prefix charge.  Hence:

* `variance_alt_ge` -- in the perfectly mixed sequence `+-+-…` the bonds at even prefix
  positions feel no field at all, so **the fluctuation never falls below `(t-1) b²`, however
  strong the field**: a well-mixed disordered region keeps a fixed fraction of its
  conformational freedom for ever;
* `variance_blk_le` -- in the diblock every prefix charge is at least one, so the fluctuation
  is at most `N b² (1 - tanh²(u b))`, which the field drives to zero: a blocky region of the
  same composition is frozen.

`fluctuation_suppression_contrast` states the two together.  Conformational entropy is what
makes a region disordered; this is an exact statement of which sequences can be deprived of
it and which cannot.
-/
import Mathlib
import RequestProject.ElectroStretching

namespace IDR
namespace Charge

open Finset
open scoped Classical

variable {N : ℕ}

/-! ## 1. The derivative of `tanh` -/

lemma hasDerivAt_tanh (x : ℝ) : HasDerivAt Real.tanh (1 - Real.tanh x ^ 2) x := by
  have hc : Real.cosh x ≠ 0 := ne_of_gt (Real.cosh_pos x)
  have h := (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hc
  have hval : (Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x) / Real.cosh x ^ 2
      = 1 - Real.tanh x ^ 2 := by
    rw [Real.tanh_eq_sinh_div_cosh]
    field_simp
  have hfun : (Real.sinh / Real.cosh) = Real.tanh := by
    funext y
    rw [Real.tanh_eq_sinh_div_cosh]
    rfl
  rw [hval, hfun] at h
  exact h

lemma hasDerivAt_tanh_aff (c d f : ℝ) :
    HasDerivAt (fun v => Real.tanh ((v - d) * c)) (c * (1 - Real.tanh ((f - d) * c) ^ 2)) f := by
  have h1 : HasDerivAt (fun v : ℝ => (v - d) * c) c f := by
    simpa using ((hasDerivAt_id f).sub_const d).mul_const c
  have h2 := (hasDerivAt_tanh ((f - d) * c)).comp f h1
  have hcomp : (Real.tanh ∘ fun v : ℝ => (v - d) * c) = fun v : ℝ => Real.tanh ((v - d) * c) := rfl
  rw [hcomp] at h2
  have hval : (1 - Real.tanh ((f - d) * c) ^ 2) * c = c * (1 - Real.tanh ((f - d) * c) ^ 2) := by
    ring
  rwa [hval] at h2

/-! ## 2. The two-field observables -/

/-- The unnormalised mean extension at stretching field `f`. -/
noncomputable def Wgen (q : ℕ → ℝ) (b u f : ℝ) (N : ℕ) : ℝ :=
  ∑ s : Chain N, endToEnd b s * Real.exp (u * dipole (N := N) q b s + f * endToEnd b s)

/-- The unnormalised mean squared extension at stretching field `f`. -/
noncomputable def Vgen (q : ℕ → ℝ) (b u f : ℝ) (N : ℕ) : ℝ :=
  ∑ s : Chain N, (endToEnd b s) ^ 2 * Real.exp (u * dipole (N := N) q b s + f * endToEnd b s)

/-- The mean extension at stretching field `f`. -/
noncomputable def extensionGen (q : ℕ → ℝ) (b u f : ℝ) (N : ℕ) : ℝ :=
  Wgen q b u f N / Zgen q b u f N

lemma Wgen_zero (q : ℕ → ℝ) (b u : ℝ) : Wgen q b u 0 N = Wext q b u N := by
  rw [Wgen, Wext]
  exact Finset.sum_congr rfl fun s _ => by norm_num

lemma extensionGen_zero (q : ℕ → ℝ) (b u : ℝ) : extensionGen q b u 0 N = extension q b u N := by
  rw [extensionGen, Wgen_zero, Zgen_zero, extension]

lemma Zgen_pos (q : ℕ → ℝ) (b u f : ℝ) : 0 < Zgen q b u f N := by
  rw [Zgen]
  exact Finset.sum_pos (fun s _ => Real.exp_pos _) Finset.univ_nonempty

lemma hasDerivAt_Zgen' (q : ℕ → ℝ) (b u f : ℝ) :
    HasDerivAt (fun v => Zgen q b u v N) (Wgen q b u f N) f := by
  have hterm : ∀ s : Chain N,
      HasDerivAt (fun v => Real.exp (u * dipole (N := N) q b s + v * endToEnd b s))
        (endToEnd b s * Real.exp (u * dipole (N := N) q b s + f * endToEnd b s)) f := by
    intro s
    have h1 : HasDerivAt
        (fun v : ℝ => u * dipole (N := N) q b s + v * endToEnd b s) (endToEnd b s) f := by
      simpa using ((hasDerivAt_id f).mul_const (endToEnd b s)).const_add
        (u * dipole (N := N) q b s)
    simpa [mul_comm] using h1.exp
  have h := hasDerivAt_finsum (univ : Finset (Chain N))
    (fun s v => Real.exp (u * dipole (N := N) q b s + v * endToEnd b s))
    (fun s => endToEnd b s * Real.exp (u * dipole (N := N) q b s + f * endToEnd b s)) f
    (fun s _ => hterm s)
  simpa [Zgen, Wgen] using h

lemma hasDerivAt_Wgen' (q : ℕ → ℝ) (b u f : ℝ) :
    HasDerivAt (fun v => Wgen q b u v N) (Vgen q b u f N) f := by
  have hterm : ∀ s : Chain N,
      HasDerivAt
        (fun v => endToEnd b s * Real.exp (u * dipole (N := N) q b s + v * endToEnd b s))
        ((endToEnd b s) ^ 2 * Real.exp (u * dipole (N := N) q b s + f * endToEnd b s)) f := by
    intro s
    have h1 : HasDerivAt
        (fun v : ℝ => u * dipole (N := N) q b s + v * endToEnd b s) (endToEnd b s) f := by
      simpa using ((hasDerivAt_id f).mul_const (endToEnd b s)).const_add
        (u * dipole (N := N) q b s)
    have h2 := (h1.exp).const_mul (endToEnd b s)
    have hval : endToEnd b s
        * (Real.exp (u * dipole (N := N) q b s + f * endToEnd b s) * endToEnd b s)
        = (endToEnd b s) ^ 2 * Real.exp (u * dipole (N := N) q b s + f * endToEnd b s) := by
      ring
    rwa [hval] at h2
  have h := hasDerivAt_finsum (univ : Finset (Chain N))
    (fun s v => endToEnd b s * Real.exp (u * dipole (N := N) q b s + v * endToEnd b s))
    (fun s => (endToEnd b s) ^ 2 * Real.exp (u * dipole (N := N) q b s + f * endToEnd b s)) f
    (fun s _ => hterm s)
  simpa [Wgen, Vgen] using h

/-! ## 3. The exact fluctuation -/

/-- The mean extension at any stretching field, in closed form. -/
theorem extensionGen_eq (q : ℕ → ℝ) (b u f : ℝ) (hQ : pre q (N + 1) = 0) :
    extensionGen q b u f N
      = ∑ i : Fin N, b * Real.tanh ((f - u * pre q ((i : ℕ) + 1)) * b) := by
  have hL : HasDerivAt (fun v => Real.log (Zgen q b u v N)) (extensionGen q b u f N) f := by
    have h := (hasDerivAt_Zgen' (N := N) q b u f).log (ne_of_gt (Zgen_pos q b u f))
    rwa [extensionGen]
  have hfun : (fun v => Real.log (Zgen q b u v N))
      = fun v => ∑ i : Fin N, Real.log (2 * Real.cosh ((v - u * pre q ((i : ℕ) + 1)) * b)) := by
    funext v
    rw [Zgen_prod q b u v hQ, Real.log_prod]
    intro i _
    exact ne_of_gt (cosh_pos' _)
  have hR : HasDerivAt
      (fun v => ∑ i : Fin N, Real.log (2 * Real.cosh ((v - u * pre q ((i : ℕ) + 1)) * b)))
      (∑ i : Fin N, b * Real.tanh ((f - u * pre q ((i : ℕ) + 1)) * b)) f :=
    hasDerivAt_finsum (univ : Finset (Fin N))
      (fun i v => Real.log (2 * Real.cosh ((v - u * pre q ((i : ℕ) + 1)) * b)))
      (fun i => b * Real.tanh ((f - u * pre q ((i : ℕ) + 1)) * b)) f
      (fun i _ => hasDerivAt_log_two_cosh_aff b (u * pre q ((i : ℕ) + 1)) f)
  rw [hfun] at hL
  exact hL.unique hR

/-- **The exact conformational fluctuation of the extension.**  Each bond contributes
`b² sech²` in the field it feels, which is the applied field times its own prefix charge. -/
theorem extension_variance_eq (q : ℕ → ℝ) (b u : ℝ) (hQ : pre q (N + 1) = 0) :
    (Vgen q b u 0 N * Zdip q b u N - Wext q b u N * Wext q b u N) / (Zdip q b u N) ^ 2
      = b ^ 2 * ∑ i : Fin N, (1 - Real.tanh (u * pre q ((i : ℕ) + 1) * b) ^ 2) := by
  have hquot := (hasDerivAt_Wgen' (N := N) q b u 0).div (hasDerivAt_Zgen' (N := N) q b u 0)
    (ne_of_gt (Zgen_pos q b u 0))
  have hfun : (fun v => extensionGen q b u v N)
      = (fun v => Wgen q b u v N) / (fun v => Zgen q b u v N) := by
    funext v
    rw [extensionGen]
    rfl
  have hL : HasDerivAt (fun v => extensionGen q b u v N)
      ((Vgen q b u 0 N * Zgen q b u 0 N - Wgen q b u 0 N * Wgen q b u 0 N)
        / (Zgen q b u 0 N) ^ 2) 0 := by
    rw [hfun]
    exact hquot
  have hclosed : (fun v => extensionGen q b u v N)
      = fun v => ∑ i : Fin N, b * Real.tanh ((v - u * pre q ((i : ℕ) + 1)) * b) := by
    funext v
    exact extensionGen_eq q b u v hQ
  have hR : HasDerivAt
      (fun v => ∑ i : Fin N, b * Real.tanh ((v - u * pre q ((i : ℕ) + 1)) * b))
      (∑ i : Fin N, b * (b * (1 - Real.tanh ((0 - u * pre q ((i : ℕ) + 1)) * b) ^ 2))) 0 :=
    hasDerivAt_finsum (univ : Finset (Fin N))
      (fun i v => b * Real.tanh ((v - u * pre q ((i : ℕ) + 1)) * b))
      (fun i => b * (b * (1 - Real.tanh ((0 - u * pre q ((i : ℕ) + 1)) * b) ^ 2))) 0
      (fun i _ => (hasDerivAt_tanh_aff b (u * pre q ((i : ℕ) + 1)) 0).const_mul b)
  rw [hclosed] at hL
  have hu := hL.unique hR
  rw [Wgen_zero, Zgen_zero] at hu
  rw [hu, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  have harg : (0 - u * pre q ((i : ℕ) + 1)) * b = -(u * pre q ((i : ℕ) + 1) * b) := by ring
  rw [harg, Real.tanh_neg]
  ring

/-! ## 4. The two extremal patterns -/

/-- **A well-mixed region keeps its conformational freedom.**  At least `t - 1` of its bonds
feel no field whatever the applied field, so the fluctuation of its extension never falls
below `(t-1) b²`. -/
theorem variance_alt_ge {N t : ℕ} (hm : N + 1 = 2 * t) (b u : ℝ) :
    ((t : ℝ) - 1) * b ^ 2
      ≤ b ^ 2 * ∑ i : Fin N,
          (1 - Real.tanh (u * pre (fun k => (alt k : ℝ)) ((i : ℕ) + 1) * b) ^ 2) := by
  have hterm : ∀ i : Fin N,
      (if ¬ Even (i : ℕ) then (1 : ℝ) else 0)
        ≤ 1 - Real.tanh (u * pre (fun k => (alt k : ℝ)) ((i : ℕ) + 1) * b) ^ 2 := by
    intro i
    rw [pre_cast, preZ_alt]
    by_cases h : Even ((i : ℕ) + 1)
    · have hi : ¬ Even (i : ℕ) := by
        rcases Nat.even_or_odd (i : ℕ) with he | ho
        · exact absurd h (by simpa [Nat.even_add_one] using he)
        · exact Nat.not_even_iff_odd.2 ho
      rw [if_pos h, if_pos hi]
      norm_num
    · have hi : Even (i : ℕ) := by
        rcases Nat.even_or_odd (i : ℕ) with he | ho
        · exact he
        · exact absurd (by simpa [Nat.even_add_one] using Nat.not_even_iff_odd.2 ho) h
      rw [if_neg h, if_neg (by simpa using hi)]
      nlinarith [abs_tanh_le_one (u * ((1 : ℤ) : ℝ) * b),
        sq_nonneg (Real.tanh (u * ((1 : ℤ) : ℝ) * b)),
        abs_nonneg (Real.tanh (u * ((1 : ℤ) : ℝ) * b)),
        sq_abs (Real.tanh (u * ((1 : ℤ) : ℝ) * b))]
  have hcount : ∑ i : Fin N, (if ¬ Even (i : ℕ) then (1 : ℝ) else 0) = ((N / 2 : ℕ) : ℝ) := by
    rw [Fin.sum_univ_eq_sum_range (fun k => if ¬ Even k then (1 : ℝ) else 0) N,
      Finset.sum_ite, Finset.sum_const, Finset.sum_const, card_odd_range]
    simp
  have hsum : ((t : ℝ) - 1) ≤ ∑ i : Fin N,
      (1 - Real.tanh (u * pre (fun k => (alt k : ℝ)) ((i : ℕ) + 1) * b) ^ 2) := by
    have hle := Finset.sum_le_sum (fun i (_ : i ∈ (univ : Finset (Fin N))) => hterm i)
    rw [hcount] at hle
    have hN2 : (N / 2 : ℕ) = t - 1 := by omega
    rw [hN2] at hle
    have ht : ((t - 1 : ℕ) : ℝ) = (t : ℝ) - 1 := by
      have h1 : 1 ≤ t := by omega
      push_cast [Nat.cast_sub h1]
      ring
    rw [ht] at hle
    exact hle
  have hb : (0 : ℝ) ≤ b ^ 2 := sq_nonneg b
  nlinarith [hsum, hb]

/-- **A blocky region is frozen by the field.**  Every prefix charge of the diblock is at
least one, so the fluctuation of its extension is at most `N b² (1 - tanh²(u b))`, which the
field drives to zero. -/
theorem variance_blk_le {N t : ℕ} (hm : N + 1 = 2 * t) {b u : ℝ} (hb : 0 ≤ b) (hu : 0 ≤ u) :
    b ^ 2 * ∑ i : Fin N,
        (1 - Real.tanh (u * pre (fun k => (blk t k : ℝ)) ((i : ℕ) + 1) * b) ^ 2)
      ≤ (N : ℝ) * b ^ 2 * (1 - Real.tanh (u * b) ^ 2) := by
  have hterm : ∀ i : Fin N,
      1 - Real.tanh (u * pre (fun k => (blk t k : ℝ)) ((i : ℕ) + 1) * b) ^ 2
        ≤ 1 - Real.tanh (u * b) ^ 2 := by
    intro i
    have hiN : (i : ℕ) < N := i.isLt
    have hle : (1 : ℝ) ≤ pre (fun k => (blk t k : ℝ)) ((i : ℕ) + 1) := by
      rw [pre_cast, preZ_blk t ((i : ℕ) + 1) (by omega)]
      have h1 : 1 ≤ min ((i : ℕ) + 1) (2 * t - ((i : ℕ) + 1)) := by omega
      exact_mod_cast h1
    have hmul : u * b ≤ u * pre (fun k => (blk t k : ℝ)) ((i : ℕ) + 1) * b := by
      have hub : 0 ≤ u * b := mul_nonneg hu hb
      nlinarith
    have hmono := tanh_mono hmul
    have hnn : 0 ≤ Real.tanh (u * b) := by
      have : Real.tanh 0 ≤ Real.tanh (u * b) := tanh_mono (mul_nonneg hu hb)
      simpa using this
    nlinarith
  have hsum : ∑ i : Fin N,
      (1 - Real.tanh (u * pre (fun k => (blk t k : ℝ)) ((i : ℕ) + 1) * b) ^ 2)
      ≤ (N : ℝ) * (1 - Real.tanh (u * b) ^ 2) := by
    calc ∑ i : Fin N,
          (1 - Real.tanh (u * pre (fun k => (blk t k : ℝ)) ((i : ℕ) + 1) * b) ^ 2)
        ≤ ∑ _i : Fin N, (1 - Real.tanh (u * b) ^ 2) :=
          Finset.sum_le_sum fun i _ => hterm i
      _ = (N : ℝ) * (1 - Real.tanh (u * b) ^ 2) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  nlinarith [hsum, sq_nonneg b]

/-- **The contrast.**  At identical charge composition and at every field strength, the
extension of the perfectly mixed sequence keeps a fluctuation of at least `(t-1) b²`, while
the diblock's fluctuation is squeezed below `N b² (1 - tanh²(u b))` and so to zero as the
field grows. -/
theorem fluctuation_suppression_contrast {N t : ℕ} (hm : N + 1 = 2 * t) {b u : ℝ} (hb : 0 ≤ b)
    (hu : 0 ≤ u) :
    ((t : ℝ) - 1) * b ^ 2
        ≤ (Vgen (fun k => (alt k : ℝ)) b u 0 N * Zdip (fun k => (alt k : ℝ)) b u N
            - Wext (fun k => (alt k : ℝ)) b u N * Wext (fun k => (alt k : ℝ)) b u N)
          / (Zdip (fun k => (alt k : ℝ)) b u N) ^ 2
      ∧ (Vgen (fun k => (blk t k : ℝ)) b u 0 N * Zdip (fun k => (blk t k : ℝ)) b u N
            - Wext (fun k => (blk t k : ℝ)) b u N * Wext (fun k => (blk t k : ℝ)) b u N)
          / (Zdip (fun k => (blk t k : ℝ)) b u N) ^ 2
        ≤ (N : ℝ) * b ^ 2 * (1 - Real.tanh (u * b) ^ 2) := by
  have heven : Even (N + 1) := ⟨t, by omega⟩
  have hQa : pre (fun k => (alt k : ℝ)) (N + 1) = 0 := by
    rw [pre_cast, alt_neutral heven]; simp
  have hQb : pre (fun k => (blk t k : ℝ)) (N + 1) = 0 := by
    rw [pre_cast, hm, blk_neutral t]; simp
  constructor
  · rw [extension_variance_eq _ b u hQa]
    exact variance_alt_ge hm b u
  · rw [extension_variance_eq _ b u hQb]
    exact variance_blk_le hm hb hu

end Charge
end IDR
