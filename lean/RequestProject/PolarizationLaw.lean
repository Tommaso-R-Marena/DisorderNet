/-
# Part CXXII  The polarization law and fluctuation--dissipation, exactly

Part CXXI solved the disordered polyampholyte in a field.  This file differentiates that
solution and closes the loop with Part CXX.

* `polarization` is the honest thermodynamic observable: the Boltzmann average of the dipole
  moment at reduced field `u`, `⟨M⟩_u = (∑_s M e^{uM}) / (∑_s e^{uM})`.
* `polarization_eq` -- **the exact polarization law**

      ⟨M⟩_u = ∑_{i<N} b S_{i+1} tanh(u b S_{i+1}),

  where `S_k` is the charge of the first `k` residues.  Every bond responds independently
  with a Langevin-type `tanh`, and its coupling to the field is its prefix charge.  This is
  an exact, closed-form, sequence-resolved response function for a disordered region.
* `polarization_zero` -- the region carries no net polarization in zero field, and
* `abs_polarization_le` -- its polarization saturates at `|b| ∑_k |S_k|`, the total absolute
  prefix charge that already controlled the free energy in Part CXXI.
* `fluctuation_dissipation` -- **the zero-field susceptibility equals the mean squared
  dipole moment**:

      d⟨M⟩_u/du at u = 0  =  b² ∑_{k ≤ N} S_k²  =  ⟨M²⟩,

  the exact sequence functional of Part CXX.  Response and fluctuation are the same number,
  proved here from the model rather than assumed: the extremal patterning law of Part CXX is
  therefore a law about a *measurable* susceptibility, maximised by the diblock and
  minimised by the perfectly mixed sequence.
-/
import Mathlib
import RequestProject.DielectricResponse

namespace IDR
namespace Charge

open Finset
open scoped Classical

variable {N : ℕ}

/-! ## 1. The Boltzmann average of the dipole -/

/-- The unnormalised mean dipole: `∑_s M(s) e^{u M(s)}`. -/
noncomputable def Wdip (q : ℕ → ℝ) (b u : ℝ) (N : ℕ) : ℝ :=
  ∑ s : Chain N, dipole (N := N) q b s * Real.exp (u * dipole (N := N) q b s)

/-- The polarization: the Boltzmann average of the dipole moment at reduced field `u`. -/
noncomputable def polarization (q : ℕ → ℝ) (b u : ℝ) (N : ℕ) : ℝ :=
  Wdip q b u N / Zdip q b u N

lemma Zdip_pos (q : ℕ → ℝ) (b u : ℝ) : 0 < Zdip q b u N := by
  rw [Zdip]
  refine Finset.sum_pos (fun s _ => Real.exp_pos _) ?_
  exact Finset.univ_nonempty

lemma Zdip_zero_field (q : ℕ → ℝ) (b : ℝ) : Zdip q b 0 N = 2 ^ N := by
  rw [Zdip]
  simp [Finset.card_univ]

/-- In zero field the dipole averages to zero: bond reversal is a symmetry. -/
lemma sum_dipole_eq_zero (q : ℕ → ℝ) (b : ℝ) (hQ : pre q (N + 1) = 0) :
    ∑ s : Chain N, dipole (N := N) q b s = 0 := by
  have hlin : ∀ s : Chain N, dipole (N := N) q b s
      = ∑ i : Fin N, (-(pre q ((i : ℕ) + 1))) * bondVec b s i :=
    fun s => dipole_eq_linear q b s hQ
  rw [Finset.sum_congr rfl (fun s (_ : s ∈ (univ : Finset (Chain N))) => hlin s),
    Finset.sum_comm]
  refine Finset.sum_eq_zero fun i _ => ?_
  rw [← Finset.mul_sum, sum_bondVec b i, mul_zero]

/-! ## 2. Differentiating the partition function -/

/-- Differentiation under a finite sum, in the `fun v => ∑ ...` form. -/
lemma hasDerivAt_finsum {ι : Type*} (t : Finset ι) (A : ι → ℝ → ℝ) (A' : ι → ℝ) (x : ℝ)
    (h : ∀ i ∈ t, HasDerivAt (A i) (A' i) x) :
    HasDerivAt (fun v => ∑ i ∈ t, A i v) (∑ i ∈ t, A' i) x := by
  have h2 := HasDerivAt.sum h
  have heq : (∑ i ∈ t, A i) = fun v => ∑ i ∈ t, A i v := by
    funext v; simp [Finset.sum_apply]
  rwa [heq] at h2

lemma hasDerivAt_Zdip (q : ℕ → ℝ) (b u : ℝ) :
    HasDerivAt (fun v => Zdip q b v N) (Wdip q b u N) u := by
  have hterm : ∀ s : Chain N, HasDerivAt (fun v => Real.exp (v * dipole (N := N) q b s))
      (dipole (N := N) q b s * Real.exp (u * dipole (N := N) q b s)) u := by
    intro s
    have h1 : HasDerivAt (fun v : ℝ => v * dipole (N := N) q b s) (dipole (N := N) q b s) u := by
      simpa using (hasDerivAt_id u).mul_const (dipole (N := N) q b s)
    simpa [mul_comm] using h1.exp
  exact hasDerivAt_finsum univ (fun s v => Real.exp (v * dipole (N := N) q b s))
    (fun s => dipole (N := N) q b s * Real.exp (u * dipole (N := N) q b s)) u
    (fun s _ => hterm s)

lemma hasDerivAt_Wdip (q : ℕ → ℝ) (b u : ℝ) :
    HasDerivAt (fun v => Wdip q b v N)
      (∑ s : Chain N, (dipole (N := N) q b s) ^ 2 * Real.exp (u * dipole (N := N) q b s)) u := by
  have hterm : ∀ s : Chain N,
      HasDerivAt (fun v => dipole (N := N) q b s * Real.exp (v * dipole (N := N) q b s))
        ((dipole (N := N) q b s) ^ 2 * Real.exp (u * dipole (N := N) q b s)) u := by
    intro s
    have h1 : HasDerivAt (fun v : ℝ => v * dipole (N := N) q b s) (dipole (N := N) q b s) u := by
      simpa using (hasDerivAt_id u).mul_const (dipole (N := N) q b s)
    have h2 := h1.exp
    have h3 := h2.const_mul (dipole (N := N) q b s)
    have hval : dipole (N := N) q b s
        * (Real.exp (u * dipole (N := N) q b s) * dipole (N := N) q b s)
        = (dipole (N := N) q b s) ^ 2 * Real.exp (u * dipole (N := N) q b s) := by ring
    rwa [hval] at h3
  exact hasDerivAt_finsum univ
    (fun s v => dipole (N := N) q b s * Real.exp (v * dipole (N := N) q b s))
    (fun s => (dipole (N := N) q b s) ^ 2 * Real.exp (u * dipole (N := N) q b s)) u
    (fun s _ => hterm s)

/-- The polarization vanishes in zero field. -/
theorem polarization_zero (q : ℕ → ℝ) (b : ℝ) (hQ : pre q (N + 1) = 0) :
    polarization q b 0 N = 0 := by
  have hW : Wdip q b 0 N = 0 := by
    rw [Wdip]
    simpa using sum_dipole_eq_zero q b hQ
  rw [polarization, hW, zero_div]

/-! ## 3. The exact polarization law -/

lemma hasDerivAt_logZ (q : ℕ → ℝ) (b u : ℝ) :
    HasDerivAt (fun v => Real.log (Zdip q b v N)) (polarization q b u N) u := by
  have h := (hasDerivAt_Zdip (N := N) q b u).log (ne_of_gt (Zdip_pos (N := N) q b u))
  rwa [polarization]

lemma logZ_eq' (q : ℕ → ℝ) (b u : ℝ) (hQ : pre q (N + 1) = 0) :
    Real.log (Zdip q b u N)
      = ∑ i : Fin N, Real.log (2 * Real.cosh (u * (b * pre q ((i : ℕ) + 1)))) := by
  rw [logZ_eq q b u hQ]
  exact Finset.sum_congr rfl fun i _ => by rw [mul_assoc]

lemma hasDerivAt_log_two_cosh (c u : ℝ) :
    HasDerivAt (fun v => Real.log (2 * Real.cosh (v * c))) (c * Real.tanh (u * c)) u := by
  have h1 : HasDerivAt (fun v : ℝ => v * c) c u := by
    simpa using (hasDerivAt_id u).mul_const c
  have h2 := ((Real.hasDerivAt_cosh (u * c)).comp u h1).const_mul (2 : ℝ)
  have h3 := h2.log (ne_of_gt (cosh_pos' (u * c)))
  have hval : 2 * (Real.sinh (u * c) * c) / (2 * Real.cosh (u * c)) = c * Real.tanh (u * c) := by
    rw [Real.tanh_eq_sinh_div_cosh]
    field_simp
  have hcomp : (Real.cosh ∘ fun v : ℝ => v * c) = fun v : ℝ => Real.cosh (v * c) := rfl
  rw [hcomp] at h3
  rwa [hval] at h3

/-- **The exact polarization law.**  Each bond responds to the field with a Langevin `tanh`
whose coupling is the charge of the sequence up to that bond. -/
theorem polarization_eq (q : ℕ → ℝ) (b u : ℝ) (hQ : pre q (N + 1) = 0) :
    polarization q b u N
      = ∑ i : Fin N, (b * pre q ((i : ℕ) + 1)) * Real.tanh (u * (b * pre q ((i : ℕ) + 1))) := by
  have hfun : (fun v => Real.log (Zdip q b v N))
      = fun v => ∑ i : Fin N, Real.log (2 * Real.cosh (v * (b * pre q ((i : ℕ) + 1)))) := by
    funext v
    exact logZ_eq' q b v hQ
  have hL : HasDerivAt (fun v => Real.log (Zdip q b v N)) (polarization q b u N) u :=
    hasDerivAt_logZ q b u
  have hR : HasDerivAt
      (fun v => ∑ i : Fin N, Real.log (2 * Real.cosh (v * (b * pre q ((i : ℕ) + 1)))))
      (∑ i : Fin N, (b * pre q ((i : ℕ) + 1)) * Real.tanh (u * (b * pre q ((i : ℕ) + 1)))) u :=
    hasDerivAt_finsum (univ : Finset (Fin N))
      (fun i v => Real.log (2 * Real.cosh (v * (b * pre q ((i : ℕ) + 1)))))
      (fun i => (b * pre q ((i : ℕ) + 1)) * Real.tanh (u * (b * pre q ((i : ℕ) + 1)))) u
      (fun i _ => hasDerivAt_log_two_cosh _ u)
  rw [hfun] at hL
  exact hL.unique hR

lemma abs_tanh_le_one (x : ℝ) : |Real.tanh x| ≤ 1 := by
  rw [abs_le]
  exact ⟨le_of_lt (Real.neg_one_lt_tanh x), le_of_lt (Real.tanh_lt_one x)⟩

/-- **Saturation.**  However strong the field, the polarization of the region cannot exceed
`|b|` times its total absolute prefix charge. -/
theorem abs_polarization_le (q : ℕ → ℝ) (b u : ℝ) (hQ : pre q (N + 1) = 0) :
    |polarization q b u N| ≤ |b| * absPre q (N + 1) := by
  rw [polarization_eq q b u hQ]
  have hbound : |∑ i : Fin N, (b * pre q ((i : ℕ) + 1))
        * Real.tanh (u * (b * pre q ((i : ℕ) + 1)))|
      ≤ ∑ i : Fin N, |b * pre q ((i : ℕ) + 1)| := by
    refine le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum fun i _ => ?_)
    rw [abs_mul]
    have h1 : |Real.tanh (u * (b * pre q ((i : ℕ) + 1)))| ≤ 1 := abs_tanh_le_one _
    nlinarith [abs_nonneg (b * pre q ((i : ℕ) + 1)), abs_nonneg
      (Real.tanh (u * (b * pre q ((i : ℕ) + 1))))]
  have hsum : ∑ i : Fin N, |b * pre q ((i : ℕ) + 1)| = |b| * absPre q (N + 1) := by
    have h := sum_abs_shift (N := N) q b 1
    simpa using h
  linarith [hbound, hsum.le, hsum.ge]

/-! ## 4. Fluctuation--dissipation -/

/-- **The zero-field susceptibility is the mean squared dipole.**  Differentiating the exact
polarization law at zero field returns exactly the sequence functional computed in Part CXX:
response and fluctuation coincide. -/
theorem fluctuation_dissipation (q : ℕ → ℝ) (b : ℝ) (hQ : pre q (N + 1) = 0) :
    HasDerivAt (fun v => polarization q b v N)
      (b ^ 2 * ∑ k ∈ range (N + 1), (pre q k) ^ 2) 0 := by
  have hZ : HasDerivAt (fun v => Zdip q b v N) (Wdip q b 0 N) 0 := hasDerivAt_Zdip q b 0
  have hW := hasDerivAt_Wdip (N := N) q b 0
  have hW0 : Wdip q b 0 N = 0 := by
    rw [Wdip]
    simpa using sum_dipole_eq_zero q b hQ
  have hZ0 : Zdip q b 0 N = 2 ^ N := Zdip_zero_field q b
  have hne : Zdip q b 0 N ≠ 0 := ne_of_gt (Zdip_pos q b 0)
  have hquot := hW.div hZ hne
  have hsum : ∑ s : Chain N, (dipole (N := N) q b s) ^ 2 * Real.exp (0 * dipole (N := N) q b s)
      = 2 ^ N * (b ^ 2 * ∑ k ∈ range (N + 1), (pre q k) ^ 2) := by
    have hzero : ∑ s : Chain N, (dipole (N := N) q b s) ^ 2 * Real.exp (0 * dipole (N := N) q b s)
        = ∑ s : Chain N, (dipole (N := N) q b s) ^ 2 :=
      Finset.sum_congr rfl fun s _ => by simp
    have hm := mean_dipole_sq (N := N) q b hQ
    rw [chainEns_expect] at hm
    have h2 : (2 : ℝ) ^ N ≠ 0 := by positivity
    rw [hzero]
    field_simp at hm
    linarith
  have hfun : (fun v => polarization q b v N)
      = (fun v => Wdip q b v N) / (fun v => Zdip q b v N) := by
    funext v
    rw [polarization]
    rfl
  rw [hfun]
  convert hquot using 1
  rw [hsum, hW0, hZ0]
  have h2 : (2 : ℝ) ^ N ≠ 0 := by positivity
  field_simp
  ring
