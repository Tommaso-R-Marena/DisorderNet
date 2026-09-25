/-
# Part X.2  Hydrodynamics: what a diffusion measurement returns

Translational diffusion (pulsed-field-gradient NMR, dynamic light scattering, analytical
ultracentrifugation) is the cheapest and most common size measurement on a disordered
region.  What it returns is *not* the radius of gyration and *not* the mean of any radius:
it is the Kirkwood--Stokes hydrodynamic radius of the ensemble, and the ensemble average is
taken on the diffusion coefficient, i.e. on `1/R`.

* `kirkwoodSum`, `hydroRadius` -- the Kirkwood formula `1/R_h = (1/N²) Σ_{i≠j} 1/r_ij` for a
  single conformer, with `kirkwoodSum_translation_invariant` and `kirkwoodSum_perm` : like
  small-angle scattering, hydrodynamics is invariant under rigid motion *and* under
  relabelling the chain.  A diffusion coefficient contains no sequence information at all.
* `kirkwood_cauchy_schwarz` and `hydroRadius_le_pairDist` -- the Kirkwood radius is a
  harmonic-type mean of the pair distances and therefore never exceeds their arithmetic
  mean (up to the exact combinatorial factor `N²/(N²-N)²`).
* `stokesEinstein`, `stokesEinstein_antitone` -- the Stokes--Einstein law and its strict
  monotonicity: `D` determines `R_h` and nothing else.
* `stokesEinstein_ensemble` -- in fast exchange the measured diffusion coefficient is the
  *weight average of the conformer diffusion coefficients*, exactly.  Hence the radius one
  reports is `appRadius w R = (Σ_k w_k/R_k)⁻¹`, a harmonic mean.
* `appRadius_le_mean` and `appRadius_lt_mean_two` -- Jensen, and an explicit strict instance:
  the reported hydrodynamic radius of a heterogeneous ensemble is *smaller* than the mean
  radius of its members, and strictly so as soon as two populated members differ.  A model
  fitted to make a single structure reproduce a measured `R_h` is therefore biased compact,
  by exactly the amount of the heterogeneity it is supposed to describe.
-/
import Mathlib

namespace IDR

open Finset

namespace Hydro

variable {E : Type*} [NormedAddCommGroup E]

/-! ## The Kirkwood radius of one conformer -/

/-- The Kirkwood double sum `Σ_{i≠j} 1/r_ij`. -/
noncomputable def kirkwoodSum {N : ℕ} (r : Fin N → E) : ℝ :=
  ∑ p ∈ Finset.univ.offDiag, ‖r p.1 - r p.2‖⁻¹

/-- The sum of all pair distances `Σ_{i≠j} r_ij`. -/
noncomputable def pairDistSum {N : ℕ} (r : Fin N → E) : ℝ :=
  ∑ p ∈ Finset.univ.offDiag, ‖r p.1 - r p.2‖

/-- The Kirkwood hydrodynamic radius `R_h = N² / Σ_{i≠j} 1/r_ij`. -/
noncomputable def hydroRadius {N : ℕ} (r : Fin N → E) : ℝ := (N : ℝ) ^ 2 / kirkwoodSum r

/-- Hydrodynamics does not see where the molecule is. -/
theorem kirkwoodSum_translation_invariant {N : ℕ} (r : Fin N → E) (t : E) :
    kirkwoodSum (fun i => r i + t) = kirkwoodSum r := by
  simp [kirkwoodSum]

/-- **Hydrodynamics does not see the sequence.**  Relabelling the monomers leaves the
Kirkwood radius unchanged: a diffusion coefficient constrains the unordered multiset of pair
distances and nothing more. -/
theorem kirkwoodSum_perm {N : ℕ} (r : Fin N → E) (sig : Equiv.Perm (Fin N)) :
    kirkwoodSum (r ∘ sig) = kirkwoodSum r := by
  unfold kirkwoodSum
  refine Finset.sum_equiv (Equiv.prodCongr sig sig) (fun p => ?_) (fun p _ => rfl)
  obtain ⟨a, b⟩ := p
  simp only [Finset.mem_offDiag, Finset.mem_univ, true_and, Equiv.prodCongr_apply,
    Prod.map_apply]
  exact ⟨fun h hc => h (sig.injective hc), fun h hc => h (by rw [hc])⟩

lemma norm_pos_of_offDiag {N : ℕ} {r : Fin N → E} (hr : Function.Injective r)
    {p : Fin N × Fin N} (hp : p ∈ Finset.univ.offDiag) : 0 < ‖r p.1 - r p.2‖ := by
  rw [Finset.mem_offDiag] at hp
  have : r p.1 ≠ r p.2 := fun h => hp.2.2 (hr h)
  simpa [sub_eq_zero] using this

lemma kirkwoodSum_pos {N : ℕ} {r : Fin N → E} (hr : Function.Injective r) (hN : 2 ≤ N) :
    0 < kirkwoodSum r := by
  have hne : (Finset.univ.offDiag : Finset (Fin N × Fin N)).Nonempty := by
    have h0 : (⟨0, by omega⟩ : Fin N) ≠ ⟨1, by omega⟩ := by
      intro h
      have := congrArg Fin.val h
      simp at this
    exact ⟨(⟨0, by omega⟩, ⟨1, by omega⟩), by simp [Finset.mem_offDiag, h0]⟩
  refine Finset.sum_pos (fun p hp => ?_) hne
  exact inv_pos.mpr (norm_pos_of_offDiag hr hp)

lemma offDiag_card_cast (N : ℕ) (hN : 1 ≤ N) :
    ((Finset.univ.offDiag : Finset (Fin N × Fin N)).card : ℝ) = (N : ℝ) ^ 2 - N := by
  rw [Finset.offDiag_card]
  simp only [Finset.card_univ, Fintype.card_fin]
  have hle : N ≤ N * N := Nat.le_mul_of_pos_left N hN
  push_cast [Nat.cast_sub hle]
  ring

/-- **The Kirkwood radius is a harmonic mean of the pair distances.**  Cauchy--Schwarz:
`(N²-N)² ≤ (Σ_{i≠j} r_ij)(Σ_{i≠j} 1/r_ij)`. -/
theorem kirkwood_cauchy_schwarz {N : ℕ} {r : Fin N → E} (hr : Function.Injective r)
    (hN : 1 ≤ N) : ((N : ℝ) ^ 2 - N) ^ 2 ≤ pairDistSum r * kirkwoodSum r := by
  set s : Finset (Fin N × Fin N) := Finset.univ.offDiag with hs
  set f : Fin N × Fin N → ℝ := fun p => Real.sqrt ‖r p.1 - r p.2‖ with hf
  set g : Fin N × Fin N → ℝ := fun p => (Real.sqrt ‖r p.1 - r p.2‖)⁻¹ with hg
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq s f g
  have hfg : ∑ p ∈ s, f p * g p = (s.card : ℝ) := by
    have hone : ∀ p ∈ s, f p * g p = 1 := by
      intro p hp
      have hpos : 0 < ‖r p.1 - r p.2‖ := norm_pos_of_offDiag hr hp
      have hsne : Real.sqrt ‖r p.1 - r p.2‖ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hpos)
      simp [hf, hg, mul_inv_cancel₀ hsne]
    rw [Finset.sum_congr rfl hone]
    simp
  have hf2 : ∑ p ∈ s, f p ^ 2 = pairDistSum r := by
    refine Finset.sum_congr rfl fun p _ => ?_
    rw [hf, Real.sq_sqrt (norm_nonneg _)]
  have hg2 : ∑ p ∈ s, g p ^ 2 = kirkwoodSum r := by
    refine Finset.sum_congr rfl fun p _ => ?_
    show (Real.sqrt ‖r p.1 - r p.2‖)⁻¹ ^ 2 = ‖r p.1 - r p.2‖⁻¹
    rw [← Real.sqrt_inv, Real.sq_sqrt (by positivity)]
  rw [hfg, hf2, hg2] at hcs
  rwa [offDiag_card_cast N hN] at hcs

/-- **The Kirkwood radius never exceeds the mean pair distance** (up to the exact
combinatorial factor).  Compactness inferred from diffusion is a lower bound on the size of
the conformer, not an estimate of it. -/
theorem hydroRadius_le_pairDist {N : ℕ} {r : Fin N → E} (hr : Function.Injective r)
    (hN : 2 ≤ N) :
    hydroRadius r * ((N : ℝ) ^ 2 - N) ^ 2 ≤ (N : ℝ) ^ 2 * pairDistSum r := by
  have hK : 0 < kirkwoodSum r := kirkwoodSum_pos hr hN
  have hcs := kirkwood_cauchy_schwarz hr (by omega)
  have : hydroRadius r * ((N : ℝ) ^ 2 - N) ^ 2
      ≤ hydroRadius r * (pairDistSum r * kirkwoodSum r) := by
    have hRh : 0 ≤ hydroRadius r := by
      unfold hydroRadius
      positivity
    exact mul_le_mul_of_nonneg_left hcs hRh
  refine this.trans_eq ?_
  unfold hydroRadius
  field_simp

/-! ## Stokes--Einstein and the ensemble average -/

/-- The Stokes--Einstein diffusion coefficient `D = k_BT/(6πηR_h)`. -/
noncomputable def stokesEinstein (kT eta Rh : ℝ) : ℝ := kT / (6 * Real.pi * eta * Rh)

/-- **A diffusion coefficient is exactly one number about the ensemble.**  `D` is strictly
decreasing in `R_h`, hence determines it uniquely. -/
theorem stokesEinstein_antitone {kT eta : ℝ} (hkT : 0 < kT) (heta : 0 < eta) {R1 R2 : ℝ}
    (hR1 : 0 < R1) (hlt : R1 < R2) :
    stokesEinstein kT eta R2 < stokesEinstein kT eta R1 := by
  have hpi : 0 < Real.pi := Real.pi_pos
  unfold stokesEinstein
  apply div_lt_div_of_pos_left hkT (by positivity)
  have : (0 : ℝ) < 6 * Real.pi * eta := by positivity
  nlinarith

theorem stokesEinstein_injective {kT eta : ℝ} (hkT : 0 < kT) (heta : 0 < eta) {R1 R2 : ℝ}
    (hR1 : 0 < R1) (hR2 : 0 < R2) (h : stokesEinstein kT eta R1 = stokesEinstein kT eta R2) :
    R1 = R2 := by
  rcases lt_trichotomy R1 R2 with hlt | heq | hgt
  · exact absurd h (ne_of_gt (stokesEinstein_antitone hkT heta hR1 hlt))
  · exact heq
  · exact absurd h.symm (ne_of_gt (stokesEinstein_antitone hkT heta hR2 hgt))

variable {m : ℕ}

/-- The apparent hydrodynamic radius reported by a diffusion measurement on a
heterogeneous, fast-exchanging ensemble: the *harmonic* mean `(Σ_k w_k/R_k)⁻¹`. -/
noncomputable def appRadius (w R : Fin m → ℝ) : ℝ := (∑ k, w k / R k)⁻¹

/-- **In fast exchange the measurement averages the diffusion coefficient, not the radius.**
An exact identity: the Stokes--Einstein coefficient of the apparent radius is the weight
average of the conformers' coefficients. -/
theorem stokesEinstein_ensemble {kT eta : ℝ} (heta : 0 < eta) (w R : Fin m → ℝ) :
    stokesEinstein kT eta (appRadius w R) = ∑ k, w k * stokesEinstein kT eta (R k) := by
  have hpi : 0 < Real.pi := Real.pi_pos
  have hc : (6 * Real.pi * eta) ≠ 0 := by positivity
  unfold stokesEinstein appRadius
  have hrhs : ∑ k, w k * (kT / (6 * Real.pi * eta * R k))
      = kT / (6 * Real.pi * eta) * ∑ k, w k / R k := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    rcases eq_or_ne (R k) 0 with h | h
    · simp [h]
    · field_simp
  rw [hrhs]
  rcases eq_or_ne (∑ k, w k / R k) 0 with h | h
  · simp [h]
  · field_simp

/-- **The reported hydrodynamic radius is biased compact.**  Jensen for the convex function
`x ↦ 1/x`: the harmonic mean of the conformer radii never exceeds their weight average. -/
theorem appRadius_le_mean {w R : Fin m → ℝ} (hw : ∀ k, 0 ≤ w k) (hsum : ∑ k, w k = 1)
    (hR : ∀ k, 0 < R k) : appRadius w R ≤ ∑ k, w k * R k := by
  obtain ⟨k0, hk0⟩ : ∃ k, 0 < w k := by
    by_contra hcon
    push_neg at hcon
    have : ∑ k, w k ≤ 0 :=
      Finset.sum_nonpos fun k _ => hcon k
    rw [hsum] at this
    linarith
  have hmeanpos : 0 < ∑ k, w k * R k :=
    lt_of_lt_of_le (mul_pos hk0 (hR k0)) (Finset.single_le_sum
      (fun k _ => mul_nonneg (hw k) (hR k).le) (Finset.mem_univ k0))
  have hconv : ConvexOn ℝ (Set.Ioi (0 : ℝ)) (fun x : ℝ => x ^ (-1 : ℤ)) := convexOn_zpow (-1)
  have hjensen : (∑ k, w k * R k) ^ (-1 : ℤ) ≤ ∑ k, w k * (R k) ^ (-1 : ℤ) := by
    have hcm := hconv.map_centerMass_le (t := Finset.univ) (w := w) (p := R)
      (fun k _ => hw k) (by rw [hsum]; norm_num) (fun k _ => Set.mem_Ioi.mpr (hR k))
    simpa [Finset.centerMass, hsum, smul_eq_mul, Function.comp] using hcm
  have hsum' : ∑ k, w k * (R k) ^ (-1 : ℤ) = ∑ k, w k / R k := by
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [zpow_neg_one, div_eq_mul_inv]
  rw [hsum'] at hjensen
  have hinvpos : 0 < (∑ k, w k * R k) ^ (-1 : ℤ) := zpow_pos hmeanpos _
  have := inv_anti₀ hinvpos hjensen
  rwa [zpow_neg_one, inv_inv] at this

/-- **Strictly compact, explicitly.**  A 50:50 ensemble of two different radii reports a
hydrodynamic radius strictly below their mean.  There is no ensemble-width correction that
can be absorbed into a single structure. -/
theorem appRadius_lt_mean_two {R1 R2 : ℝ} (h1 : 0 < R1) (h2 : 0 < R2) (hne : R1 ≠ R2) :
    appRadius ![1 / 2, 1 / 2] ![R1, R2] < ∑ k, (![1 / 2, 1 / 2] : Fin 2 → ℝ) k * ![R1, R2] k := by
  have hsum : (0 : ℝ) < 1 / 2 / R1 + 1 / 2 / R2 := by positivity
  have hkey : 2 * R1 * R2 / (R1 + R2) < (R1 + R2) / 2 := by
    rw [div_lt_div_iff₀ (by linarith) (by norm_num)]
    have hsub : R1 - R2 ≠ 0 := sub_ne_zero.mpr hne
    have hpos : 0 < (R1 - R2) ^ 2 := by positivity
    nlinarith [hpos]
  have happ : appRadius ![1 / 2, 1 / 2] ![R1, R2] = 2 * R1 * R2 / (R1 + R2) := by
    unfold appRadius
    rw [Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [show (1 : ℝ) / 2 / R1 + 1 / 2 / R2 = (R1 + R2) / (2 * R1 * R2) by field_simp; ring]
    rw [inv_div]
  rw [happ, Fin.sum_univ_two]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
  calc 2 * R1 * R2 / (R1 + R2) < (R1 + R2) / 2 := hkey
    _ = 1 / 2 * R1 + 1 / 2 * R2 := by ring

end Hydro

end IDR
