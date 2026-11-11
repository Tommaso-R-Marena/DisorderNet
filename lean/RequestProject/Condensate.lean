/-
# Part VII.1  Collective behaviour: when does a disordered protein demix?

Parts I--VI treat a disordered region one chain at a time: the object to be modelled is a
distribution over the conformations of a single molecule, in a fixed context.  The most
conspicuous biology of intrinsically disordered regions is not of that kind.  Multivalent
disordered proteins *condense*: above a threshold the homogeneous solution splits into a
dilute and a dense phase.  This file asks what a model must contain in order to predict
that, and proves that the answer is exactly the part of the free energy which is **not**
affine in the concentration -- so that no amount of single-chain information, however
accurate, can decide the question.

* `PhaseSeparates` -- a system whose free-energy density is `f` demixes at overall
  composition `c` when splitting into two coexisting phases of different compositions
  strictly lowers the total free energy (the compositions mixing back to `c`: mass is
  conserved).
* `not_phaseSeparates_of_convexOn` and `exists_phaseSeparates_iff_not_convexOn` --
  **demixing is exactly non-convexity of the free-energy density.**  A convex free energy
  never demixes, and conversely any failure of convexity *is* a demixing composition.  This
  is the common-tangent construction, stated as an equivalence.
* `lever_rule` -- for coexisting compositions `c₁ < c₂` and any overall `c` between them the
  phase fractions are forced: `t = (c₂ - c)/(c₂ - c₁)`.
* `phaseSeparates_add_affine` -- **the phase diagram is invariant under adding any affine
  function of the concentration.**  The single-chain free energy enters the free-energy
  density multiplied by the concentration, i.e. affinely; hence
  `chain_free_energy_blind_to_demixing`: two systems with exactly the same single-chain
  thermodynamics, differing only in their interchain coupling, can sit on opposite sides of
  the phase boundary.  Condensation is a property of the interaction, not of the ensemble
  of the isolated chain.
* `floryFE`, `floryFE_convexOn`, `no_demixing_of_weak_coupling`, `flory_demixes` -- the
  Flory--Huggins free-energy density made explicit: for coupling `chi ≤ 2` it is convex, so
  the solution is stable at every composition, while at `chi = 4` the half-filled solution
  demixes.  Both bounds are proved, the first from the second derivative
  `1/c + 1/(1-c) - 2·chi ≥ 4 - 2·chi`, the second from `log 2 < 1`.
-/
import Mathlib
import RequestProject.FreeEnergy

namespace IDR

namespace Phase

open Set

/-- **Demixing.**  A system whose homogeneous free-energy density is `f`, with compositions
constrained to `s`, phase separates at overall composition `c` when there are two distinct
compositions `c₁ ≠ c₂` in `s` and a phase fraction `t ∈ (0,1)` which conserve the material,
`c = t·c₁ + (1-t)·c₂`, and whose total free energy is strictly lower than that of the
homogeneous state. -/
def PhaseSeparates (s : Set ℝ) (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∃ c₁ c₂ t : ℝ, c₁ ∈ s ∧ c₂ ∈ s ∧ 0 < t ∧ t < 1 ∧ c₁ ≠ c₂ ∧
    c = t * c₁ + (1 - t) * c₂ ∧ t * f c₁ + (1 - t) * f c₂ < f c

/-- **A convex free-energy density never demixes**: the homogeneous state is the best one at
every composition. -/
theorem not_phaseSeparates_of_convexOn {s : Set ℝ} {f : ℝ → ℝ} (hf : ConvexOn ℝ s f) (c : ℝ) :
    ¬ PhaseSeparates s f c := by
  rintro ⟨c₁, c₂, t, hc₁, hc₂, ht0, ht1, -, hc, hlt⟩
  have h := hf.2 hc₁ hc₂ ht0.le (by linarith : (0:ℝ) ≤ 1 - t) (by ring)
  simp only [smul_eq_mul] at h
  rw [hc] at hlt
  linarith

/-- **Demixing is exactly the failure of convexity.**  Some composition phase separates if
and only if the free-energy density is not convex; this is the common-tangent construction
read as an equivalence. -/
theorem exists_phaseSeparates_iff_not_convexOn {s : Set ℝ} (hs : Convex ℝ s) (f : ℝ → ℝ) :
    (∃ c, PhaseSeparates s f c) ↔ ¬ ConvexOn ℝ s f := by
  constructor
  · rintro ⟨c, hc⟩ hconv
    exact not_phaseSeparates_of_convexOn hconv c hc
  · intro h
    rw [ConvexOn] at h
    push_neg at h
    obtain ⟨x, hx, y, hy, a, b, ha, hb, hab, hlt⟩ := h hs
    simp only [smul_eq_mul] at hlt
    have hb' : b = 1 - a := by linarith
    subst hb'
    have ha0 : 0 < a := by
      rcases ha.lt_or_eq with h' | h'
      · exact h'
      · exfalso; rw [← h'] at hlt; simp at hlt
    have ha1 : a < 1 := by
      rcases hb.lt_or_eq with h' | h'
      · linarith
      · exfalso
        have hA : a = 1 := by linarith
        rw [hA] at hlt; simp at hlt
    have hxy : x ≠ y := by
      rintro rfl
      have : a * x + (1 - a) * x = x := by ring
      rw [this] at hlt
      nlinarith
    exact ⟨a * x + (1 - a) * y, x, y, a, hx, hy, ha0, ha1, hxy, rfl, hlt⟩

/-- **The lever rule.**  Once the two coexisting compositions are fixed, the amount of each
phase is determined by the overall composition: no further modelling freedom remains. -/
theorem lever_rule {c₁ c₂ c : ℝ} (h : c₁ < c₂) (hc : c ∈ Icc c₁ c₂) :
    ∃ t : ℝ, t ∈ Icc (0:ℝ) 1 ∧ t = (c₂ - c) / (c₂ - c₁) ∧ c = t * c₁ + (1 - t) * c₂ := by
  have hpos : 0 < c₂ - c₁ := by linarith
  refine ⟨(c₂ - c) / (c₂ - c₁), ⟨?_, ?_⟩, rfl, ?_⟩
  · exact div_nonneg (by linarith [hc.2]) hpos.le
  · rw [div_le_one hpos]; linarith [hc.1]
  · field_simp
    ring

/-- **The phase diagram is blind to any affine term.**  Adding `a·c + b` to the free-energy
density -- in particular the free energy of the isolated chain, which enters proportionally
to the concentration -- changes no demixing statement whatsoever. -/
theorem phaseSeparates_add_affine {s : Set ℝ} (f : ℝ → ℝ) (a b c : ℝ) :
    PhaseSeparates s (fun x => f x + (a * x + b)) c ↔ PhaseSeparates s f c := by
  constructor
  · rintro ⟨c₁, c₂, t, hc₁, hc₂, ht0, ht1, hne, hmass, hlt⟩
    refine ⟨c₁, c₂, t, hc₁, hc₂, ht0, ht1, hne, hmass, ?_⟩
    simp only at hlt
    rw [hmass] at hlt ⊢
    nlinarith [hlt]
  · rintro ⟨c₁, c₂, t, hc₁, hc₂, ht0, ht1, hne, hmass, hlt⟩
    refine ⟨c₁, c₂, t, hc₁, hc₂, ht0, ht1, hne, hmass, ?_⟩
    simp only
    rw [hmass] at hlt ⊢
    nlinarith [hlt]

/-- The Flory--Huggins free-energy density of a solution at volume fraction `c`: the mixing
entropy `c log c + (1-c) log (1-c)` plus an interchain coupling `chi · c(1-c)`. -/
noncomputable def floryFE (chi c : ℝ) : ℝ :=
  c * Real.log c + (1 - c) * Real.log (1 - c) + chi * (c * (1 - c))

/-- The derivative of the Flory--Huggins density on the open interval of compositions:
the exchange chemical potential. -/
theorem floryFE_hasDerivAt (chi x : ℝ) (hx : x ∈ Ioo (0:ℝ) 1) :
    HasDerivAt (floryFE chi) (Real.log x - Real.log (1 - x) + chi * (1 - 2 * x)) x := by
  obtain ⟨hx0, hx1⟩ := hx
  have hne : (1:ℝ) - x ≠ 0 := by linarith
  have h1 : HasDerivAt (fun c : ℝ => c * Real.log c) (Real.log x + 1) x := by
    have := (hasDerivAt_id x).mul (Real.hasDerivAt_log hx0.ne')
    convert this using 1
    simp only [id_eq]
    field_simp
  have hlin : HasDerivAt (fun c : ℝ => 1 - c) (-1) x := by
    simpa using (hasDerivAt_const x (1:ℝ)).sub (hasDerivAt_id x)
  have h2 : HasDerivAt (fun c : ℝ => (1 - c) * Real.log (1 - c))
      ((Real.log (1 - x) + 1) * (-1)) x := by
    have hb : HasDerivAt (fun u : ℝ => u * Real.log u) (Real.log (1 - x) + 1) (1 - x) := by
      have := (hasDerivAt_id (1 - x)).mul (Real.hasDerivAt_log hne)
      convert this using 1
      simp only [id_eq]
      field_simp
    exact hb.comp x hlin
  have h3 : HasDerivAt (fun c : ℝ => chi * (c * (1 - c))) (chi * (1 - 2 * x)) x := by
    have h : HasDerivAt (fun c : ℝ => c * (1 - c)) (1 * (1 - x) + x * (-1)) x :=
      (hasDerivAt_id x).mul hlin
    have := h.const_mul chi
    convert this using 1
    ring
  have := (h1.add h2).add h3
  convert this using 1
  ring

/-- The second derivative of the Flory--Huggins density: `1/c + 1/(1-c) - 2·chi`, the inverse
osmotic compressibility. -/
theorem floryFE_hasDerivAt2 (chi x : ℝ) (hx : x ∈ Ioo (0:ℝ) 1) :
    HasDerivAt (fun c => Real.log c - Real.log (1 - c) + chi * (1 - 2 * c))
      (1 / x + 1 / (1 - x) - 2 * chi) x := by
  obtain ⟨hx0, hx1⟩ := hx
  have hne : (1:ℝ) - x ≠ 0 := by linarith
  have hlin : HasDerivAt (fun c : ℝ => 1 - c) (-1) x := by
    simpa using (hasDerivAt_const x (1:ℝ)).sub (hasDerivAt_id x)
  have h1 : HasDerivAt (fun c : ℝ => Real.log c) x⁻¹ x := Real.hasDerivAt_log hx0.ne'
  have h2 : HasDerivAt (fun c : ℝ => Real.log (1 - c)) ((1 - x)⁻¹ * (-1)) x :=
    (Real.hasDerivAt_log hne).comp x hlin
  have h3 : HasDerivAt (fun c : ℝ => chi * (1 - 2 * c)) (chi * (-2)) x := by
    have h : HasDerivAt (fun c : ℝ => 1 - 2 * c) (-2 : ℝ) x := by
      simpa using (hasDerivAt_const x (1:ℝ)).sub ((hasDerivAt_id x).const_mul (2:ℝ))
    simpa using h.const_mul chi
  have := (h1.sub h2).add h3
  convert this using 1
  field_simp
  ring

/-- The Flory--Huggins density is continuous on the whole line (the mixing entropy extends
continuously to the pure phases). -/
theorem floryFE_continuous (chi : ℝ) : Continuous (floryFE chi) := by
  have h1 : Continuous fun c : ℝ => c * Real.log c := Real.continuous_mul_log
  have h2 : Continuous fun c : ℝ => (1 - c) * Real.log (1 - c) :=
    Real.continuous_mul_log.comp (by fun_prop)
  exact (h1.add h2).add (by fun_prop)

/-- **Below the critical coupling the free-energy density is convex.**  The mixing entropy
contributes `1/c + 1/(1-c) ≥ 4` to the curvature and the coupling `-2·chi`. -/
theorem floryFE_convexOn (chi : ℝ) (hchi : chi ≤ 2) : ConvexOn ℝ (Icc (0:ℝ) 1) (floryFE chi) := by
  have hint : interior (Icc (0:ℝ) 1) = Ioo (0:ℝ) 1 := interior_Icc
  refine convexOn_of_hasDerivWithinAt2_nonneg (convex_Icc 0 1)
    (f' := fun c => Real.log c - Real.log (1 - c) + chi * (1 - 2 * c))
    (f'' := fun c => 1 / c + 1 / (1 - c) - 2 * chi)
    (floryFE_continuous chi).continuousOn ?_ ?_ ?_
  · intro x hx
    rw [hint] at hx
    exact (floryFE_hasDerivAt chi x hx).hasDerivWithinAt
  · intro x hx
    rw [hint] at hx
    exact (floryFE_hasDerivAt2 chi x hx).hasDerivWithinAt
  · intro x hx
    rw [hint] at hx
    obtain ⟨hx0, hx1⟩ := hx
    have h : 4 ≤ 1 / x + 1 / (1 - x) := by
      rw [div_add_div _ _ (ne_of_gt hx0) (by linarith), le_div_iff₀ (by nlinarith)]
      nlinarith [sq_nonneg (1 - 2 * x)]
    linarith

/-- **A weakly coupled solution of disordered chains is stable at every composition**: no
condensate forms below the critical coupling, whatever the chains' internal statistics. -/
theorem no_demixing_of_weak_coupling {chi : ℝ} (hchi : chi ≤ 2) (c : ℝ) :
    ¬ PhaseSeparates (Icc (0:ℝ) 1) (floryFE chi) c :=
  not_phaseSeparates_of_convexOn (floryFE_convexOn chi hchi) c

@[simp] theorem floryFE_zero (chi : ℝ) : floryFE chi 0 = 0 := by
  simp [floryFE]

@[simp] theorem floryFE_one (chi : ℝ) : floryFE chi 1 = 0 := by
  simp [floryFE]

/-- **Above the critical coupling the solution demixes.**  At `chi = 4` the half-filled
homogeneous solution has free energy `1 - log 2 > 0`, while the demixed state costs nothing:
the system condenses. -/
theorem flory_demixes : PhaseSeparates (Icc (0:ℝ) 1) (floryFE 4) (1/2) := by
  refine ⟨0, 1, 1/2, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, ?_⟩
  have hmid : floryFE 4 (1/2) = 1 - Real.log 2 := by
    have h : Real.log (1/2 : ℝ) = - Real.log 2 := by
      rw [one_div, Real.log_inv]
    simp only [floryFE]
    rw [show (1 : ℝ) - 1/2 = 1/2 by norm_num, h]
    ring
  have hlog : Real.log 2 < 1 := by nlinarith [Real.log_two_lt_d9]
  rw [hmid]
  simp only [floryFE_zero, floryFE_one]
  linarith

/-- **The single-chain ensemble cannot decide whether a condensate forms.**  Whatever affine
contribution `a·c + b` the isolated chain makes to the free-energy density -- and its
contribution *is* affine, being proportional to the amount of material -- the demixing
behaviour is decided by the interchain coupling alone: at `chi = 4` the solution demixes and
at any `chi ≤ 2` it does not, with the same chain term in both. -/
theorem chain_free_energy_blind_to_demixing (a b : ℝ) {chi : ℝ} (hchi : chi ≤ 2) :
    PhaseSeparates (Icc (0:ℝ) 1) (fun c => floryFE 4 c + (a * c + b)) (1/2) ∧
      ∀ c, ¬ PhaseSeparates (Icc (0:ℝ) 1) (fun x => floryFE chi x + (a * x + b)) c := by
  refine ⟨(phaseSeparates_add_affine (floryFE 4) a b (1/2)).2 flory_demixes, fun c hc => ?_⟩
  exact no_demixing_of_weak_coupling hchi c ((phaseSeparates_add_affine _ a b c).1 hc)

end Phase

end IDR
