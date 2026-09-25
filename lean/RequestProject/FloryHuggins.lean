/-
# Part LXXV  Chain length and the critical point of a condensate

Part VII.1 treats demixing with the symmetric Flory--Huggins density, in which solute and solvent
occupy equally many lattice sites.  A disordered region is not symmetric with the solvent: it is a
chain of `N` segments, and that asymmetry is the whole reason biological condensates form at
micromolar concentrations rather than at volume fraction one half.  This file carries the chain
length through.

The free-energy density at segment volume fraction `phi` for chains of length `N` is

  `f(phi) = (phi/N) log phi + (1-phi) log(1-phi) + chi phi (1-phi)`,

the mixing entropy of the chain being reduced by the factor `1/N` because `N` segments move
together.  The results:

* `fh_hasDerivAt`, `fh_hasDerivAt2` -- the exchange chemical potential and the curvature
  `1/(N phi) + 1/(1-phi) - 2 chi`, the inverse osmotic compressibility.
* `curvature_min` -- **the critical point, exactly.**  The entropic curvature
  `1/(N phi) + 1/(1-phi)` is minimised over `(0,1)` at `phiC N = 1/(1 + sqrt N)` with value
  `2 chiC N`, where `chiC N = (1 + sqrt N)^2 / (2N)`.  The proof is the identity
  `1/(N phi) + 1/(1-phi) - 2 chiC N = ((1 + sqrt N) phi - 1)^2 / (N phi (1-phi))`: a perfect
  square, so the minimum and the minimiser come out together.
* `fh_convexOn`, `no_demixing_below_chiC` -- below `chiC N` the density is convex, so the
  solution is stable at every composition.
* `fh_demixes_above_chiC` -- above `chiC N` it is strictly concave near `phiC N`, and an
  explicit pair of coexisting compositions with equal phase fractions beats the homogeneous
  state.  `chiC N` is therefore the exact critical coupling, not a bound.
* `chiC_strictAnti`, `chiC_gt_half`, `chiC_tendsto_half`, `phiC_tendsto_zero` -- the biology.
  The critical coupling *decreases* strictly with chain length and tends to `1/2` from above,
  while the critical volume fraction tends to `0` like `1/sqrt N`: long disordered chains
  condense at weak coupling and at vanishing concentration, and the same interaction that
  leaves short peptides mixed demixes a long region.
* `fh_one`, `chiC_one` -- for `N = 1` the density is exactly the symmetric one of Part VII.1
  and `chiC 1 = 2`, its threshold; the present part strictly contains it.

The design consequence is that a model of an intrinsically disordered region cannot report a
coupling strength without reporting the length it was fitted at: `chi` and `N` enter the phase
boundary only through the combination `2 chi N/(1 + sqrt N)^2`, and the same `chi` means
"stable" at one length and "condensed" at another.
-/
import Mathlib
import RequestProject.Condensate

set_option autoImplicit false

namespace IDR

namespace FH

open Set IDR.Phase

/-- The Flory--Huggins free-energy density for chains of `N` segments at segment volume
fraction `phi`. -/
noncomputable def fh (N chi phi : ℝ) : ℝ :=
  (phi / N) * Real.log phi + (1 - phi) * Real.log (1 - phi) + chi * (phi * (1 - phi))

/-- The exchange chemical potential: the first derivative of `fh` in `phi`. -/
noncomputable def fh' (N chi phi : ℝ) : ℝ :=
  Real.log phi / N + 1 / N - Real.log (1 - phi) - 1 + chi * (1 - 2 * phi)

/-- The curvature of `fh`: the inverse osmotic compressibility. -/
noncomputable def curvature (N chi phi : ℝ) : ℝ := 1 / (N * phi) + 1 / (1 - phi) - 2 * chi

/-- The critical volume fraction `1/(1 + sqrt N)`. -/
noncomputable def phiC (N : ℝ) : ℝ := 1 / (1 + Real.sqrt N)

/-- The critical coupling `(1 + sqrt N)^2/(2N)`. -/
noncomputable def chiC (N : ℝ) : ℝ := (1 + Real.sqrt N) ^ 2 / (2 * N)

/-! ## Derivatives -/

theorem fh_hasDerivAt {N : ℝ} (hN : N ≠ 0) (chi : ℝ) {x : ℝ} (hx : x ∈ Ioo (0 : ℝ) 1) :
    HasDerivAt (fh N chi) (fh' N chi x) x := by
  obtain ⟨hx0, hx1⟩ := hx
  have hne : (1 : ℝ) - x ≠ 0 := by linarith
  have h1 : HasDerivAt (fun c : ℝ => (c / N) * Real.log c)
      ((1 / N) * Real.log x + (x / N) * (1 / x)) x := by
    have hd : HasDerivAt (fun c : ℝ => c / N) (1 / N) x := by
      simpa using (hasDerivAt_id x).div_const N
    have := hd.mul (Real.hasDerivAt_log hx0.ne')
    convert this using 1
    field_simp
  have hlin : HasDerivAt (fun c : ℝ => 1 - c) (-1) x := by
    simpa using (hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_id x)
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
  rw [fh']
  field_simp
  ring

theorem fh_hasDerivAt2 {N : ℝ} (hN : N ≠ 0) (chi : ℝ) {x : ℝ} (hx : x ∈ Ioo (0 : ℝ) 1) :
    HasDerivAt (fh' N chi) (curvature N chi x) x := by
  obtain ⟨hx0, hx1⟩ := hx
  have hne : (1 : ℝ) - x ≠ 0 := by linarith
  have hlin : HasDerivAt (fun c : ℝ => 1 - c) (-1) x := by
    simpa using (hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_id x)
  have h1 : HasDerivAt (fun c : ℝ => Real.log c / N) (x⁻¹ / N) x :=
    (Real.hasDerivAt_log hx0.ne').div_const N
  have h2 : HasDerivAt (fun c : ℝ => Real.log (1 - c)) ((1 - x)⁻¹ * (-1)) x :=
    (Real.hasDerivAt_log hne).comp x hlin
  have h3 : HasDerivAt (fun c : ℝ => chi * (1 - 2 * c)) (chi * (-2)) x := by
    have h : HasDerivAt (fun c : ℝ => 1 - 2 * c) (-2 : ℝ) x := by
      simpa using (hasDerivAt_const x (1 : ℝ)).sub ((hasDerivAt_id x).const_mul (2 : ℝ))
    simpa using h.const_mul chi
  have hsum := (((h1.add_const (1 / N)).sub h2).sub_const 1).add h3
  convert hsum using 1
  rw [curvature]
  field_simp
  ring

/-! ## The critical point -/

theorem sqrt_pos_of {N : ℝ} (hN : 0 < N) : 0 < Real.sqrt N := Real.sqrt_pos.mpr hN

theorem phiC_mem {N : ℝ} (hN : 0 < N) : phiC N ∈ Ioo (0 : ℝ) 1 := by
  have hs := sqrt_pos_of hN
  constructor
  · rw [phiC]; positivity
  · rw [phiC, div_lt_one (by linarith)]
    linarith

/-- **The exact spinodal identity.**  The entropic curvature exceeds `2 chiC N` by a perfect
square: `1/(N phi) + 1/(1-phi) - 2 chiC N = ((1 + sqrt N) phi - 1)^2 / (N phi (1-phi))`. -/
theorem curvature_identity {N : ℝ} (hN : 0 < N) {phi : ℝ} (h0 : 0 < phi) (h1 : phi < 1) :
    1 / (N * phi) + 1 / (1 - phi) - 2 * chiC N
      = ((1 + Real.sqrt N) * phi - 1) ^ 2 / (N * phi * (1 - phi)) := by
  have hs := sqrt_pos_of hN
  have hsq : Real.sqrt N ^ 2 = N := Real.sq_sqrt hN.le
  have h1' : (0 : ℝ) < 1 - phi := by linarith
  rw [chiC]
  field_simp
  nlinarith [hsq, h0, h1', hN]

/-- **The critical point.**  `1/(N phi) + 1/(1-phi) >= 2 chiC N` on `(0,1)`. -/
theorem curvature_min {N : ℝ} (hN : 0 < N) {phi : ℝ} (h0 : 0 < phi) (h1 : phi < 1) :
    2 * chiC N ≤ 1 / (N * phi) + 1 / (1 - phi) := by
  have hid := curvature_identity hN h0 h1
  have hpos : (0 : ℝ) < N * phi * (1 - phi) := by
    have : (0 : ℝ) < 1 - phi := by linarith
    positivity
  have : 0 ≤ ((1 + Real.sqrt N) * phi - 1) ^ 2 / (N * phi * (1 - phi)) :=
    div_nonneg (sq_nonneg _) hpos.le
  linarith

/-- And the minimum is attained at `phiC N`: the critical composition. -/
theorem curvature_at_phiC {N : ℝ} (hN : 0 < N) :
    1 / (N * phiC N) + 1 / (1 - phiC N) = 2 * chiC N := by
  have hs := sqrt_pos_of hN
  have hmem := phiC_mem hN
  have hid := curvature_identity hN hmem.1 hmem.2
  have hzero : (1 + Real.sqrt N) * phiC N - 1 = 0 := by
    rw [phiC]
    field_simp
    ring
  rw [hzero] at hid
  have hz : ((0 : ℝ)) ^ 2 / (N * phiC N * (1 - phiC N)) = 0 := by simp
  rw [hz] at hid
  linarith

/-! ## Convexity below the critical coupling -/

theorem fh_continuous {N : ℝ} (hN : N ≠ 0) (chi : ℝ) : Continuous (fh N chi) := by
  have h1 : Continuous fun c : ℝ => (c / N) * Real.log c := by
    have : (fun c : ℝ => (c / N) * Real.log c) = fun c : ℝ => (1 / N) * (c * Real.log c) := by
      funext c; field_simp
    rw [this]
    exact Real.continuous_mul_log.const_smul (1 / N)
  have h2 : Continuous fun c : ℝ => (1 - c) * Real.log (1 - c) :=
    Real.continuous_mul_log.comp (by fun_prop)
  exact (h1.add h2).add (by fun_prop)

/-- **Below the critical coupling the free-energy density is convex**, so the solution is stable
at every composition. -/
theorem fh_convexOn {N chi : ℝ} (hN : 0 < N) (hchi : chi ≤ chiC N) :
    ConvexOn ℝ (Icc (0 : ℝ) 1) (fh N chi) := by
  have hint : interior (Icc (0 : ℝ) 1) = Ioo (0 : ℝ) 1 := interior_Icc
  refine convexOn_of_hasDerivWithinAt2_nonneg (convex_Icc 0 1)
    (f' := fh' N chi) (f'' := curvature N chi)
    (fh_continuous hN.ne' chi).continuousOn ?_ ?_ ?_
  · intro x hx
    rw [hint] at hx
    exact (fh_hasDerivAt hN.ne' chi hx).hasDerivWithinAt
  · intro x hx
    rw [hint] at hx
    exact (fh_hasDerivAt2 hN.ne' chi hx).hasDerivWithinAt
  · intro x hx
    rw [hint] at hx
    have := curvature_min hN hx.1 hx.2
    rw [curvature]
    linarith

/-- No condensate below the critical coupling, at any composition. -/
theorem no_demixing_below_chiC {N chi : ℝ} (hN : 0 < N) (hchi : chi ≤ chiC N) (c : ℝ) :
    ¬ PhaseSeparates (Icc (0 : ℝ) 1) (fh N chi) c :=
  not_phaseSeparates_of_convexOn (fh_convexOn hN hchi) c

/-! ## Demixing above the critical coupling -/

/-- **Above the critical coupling the solution demixes.**  The curvature is negative at `phiC N`,
hence on a neighbourhood, and the two compositions symmetric about `phiC N` in that neighbourhood
beat the homogeneous state in equal proportion. -/
theorem fh_demixes_above_chiC {N chi : ℝ} (hN : 0 < N) (hchi : chiC N < chi) :
    ∃ c, PhaseSeparates (Icc (0 : ℝ) 1) (fh N chi) c := by
  have hmem := phiC_mem hN
  -- the curvature is a continuous function of the composition on `(0,1)`
  have hcont : ContinuousAt (fun y : ℝ => curvature N chi y) (phiC N) := by
    have h1 : ContinuousAt (fun y : ℝ => 1 / (N * y)) (phiC N) := by
      apply ContinuousAt.div continuousAt_const (by fun_prop)
      have : (0 : ℝ) < N * phiC N := mul_pos hN hmem.1
      exact this.ne'
    have h2 : ContinuousAt (fun y : ℝ => 1 / (1 - y)) (phiC N) := by
      apply ContinuousAt.div continuousAt_const (by fun_prop)
      have : (0 : ℝ) < 1 - phiC N := by linarith [hmem.2]
      exact this.ne'
    exact (h1.add h2).sub continuousAt_const
  have hneg : curvature N chi (phiC N) < 0 := by
    rw [curvature, curvature_at_phiC hN]
    linarith
  have hev : ∀ᶠ y in nhds (phiC N), curvature N chi y < 0 := hcont (gt_mem_nhds hneg)
  have hev2 : ∀ᶠ y in nhds (phiC N), y ∈ Ioo (0 : ℝ) 1 :=
    (isOpen_Ioo.mem_nhds hmem)
  obtain ⟨eps, heps, hball⟩ := Metric.mem_nhds_iff.mp (hev.and hev2)
  set a : ℝ := phiC N - eps / 2 with ha
  set b : ℝ := phiC N + eps / 2 with hb
  have hmemball : ∀ y ∈ Icc a b, curvature N chi y < 0 ∧ y ∈ Ioo (0 : ℝ) 1 := by
    intro y hy
    refine hball ?_
    rw [Metric.mem_ball, Real.dist_eq, abs_lt]
    constructor <;> [linarith [hy.1]; linarith [hy.2]]
  have hab : a < b := by simp only [ha, hb]; linarith
  -- on `[a,b]` the density is strictly concave
  have hderiv1 : ∀ y ∈ Ioo a b, HasDerivAt (fh N chi) (fh' N chi y) y := by
    intro y hy
    exact fh_hasDerivAt hN.ne' chi (hmemball y ⟨hy.1.le, hy.2.le⟩).2
  have hderiv2 : ∀ y ∈ Ioo a b, HasDerivAt (fh' N chi) (curvature N chi y) y := by
    intro y hy
    exact fh_hasDerivAt2 hN.ne' chi (hmemball y ⟨hy.1.le, hy.2.le⟩).2
  have hderiv_eq : ∀ y ∈ Ioo a b, deriv (fh N chi) y = fh' N chi y :=
    fun y hy => (hderiv1 y hy).deriv
  have hd2 : ∀ y ∈ interior (Icc a b), deriv^[2] (fh N chi) y < 0 := by
    intro y hy
    rw [interior_Icc] at hy
    have heq : deriv (fh N chi) =ᶠ[nhds y] fh' N chi := by
      filter_upwards [isOpen_Ioo.mem_nhds hy] with z hz using hderiv_eq z hz
    have : deriv (deriv (fh N chi)) y = curvature N chi y := by
      rw [heq.deriv_eq, (hderiv2 y hy).deriv]
    simpa [Function.iterate_succ, Function.comp] using
      this ▸ (hmemball y ⟨hy.1.le, hy.2.le⟩).1
  have hconc : StrictConcaveOn ℝ (Icc a b) (fh N chi) :=
    strictConcaveOn_of_deriv2_neg (convex_Icc a b)
      (fh_continuous hN.ne' chi).continuousOn hd2
  -- the midpoint beats the two ends
  have hmid := hconc.2 (left_mem_Icc.mpr hab.le) (right_mem_Icc.mpr hab.le) (ne_of_lt hab)
    (by norm_num : (0:ℝ) < 1/2) (by norm_num : (0:ℝ) < 1/2) (by norm_num)
  simp only [smul_eq_mul] at hmid
  refine ⟨(1/2 : ℝ) * a + (1 - 1/2) * b, a, b, 1/2, ?_, ?_, by norm_num, by norm_num,
    ne_of_lt hab, by ring, ?_⟩
  · exact ⟨(hmemball a (left_mem_Icc.mpr hab.le)).2.1.le,
      (hmemball a (left_mem_Icc.mpr hab.le)).2.2.le⟩
  · exact ⟨(hmemball b (right_mem_Icc.mpr hab.le)).2.1.le,
      (hmemball b (right_mem_Icc.mpr hab.le)).2.2.le⟩
  · have : (1/2 : ℝ) * a + (1 - 1/2) * b = 1/2 * a + 1/2 * b := by ring
    rw [this]
    linarith [hmid]

/-! ## The chain-length dependence -/

theorem chiC_one : chiC 1 = 2 := by
  rw [chiC, Real.sqrt_one]
  norm_num

theorem fh_one (chi : ℝ) : fh 1 chi = floryFE chi := by
  funext phi
  rw [fh, floryFE]
  norm_num

/-- **Longer chains condense at weaker coupling.**  `chiC` is strictly decreasing in `N`. -/
theorem chiC_strictAnti {N M : ℝ} (hN : 0 < N) (hNM : N < M) : chiC M < chiC N := by
  have hM : 0 < M := lt_trans hN hNM
  have hsN : 0 < Real.sqrt N := sqrt_pos_of hN
  have hsM : 0 < Real.sqrt M := sqrt_pos_of hM
  have hsqN : Real.sqrt N ^ 2 = N := Real.sq_sqrt hN.le
  have hsqM : Real.sqrt M ^ 2 = M := Real.sq_sqrt hM.le
  have hlt : Real.sqrt N < Real.sqrt M := by
    apply Real.sqrt_lt_sqrt hN.le hNM
  have key : chiC N = (1 + 1 / Real.sqrt N) ^ 2 / 2 := by
    have h : (1 + 1 / Real.sqrt N) ^ 2 / 2 = (1 + Real.sqrt N) ^ 2 / (2 * Real.sqrt N ^ 2) := by
      field_simp
      ring
    rw [chiC, h, hsqN]
  have keyM : chiC M = (1 + 1 / Real.sqrt M) ^ 2 / 2 := by
    have h : (1 + 1 / Real.sqrt M) ^ 2 / 2 = (1 + Real.sqrt M) ^ 2 / (2 * Real.sqrt M ^ 2) := by
      field_simp
      ring
    rw [chiC, h, hsqM]
  rw [key, keyM]
  have hinv : 1 / Real.sqrt M < 1 / Real.sqrt N := by
    apply one_div_lt_one_div_of_lt hsN hlt
  have hpos : 0 < 1 + 1 / Real.sqrt M := by positivity
  have : (1 + 1 / Real.sqrt M) ^ 2 < (1 + 1 / Real.sqrt N) ^ 2 := by nlinarith
  linarith

/-- The critical coupling is always above the incompressible-solvent value `1/2`. -/
theorem chiC_gt_half {N : ℝ} (hN : 0 < N) : 1 / 2 < chiC N := by
  have hs : 0 < Real.sqrt N := sqrt_pos_of hN
  have hsq : Real.sqrt N ^ 2 = N := Real.sq_sqrt hN.le
  rw [chiC, lt_div_iff₀ (by linarith)]
  nlinarith [hsq, hs]

/-- **The long-chain limit.**  `chiC N -> 1/2` as the chain grows. -/
theorem chiC_tendsto_half : Filter.Tendsto chiC Filter.atTop (nhds (1 / 2)) := by
  have heq : ∀ᶠ N : ℝ in Filter.atTop, chiC N = (1 + 1 / Real.sqrt N) ^ 2 / 2 := by
    filter_upwards [Filter.eventually_gt_atTop (0 : ℝ)] with N hN
    have hsq : Real.sqrt N ^ 2 = N := Real.sq_sqrt hN.le
    have hs : 0 < Real.sqrt N := sqrt_pos_of hN
    have h : (1 + 1 / Real.sqrt N) ^ 2 / 2 = (1 + Real.sqrt N) ^ 2 / (2 * Real.sqrt N ^ 2) := by
      field_simp
      ring
    rw [chiC, h, hsq]
  have hsqrt : Filter.Tendsto (fun N : ℝ => Real.sqrt N) Filter.atTop Filter.atTop :=
    Real.tendsto_sqrt_atTop
  have hinv : Filter.Tendsto (fun N : ℝ => 1 / Real.sqrt N) Filter.atTop (nhds 0) := by
    simpa using hsqrt.inv_tendsto_atTop
  have : Filter.Tendsto (fun N : ℝ => (1 + 1 / Real.sqrt N) ^ 2 / 2) Filter.atTop
      (nhds ((1 + 0) ^ 2 / 2)) := by
    exact ((tendsto_const_nhds.add hinv).pow 2).div_const 2
  rw [show ((1 : ℝ) + 0) ^ 2 / 2 = 1 / 2 by norm_num] at this
  exact this.congr' (heq.mono fun N h => h.symm)

/-- **The critical composition vanishes.**  Long disordered chains condense out of ever more
dilute solutions. -/
theorem phiC_tendsto_zero : Filter.Tendsto phiC Filter.atTop (nhds 0) := by
  have hsqrt : Filter.Tendsto (fun N : ℝ => 1 + Real.sqrt N) Filter.atTop Filter.atTop :=
    Filter.tendsto_atTop_add_const_left _ 1 Real.tendsto_sqrt_atTop
  have h := hsqrt.inv_tendsto_atTop
  refine h.congr fun N => ?_
  simp [phiC, one_div]

end FH

end IDR
