import Mathlib

/-!
# Part CXXXVII — The radial screened equation `u'' = κ² u`

The Debye–Hückel (linearised Poisson–Boltzmann) equation for a spherically symmetric
potential `ψ` reduces, under the substitution `u(r) = r ψ(r)`, to the one-dimensional
equation `u'' = κ² u`.  This file develops that ODE from scratch:

* existence of the explicit two-parameter family (`solvesScreened_exp`),
* uniqueness from Cauchy data (`screened_uniqueness`), by an energy/Grönwall argument,
* the classification of all solutions (`screened_classification`),
* the fact that a *bounded* solution on a half line is a pure decaying exponential
  (`screened_bounded_decays`) — this is what selects the physical Debye solution,
* the maximum principle for subsolutions (`screened_maximum_principle`), and
* the nonlinear Poisson–Boltzmann comparison theorem
  (`nonlinear_pb_le_linear`): a nonnegative solution of the *full* nonlinear equation
  `w'' = κ² sinh w` never exceeds the linearised solution with the same data, i.e.
  Debye–Hückel theory always overestimates the potential.
-/

noncomputable section

namespace RequestProject.Physics

open Set Real Filter Topology

/-- `u` solves the screened equation `u'' = κ² u` on `s`, with derivative `up`. -/
structure SolvesScreened (kappa : ℝ) (u up : ℝ → ℝ) (s : Set ℝ) : Prop where
  deriv_u : ∀ r ∈ s, HasDerivAt u (up r) r
  deriv_up : ∀ r ∈ s, HasDerivAt up (kappa ^ 2 * u r) r

lemma SolvesScreened.mono {kappa : ℝ} {u up : ℝ → ℝ} {s t : Set ℝ}
    (h : SolvesScreened kappa u up s) (hts : t ⊆ s) : SolvesScreened kappa u up t :=
  ⟨fun r hr => h.deriv_u r (hts hr), fun r hr => h.deriv_up r (hts hr)⟩

/-- The explicit exponential family solves the screened equation. -/
theorem solvesScreened_exp (kappa A B a : ℝ) :
    SolvesScreened kappa
      (fun r => A * exp (kappa * (r - a)) + B * exp (-kappa * (r - a)))
      (fun r => A * kappa * exp (kappa * (r - a)) - B * kappa * exp (-kappa * (r - a)))
      univ := by
  constructor
  · intro r _
    have h1 : HasDerivAt (fun s : ℝ => kappa * (s - a)) kappa r := by
      simpa using ((hasDerivAt_id r).sub_const a).const_mul kappa
    have h2 : HasDerivAt (fun s : ℝ => -kappa * (s - a)) (-kappa) r := by
      simpa using ((hasDerivAt_id r).sub_const a).const_mul (-kappa)
    have e1 : HasDerivAt (fun s : ℝ => A * exp (kappa * (s - a)))
        (A * (exp (kappa * (r - a)) * kappa)) r := (h1.exp).const_mul A
    have e2 : HasDerivAt (fun s : ℝ => B * exp (-kappa * (s - a)))
        (B * (exp (-kappa * (r - a)) * -kappa)) r := (h2.exp).const_mul B
    have := e1.add e2
    convert this using 1
    ring
  · intro r _
    have h1 : HasDerivAt (fun s : ℝ => kappa * (s - a)) kappa r := by
      simpa using ((hasDerivAt_id r).sub_const a).const_mul kappa
    have h2 : HasDerivAt (fun s : ℝ => -kappa * (s - a)) (-kappa) r := by
      simpa using ((hasDerivAt_id r).sub_const a).const_mul (-kappa)
    have e1 : HasDerivAt (fun s : ℝ => A * kappa * exp (kappa * (s - a)))
        (A * kappa * (exp (kappa * (r - a)) * kappa)) r := (h1.exp).const_mul (A * kappa)
    have e2 : HasDerivAt (fun s : ℝ => B * kappa * exp (-kappa * (s - a)))
        (B * kappa * (exp (-kappa * (r - a)) * -kappa)) r := (h2.exp).const_mul (B * kappa)
    have := e1.sub e2
    convert this using 1
    ring

/-- **Uniqueness from Cauchy data.**  Two solutions of the screened equation on a half line
that agree to first order at the left end agree everywhere. -/
theorem screened_uniqueness {kappa a : ℝ} (hk : 0 < kappa) {u up v vp : ℝ → ℝ}
    (hu : SolvesScreened kappa u up (Ici a)) (hv : SolvesScreened kappa v vp (Ici a))
    (h0 : u a = v a) (h1 : up a = vp a) :
    ∀ r ∈ Ici a, u r = v r := by
  set w : ℝ → ℝ := fun r => u r - v r with hwdef
  set wp : ℝ → ℝ := fun r => up r - vp r with hwpdef
  have hw : ∀ r ∈ Ici a, HasDerivAt w (wp r) r := fun r hr =>
    (hu.deriv_u r hr).sub (hv.deriv_u r hr)
  have hwp : ∀ r ∈ Ici a, HasDerivAt wp (kappa ^ 2 * w r) r := by
    intro r hr
    have := (hu.deriv_up r hr).sub (hv.deriv_up r hr)
    convert this using 1
    simp [hwdef]
    ring
  have hk2 : (0 : ℝ) < kappa ^ 2 := by positivity
  set G : ℝ → ℝ := fun r => w r ^ 2 + (wp r) ^ 2 / kappa ^ 2 with hGdef
  set H : ℝ → ℝ := fun r => G r * exp (-2 * kappa * (r - a)) with hHdef
  have hGderiv : ∀ r ∈ Ici a, HasDerivAt G (4 * w r * wp r) r := by
    intro r hr
    have hsq : HasDerivAt (fun s => w s ^ 2) (2 * w r * wp r) r := by
      have := (hw r hr).pow 2
      convert this using 1
      ring
    have hsq2 : HasDerivAt (fun s => (wp s) ^ 2 / kappa ^ 2)
        (2 * wp r * (kappa ^ 2 * w r) / kappa ^ 2) r := by
      have := ((hwp r hr).pow 2).div_const (kappa ^ 2)
      convert this using 1
      ring
    have := hsq.add hsq2
    convert this using 1
    field_simp
    ring
  have hHderiv : ∀ r ∈ Ici a, HasDerivAt H
      ((4 * w r * wp r - 2 * kappa * G r) * exp (-2 * kappa * (r - a))) r := by
    intro r hr
    have hexp : HasDerivAt (fun s : ℝ => exp (-2 * kappa * (s - a)))
        (exp (-2 * kappa * (r - a)) * (-2 * kappa)) r := by
      have h1 : HasDerivAt (fun s : ℝ => -2 * kappa * (s - a)) (-2 * kappa) r := by
        simpa using ((hasDerivAt_id r).sub_const a).const_mul (-2 * kappa)
      exact h1.exp
    have := (hGderiv r hr).mul hexp
    convert this using 1
    ring
  have hHnonpos : ∀ r ∈ Ici a, (4 * w r * wp r - 2 * kappa * G r) * exp (-2 * kappa * (r - a)) ≤ 0 := by
    intro r hr
    have hkey : 4 * w r * wp r - 2 * kappa * G r = -(2 * kappa * (w r - wp r / kappa) ^ 2) := by
      simp only [hGdef]
      field_simp
      ring
    rw [hkey]
    have : 0 ≤ 2 * kappa * (w r - wp r / kappa) ^ 2 := by positivity
    have hexp : 0 < exp (-2 * kappa * (r - a)) := Real.exp_pos _
    nlinarith
  have hcont : ContinuousOn H (Ici a) := fun r hr =>
    ((hHderiv r hr).continuousAt).continuousWithinAt
  have hdiff : DifferentiableOn ℝ H (interior (Ici a)) := by
    intro r hr
    rw [interior_Ici] at hr
    exact ((hHderiv r (Set.mem_Ici.mpr (le_of_lt hr))).differentiableAt).differentiableWithinAt
  have hderivle : ∀ r ∈ interior (Ici a), deriv H r ≤ 0 := by
    intro r hr
    rw [interior_Ici] at hr
    rw [(hHderiv r (Set.mem_Ici.mpr (le_of_lt hr))).deriv]
    exact hHnonpos r (Set.mem_Ici.mpr (le_of_lt hr))
  have hanti : AntitoneOn H (Ici a) :=
    antitoneOn_of_deriv_nonpos (convex_Ici a) hcont hdiff hderivle
  have hHa : H a = 0 := by
    have hwa : w a = 0 := by simp [hwdef, h0]
    have hwpa : wp a = 0 := by simp [hwpdef, h1]
    simp [hHdef, hGdef, hwa, hwpa]
  intro r hr
  have hHr : H r ≤ 0 := by
    have := hanti (Set.self_mem_Ici) hr hr
    rw [hHa] at this
    exact this
  have hGr : G r ≤ 0 := by
    by_contra hcon
    push_neg at hcon
    have hexp : 0 < exp (-2 * kappa * (r - a)) := Real.exp_pos _
    nlinarith [hHr]
  have hGnn : 0 ≤ G r := by
    have h1 : 0 ≤ w r ^ 2 := sq_nonneg _
    have h2 : 0 ≤ (wp r) ^ 2 / kappa ^ 2 := by positivity
    simp [hGdef]
    linarith
  have hG0 : G r = 0 := le_antisymm hGr hGnn
  have hw0 : w r = 0 := by
    have h2 : 0 ≤ (wp r) ^ 2 / kappa ^ 2 := by positivity
    have hsq : w r ^ 2 = 0 := by
      have : w r ^ 2 + (wp r) ^ 2 / kappa ^ 2 = 0 := hG0
      nlinarith [sq_nonneg (w r)]
    exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp hsq
  have : u r - v r = 0 := hw0
  linarith

/-- **Classification.** Every solution on a half line is the explicit exponential
combination determined by its Cauchy data. -/
theorem screened_classification {kappa a : ℝ} (hk : 0 < kappa) {u up : ℝ → ℝ}
    (hu : SolvesScreened kappa u up (Ici a)) :
    ∀ r ∈ Ici a, u r =
      ((u a + up a / kappa) / 2) * exp (kappa * (r - a)) +
      ((u a - up a / kappa) / 2) * exp (-kappa * (r - a)) := by
  set A := (u a + up a / kappa) / 2 with hA
  set B := (u a - up a / kappa) / 2 with hB
  have hv : SolvesScreened kappa
      (fun r => A * exp (kappa * (r - a)) + B * exp (-kappa * (r - a)))
      (fun r => A * kappa * exp (kappa * (r - a)) - B * kappa * exp (-kappa * (r - a)))
      (Ici a) := (solvesScreened_exp kappa A B a).mono (subset_univ _)
  have h0 : u a = A * exp (kappa * (a - a)) + B * exp (-kappa * (a - a)) := by
    simp [hA, hB]
    field_simp
    ring
  have h1 : up a = A * kappa * exp (kappa * (a - a)) - B * kappa * exp (-kappa * (a - a)) := by
    simp [hA, hB]
    field_simp
    ring
  exact screened_uniqueness hk hu hv h0 h1

/-- **The physical (bounded) solution is the decaying exponential.**  This is the step that
picks the Debye–Hückel screened potential out of the two-dimensional solution space. -/
theorem screened_bounded_decays {kappa a : ℝ} (hk : 0 < kappa) {u up : ℝ → ℝ}
    (hu : SolvesScreened kappa u up (Ici a)) {M : ℝ} (hbdd : ∀ r ∈ Ici a, |u r| ≤ M) :
    ∀ r ∈ Ici a, u r = u a * exp (-kappa * (r - a)) := by
  set A := (u a + up a / kappa) / 2 with hA
  set B := (u a - up a / kappa) / 2 with hB
  have hclass := screened_classification hk hu
  have hA0 : A = 0 := by
    by_contra hne
    have habs : 0 < |A| := abs_pos.mpr hne
    have htend : Tendsto (fun r : ℝ => |A| * exp (kappa * (r - a)) - |B|) atTop atTop := by
      have h0 : Tendsto (fun r : ℝ => r - a) atTop atTop := by
        simpa [sub_eq_add_neg] using tendsto_atTop_add_const_right atTop (-a) tendsto_id
      have h1 : Tendsto (fun r : ℝ => kappa * (r - a)) atTop atTop :=
        Filter.Tendsto.const_mul_atTop hk h0
      have h2 : Tendsto (fun r : ℝ => exp (kappa * (r - a))) atTop atTop :=
        Real.tendsto_exp_atTop.comp h1
      have h3 : Tendsto (fun r : ℝ => |A| * exp (kappa * (r - a))) atTop atTop :=
        Filter.Tendsto.const_mul_atTop habs h2
      have h4 := tendsto_atTop_add_const_right atTop (-|B|) h3
      exact h4.congr (fun r => by ring)
    have hev : ∀ᶠ r in atTop, M < |A| * exp (kappa * (r - a)) - |B| := htend.eventually_gt_atTop M
    have hev2 : ∀ᶠ r in atTop, a ≤ r := eventually_ge_atTop a
    obtain ⟨r, hr1, hr2⟩ := (hev.and hev2).exists
    have hur := hclass r hr2
    have hexpB : exp (-kappa * (r - a)) ≤ 1 := by
      rw [Real.exp_le_one_iff]
      have : 0 ≤ r - a := by linarith
      nlinarith
    have hexpBpos : 0 < exp (-kappa * (r - a)) := Real.exp_pos _
    have hbound := hbdd r hr2
    have hge : |A| * exp (kappa * (r - a)) - |B| ≤ |u r| := by
      have h1 : |u r| = |A * exp (kappa * (r - a)) + B * exp (-kappa * (r - a))| := by
        rw [hur]
      rw [h1]
      have h2 : |A * exp (kappa * (r - a))| - |B * exp (-kappa * (r - a))| ≤
          |A * exp (kappa * (r - a)) + B * exp (-kappa * (r - a))| := by
        have h := abs_add_le (A * exp (kappa * (r - a)) + B * exp (-kappa * (r - a)))
          (-(B * exp (-kappa * (r - a))))
        simp only [add_neg_cancel_right, abs_neg] at h
        linarith
      have h3 : |A * exp (kappa * (r - a))| = |A| * exp (kappa * (r - a)) := by
        rw [abs_mul, abs_of_pos (Real.exp_pos _)]
      have h4 : |B * exp (-kappa * (r - a))| ≤ |B| := by
        rw [abs_mul, abs_of_pos hexpBpos]
        nlinarith [abs_nonneg B]
      linarith
    linarith
  intro r hr
  have hur := hclass r hr
  rw [← hA, ← hB, hA0] at hur
  have hBval : B = u a := by
    have : A + B = u a := by
      rw [hA, hB]; ring
    rw [hA0] at this
    linarith
  rw [hur, hBval]
  ring

/-- **Maximum principle** for the operator `d ↦ d'' - κ² d`: a subsolution which is
nonpositive at both ends of an interval is nonpositive throughout. -/
theorem screened_maximum_principle {kappa a b : ℝ} (hk : 0 < kappa) (hab : a ≤ b)
    {d dp dpp : ℝ → ℝ}
    (hd : ∀ r ∈ Icc a b, HasDerivAt d (dp r) r)
    (hdp : ∀ r ∈ Icc a b, HasDerivAt dp (dpp r) r)
    (hsub : ∀ r ∈ Icc a b, kappa ^ 2 * d r ≤ dpp r)
    (ha : d a ≤ 0) (hb : d b ≤ 0) :
    ∀ r ∈ Icc a b, d r ≤ 0 := by
  by_contra hcon
  push_neg at hcon
  obtain ⟨r0, hr0mem, hr0pos⟩ := hcon
  have hcont : ContinuousOn d (Icc a b) := fun r hr => ((hd r hr).continuousAt).continuousWithinAt
  obtain ⟨c, hcmem, hcmax⟩ :=
    isCompact_Icc.exists_isMaxOn (Set.nonempty_Icc.mpr hab) hcont
  have hdc : 0 < d c := lt_of_lt_of_le hr0pos (hcmax hr0mem)
  have hca : c ≠ a := by intro h; rw [h] at hdc; linarith
  have hcb : c ≠ b := by intro h; rw [h] at hdc; linarith
  have hcioo : c ∈ Ioo a b := by
    rcases hcmem with ⟨h1, h2⟩
    exact ⟨lt_of_le_of_ne h1 (Ne.symm hca), lt_of_le_of_ne h2 hcb⟩
  have hnhds : Icc a b ∈ 𝓝 c := by
    have : Ioo a b ⊆ Icc a b := Ioo_subset_Icc_self
    exact Filter.mem_of_superset (Ioo_mem_nhds hcioo.1 hcioo.2) this
  have hlocmax : IsLocalMax d c := hcmax.isLocalMax hnhds
  have hdpc : dp c = 0 := hlocmax.hasDerivAt_eq_zero (hd c hcmem)
  have hdppc : 0 < dpp c := by
    have hle := hsub c hcmem
    have hk2 : (0 : ℝ) < kappa ^ 2 := by positivity
    nlinarith
  -- the derivative becomes strictly positive just to the right of `c`
  have hslope : Tendsto (slope dp c) (𝓝[≠] c) (𝓝 (dpp c)) :=
    hasDerivAt_iff_tendsto_slope.mp (hdp c hcmem)
  have hpos : ∀ᶠ t in 𝓝[≠] c, 0 < slope dp c t := hslope.eventually (eventually_gt_nhds hdppc)
  have hposGT : ∀ᶠ t in 𝓝[>] c, 0 < slope dp c t :=
    hpos.filter_mono (nhdsWithin_mono c (fun t ht => ne_of_gt ht))
  have hdpposGT : ∀ᶠ t in 𝓝[>] c, 0 < dp t := by
    filter_upwards [hposGT, self_mem_nhdsWithin] with t hslopet htgt
    have htne : 0 < t - c := sub_pos.mpr htgt
    have hs : slope dp c t = dp t / (t - c) := by
      rw [slope_def_field, hdpc, sub_zero]
    rw [hs] at hslopet
    rcases div_pos_iff.mp hslopet with ⟨h1, _⟩ | ⟨_, h2⟩
    · exact h1
    · linarith
  obtain ⟨e, hegt, hsub2⟩ := mem_nhdsGT_iff_exists_Ioc_subset.mp hdpposGT
  set delta := min e b with hdelta
  have hcd : c < delta := lt_min hegt hcioo.2
  have hdeltab : delta ≤ b := min_le_right _ _
  have hsubset : Icc c delta ⊆ Icc a b := by
    intro t ht
    exact ⟨le_trans hcmem.1 ht.1, le_trans ht.2 hdeltab⟩
  have hmono : StrictMonoOn d (Icc c delta) := by
    refine strictMonoOn_of_deriv_pos (convex_Icc c delta) ?_ ?_
    · exact fun t ht => ((hd t (hsubset ht)).continuousAt).continuousWithinAt
    · intro t ht
      rw [interior_Icc] at ht
      have htmem : t ∈ Icc a b := hsubset ⟨le_of_lt ht.1, le_of_lt ht.2⟩
      rw [(hd t htmem).deriv]
      have : t ∈ Ioc c e := ⟨ht.1, le_trans (le_of_lt ht.2) (min_le_left _ _)⟩
      exact hsub2 this
  have hlt : d c < d delta :=
    hmono (left_mem_Icc.mpr (le_of_lt hcd)) (right_mem_Icc.mpr (le_of_lt hcd)) hcd
  have hle : d delta ≤ d c := hcmax (hsubset (right_mem_Icc.mpr (le_of_lt hcd)))
  linarith

/-- **Nonlinear Poisson–Boltzmann comparison.**  If `w` solves the full nonlinear
Poisson–Boltzmann equation `w'' = κ² sinh w` and is nonnegative, and `v` solves the
linearised (Debye–Hückel) equation `v'' = κ² v` with the same value at the inner boundary
and at least the value of `w` at the outer boundary, then `w ≤ v` throughout:
linearisation *overestimates* the screened potential. -/
theorem nonlinear_pb_le_linear {kappa a b : ℝ} (hk : 0 < kappa) (hab : a ≤ b)
    {w wp v vp : ℝ → ℝ}
    (hw : ∀ r ∈ Icc a b, HasDerivAt w (wp r) r)
    (hwp : ∀ r ∈ Icc a b, HasDerivAt wp (kappa ^ 2 * sinh (w r)) r)
    (hv : ∀ r ∈ Icc a b, HasDerivAt v (vp r) r)
    (hvp : ∀ r ∈ Icc a b, HasDerivAt vp (kappa ^ 2 * v r) r)
    (hwnn : ∀ r ∈ Icc a b, 0 ≤ w r)
    (hA : w a = v a) (hB : w b ≤ v b) :
    ∀ r ∈ Icc a b, w r ≤ v r := by
  set d : ℝ → ℝ := fun r => w r - v r with hddef
  set dp : ℝ → ℝ := fun r => wp r - vp r with hdpdef
  set dpp : ℝ → ℝ := fun r => kappa ^ 2 * sinh (w r) - kappa ^ 2 * v r with hdppdef
  have hd : ∀ r ∈ Icc a b, HasDerivAt d (dp r) r := fun r hr => (hw r hr).sub (hv r hr)
  have hdp' : ∀ r ∈ Icc a b, HasDerivAt dp (dpp r) r := fun r hr => (hwp r hr).sub (hvp r hr)
  have hsub : ∀ r ∈ Icc a b, kappa ^ 2 * d r ≤ dpp r := by
    intro r hr
    have hs : w r ≤ sinh (w r) := Real.self_le_sinh_iff.mpr (hwnn r hr)
    have hk2 : (0 : ℝ) < kappa ^ 2 := by positivity
    have : kappa ^ 2 * w r ≤ kappa ^ 2 * sinh (w r) := by nlinarith
    simp only [hddef, hdppdef]
    nlinarith
  have hda : d a ≤ 0 := by simp [hddef, hA]
  have hdb : d b ≤ 0 := by simp [hddef]; linarith
  intro r hr
  have := screened_maximum_principle hk hab hd hdp' hsub hda hdb r hr
  simp only [hddef] at this
  linarith

end RequestProject.Physics
