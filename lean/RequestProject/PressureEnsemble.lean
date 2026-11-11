/-
# Part CXXXVIII  Pressure: the thermodynamic axis a solution model of a disordered region must carry

Every earlier part perturbed a disordered region with temperature, salt, denaturant or a binding
partner.  Hydrostatic pressure is the one remaining *thermodynamic* axis, and it is the one that
couples to the quantity a coarse model most often gets wrong: the partial molar volume of a
conformer, i.e. how much solvent is excluded, how many voids are packed into the chain, and how
much water is bound at the surface.  A pressure axis is therefore the sharpest available test of
whether a model represents the solvent at all.

This part builds the pressure axis from scratch, for a finite conformer ensemble whose members
carry a reference free energy `G i` and a partial molar volume `V i` (`kT = 1`, volumes in units
of `kT` per unit pressure), so that the Boltzmann weight at pressure `p` is
`exp(-(G i + p·V i))`.

* `Z_pos`, `mean_one` — the isothermal–isobaric ensemble is well posed.
* `hasDerivAt_mean` — **the exact fluctuation–response identity**: `d⟨f⟩/dp = −Cov(V, f)`.  With
  `f = V` this is `mean_volume_deriv`: the pressure derivative of the mean volume is minus the
  volume variance, the microscopic definition of the compressibility, and `var_nonneg` makes it a
  sign statement.
* `mean_antitone_of_comonotone` — **Le Chatelier, exactly and without calculus.**  If an
  observable is comonotone with volume then its ensemble average never increases with pressure;
  `mean_volume_antitone` is the case `f = V`, and `mean_volume_strict_anti` says the decrease is
  strict as soon as two conformers of different volume are present.
* `population_tendsto_zero` — at high pressure the ensemble collapses onto the conformers of
  least partial molar volume, whatever their reference free energies.
* Two-state section: `pop_half_iff`, `pop_strictMono`, `pop_tendsto_one`, and the exact estimator
  `volume_change_recovery` — a pressure series determines `ΔV` from the slope of `log K`.
* `pressure_curve_two_point_underdetermined` with `pressure_curve_three_point_unique` — the design
  clause.  A pressure titration read at **two** pressures determines the compressibility change
  `Δβ` not at all: for *every* value of `Δβ` there is an exactly fitting `(ΔG₀, ΔV)`, and those
  fits disagree at every other pressure.  **Three** pressures pin all three coefficients.  So a
  model that reports a partial molar volume from a two-point pressure experiment is reporting a
  number the experiment does not contain.
* `activation_volume_recovery` — the same for kinetics: the pressure dependence of a rate gives
  the activation volume exactly.
* `pressure_design_law` bundles the clauses.
-/
import Mathlib

namespace RequestProject.PressureEnsemble

open Finset

/-- A finite conformer ensemble on the pressure axis: each conformer has a reference free energy
`G` (in `kT`) and a partial molar volume `V` (in `kT` per unit pressure). -/
structure Ensemble (ι : Type*) where
  /-- Reference free energy of each conformer, in units of `kT`, at zero pressure. -/
  G : ι → ℝ
  /-- Partial molar volume of each conformer. -/
  V : ι → ℝ

variable {ι : Type*} [Fintype ι] {E : Ensemble ι} {p q : ℝ}

/-- Unnormalised Boltzmann weight of conformer `i` at pressure `p`. -/
noncomputable def wt (E : Ensemble ι) (p : ℝ) (i : ι) : ℝ := Real.exp (-(E.G i + p * E.V i))

/-- Isothermal–isobaric partition function. -/
noncomputable def Z (E : Ensemble ι) (p : ℝ) : ℝ := ∑ i, wt E p i

/-- Ensemble average of an observable at pressure `p`. -/
noncomputable def mean (E : Ensemble ι) (p : ℝ) (f : ι → ℝ) : ℝ :=
  (∑ i, wt E p i * f i) / Z E p

/-- Ensemble covariance of two observables at pressure `p`. -/
noncomputable def cov (E : Ensemble ι) (p : ℝ) (f g : ι → ℝ) : ℝ :=
  mean E p (fun i => f i * g i) - mean E p f * mean E p g

/-- Ensemble variance of an observable. -/
noncomputable def var (E : Ensemble ι) (p : ℝ) (f : ι → ℝ) : ℝ := cov E p f f

omit [Fintype ι] in
lemma wt_pos (E : Ensemble ι) (p : ℝ) (i : ι) : 0 < wt E p i := Real.exp_pos _

lemma Z_pos [Nonempty ι] (E : Ensemble ι) (p : ℝ) : 0 < Z E p :=
  Finset.sum_pos (fun i _ => wt_pos E p i) Finset.univ_nonempty

lemma Z_ne_zero [Nonempty ι] (E : Ensemble ι) (p : ℝ) : Z E p ≠ 0 := (Z_pos E p).ne'

@[simp] lemma mean_one [Nonempty ι] (E : Ensemble ι) (p : ℝ) : mean E p (fun _ => 1) = 1 := by
  simp only [mean, mul_one]
  exact div_self (Z_ne_zero E p)

lemma mean_nonneg [Nonempty ι] (E : Ensemble ι) (p : ℝ) {f : ι → ℝ} (hf : ∀ i, 0 ≤ f i) :
    0 ≤ mean E p f :=
  div_nonneg (Finset.sum_nonneg fun i _ => mul_nonneg (wt_pos E p i).le (hf i)) (Z_pos E p).le

omit [Fintype ι] in
/-- The factorisation that drives every monotonicity statement: raising the pressure from `p`
to `q` reweights each conformer by `exp(-(q-p)·V i)`. -/
lemma wt_shift (E : Ensemble ι) (p q : ℝ) (i : ι) :
    wt E q i = wt E p i * Real.exp (-((q - p) * E.V i)) := by
  simp only [wt, ← Real.exp_add]
  ring_nf

/-! ## Variance and the fluctuation–response identity -/

/-- `Z · Var(f)` is the weighted sum of squared deviations, hence the variance is nonnegative. -/
lemma Z_mul_var [Nonempty ι] (E : Ensemble ι) (p : ℝ) (f : ι → ℝ) :
    Z E p * var E p f = ∑ i, wt E p i * (f i - mean E p f) ^ 2 := by
  have hZ : Z E p ≠ 0 := Z_ne_zero E p
  have hpt : ∀ i : ι, wt E p i * (f i - mean E p f) ^ 2
      = wt E p i * (f i * f i) - 2 * mean E p f * (wt E p i * f i)
        + (mean E p f) ^ 2 * wt E p i := fun i => by ring
  have hexp : ∑ i, wt E p i * (f i - mean E p f) ^ 2
      = (∑ i, wt E p i * (f i * f i)) - 2 * mean E p f * (∑ i, wt E p i * f i)
        + (mean E p f) ^ 2 * Z E p := by
    simp_rw [hpt]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, Z]
  rw [hexp]
  simp only [var, cov, mean]
  field_simp
  ring

lemma var_nonneg [Nonempty ι] (E : Ensemble ι) (p : ℝ) (f : ι → ℝ) : 0 ≤ var E p f := by
  have h := Z_mul_var E p f
  have hs : 0 ≤ ∑ i, wt E p i * (f i - mean E p f) ^ 2 :=
    Finset.sum_nonneg fun i _ => mul_nonneg (wt_pos E p i).le (sq_nonneg _)
  nlinarith [Z_pos E p, h, hs]

omit [Fintype ι] in
lemma hasDerivAt_wt (E : Ensemble ι) (p : ℝ) (i : ι) :
    HasDerivAt (fun p => wt E p i) (-(E.V i) * wt E p i) p := by
  have h0 : HasDerivAt (fun p : ℝ => E.G i + p * E.V i) (E.V i) p := by
    simpa using ((hasDerivAt_id p).mul_const (E.V i)).const_add (E.G i)
  simpa [wt, mul_comm] using h0.neg.exp

private lemma hasDerivAt_finsum (g : ι → ℝ → ℝ) (g' : ι → ℝ) (p : ℝ)
    (h : ∀ i, HasDerivAt (fun y => g i y) (g' i) p) :
    HasDerivAt (fun y => ∑ i, g i y) (∑ i, g' i) p := by
  have h2 := HasDerivAt.sum (u := (Finset.univ : Finset ι)) (fun i _ => h i)
  have heq : (∑ i ∈ (Finset.univ : Finset ι), fun y : ℝ => g i y) = fun y => ∑ i, g i y := by
    funext y; simp
  rwa [heq] at h2

lemma hasDerivAt_sum_wt (E : Ensemble ι) (p : ℝ) (f : ι → ℝ) :
    HasDerivAt (fun p => ∑ i, wt E p i * f i) (-∑ i, wt E p i * (E.V i * f i)) p := by
  have h : HasDerivAt (fun p => ∑ i, wt E p i * f i) (∑ i, (-(E.V i) * wt E p i) * f i) p :=
    hasDerivAt_finsum (fun i p => wt E p i * f i) _ p
      (fun i => (hasDerivAt_wt E p i).mul_const (f i))
  refine h.congr_deriv ?_
  rw [← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

lemma hasDerivAt_Z (E : Ensemble ι) (p : ℝ) :
    HasDerivAt (fun p => Z E p) (-∑ i, wt E p i * E.V i) p := by
  have h := hasDerivAt_sum_wt E p (fun _ => 1)
  simpa [Z] using h

/-- **Fluctuation–response.**  The pressure derivative of any ensemble average is minus its
covariance with the partial molar volume. -/
theorem hasDerivAt_mean [Nonempty ι] (E : Ensemble ι) (p : ℝ) (f : ι → ℝ) :
    HasDerivAt (fun p => mean E p f) (-cov E p E.V f) p := by
  have hZ : Z E p ≠ 0 := Z_ne_zero E p
  have h := (hasDerivAt_sum_wt E p f).div (hasDerivAt_Z E p) hZ
  refine h.congr_deriv ?_
  simp only [cov, mean, Z] at *
  field_simp
  ring

/-- With `f = V`: the mean partial molar volume decreases with pressure at a rate equal to the
volume variance — the microscopic compressibility. -/
theorem mean_volume_deriv [Nonempty ι] (E : Ensemble ι) (p : ℝ) :
    HasDerivAt (fun p => mean E p E.V) (-var E p E.V) p :=
  hasDerivAt_mean E p E.V

/-! ## Le Chatelier without calculus -/

/-- An observable is comonotone with volume when it never moves against it across conformers. -/
def Comonotone (V f : ι → ℝ) : Prop := ∀ i j, 0 ≤ (f i - f j) * (V i - V j)

omit [Fintype ι] in
lemma comonotone_self (V : ι → ℝ) : Comonotone V V := fun _ _ => mul_self_nonneg _

omit [Fintype ι] in
private lemma pair_le (E : Ensemble ι) {f : ι → ℝ} (hf : Comonotone E.V f) (hpq : p ≤ q)
    (i j : ι) :
    (f i - f j) * (wt E q i * wt E p j - wt E p i * wt E q j) ≤ 0 := by
  have hc0 : 0 ≤ q - p := by linarith
  have hfac : wt E q i * wt E p j - wt E p i * wt E q j
      = wt E p i * wt E p j *
        (Real.exp (-((q - p) * E.V i)) - Real.exp (-((q - p) * E.V j))) := by
    rw [wt_shift E p q i, wt_shift E p q j]
    ring
  rw [hfac]
  have hpos : 0 < wt E p i * wt E p j := mul_pos (wt_pos E p i) (wt_pos E p j)
  have key : (f i - f j) *
      (Real.exp (-((q - p) * E.V i)) - Real.exp (-((q - p) * E.V j))) ≤ 0 := by
    rcases lt_trichotomy (E.V i) (E.V j) with h | h | h
    · have hfle : f i - f j ≤ 0 := by nlinarith [hf i j]
      have hg : Real.exp (-((q - p) * E.V j)) ≤ Real.exp (-((q - p) * E.V i)) := by
        apply Real.exp_le_exp.2
        nlinarith
      nlinarith
    · simp [h]
    · have hfge : 0 ≤ f i - f j := by nlinarith [hf i j]
      have hg : Real.exp (-((q - p) * E.V i)) ≤ Real.exp (-((q - p) * E.V j)) := by
        apply Real.exp_le_exp.2
        nlinarith
      nlinarith
  nlinarith

private lemma cross_sum_gen (a b f : ι → ℝ) :
    2 * ((∑ i, b i * f i) * (∑ j, a j) - (∑ i, a i * f i) * (∑ j, b j))
      = ∑ i, ∑ j, (f i - f j) * (b i * a j - a i * b j) := by
  have h1 : ∑ i, ∑ j, f i * (b i * a j - a i * b j)
      = (∑ i, b i * f i) * (∑ j, a j) - (∑ i, a i * f i) * (∑ j, b j) := by
    have hrow : ∀ i : ι, ∑ j, f i * (b i * a j - a i * b j)
        = (b i * f i) * (∑ j, a j) - (a i * f i) * (∑ j, b j) := by
      intro i
      have hj : ∀ j : ι, f i * (b i * a j - a i * b j)
          = (b i * f i) * a j - (a i * f i) * b j := fun j => by ring
      simp_rw [hj]
      rw [Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    simp_rw [hrow]
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, ← Finset.sum_mul]
  have h2 : ∑ i, ∑ j, f j * (b i * a j - a i * b j)
      = -((∑ i, b i * f i) * (∑ j, a j) - (∑ i, a i * f i) * (∑ j, b j)) := by
    rw [Finset.sum_comm, ← h1, ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  have hsplit : ∑ i, ∑ j, (f i - f j) * (b i * a j - a i * b j)
      = (∑ i, ∑ j, f i * (b i * a j - a i * b j))
        - ∑ i, ∑ j, f j * (b i * a j - a i * b j) := by
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [hsplit, h1, h2]
  ring

/-- **Le Chatelier's principle for a disordered ensemble.**  Any observable comonotone with the
partial molar volume has an ensemble average that never increases with pressure. -/
theorem mean_antitone_of_comonotone [Nonempty ι] (E : Ensemble ι) {f : ι → ℝ}
    (hf : Comonotone E.V f) (hpq : p ≤ q) : mean E q f ≤ mean E p f := by
  have hsum : ∑ i, ∑ j, (f i - f j) * (wt E q i * wt E p j - wt E p i * wt E q j) ≤ 0 :=
    Finset.sum_nonpos fun i _ => Finset.sum_nonpos fun j _ => pair_le E hf hpq i j
  have hcross := cross_sum_gen (fun i => wt E p i) (fun i => wt E q i) f
  have hkey : (∑ i, wt E q i * f i) * (∑ j, wt E p j)
      - (∑ i, wt E p i * f i) * (∑ j, wt E q j) ≤ 0 := by
    nlinarith [hcross, hsum]
  rw [mean, mean, div_le_div_iff₀ (Z_pos E q) (Z_pos E p), Z, Z]
  linarith

/-- Pressure never increases the mean partial molar volume. -/
theorem mean_volume_antitone [Nonempty ι] (E : Ensemble ι) (hpq : p ≤ q) :
    mean E q E.V ≤ mean E p E.V :=
  mean_antitone_of_comonotone E (comonotone_self E.V) hpq

/-- And the decrease is strict as soon as the ensemble contains two conformers of different
volume and the pressure genuinely increases. -/
theorem mean_volume_strict_anti [Nonempty ι] (E : Ensemble ι) (hpq : p < q)
    {a b : ι} (hab : E.V a < E.V b) : mean E q E.V < mean E p E.V := by
  have hterm : ∀ i j : ι,
      (E.V i - E.V j) * (wt E q i * wt E p j - wt E p i * wt E q j) ≤ 0 :=
    fun i j => pair_le E (comonotone_self E.V) hpq.le i j
  have hab' : (E.V a - E.V b) * (wt E q a * wt E p b - wt E p a * wt E q b) < 0 := by
    have hc0 : 0 < q - p := by linarith
    have hfac : wt E q a * wt E p b - wt E p a * wt E q b
        = wt E p a * wt E p b *
          (Real.exp (-((q - p) * E.V a)) - Real.exp (-((q - p) * E.V b))) := by
      rw [wt_shift E p q a, wt_shift E p q b]
      ring
    rw [hfac]
    have hpos : 0 < wt E p a * wt E p b := mul_pos (wt_pos E p a) (wt_pos E p b)
    have hg : Real.exp (-((q - p) * E.V b)) < Real.exp (-((q - p) * E.V a)) := by
      apply Real.exp_lt_exp.2
      nlinarith
    have hprod : 0 < wt E p a * wt E p b *
        (Real.exp (-((q - p) * E.V a)) - Real.exp (-((q - p) * E.V b))) :=
      mul_pos hpos (by linarith)
    exact mul_neg_of_neg_of_pos (by linarith) hprod
  have hsum : ∑ i, ∑ j, (E.V i - E.V j) * (wt E q i * wt E p j - wt E p i * wt E q j) < 0 := by
    have hrowa : ∑ j, (E.V a - E.V j) * (wt E q a * wt E p j - wt E p a * wt E q j) < 0 := by
      have := Finset.sum_lt_sum (s := (Finset.univ : Finset ι))
        (f := fun j => (E.V a - E.V j) * (wt E q a * wt E p j - wt E p a * wt E q j))
        (g := fun _ => (0:ℝ)) (fun j _ => hterm a j) ⟨b, Finset.mem_univ b, hab'⟩
      simpa using this
    have := Finset.sum_lt_sum (s := (Finset.univ : Finset ι))
      (f := fun i => ∑ j, (E.V i - E.V j) * (wt E q i * wt E p j - wt E p i * wt E q j))
      (g := fun _ => (0:ℝ)) (fun i _ => Finset.sum_nonpos fun j _ => hterm i j)
      ⟨a, Finset.mem_univ a, hrowa⟩
    simpa using this
  have hcross := cross_sum_gen (fun i => wt E p i) (fun i => wt E q i) E.V
  have hkey : (∑ i, wt E q i * E.V i) * (∑ j, wt E p j)
      - (∑ i, wt E p i * E.V i) * (∑ j, wt E q j) < 0 := by
    nlinarith [hcross, hsum]
  rw [mean, mean, div_lt_div_iff₀ (Z_pos E q) (Z_pos E p), Z, Z]
  linarith

/-! ## The high-pressure limit -/

/-- A conformer of strictly larger volume than some other conformer is depopulated as the
pressure grows: pressure selects for small partial molar volume, whatever the free energies. -/
theorem population_tendsto_zero [Nonempty ι] (E : Ensemble ι) {i j : ι} (hij : E.V j < E.V i) :
    Filter.Tendsto (fun p => wt E p i / Z E p) Filter.atTop (nhds 0) := by
  have hbound : ∀ p : ℝ, wt E p i / Z E p ≤
      Real.exp (-(E.G i - E.G j) - p * (E.V i - E.V j)) := by
    intro p
    have hZ : wt E p j ≤ Z E p := Finset.single_le_sum (f := fun k => wt E p k)
      (fun k _ => (wt_pos E p k).le) (Finset.mem_univ j)
    have hposj : 0 < wt E p j := wt_pos E p j
    have h1 : wt E p i / Z E p ≤ wt E p i / wt E p j :=
      div_le_div_of_nonneg_left (wt_pos E p i).le hposj hZ
    have h2 : wt E p i / wt E p j = Real.exp (-(E.G i - E.G j) - p * (E.V i - E.V j)) := by
      rw [wt, wt, ← Real.exp_sub]
      congr 1
      ring
    linarith [h2 ▸ h1]
  have hlim : Filter.Tendsto
      (fun p : ℝ => Real.exp (-(E.G i - E.G j) - p * (E.V i - E.V j))) Filter.atTop (nhds 0) := by
    have hd : 0 < E.V i - E.V j := by linarith
    have h1 : Filter.Tendsto (fun p : ℝ => p * (E.V i - E.V j)) Filter.atTop Filter.atTop :=
      Filter.tendsto_id.atTop_mul_const hd
    have h2 : Filter.Tendsto (fun p : ℝ => -(p * (E.V i - E.V j))) Filter.atTop Filter.atBot :=
      Filter.tendsto_neg_atTop_atBot.comp h1
    have h3 : Filter.Tendsto (fun p : ℝ => -(E.G i - E.G j) + -(p * (E.V i - E.V j)))
        Filter.atTop Filter.atBot :=
      Filter.tendsto_atBot_add_const_left Filter.atTop _ h2
    have h4 : Filter.Tendsto (fun p : ℝ => -(E.G i - E.G j) - p * (E.V i - E.V j))
        Filter.atTop Filter.atBot := by simpa [sub_eq_add_neg] using h3
    exact Real.tendsto_exp_atBot.comp h4
  exact squeeze_zero (fun p => div_nonneg (wt_pos E p i).le (Z_pos E p).le) hbound hlim

/-! ## The two-state pressure titration -/

/-- Population of the second state of a two-state equilibrium whose free-energy difference is
`ΔG(p) = dG0 + p·dV` (state 2 minus state 1). -/
noncomputable def pop (dG0 dV p : ℝ) : ℝ := 1 / (1 + Real.exp (dG0 + p * dV))

lemma pop_pos (dG0 dV p : ℝ) : 0 < pop dG0 dV p := by
  have h : 0 < 1 + Real.exp (dG0 + p * dV) := by positivity
  simpa [pop] using div_pos one_pos h

lemma pop_lt_one (dG0 dV p : ℝ) : pop dG0 dV p < 1 := by
  have h : 0 < Real.exp (dG0 + p * dV) := Real.exp_pos _
  rw [pop, div_lt_one (by linarith)]
  linarith

/-- The midpoint pressure of a pressure titration is exactly `−ΔG₀/ΔV`. -/
theorem pop_half_iff {dG0 dV p : ℝ} (hdV : dV ≠ 0) :
    pop dG0 dV p = 1 / 2 ↔ p = -dG0 / dV := by
  have hden : 0 < 1 + Real.exp (dG0 + p * dV) := by positivity
  constructor
  · intro h
    have he : Real.exp (dG0 + p * dV) = 1 := by
      rw [pop, div_eq_div_iff hden.ne' (by norm_num : (2:ℝ) ≠ 0)] at h
      linarith
    have h0 : dG0 + p * dV = 0 := (Real.exp_eq_one_iff _).1 he
    field_simp
    linarith
  · intro h
    subst h
    have h0 : dG0 + (-dG0 / dV) * dV = 0 := by field_simp; ring
    rw [pop, h0]
    norm_num

/-- If the second state has the smaller volume (`dV < 0`) its population increases with
pressure — pressure denaturation, in its exact form. -/
theorem pop_strictMono {dG0 dV : ℝ} (hdV : dV < 0) : StrictMono (pop dG0 dV) := by
  intro a b hab
  have hlt : dG0 + b * dV < dG0 + a * dV := by nlinarith
  have h1 : Real.exp (dG0 + b * dV) < Real.exp (dG0 + a * dV) := Real.exp_lt_exp.2 hlt
  have hpa : 0 < 1 + Real.exp (dG0 + a * dV) := by positivity
  have hpb : 0 < 1 + Real.exp (dG0 + b * dV) := by positivity
  rw [pop, pop, div_lt_div_iff₀ hpa hpb]
  linarith

/-- And that population saturates: at high pressure the small-volume state is all there is. -/
theorem pop_tendsto_one {dG0 dV : ℝ} (hdV : dV < 0) :
    Filter.Tendsto (pop dG0 dV) Filter.atTop (nhds 1) := by
  have h1 : Filter.Tendsto (fun p : ℝ => p * (-dV)) Filter.atTop Filter.atTop :=
    Filter.tendsto_id.atTop_mul_const (by linarith)
  have h2 : Filter.Tendsto (fun p : ℝ => -(p * (-dV))) Filter.atTop Filter.atBot :=
    Filter.tendsto_neg_atTop_atBot.comp h1
  have h3 : Filter.Tendsto (fun p : ℝ => dG0 + -(p * (-dV))) Filter.atTop Filter.atBot :=
    Filter.tendsto_atBot_add_const_left Filter.atTop dG0 h2
  have hlin : Filter.Tendsto (fun p : ℝ => dG0 + p * dV) Filter.atTop Filter.atBot := by
    simpa using h3
  have hexp : Filter.Tendsto (fun p : ℝ => Real.exp (dG0 + p * dV)) Filter.atTop (nhds 0) :=
    Real.tendsto_exp_atBot.comp hlin
  have hlim : Filter.Tendsto (fun p : ℝ => 1 / (1 + Real.exp (dG0 + p * dV)))
      Filter.atTop (nhds (1 / (1 + 0))) :=
    Filter.Tendsto.div tendsto_const_nhds (tendsto_const_nhds.add hexp) (by norm_num)
  have hpop : pop dG0 dV = fun p => 1 / (1 + Real.exp (dG0 + p * dV)) := rfl
  rw [hpop]
  simpa using hlim

/-- **The volume change is an exact slope.**  Two pressures suffice to read `ΔV` off a linear
free-energy series. -/
theorem volume_change_recovery (dG0 dV p₁ p₂ : ℝ) (h : p₁ ≠ p₂) :
    ((dG0 + p₂ * dV) - (dG0 + p₁ * dV)) / (p₂ - p₁) = dV := by
  have hne : p₂ - p₁ ≠ 0 := sub_ne_zero.2 (Ne.symm h)
  field_simp
  ring

/-- The same statement for rates: the pressure dependence of `log k` gives the activation
volume `ΔV‡` exactly. -/
theorem activation_volume_recovery (k0 dVa p₁ p₂ : ℝ) (hk0 : 0 < k0) (h : p₁ ≠ p₂) :
    (Real.log (k0 * Real.exp (-(p₁ * dVa))) - Real.log (k0 * Real.exp (-(p₂ * dVa))))
        / (p₂ - p₁) = dVa := by
  have hne : p₂ - p₁ ≠ 0 := sub_ne_zero.2 (Ne.symm h)
  rw [Real.log_mul hk0.ne' (Real.exp_ne_zero _), Real.log_mul hk0.ne' (Real.exp_ne_zero _),
    Real.log_exp, Real.log_exp]
  field_simp
  ring

/-! ## Volume versus compressibility: what two pressures cannot separate -/

/-- Second-order pressure expansion of a free-energy difference: `ΔG(p) = a + b·p − (c/2)p²`,
with `b = ΔV` the volume change and `c = Δβ` the compressibility change. -/
noncomputable def dGq (a b c p : ℝ) : ℝ := a + b * p - c * p ^ 2 / 2

/-- **Two pressures do not constrain the compressibility at all.**  For every target value `c'`
of the compressibility change there are `a'`, `b'` reproducing the measured free-energy
differences exactly at both pressures; and if `c' ≠ c` the two models disagree at every other
pressure, so the ambiguity is not a reparametrisation but a genuine loss of prediction. -/
theorem pressure_curve_two_point_underdetermined (a b c c' p₁ p₂ : ℝ) :
    ∃ a' b' : ℝ, dGq a' b' c' p₁ = dGq a b c p₁ ∧ dGq a' b' c' p₂ = dGq a b c p₂ ∧
      ∀ p : ℝ, dGq a' b' c' p - dGq a b c p = -((c' - c) / 2) * (p - p₁) * (p - p₂) := by
  refine ⟨a - ((c' - c) / 2) * (p₁ * p₂), b + ((c' - c) / 2) * (p₁ + p₂), ?_, ?_, ?_⟩ <;>
    simp only [dGq] <;> intros <;> ring

/-- Three distinct pressures do determine all three coefficients. -/
theorem pressure_curve_three_point_unique {a b c a' b' c' p₁ p₂ p₃ : ℝ}
    (h₁₂ : p₁ ≠ p₂) (h₁₃ : p₁ ≠ p₃) (h₂₃ : p₂ ≠ p₃)
    (e₁ : dGq a' b' c' p₁ = dGq a b c p₁) (e₂ : dGq a' b' c' p₂ = dGq a b c p₂)
    (e₃ : dGq a' b' c' p₃ = dGq a b c p₃) : a' = a ∧ b' = b ∧ c' = c := by
  simp only [dGq] at e₁ e₂ e₃
  have d₁₂ : (b' - b) * (p₁ - p₂) - ((c' - c) / 2) * (p₁ ^ 2 - p₂ ^ 2) = 0 := by linarith
  have d₁₃ : (b' - b) * (p₁ - p₃) - ((c' - c) / 2) * (p₁ ^ 2 - p₃ ^ 2) = 0 := by linarith
  have h12 : p₁ - p₂ ≠ 0 := sub_ne_zero.2 h₁₂
  have h13 : p₁ - p₃ ≠ 0 := sub_ne_zero.2 h₁₃
  have h23 : p₂ - p₃ ≠ 0 := sub_ne_zero.2 h₂₃
  have s₁₂ : (b' - b) - ((c' - c) / 2) * (p₁ + p₂) = 0 := by
    have hz : ((b' - b) - ((c' - c) / 2) * (p₁ + p₂)) * (p₁ - p₂) = 0 := by
      rw [← d₁₂]; ring
    rcases mul_eq_zero.1 hz with h | h
    · exact h
    · exact absurd h h12
  have s₁₃ : (b' - b) - ((c' - c) / 2) * (p₁ + p₃) = 0 := by
    have hz : ((b' - b) - ((c' - c) / 2) * (p₁ + p₃)) * (p₁ - p₃) = 0 := by
      rw [← d₁₃]; ring
    rcases mul_eq_zero.1 hz with h | h
    · exact h
    · exact absurd h h13
  have hc : c' = c := by
    have hz : ((c' - c) / 2) * (p₂ - p₃) = 0 := by linarith
    rcases mul_eq_zero.1 hz with h | h
    · linarith
    · exact absurd h h23
  have hb : b' = b := by
    rw [hc] at s₁₂
    simp at s₁₂
    linarith
  refine ⟨?_, hb, hc⟩
  rw [hc, hb] at e₁
  linarith

/-! ## Capstone -/

/-- **The pressure design law.**  A model of a disordered region that carries a pressure axis
must satisfy all of: the exact fluctuation–response identity (1), the sign law that averages
comonotone with volume fall under pressure (2), collapse onto the least-volume conformers at
high pressure (3), exact recovery of the volume change from a linear series (4), and the
identifiability boundary — a two-pressure experiment leaves the compressibility change entirely
free while three pressures fix it (5), (6). -/
theorem pressure_design_law [Nonempty ι] (E : Ensemble ι) :
    (∀ p : ℝ, ∀ f : ι → ℝ, HasDerivAt (fun p => mean E p f) (-cov E p E.V f) p) ∧
    (∀ f : ι → ℝ, Comonotone E.V f → ∀ p q : ℝ, p ≤ q → mean E q f ≤ mean E p f) ∧
    (∀ i j : ι, E.V j < E.V i →
      Filter.Tendsto (fun p => wt E p i / Z E p) Filter.atTop (nhds 0)) ∧
    (∀ dG0 dV p₁ p₂ : ℝ, p₁ ≠ p₂ →
      ((dG0 + p₂ * dV) - (dG0 + p₁ * dV)) / (p₂ - p₁) = dV) ∧
    (∀ a b c c' p₁ p₂ : ℝ, p₁ ≠ p₂ → ∃ a' b' : ℝ,
      dGq a' b' c' p₁ = dGq a b c p₁ ∧ dGq a' b' c' p₂ = dGq a b c p₂) ∧
    (∀ a b c a' b' c' p₁ p₂ p₃ : ℝ, p₁ ≠ p₂ → p₁ ≠ p₃ → p₂ ≠ p₃ →
      dGq a' b' c' p₁ = dGq a b c p₁ → dGq a' b' c' p₂ = dGq a b c p₂ →
      dGq a' b' c' p₃ = dGq a b c p₃ → a' = a ∧ b' = b ∧ c' = c) := by
  refine ⟨fun p f => hasDerivAt_mean E p f,
    fun f hf p q hpq => mean_antitone_of_comonotone E hf hpq,
    fun i j hij => population_tendsto_zero E hij,
    fun dG0 dV p₁ p₂ h => volume_change_recovery dG0 dV p₁ p₂ h, ?_, ?_⟩
  · intro a b c c' p₁ p₂ _
    obtain ⟨a', b', h1, h2, _⟩ := pressure_curve_two_point_underdetermined a b c c' p₁ p₂
    exact ⟨a', b', h1, h2⟩
  · intro a b c a' b' c' p₁ p₂ p₃ h₁₂ h₁₃ h₂₃ e₁ e₂ e₃
    exact pressure_curve_three_point_unique h₁₂ h₁₃ h₂₃ e₁ e₂ e₃

end RequestProject.PressureEnsemble
