/-
# Part CXXXIX  Solvent quality is part of the distance law: what a titration says about the
# swelling exponent

Part CXXXVIII fixed the internal distance law of the disordered region to that of an *ideal*
chain, `R(d) = b√d`, and showed that inside that law the bond length `b` trades one-for-one
against sequence separation unless the correlation profile is known independently.  Real
intrinsically disordered regions are not ideal chains.  Their internal distances follow a
polymer scaling law `R(d) = b·d^ν` whose exponent measures solvent quality: `ν ≈ 3/5` for a
swollen, highly charged region in good solvent, `ν = 1/2` at the theta point, `ν ≈ 1/3` for a
collapsed globule.  The exponent is exactly the quantity a model of a disordered region is
usually asked to report, so the question of this part is whether a salt titration can supply it.

The screened electrostatic reading of a charge correlation at sequence separation `d` becomes

  `skern b ν κ d = e^{−κ·b·d^ν} / (b·d^ν)`,

a two-parameter distance law that contains the Gaussian-chain kernel of Part CXXXVIII at
`ν = 1/2` (`swellCurve_half_eq_debyeCurve`).  Four results.

* **The profile is still identified, for a declared law.**  `swell_profile_identifiable` — for any
  `b > 0`, `ν > 0`, a complete titration determines every charge correlation in the window.  So
  nothing is lost by moving to the realistic law; what changes is what the *law itself* costs.

* **The confound of Part CXXXVIII survives, deformed.**  `even_lag_swelling_confound` — a profile
  supported on even lags, read at `(b, ν)`, is reproduced exactly, at every ionic strength, by the
  profile of its even lags read at `(b·2^ν, ν)`: the ideal-chain factor `√2` becomes `2^ν`.

* **A single separation determines nothing about solvent quality.**  `single_lag_exponent_confound`
  — if the region carries one correlation, at lag `D`, then for *every* exponent `ν'` there is a
  bond length `b' = b·D^ν/D^{ν'}` reproducing the titration curve identically.  The whole
  `(b, ν)` line is degenerate: `swollen_and_collapsed_indistinguishable_at_one_lag` exhibits a
  good-solvent model (`ν = 3/5`) and a collapsed-globule model (`ν = 1/3`) with literally the same
  curve at every condition.  Solvent quality is not a one-separation measurement.

* **Two separations suffice, and this is the new positive result.**
  `swell_bond_and_exponent_identified` — if the correlation profile is known independently (the
  bond-length-free separation-resolved panel of Part CXXXVI) and is non-zero at lags `1` *and* `2`,
  then a complete titration determines the bond length *and* the swelling exponent.  The mechanism
  is a two-stage slowest-rate argument: lag 1 has rate `b·1^ν = b` whatever the exponent, so it
  fixes the bond length first; with `b` known, the lag-1 terms cancel identically and lag 2 becomes
  the slowest surviving rate, `b·2^ν`, which fixes the exponent.  Hence
  `good_solvent_distinguishable_from_theta`: a swollen region and a theta-state region with the
  same known, short-range-active profile are separated by some ionic strength, at *any* pair of
  bond lengths.

The design consequence, on top of Parts CXXXIV–CXXXVIII: a model may report a swelling exponent
fitted from a salt titration only if the correlation profile comes from an independent probe and
is active at two distinct short separations.  Fitted from one separation the exponent is pure
prior — a good-solvent and a globule model then differ by no measurement, only by the bond length
each is allowed to absorb.
-/
import Mathlib
import RequestProject.DistanceLaw

set_option autoImplicit false

namespace IDR
namespace SwellingExponent

open Finset

/-! ## 0. The polymer-scaling distance law -/

/-- The screened kernel of a charge correlation at sequence separation `d` for a chain whose
internal distance law is `R(d) = b·d^ν`. -/
noncomputable def skern (b nu kappa : ℝ) (d : ℕ) : ℝ :=
  Real.exp (-(kappa * (b * (d : ℝ) ^ nu))) / (b * (d : ℝ) ^ nu)

/-- The titration curve of a correlation profile `c`, read over separations `m ≤ d < N`. -/
noncomputable def swellTail (m N : ℕ) (b nu kappa : ℝ) (c : ℕ → ℝ) : ℝ :=
  ∑ d ∈ Ico m N, skern b nu kappa d * c d

/-- The titration curve of a correlation profile `c` over a window of `N` residues, under the
polymer distance law with bond length `b` and swelling exponent `ν`. -/
noncomputable def swellCurve (N : ℕ) (b nu kappa : ℝ) (c : ℕ → ℝ) : ℝ :=
  swellTail 1 N b nu kappa c

lemma swellCurve_eq_tail (N : ℕ) (b nu kappa : ℝ) (c : ℕ → ℝ) :
    swellCurve N b nu kappa c = swellTail 1 N b nu kappa c := rfl

/-- At `ν = 1/2` the polymer law is the Gaussian-chain Debye law of Part CXXXVIII. -/
lemma skern_half_eq_dkern (b kappa : ℝ) (d : ℕ) :
    skern b (1 / 2) kappa d = DebyeResolution.dkern b kappa d := by
  rw [skern, DebyeResolution.dkern, Real.sqrt_eq_rpow]

/-- **The ideal chain is the special case `ν = 1/2`.** -/
lemma swellCurve_half_eq_debyeCurve (N : ℕ) (b kappa : ℝ) (c : ℕ → ℝ) :
    swellCurve N b (1 / 2) kappa c = DebyeResolution.debyeCurve N b kappa c := by
  rw [swellCurve, swellTail, DebyeResolution.debyeCurve]
  exact Finset.sum_congr rfl fun d _ => by rw [skern_half_eq_dkern]

/-! ## 1. Elementary facts about the rates `b·d^ν` -/

lemma one_le_rpow_nat {nu : ℝ} (hnu : 0 ≤ nu) {d : ℕ} (hd : 1 ≤ d) : (1 : ℝ) ≤ (d : ℝ) ^ nu := by
  have hd1 : (1 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
  calc (1 : ℝ) = (1 : ℝ) ^ nu := (Real.one_rpow nu).symm
    _ ≤ (d : ℝ) ^ nu := Real.rpow_le_rpow (by norm_num) hd1 hnu

lemma rpow_nat_pos {nu : ℝ} {d : ℕ} (hd : 1 ≤ d) : (0 : ℝ) < (d : ℝ) ^ nu := by
  have hd0 : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
  exact Real.rpow_pos_of_pos hd0 nu

lemma rpow_nat_lt_rpow_nat {nu : ℝ} (hnu : 0 < nu) {d e : ℕ} (hde : d < e) :
    ((d : ℝ)) ^ nu < ((e : ℝ)) ^ nu := by
  have hd0 : (0 : ℝ) ≤ (d : ℝ) := Nat.cast_nonneg d
  have hde' : (d : ℝ) < (e : ℝ) := by exact_mod_cast hde
  exact Real.rpow_lt_rpow hd0 hde' hnu

/-- The rate of a separation is strictly increasing in the separation. -/
lemma rate_lt_rate {b nu : ℝ} (hb : 0 < b) (hnu : 0 < nu) {d e : ℕ} (hde : d < e) :
    b * (d : ℝ) ^ nu < b * (e : ℝ) ^ nu :=
  mul_lt_mul_of_pos_left (rpow_nat_lt_rpow_nat hnu hde) hb

/-! ## 2. A complete titration still identifies the profile, for a declared law -/

lemma swellCurve_eq_genCurve (N : ℕ) (b nu kappa : ℝ) (c : ℕ → ℝ) :
    swellCurve N b nu kappa c
      = DistanceLaw.genCurve N (fun d => 1 / (b * (d : ℝ) ^ nu))
          (fun d => b * (d : ℝ) ^ nu) c kappa := by
  rw [swellCurve, swellTail, DistanceLaw.genCurve]
  exact Finset.sum_congr rfl fun d _ => by rw [skern]; ring

/-- **The polymer-law titration determines every charge correlation**, once bond length and
swelling exponent are declared. -/
theorem swell_profile_identifiable {N : ℕ} {b nu kappa0 : ℝ} (hb : 0 < b) (hnu : 0 < nu)
    {c c' : ℕ → ℝ}
    (h : ∀ kappa, kappa0 ≤ kappa → swellCurve N b nu kappa c = swellCurve N b nu kappa c') :
    ∀ d ∈ Ico 1 N, c d = c' d := by
  refine DistanceLaw.genCurve_identifies (fun d hd e he hde => ?_) (fun d hd => ?_)
    (fun k hk => by simpa [swellCurve_eq_genCurve] using h k hk)
  · simp only at hde
    rcases lt_trichotomy d e with hlt | heq | hgt
    · exact absurd hde (ne_of_lt (rate_lt_rate hb hnu hlt))
    · exact heq
    · exact absurd hde.symm (ne_of_lt (rate_lt_rate hb hnu hgt))
  · have hd1 : 1 ≤ d := (Finset.mem_Ico.1 hd).1
    have : (0 : ℝ) < b * (d : ℝ) ^ nu := mul_pos hb (rpow_nat_pos hd1)
    positivity

/-! ## 3. Bond length still trades against sequence separation -/

/-- **The even-lag confound of Part CXXXVIII, at general solvent quality.**  A profile supported
on even separations, read with bond length `b` and exponent `ν`, produces exactly the same
titration curve, at every ionic strength, as the profile of its even lags read with bond length
`b·2^ν` and the same exponent.  For the ideal chain (`ν = 1/2`) the factor is `√2`. -/
theorem even_lag_swelling_confound {N : ℕ} (b nu kappa : ℝ) {c : ℕ → ℝ}
    (hodd : ∀ d, ¬ (2 ∣ d) → c d = 0) :
    swellCurve (2 * N) b nu kappa c
      = swellCurve N (b * (2 : ℝ) ^ nu) nu kappa (fun d => c (2 * d)) := by
  classical
  have hterm : ∀ d : ℕ, 1 ≤ d →
      skern b nu kappa (2 * d) * c (2 * d)
        = skern (b * (2 : ℝ) ^ nu) nu kappa d * c (2 * d) := by
    intro d hd
    have hcast : ((2 * d : ℕ) : ℝ) = (2 : ℝ) * (d : ℝ) := by push_cast; ring
    have hd0 : (0 : ℝ) ≤ (d : ℝ) := Nat.cast_nonneg d
    have hrp : ((2 * d : ℕ) : ℝ) ^ nu = (2 : ℝ) ^ nu * (d : ℝ) ^ nu := by
      rw [hcast, Real.mul_rpow (by norm_num) hd0]
    have hassoc : b * ((2 : ℝ) ^ nu * (d : ℝ) ^ nu) = b * (2 : ℝ) ^ nu * (d : ℝ) ^ nu := by ring
    rw [skern, skern, hrp, hassoc]
  have hset : (Ico 1 (2 * N)).filter (fun d => 2 ∣ d) = (Ico 1 N).image (fun e => 2 * e) := by
    ext d
    simp only [Finset.mem_filter, Finset.mem_Ico, Finset.mem_image]
    constructor
    · rintro ⟨⟨h1, h2⟩, k, rfl⟩
      exact ⟨k, ⟨by omega, by omega⟩, rfl⟩
    · rintro ⟨k, ⟨h1, h2⟩, rfl⟩
      exact ⟨⟨by omega, by omega⟩, ⟨k, rfl⟩⟩
  have hfilter : ∑ d ∈ Ico 1 (2 * N), skern b nu kappa d * c d
      = ∑ d ∈ (Ico 1 N).image (fun e => 2 * e), skern b nu kappa d * c d := by
    rw [← hset]
    refine (Finset.sum_filter_of_ne ?_).symm
    intro d _ hne
    by_contra hdvd
    exact hne (by rw [hodd d hdvd, mul_zero])
  rw [swellCurve, swellCurve, swellTail, swellTail, hfilter,
    Finset.sum_image (by intro x _ y _ h; dsimp only at h; omega)]
  exact Finset.sum_congr rfl fun d hd => hterm d (Finset.mem_Ico.1 hd).1

/-- A profile carrying a single correlation `t` at separation `D` (Part CXXXVIII). -/
noncomputable def single (D : ℕ) (t : ℝ) : ℕ → ℝ := DistanceLaw.single D t

/-- **A single separation determines nothing about solvent quality.**  A region whose only charge
correlation sits at separation `D`, read with bond length `b` and exponent `ν`, produces literally
the same titration curve — at every ionic strength — as the same correlation read with *any* other
exponent `ν'`, at the bond length `b·D^ν/D^{ν'}` that matches the screening rate. -/
theorem single_lag_exponent_confound {N D : ℕ} (hD : 1 ≤ D) (b nu nu' kappa t : ℝ) :
    swellCurve N b nu kappa (single D t)
      = swellCurve N (b * (D : ℝ) ^ nu / (D : ℝ) ^ nu') nu' kappa (single D t) := by
  have hDpos : (0 : ℝ) < (D : ℝ) ^ nu' := rpow_nat_pos hD
  have hrate : b * (D : ℝ) ^ nu / (D : ℝ) ^ nu' * (D : ℝ) ^ nu' = b * (D : ℝ) ^ nu := by
    field_simp
  rw [swellCurve, swellCurve, swellTail, swellTail]
  refine Finset.sum_congr rfl fun d hd => ?_
  by_cases hdD : d = D
  · subst hdD
    rw [skern, skern, hrate]
  · simp [single, DistanceLaw.single, hdD]

/-- **A swollen region and a collapsed globule can be indistinguishable.**  With a single
correlation at separation `2`, the good-solvent model (`ν = 3/5`) and the globule model
(`ν = 1/3`) give the same titration curve at every ionic strength, at bond lengths differing by
the factor `2^{3/5−1/3}`.  No experiment of this kind reports solvent quality from one
separation. -/
theorem swollen_and_collapsed_indistinguishable_at_one_lag {N : ℕ} {b : ℝ} (hb : 0 < b) (t : ℝ) :
    ∃ b' : ℝ, 0 < b' ∧ b ≠ b' ∧
      ∀ kappa : ℝ, swellCurve N b (3 / 5) kappa (single 2 t)
        = swellCurve N b' (1 / 3) kappa (single 2 t) := by
  have h1 : (0 : ℝ) < (2 : ℝ) ^ (3 / 5 : ℝ) := Real.rpow_pos_of_pos (by norm_num) _
  have h2 : (0 : ℝ) < (2 : ℝ) ^ (1 / 3 : ℝ) := Real.rpow_pos_of_pos (by norm_num) _
  have hlt : (2 : ℝ) ^ (1 / 3 : ℝ) < (2 : ℝ) ^ (3 / 5 : ℝ) :=
    (Real.rpow_lt_rpow_left_iff (by norm_num)).2 (by norm_num)
  refine ⟨b * (2 : ℝ) ^ (3 / 5 : ℝ) / (2 : ℝ) ^ (1 / 3 : ℝ), by positivity, ?_, fun kappa => ?_⟩
  · intro hcon
    have : b * (2 : ℝ) ^ (3 / 5 : ℝ) = b * (2 : ℝ) ^ (1 / 3 : ℝ) := by
      field_simp at hcon
      linarith [hcon]
    nlinarith
  · have := single_lag_exponent_confound (N := N) (D := 2) (by norm_num) b (3 / 5) (1 / 3) kappa t
    simpa using this

/-! ## 4. Two separations identify both the bond length and the exponent -/

/-- The screening rates of two competing polymer laws, indexed so that separation `d` of the
`(b, ν)`-model has index `d` and separation `d` of the `(b', ν')`-model has index `N + d`. -/
noncomputable def jrate (N : ℕ) (b nu b' nu' : ℝ) : ℕ → ℝ :=
  fun j => if j < N then b * (j : ℝ) ^ nu else b' * ((j : ℝ) - N) ^ nu'

/-- The amplitudes of the difference of the two models, in the same indexing. -/
noncomputable def jamp (N : ℕ) (b nu b' nu' : ℝ) (c c' : ℕ → ℝ) : ℕ → ℝ :=
  fun j => if j < N then c j / (b * (j : ℝ) ^ nu)
    else -(c' (j - N) / (b' * ((j : ℝ) - N) ^ nu'))

/-- The difference of two polymer-law titration curves, with different bond lengths and different
swelling exponents, is an exponential sum in the ionic strength. -/
theorem swellTail_sub_eq_expSum (m N : ℕ) (b nu b' nu' : ℝ) (c c' : ℕ → ℝ) (kappa : ℝ) :
    swellTail m N b nu kappa c - swellTail m N b' nu' kappa c'
      = DistanceLaw.expSum ((Ico m N) ∪ (Ico m N).image (fun d => N + d))
          (jamp N b nu b' nu' c c') (jrate N b nu b' nu') kappa := by
  classical
  have hdisj : Disjoint (Ico m N) ((Ico m N).image (fun d => N + d)) := by
    rw [Finset.disjoint_left]
    intro x hx hx'
    rw [Finset.mem_Ico] at hx
    simp only [Finset.mem_image, Finset.mem_Ico] at hx'
    obtain ⟨e, he, rfl⟩ := hx'
    omega
  have e1 : ∑ d ∈ Ico m N,
      jamp N b nu b' nu' c c' d * Real.exp (-(kappa * jrate N b nu b' nu' d))
        = swellTail m N b nu kappa c := by
    rw [swellTail]
    refine Finset.sum_congr rfl fun d hd => ?_
    have h1 : d < N := (Finset.mem_Ico.1 hd).2
    simp only [jamp, jrate, if_pos h1, skern]
    ring
  have e2 : ∑ d ∈ Ico m N,
      jamp N b nu b' nu' c c' (N + d) * Real.exp (-(kappa * jrate N b nu b' nu' (N + d)))
        = -swellTail m N b' nu' kappa c' := by
    rw [swellTail, ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun d _ => ?_
    have h2 : ¬ (N + d < N) := by omega
    have hc : ((N + d : ℕ) : ℝ) - (N : ℝ) = (d : ℝ) := by push_cast; ring
    simp only [jamp, jrate, if_neg h2, hc, Nat.add_sub_cancel_left, skern]
    ring
  rw [DistanceLaw.expSum, Finset.sum_union hdisj,
    Finset.sum_image (by intro x _ y _ h; dsimp only at h; omega), e1, e2]
  ring

/-- **The slowest surviving rate.**  If the separation-`m` rate of the first model is strictly
smaller than every other rate in play, then the two curves can agree at every ionic strength only
if the first model's correlation at separation `m` vanishes. -/
theorem tail_min_rate {m N : ℕ} (hmN : m < N) {b nu b' nu' kappa0 : ℝ} {c c' : ℕ → ℝ}
    (hb : 0 < b) (hm1 : 1 ≤ m)
    (hmin : ∀ d, m ≤ d → d < N → d ≠ m → b * (m : ℝ) ^ nu < b * (d : ℝ) ^ nu)
    (hmin' : ∀ d, m ≤ d → d < N → b * (m : ℝ) ^ nu < b' * (d : ℝ) ^ nu')
    (h : ∀ kappa, kappa0 ≤ kappa →
      swellTail m N b nu kappa c = swellTail m N b' nu' kappa c') :
    c m = 0 := by
  classical
  set S : Finset ℕ := (Ico m N) ∪ (Ico m N).image (fun d => N + d) with hS
  have hzero : ∀ kappa, kappa0 ≤ kappa →
      DistanceLaw.expSum S (jamp N b nu b' nu' c c') (jrate N b nu b' nu') kappa = 0 := by
    intro k hk
    rw [hS, ← swellTail_sub_eq_expSum, h k hk, sub_self]
  have hmS : m ∈ S := Finset.mem_union_left _ (Finset.mem_Ico.2 ⟨le_refl m, hmN⟩)
  have hratem : jrate N b nu b' nu' m = b * (m : ℝ) ^ nu := by
    simp [jrate, hmN]
  have hmin_all : ∀ j ∈ S, j ≠ m → jrate N b nu b' nu' m < jrate N b nu b' nu' j := by
    intro j hj hne
    rw [hratem]
    rcases Finset.mem_union.1 hj with hj1 | hj2
    · rw [Finset.mem_Ico] at hj1
      have hval : jrate N b nu b' nu' j = b * (j : ℝ) ^ nu := by simp [jrate, hj1.2]
      rw [hval]
      exact hmin j hj1.1 hj1.2 hne
    · simp only [Finset.mem_image, Finset.mem_Ico] at hj2
      obtain ⟨e, he, rfl⟩ := hj2
      have hlt2 : ¬ (N + e < N) := by omega
      have hcast : ((N + e : ℕ) : ℝ) - (N : ℝ) = (e : ℝ) := by push_cast; ring
      have hval : jrate N b nu b' nu' (N + e) = b' * (e : ℝ) ^ nu' := by
        simp [jrate, hlt2]
      rw [hval]
      exact hmin' e he.1 he.2
  have hamp := DistanceLaw.expSum_min_rate_zero hmS hmin_all hzero
  have hampval : jamp N b nu b' nu' c c' m = c m / (b * (m : ℝ) ^ nu) := by
    simp [jamp, hmN]
  rw [hampval, div_eq_zero_iff] at hamp
  rcases hamp with h1 | h2
  · exact h1
  · exact absurd h2 (ne_of_gt (mul_pos hb (rpow_nat_pos hm1)))

/-- **The bond length is fixed by the shortest separation, whatever the exponent.**  If the two
models' curves agree at every condition and the first model has the smaller bond length, then its
lag-1 correlation vanishes: the rate of separation `1` is `b·1^ν = b` for every exponent, so it is
strictly the slowest rate in play and nothing can cancel it. -/
theorem lag1_zero_of_bond_length_lt {N : ℕ} (hN : 2 ≤ N) {b b' nu nu' kappa0 : ℝ} (hb : 0 < b)
    (hnu : 0 < nu) (hnu' : 0 < nu') (hlt : b < b') {c c' : ℕ → ℝ}
    (h : ∀ kappa, kappa0 ≤ kappa → swellCurve N b nu kappa c = swellCurve N b' nu' kappa c') :
    c 1 = 0 := by
  have hone : ((1 : ℕ) : ℝ) ^ nu = 1 := by
    simp
  refine tail_min_rate (m := 1) (by omega) hb (le_refl 1) ?_ ?_ (fun k hk => h k hk)
  · intro d _ hdN hne
    exact rate_lt_rate hb hnu (by omega)
  · intro d hd1 _
    have h1 : (1 : ℝ) ≤ (d : ℝ) ^ nu' := one_le_rpow_nat hnu'.le hd1
    rw [hone, mul_one]
    nlinarith

/-- With the bond length known, the lag-1 terms cancel identically and the *second* separation
carries the slowest surviving rate; so if the first model has the smaller exponent, its lag-2
correlation vanishes. -/
theorem lag2_zero_of_exponent_lt {N : ℕ} (hN : 3 ≤ N) {b nu nu' kappa0 : ℝ} (hb : 0 < b)
    (hnu : 0 < nu) (hlt : nu < nu') {c : ℕ → ℝ}
    (h : ∀ kappa, kappa0 ≤ kappa → swellCurve N b nu kappa c = swellCurve N b nu' kappa c) :
    c 2 = 0 := by
  have hone : ((1 : ℕ) : ℝ) ^ nu = 1 := by simp
  have hone' : ((1 : ℕ) : ℝ) ^ nu' = 1 := by simp
  have hnu'pos : 0 < nu' := lt_trans hnu hlt
  -- The two lag-1 terms are identical, so the tails from separation 2 agree.
  have htail : ∀ kappa, kappa0 ≤ kappa →
      swellTail 2 N b nu kappa c = swellTail 2 N b nu' kappa c := by
    intro k hk
    have hsplit : ∀ e : ℝ, swellCurve N b e k c = skern b e k 1 * c 1 + swellTail 2 N b e k c := by
      intro e
      rw [swellCurve, swellTail, swellTail]
      exact Finset.sum_eq_sum_Ico_succ_bot (by omega) _
    have h1 : skern b nu k 1 = skern b nu' k 1 := by
      rw [skern, skern, hone, hone']
    have := h k hk
    rw [hsplit nu, hsplit nu', h1] at this
    linarith
  have h2rate : (2 : ℝ) ^ nu < (2 : ℝ) ^ nu' :=
    (Real.rpow_lt_rpow_left_iff (by norm_num)).2 hlt
  refine tail_min_rate (m := 2) (by omega) hb (by omega) ?_ ?_ htail
  · intro d hd2 _ hne
    have : (2 : ℕ) < d := by omega
    exact rate_lt_rate hb hnu this
  · intro d hd2 _
    have hmono : ((2 : ℕ) : ℝ) ^ nu' ≤ (d : ℝ) ^ nu' := by
      rcases eq_or_lt_of_le hd2 with heq | hlt2
      · rw [heq]
      · exact le_of_lt (rpow_nat_lt_rpow_nat hnu'pos hlt2)
    have hcast : ((2 : ℕ) : ℝ) = (2 : ℝ) := by norm_num
    rw [hcast] at hmono ⊢
    nlinarith

/-- **The main result: two active separations identify the whole distance law.**  If the
correlation profile is known independently — by the bond-length-free separation-resolved panel of
Part CXXXVI — and is non-zero at separations `1` and `2`, then agreement of the titration curves
at every ionic strength forces the same bond length *and* the same swelling exponent.  Solvent
quality is therefore a measurable parameter of the model, not a modelling choice, as soon as two
short separations are active. -/
theorem swell_bond_and_exponent_identified {N : ℕ} (hN : 3 ≤ N) {b b' nu nu' kappa0 : ℝ}
    (hb : 0 < b) (hb' : 0 < b') (hnu : 0 < nu) (hnu' : 0 < nu') {c : ℕ → ℝ}
    (hc1 : c 1 ≠ 0) (hc2 : c 2 ≠ 0)
    (h : ∀ kappa, kappa0 ≤ kappa → swellCurve N b nu kappa c = swellCurve N b' nu' kappa c) :
    b = b' ∧ nu = nu' := by
  have hbb : b = b' := by
    rcases lt_trichotomy b b' with hlt | heq | hgt
    · exact absurd (lag1_zero_of_bond_length_lt (by omega) hb hnu hnu' hlt h) hc1
    · exact heq
    · exact absurd
        (lag1_zero_of_bond_length_lt (by omega) hb' hnu' hnu hgt (fun k hk => (h k hk).symm)) hc1
  subst hbb
  refine ⟨rfl, ?_⟩
  rcases lt_trichotomy nu nu' with hlt | heq | hgt
  · exact absurd (lag2_zero_of_exponent_lt hN hb hnu hlt h) hc2
  · exact heq
  · exact absurd (lag2_zero_of_exponent_lt hN hb hnu' hgt (fun k hk => (h k hk).symm)) hc2

/-- **Good solvent is distinguishable from the theta state.**  A swollen region (`ν = 3/5`) and a
theta-state region (`ν = 1/2`) carrying the same known profile, active at separations `1` and `2`,
are separated by some ionic strength — at *any* pair of bond lengths.  Contrast
`swollen_and_collapsed_indistinguishable_at_one_lag`: with a single active separation they are
not. -/
theorem good_solvent_distinguishable_from_theta {N : ℕ} (hN : 3 ≤ N) {b b' kappa0 : ℝ}
    (hb : 0 < b) (hb' : 0 < b') {c : ℕ → ℝ} (hc1 : c 1 ≠ 0) (hc2 : c 2 ≠ 0) :
    ∃ kappa, kappa0 ≤ kappa ∧
      swellCurve N b (3 / 5) kappa c ≠ swellCurve N b' (1 / 2) kappa c := by
  by_contra hcon
  push_neg at hcon
  have := swell_bond_and_exponent_identified hN hb hb' (by norm_num) (by norm_num) hc1 hc2
    (fun k hk => hcon k hk)
  norm_num at this

/-! ## 4½. What identification costs: solvent quality is hidden at high salt -/

/-- Splitting the titration curve into its shortest-separation term and the rest. -/
lemma swellCurve_split {N : ℕ} (hN : 1 < N) (b nu kappa : ℝ) (c : ℕ → ℝ) :
    swellCurve N b nu kappa c = skern b nu kappa 1 * c 1 + swellTail 2 N b nu kappa c := by
  rw [swellCurve, swellTail, swellTail]
  exact Finset.sum_eq_sum_Ico_succ_bot hN _

/-- Every separation beyond the first is screened at least as hard as the rate `b·2^{ν_min}`. -/
lemma skern_le_of_two_le {b nu kappa numin : ℝ} {d : ℕ} (hb : 0 < b) (hnumin : 0 < numin)
    (hnu : numin ≤ nu) (hd : 2 ≤ d) (hk : 0 ≤ kappa) :
    |skern b nu kappa d|
      ≤ Real.exp (-(kappa * (b * (2 : ℝ) ^ numin))) / (b * (2 : ℝ) ^ numin) := by
  have h2pos : (0 : ℝ) < (2 : ℝ) ^ numin := Real.rpow_pos_of_pos (by norm_num) _
  have hrpos : (0 : ℝ) < b * (2 : ℝ) ^ numin := mul_pos hb h2pos
  have hmono : (2 : ℝ) ^ numin ≤ (d : ℝ) ^ nu := by
    have h1 : (2 : ℝ) ^ numin ≤ (2 : ℝ) ^ nu :=
      Real.rpow_le_rpow_left_iff (by norm_num) |>.2 hnu
    have h2 : (2 : ℝ) ^ nu ≤ (d : ℝ) ^ nu := by
      have hd' : (2 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
      exact Real.rpow_le_rpow (by norm_num) hd' (lt_of_lt_of_le hnumin hnu).le
    linarith
  have hdpos : (0 : ℝ) < b * (d : ℝ) ^ nu := by nlinarith
  have hexp : Real.exp (-(kappa * (b * (d : ℝ) ^ nu)))
      ≤ Real.exp (-(kappa * (b * (2 : ℝ) ^ numin))) := by
    refine Real.exp_le_exp.2 ?_
    have hbm : b * (2 : ℝ) ^ numin ≤ b * (d : ℝ) ^ nu := mul_le_mul_of_nonneg_left hmono hb.le
    have := mul_le_mul_of_nonneg_left hbm hk
    linarith
  have habs : |skern b nu kappa d| = skern b nu kappa d := by
    rw [abs_of_pos]
    rw [skern]
    exact div_pos (Real.exp_pos _) hdpos
  rw [habs, skern]
  exact div_le_div₀ (le_of_lt (Real.exp_pos _)) hexp hrpos (by nlinarith)

/-- The part of the curve carried by separations `2` and beyond is bounded by the total
correlation mass times the screening factor of the slowest of those separations. -/
lemma swellTail_two_abs_le {N : ℕ} {b nu kappa numin : ℝ} (hb : 0 < b) (hnumin : 0 < numin)
    (hnu : numin ≤ nu) (hk : 0 ≤ kappa) (c : ℕ → ℝ) :
    |swellTail 2 N b nu kappa c|
      ≤ (∑ d ∈ Ico 2 N, |c d|)
          * (Real.exp (-(kappa * (b * (2 : ℝ) ^ numin))) / (b * (2 : ℝ) ^ numin)) := by
  have h2pos : (0 : ℝ) < (2 : ℝ) ^ numin := Real.rpow_pos_of_pos (by norm_num) _
  have hrpos : (0 : ℝ) < b * (2 : ℝ) ^ numin := mul_pos hb h2pos
  rw [swellTail, Finset.sum_mul]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun d hd => ?_)
  have hd2 : 2 ≤ d := (Finset.mem_Ico.1 hd).1
  rw [abs_mul, mul_comm (|c d|)]
  exact mul_le_mul_of_nonneg_right (skern_le_of_two_le hb hnumin hnu hd2 hk) (abs_nonneg _)

/-- **Reading the swelling exponent costs exponential precision.**  Two models with the same bond
length and the same, known correlation profile but different exponents `ν, ν' ≥ ν_min` differ, at
ionic strength `κ`, by at most `2·(total correlation mass)·e^{−κ·b·2^{ν_min}}/(b·2^{ν_min})`.  The
shortest separation cancels exactly — its rate is `b` whatever the exponent — so all the
information about solvent quality sits behind the screening factor of separation `2`. -/
theorem exponent_difference_bound {N : ℕ} (hN : 1 < N) {b nu nu' kappa numin : ℝ} (hb : 0 < b)
    (hnumin : 0 < numin) (hnu : numin ≤ nu) (hnu' : numin ≤ nu') (hk : 0 ≤ kappa) (c : ℕ → ℝ) :
    |swellCurve N b nu kappa c - swellCurve N b nu' kappa c|
      ≤ 2 * (∑ d ∈ Ico 2 N, |c d|)
          * (Real.exp (-(kappa * (b * (2 : ℝ) ^ numin))) / (b * (2 : ℝ) ^ numin)) := by
  have hone : ((1 : ℕ) : ℝ) ^ nu = 1 := by simp
  have hone' : ((1 : ℕ) : ℝ) ^ nu' = 1 := by simp
  have hlag1 : skern b nu kappa 1 = skern b nu' kappa 1 := by
    rw [skern, skern, hone, hone']
  rw [swellCurve_split hN b nu kappa c, swellCurve_split hN b nu' kappa c, hlag1]
  have hsub : skern b nu' kappa 1 * c 1 + swellTail 2 N b nu kappa c
      - (skern b nu' kappa 1 * c 1 + swellTail 2 N b nu' kappa c)
      = swellTail 2 N b nu kappa c - swellTail 2 N b nu' kappa c := by ring
  rw [hsub]
  have h1 := swellTail_two_abs_le (N := N) hb hnumin hnu hk c
  have h2 := swellTail_two_abs_le (N := N) hb hnumin hnu' hk c
  calc |swellTail 2 N b nu kappa c - swellTail 2 N b nu' kappa c|
      ≤ |swellTail 2 N b nu kappa c| + |swellTail 2 N b nu' kappa c| := abs_sub _ _
    _ ≤ 2 * (∑ d ∈ Ico 2 N, |c d|)
          * (Real.exp (-(kappa * (b * (2 : ℝ) ^ numin))) / (b * (2 : ℝ) ^ numin)) := by
        linarith

/-- **A resolution horizon for solvent quality.**  Above the ionic strength
`κ* = log(2·M/(b·2^{ν_min}·ε)) / (b·2^{ν_min})`, where `M` is the total correlation mass beyond the
first separation, *no* pair of exponents `ν, ν' ≥ ν_min` can be told apart by a titration read to
precision `ε`.  Identification (`swell_bond_and_exponent_identified`) therefore has to be done at
low salt: the exact result is a statement about the whole half-line of conditions, and the price
of moving up it is exponential. -/
theorem solvent_quality_resolution_horizon {N : ℕ} (hN : 1 < N) {b numin eps : ℝ} (hb : 0 < b)
    (hnumin : 0 < numin) (heps : 0 < eps) (c : ℕ → ℝ) (hM : 0 < ∑ d ∈ Ico 2 N, |c d|) :
    ∀ nu nu' kappa : ℝ, numin ≤ nu → numin ≤ nu' →
      Real.log (2 * (∑ d ∈ Ico 2 N, |c d|) / (b * (2 : ℝ) ^ numin * eps))
          / (b * (2 : ℝ) ^ numin) ≤ kappa →
      0 ≤ kappa →
      |swellCurve N b nu kappa c - swellCurve N b nu' kappa c| ≤ eps := by
  intro nu nu' kappa hnu hnu' hkappa hk
  set r : ℝ := b * (2 : ℝ) ^ numin with hr
  have h2pos : (0 : ℝ) < (2 : ℝ) ^ numin := Real.rpow_pos_of_pos (by norm_num) _
  have hrpos : (0 : ℝ) < r := mul_pos hb h2pos
  set M : ℝ := ∑ d ∈ Ico 2 N, |c d| with hMdef
  have hquot : (0 : ℝ) < 2 * M / (r * eps) := by positivity
  have hlog : Real.log (2 * M / (r * eps)) ≤ kappa * r := by
    have := (div_le_iff₀ hrpos).1 hkappa
    linarith
  have hexp : Real.exp (-(kappa * r)) ≤ r * eps / (2 * M) := by
    have h1 : Real.exp (-(kappa * r)) ≤ Real.exp (-Real.log (2 * M / (r * eps))) :=
      Real.exp_le_exp.2 (by linarith)
    have h2 : Real.exp (-Real.log (2 * M / (r * eps))) = r * eps / (2 * M) := by
      rw [Real.exp_neg, Real.exp_log hquot]
      field_simp
    linarith [h1, h2.le, h2.ge]
  have hbound := exponent_difference_bound hN hb hnumin hnu hnu' hk c
  have hfinal : 2 * M * (Real.exp (-(kappa * r)) / r) ≤ eps := by
    calc 2 * M * (Real.exp (-(kappa * r)) / r)
        ≤ 2 * M * ((r * eps / (2 * M)) / r) := by gcongr
      _ = eps := by field_simp
  exact le_trans hbound hfinal

/-! ## 5. The law -/

/-- **The solvent-quality law.**  Under the realistic polymer distance law `R(d) = b·d^ν`:
the correlation profile is identified by a complete titration once the law is declared; a profile
active at a single separation leaves the entire `(b, ν)` line degenerate, so an exponent fitted
from one separation is prior, not measurement; a profile known independently and active at
separations `1` and `2` determines the bond length and the swelling exponent together; and that
identification is a low-salt statement — above an explicit ionic strength no two exponents can be
separated at finite precision. -/
theorem swelling_exponent_law {N : ℕ} {b kappa0 : ℝ} (hN : 3 ≤ N) (hb : 0 < b) :
    (∀ (nu : ℝ) (c c' : ℕ → ℝ), 0 < nu → (∀ kappa, kappa0 ≤ kappa →
        swellCurve N b nu kappa c = swellCurve N b nu kappa c') → ∀ d ∈ Ico 1 N, c d = c' d) ∧
    (∀ (nu nu' t : ℝ) (D : ℕ), 1 ≤ D → ∃ b' : ℝ, ∀ kappa : ℝ,
        swellCurve N b nu kappa (single D t) = swellCurve N b' nu' kappa (single D t)) ∧
    (∀ (b' nu nu' : ℝ) (c : ℕ → ℝ), 0 < b' → 0 < nu → 0 < nu' → c 1 ≠ 0 → c 2 ≠ 0 →
        (∀ kappa, kappa0 ≤ kappa → swellCurve N b nu kappa c = swellCurve N b' nu' kappa c) →
        b = b' ∧ nu = nu') ∧
    (∀ (numin eps : ℝ) (c : ℕ → ℝ), 0 < numin → 0 < eps → 0 < ∑ d ∈ Ico 2 N, |c d| →
        ∀ nu nu' kappa : ℝ, numin ≤ nu → numin ≤ nu' →
          Real.log (2 * (∑ d ∈ Ico 2 N, |c d|) / (b * (2 : ℝ) ^ numin * eps))
              / (b * (2 : ℝ) ^ numin) ≤ kappa → 0 ≤ kappa →
          |swellCurve N b nu kappa c - swellCurve N b nu' kappa c| ≤ eps) :=
  ⟨fun _ _ _ hnu h => swell_profile_identifiable hb hnu h,
   fun nu nu' t D hD => ⟨b * (D : ℝ) ^ nu / (D : ℝ) ^ nu',
     fun kappa => single_lag_exponent_confound hD b nu nu' kappa t⟩,
   fun _ _ _ _ hb' hnu hnu' hc1 hc2 h =>
     swell_bond_and_exponent_identified hN hb hb' hnu hnu' hc1 hc2 h,
   fun _ _ c hnumin heps hM =>
     solvent_quality_resolution_horizon (by omega) hb hnumin heps c hM⟩

end SwellingExponent
end IDR
