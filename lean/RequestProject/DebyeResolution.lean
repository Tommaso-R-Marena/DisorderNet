/-
# Part CXXXVII  How severe is the ill-conditioning?  The Debye–Hückel kernel

Part CXXXIV proved that reading charge correlations off a salt titration is ill-conditioned, with
a resolution floor `eps·e^{κ₀D}/D` at lag `D`.  That was computed for the Debye-damped *linear
chain* kernel `d·e^{−κd}` of Part CXXVI, in which the screening exponent is proportional to the
sequence separation.  The standard Debye–Hückel coupling on a Gaussian chain (Part LXXIII,
`Pattern.debye`) has exponent `κ b √d` instead, because the spatial distance between residues `d`
apart scales like `b√d`.  This part redoes the analysis for that kernel, and the answer is a
genuinely different rate.

* `debye_single_lag_gap`, `debye_single_lag_invisible`, `debye_resolution_floor` — the same three
  statements as in Part CXXXIV, with floor `eps · b√D · e^{κ₀ b √D}`;
* `debye_detectable` — and its converse: a discrepancy at or above the floor *is* seen at the
  condition in question, so the floor is the exact resolution;
* `debye_high_lags_unconstrained` — the horizon still exists: beyond a lag of order
  `4B/(eps κ₀² b³)` nothing is constrained;
* `demo_debye_lag16_resolved` — but it is much further out.  For `b = 1`, `κ₀ = 1` and a
  resolution of `10⁻³`, the lag-16 correlation is pinned to `±1/4` by the single condition
  `κ = 1`, where the exponentially screened chain kernel leaves an entire physical range free at
  lag 15 (Part CXXXV).

The moral is quantitative, and worth stating precisely because it is easy to get backwards.
Ill-conditioning is generic — any kernel that decays in separation filters the long lags — but
its *severity* is a property of the kernel.  For a kernel whose exponent is `κd` the horizon sits
at separation `≈ log(B/eps)/κ`; for one whose exponent is `κb√d` it sits at
`≈ (log(B/eps)/(κb))²`, the square.  A model of a charged disordered region must therefore
declare which distance law it assumes before it can quote a resolution on a long-range
correlation: the same data, read with the two kernels, support conclusions about very different
ranges of sequence separation.
-/
import Mathlib
import RequestProject.ChargePatterning
import RequestProject.TitrationResolution

set_option autoImplicit false

namespace IDR
namespace DebyeResolution

open Finset

/-! ## 1. The Debye–Hückel titration curve as a functional of the correlation profile -/

/-- The Debye–Hückel coupling between two charges `d` residues apart on a Gaussian chain of bond
length `b`, at inverse screening length `κ`. -/
noncomputable def dkern (b kappa : ℝ) (d : ℕ) : ℝ :=
  Real.exp (-(kappa * (b * Real.sqrt d))) / (b * Real.sqrt d)

/-- The Debye–Hückel energy as a functional of the charge-correlation profile. -/
noncomputable def debyeCurve (N : ℕ) (b kappa : ℝ) (c : ℕ → ℝ) : ℝ :=
  ∑ d ∈ Ico 1 N, dkern b kappa d * c d

/-- The measured Debye–Hückel energy of a sequence is the curve of its autocorrelation. -/
theorem debye_eq_curve (N : ℕ) (b kappa : ℝ) (q : ℕ → ℝ) :
    Pattern.debye N b kappa q = debyeCurve N b kappa (Pattern.autocorr N q) :=
  Pattern.pairEnergy_eq_sum_autocorr N _ q

lemma debyeCurve_bump {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N) (b kappa delta : ℝ) :
    debyeCurve N b kappa (Resolution.bump D delta) = dkern b kappa D * delta := by
  rw [debyeCurve, Finset.sum_eq_single D]
  · rw [Resolution.bump, if_pos rfl]
  · intro e _ he
    rw [Resolution.bump, if_neg he, mul_zero]
  · intro h
    exact absurd (Finset.mem_Ico.mpr ⟨hD1, hDN⟩) h

/-- **The single-lag gap for the Debye–Hückel kernel.** -/
theorem debye_single_lag_gap {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N) (b kappa : ℝ) (c : ℕ → ℝ)
    (delta : ℝ) :
    debyeCurve N b kappa (fun d => c d + Resolution.bump D delta d) - debyeCurve N b kappa c
      = dkern b kappa D * delta := by
  have h : debyeCurve N b kappa (fun d => c d + Resolution.bump D delta d)
      = debyeCurve N b kappa c + debyeCurve N b kappa (Resolution.bump D delta) := by
    rw [debyeCurve, debyeCurve, debyeCurve, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun d _ => by ring
  rw [h, debyeCurve_bump hD1 hDN, add_sub_cancel_left]

/-! ## 2. The floor, and that it is exact -/

lemma sqrt_pos_of_one_le {D : ℕ} (hD1 : 1 ≤ D) : 0 < Real.sqrt D := by
  have : (1 : ℝ) ≤ (D : ℝ) := by exact_mod_cast hD1
  exact Real.sqrt_pos.2 (by linarith)

/-- **The resolution floor of a Debye–Hückel titration.**  A perturbation of the lag-`D`
correlation by at most `eps · b√D · e^{κ₀ b √D}` moves the measured energy by at most `eps` at
every accessible ionic strength `κ ≥ κ₀`. -/
theorem debye_single_lag_invisible {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N) {b eps kappa0 kappa
    delta : ℝ} (hb : 0 < b) (hk : kappa0 ≤ kappa)
    (hdelta : |delta| ≤ eps * (b * Real.sqrt D) * Real.exp (kappa0 * (b * Real.sqrt D)))
    (c : ℕ → ℝ) :
    |debyeCurve N b kappa (fun d => c d + Resolution.bump D delta d)
      - debyeCurve N b kappa c| ≤ eps := by
  have hs : 0 < Real.sqrt D := sqrt_pos_of_one_le hD1
  have hbs : 0 < b * Real.sqrt D := by positivity
  rw [debye_single_lag_gap hD1 hDN b kappa c delta, dkern, abs_mul, abs_div,
    abs_of_pos (Real.exp_pos _), abs_of_pos hbs]
  have h1 : Real.exp (-(kappa * (b * Real.sqrt D)))
      ≤ Real.exp (-(kappa0 * (b * Real.sqrt D))) :=
    Real.exp_le_exp.mpr (by nlinarith)
  have h2 : Real.exp (-(kappa0 * (b * Real.sqrt D))) * Real.exp (kappa0 * (b * Real.sqrt D))
      = 1 := by
    rw [← Real.exp_add]; simp
  have hstep : Real.exp (-(kappa * (b * Real.sqrt D))) / (b * Real.sqrt D) * |delta|
      ≤ Real.exp (-(kappa0 * (b * Real.sqrt D))) / (b * Real.sqrt D)
        * (eps * (b * Real.sqrt D) * Real.exp (kappa0 * (b * Real.sqrt D))) := by
    gcongr
  have hfin : Real.exp (-(kappa0 * (b * Real.sqrt D))) / (b * Real.sqrt D)
      * (eps * (b * Real.sqrt D) * Real.exp (kappa0 * (b * Real.sqrt D)))
      = eps * (Real.exp (-(kappa0 * (b * Real.sqrt D)))
        * Real.exp (kappa0 * (b * Real.sqrt D))) := by
    field_simp
  rw [hfin, h2, mul_one] at hstep
  exact hstep

/-- The floor, in existential form: at every lag there is a competing profile, differing by the
floor, that no accessible condition can rule out. -/
theorem debye_resolution_floor {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N) {b eps kappa0 : ℝ}
    (hb : 0 < b) (heps : 0 ≤ eps) (c : ℕ → ℝ) :
    ∃ c' : ℕ → ℝ, (∀ d, d ≠ D → c' d = c d) ∧
      c' D - c D = eps * (b * Real.sqrt D) * Real.exp (kappa0 * (b * Real.sqrt D)) ∧
      ∀ kappa, kappa0 ≤ kappa →
        |debyeCurve N b kappa c' - debyeCurve N b kappa c| ≤ eps := by
  have hs : 0 < Real.sqrt D := sqrt_pos_of_one_le hD1
  set delta : ℝ := eps * (b * Real.sqrt D) * Real.exp (kappa0 * (b * Real.sqrt D)) with hdel
  have hnn : 0 ≤ delta := by
    rw [hdel]; positivity
  refine ⟨fun d => c d + Resolution.bump D delta d, ?_, ?_, ?_⟩
  · intro d hd; simp [Resolution.bump, hd]
  · simp [Resolution.bump]
  · intro kappa hkappa
    simpa using
      debye_single_lag_invisible hD1 hDN hb hkappa (by rw [abs_of_nonneg hnn]) c

/-- **And the floor is exact.**  A lag-`D` discrepancy at or above `eps · b√D · e^{κ b √D}` is
seen at the condition `κ`: the bound of `debye_single_lag_invisible` cannot be improved. -/
theorem debye_detectable {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N) {b eps kappa delta : ℝ}
    (hb : 0 < b)
    (hdelta : eps * (b * Real.sqrt D) * Real.exp (kappa * (b * Real.sqrt D)) ≤ |delta|)
    (c : ℕ → ℝ) :
    eps ≤ |debyeCurve N b kappa (fun d => c d + Resolution.bump D delta d)
      - debyeCurve N b kappa c| := by
  have hs : 0 < Real.sqrt D := sqrt_pos_of_one_le hD1
  have hbs : 0 < b * Real.sqrt D := by positivity
  rw [debye_single_lag_gap hD1 hDN b kappa c delta, dkern, abs_mul, abs_div,
    abs_of_pos (Real.exp_pos _), abs_of_pos hbs]
  have hE : 0 < Real.exp (-(kappa * (b * Real.sqrt D))) := Real.exp_pos _
  have hprod : Real.exp (-(kappa * (b * Real.sqrt D))) * Real.exp (kappa * (b * Real.sqrt D))
      = 1 := by
    rw [← Real.exp_add]; simp
  have hmul := mul_le_mul_of_nonneg_left hdelta
    (le_of_lt (div_pos hE hbs))
  have hid : Real.exp (-(kappa * (b * Real.sqrt D))) / (b * Real.sqrt D)
      * (eps * (b * Real.sqrt D) * Real.exp (kappa * (b * Real.sqrt D)))
      = eps * (Real.exp (-(kappa * (b * Real.sqrt D)))
        * Real.exp (kappa * (b * Real.sqrt D))) := by
    field_simp
  calc eps = Real.exp (-(kappa * (b * Real.sqrt D))) / (b * Real.sqrt D)
        * (eps * (b * Real.sqrt D) * Real.exp (kappa * (b * Real.sqrt D))) := by
        rw [hid, hprod, mul_one]
    _ ≤ Real.exp (-(kappa * (b * Real.sqrt D))) / (b * Real.sqrt D) * |delta| := hmul

/-! ## 3. The horizon is further out -/

lemma sq_div_four_le_exp {x : ℝ} (hx : 0 ≤ x) : x ^ 2 / 4 ≤ Real.exp x := by
  have h1 : 1 + x / 2 ≤ Real.exp (x / 2) := Real.add_one_le_exp _ |>.trans_eq' (by ring)
  have h2 : Real.exp (x / 2) * Real.exp (x / 2) = Real.exp x := by
    rw [← Real.exp_add]; ring_nf
  nlinarith [h1, h2]

/-- **The horizon for the Debye–Hückel kernel.**  Beyond a lag of order `4B/(eps κ₀² b³)` the
titration constrains nothing, whatever the a-priori range `B` of the correlation.  The threshold
is the same shape as for the exponentially screened chain kernel; what differs is how quickly the
floor grows before it — and here that is `e^{κ₀b√D}`, not `e^{κ₀D}`. -/
theorem debye_high_lags_unconstrained {b eps kappa0 : ℝ} (hb : 0 < b) (heps : 0 < eps)
    (hk0 : 0 < kappa0) (B : ℝ) :
    ∃ D₀ : ℕ, ∀ N D : ℕ, D₀ ≤ D → D < N → ∀ (c : ℕ → ℝ) (delta : ℝ), |delta| ≤ B →
      ∀ kappa, kappa0 ≤ kappa →
        |debyeCurve N b kappa (fun d => c d + Resolution.bump D delta d)
          - debyeCurve N b kappa c| ≤ eps := by
  obtain ⟨m, hm⟩ := exists_nat_ge (4 * B / (eps * kappa0 ^ 2 * b ^ 3))
  refine ⟨m + 1, fun N D hD hDN c delta hdelta kappa hk => ?_⟩
  have hD1 : 1 ≤ D := le_trans (Nat.le_add_left 1 m) hD
  have hmD : (m : ℝ) ≤ D := by exact_mod_cast le_trans (Nat.le_succ m) hD
  have hDR : (1 : ℝ) ≤ (D : ℝ) := by exact_mod_cast hD1
  have hs : 0 < Real.sqrt D := sqrt_pos_of_one_le hD1
  have hs1 : 1 ≤ Real.sqrt D := by
    rw [show (1 : ℝ) = Real.sqrt 1 by simp]
    exact Real.sqrt_le_sqrt hDR
  have hsq : Real.sqrt D ^ 2 = (D : ℝ) := Real.sq_sqrt (by linarith)
  have hexp : (kappa0 * (b * Real.sqrt D)) ^ 2 / 4
      ≤ Real.exp (kappa0 * (b * Real.sqrt D)) := sq_div_four_le_exp (by positivity)
  have hfloor : B ≤ eps * (b * Real.sqrt D) * Real.exp (kappa0 * (b * Real.sqrt D)) := by
    have h2 : 4 * B / (eps * kappa0 ^ 2 * b ^ 3) ≤ (D : ℝ) := le_trans hm hmD
    have hden : 0 < eps * kappa0 ^ 2 * b ^ 3 := by positivity
    rw [div_le_iff₀ hden] at h2
    have hkey : eps * (b * Real.sqrt D) * ((kappa0 * (b * Real.sqrt D)) ^ 2 / 4)
        = eps * kappa0 ^ 2 * b ^ 3 * (D : ℝ) * Real.sqrt D / 4 := by
      have : (kappa0 * (b * Real.sqrt D)) ^ 2 = kappa0 ^ 2 * b ^ 2 * Real.sqrt D ^ 2 := by ring
      rw [this, hsq]; ring
    have hchain : eps * (b * Real.sqrt D) * ((kappa0 * (b * Real.sqrt D)) ^ 2 / 4)
        ≤ eps * (b * Real.sqrt D) * Real.exp (kappa0 * (b * Real.sqrt D)) := by
      gcongr
    have hpos : 0 < eps * kappa0 ^ 2 * b ^ 3 * (D : ℝ) := by positivity
    have hstep : eps * kappa0 ^ 2 * b ^ 3 * (D : ℝ) * 1
        ≤ eps * kappa0 ^ 2 * b ^ 3 * (D : ℝ) * Real.sqrt D :=
      mul_le_mul_of_nonneg_left hs1 hpos.le
    rw [mul_one] at hstep
    have hB : B ≤ eps * kappa0 ^ 2 * b ^ 3 * (D : ℝ) * Real.sqrt D / 4 := by linarith
    linarith [hkey ▸ hchain]
  exact debye_single_lag_invisible hD1 hDN hb hk (hdelta.trans hfloor) c

/-! ## 4. The worked contrast -/

lemma exp_four_le : Real.exp 4 ≤ 62 := by
  have he : Real.exp 1 < 2.7182818286 := Real.exp_one_lt_d9
  have h : Real.exp 4 = (Real.exp 1) ^ (4 : ℕ) := by
    rw [← Real.exp_nat_mul]; norm_num
  rw [h]
  calc (Real.exp 1) ^ (4 : ℕ) ≤ (2.7182818286 : ℝ) ^ (4 : ℕ) :=
        pow_le_pow_left₀ (Real.exp_pos 1).le he.le 4
    _ ≤ 62 := by norm_num

lemma sqrt_sixteen : Real.sqrt ((16 : ℕ) : ℝ) = 4 := by
  rw [show (((16 : ℕ) : ℝ)) = 4 ^ 2 by norm_num]
  exact Real.sqrt_sq (by norm_num)

/-- **The worked contrast.**  With the Debye–Hückel Gaussian-chain kernel at bond length `b = 1`
and inverse screening length `κ = 1`, a discrepancy of `1/4` or more in the lag-16 correlation of
a twenty-residue region is detected at that single condition, at a resolution of `10⁻³`.  Under
the exponentially screened chain kernel of Part CXXXV, at the same `κ₀` and the same resolution,
even the largest physically possible lag-15 correlation is invisible.  The severity of the
ill-conditioning is a property of the assumed distance law, not of the region. -/
theorem demo_debye_lag16_resolved (c : ℕ → ℝ) {delta : ℝ} (hdelta : 1 / 4 ≤ |delta|) :
    1 / 1000 ≤ |debyeCurve 20 1 1 (fun d => c d + Resolution.bump 16 delta d)
      - debyeCurve 20 1 1 c| := by
  refine debye_detectable (N := 20) (D := 16) (by norm_num) (by norm_num) (by norm_num) ?_ c
  refine le_trans ?_ hdelta
  rw [sqrt_sixteen]
  have h : Real.exp ((1 : ℝ) * (1 * 4)) = Real.exp 4 := by norm_num
  rw [h]
  have := exp_four_le
  nlinarith

end DebyeResolution
end IDR
