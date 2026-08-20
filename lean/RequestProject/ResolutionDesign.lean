/-
# Part CXXXV  The resolution law in numbers: what a titration at physiological salt can say

Part CXXXIV (`TitrationResolution.lean`) proved that a salt titration confined to ionic
strengths `κ ≥ κ₀` determines the lag-`D` charge correlation of a disordered region only to
within `eps·e^{κ₀D}/D`, where `eps` is the energy resolution.  This part draws the practical
consequence, first in general and then for a worked case.

* `lag_unconstrained_of_bounded` — **the general criterion.**  If the a-priori range `B` of a
  correlation coefficient is already below the resolution floor, `B ≤ eps·e^{κ₀D}/D`, then the
  titration says *nothing whatever* about lag `D`: every admissible value of that coefficient is
  consistent with the data to within the resolution, at every accessible condition.
* `high_lags_unconstrained` — and since the floor grows exponentially, this happens for all lags
  beyond an explicit `D₀ = 4B/(eps·κ₀²)`, whatever the region, the instrument and the buffer.
  There is a hard horizon on the sequence separation a salt series can probe.

The worked case is a twenty-residue charged region, read to `10⁻³ kT`, with the lowest
accessible condition at `κ₀ = 1` in inverse residue units (a Debye length of one residue
spacing, comparable to physiological buffer).  Charge autocorrelations of a region of unit
charges are bounded by `20` in absolute value, so:

* `demo_lag10_floor` — the lag-10 correlation is undetermined over a range of at least `2`
  charge units, a substantial fraction of its physical range;
* `demo_lag15_unconstrained` — the lag-15 correlation is *completely* undetermined: every value
  in its whole physical range `[−20, 20]` fits the data at every accessible ionic strength;
* `demo_endpoint_invisible` with `demo_endpoint_zero_salt` — the sharpest form.  A contact
  between the two ends of the region is worth exactly `19 kT` in the unscreened model, and is
  invisible — below `10⁻³ kT` — at every condition of the titration;
* `physiological_resolution_report` — the three statements together.

The design consequence is the counterpart of Part CXXXIII.  That part said: use at least `N − 1`
salt conditions.  This one says: only the first handful of lags will be identified whatever you
do, so a model of a charged disordered region must either (i) get its long-range correlations
from an observable that is not a screened energy — a distance measurement, a contact frequency —
or (ii) report them as free parameters with the floor `eps·e^{κ₀D}/D` attached, and never as
fitted values.
-/
import Mathlib
import RequestProject.TitrationResolution

set_option autoImplicit false

namespace IDR
namespace ResolutionDesign

open Finset Titration Resolution

/-! ## 1. The general criterion -/

/-- **A lag below the resolution floor is unconstrained.**  If every admissible value of the
lag-`D` correlation lies within `B` of the fitted one and `B ≤ eps·e^{κ₀D}/D`, then every such
value reproduces the titration curve to within the resolution `eps` at every accessible
condition: the data contain no information about that lag. -/
theorem lag_unconstrained_of_bounded {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N) {eps kappa0 B : ℝ}
    (hB : B ≤ eps * Real.exp (kappa0 * D) / D) (c : ℕ → ℝ) :
    ∀ delta, |delta| ≤ B → ∀ kappa, kappa0 ≤ kappa →
      |curve N (fun d => c d + bump D delta d) kappa - curve N c kappa| ≤ eps := by
  intro delta hdelta kappa hk
  exact single_lag_invisible hD1 hDN (hdelta.trans hB) hk c

/-- **The horizon.**  For any resolution `eps > 0`, any lowest accessible condition `κ₀ > 0` and
any a-priori range `B`, every lag beyond `4B/(eps·κ₀²)` is unconstrained by the titration. -/
theorem high_lags_unconstrained {eps kappa0 : ℝ} (heps : 0 < eps) (hk0 : 0 < kappa0) (B : ℝ) :
    ∃ D₀ : ℕ, ∀ N D : ℕ, D₀ ≤ D → D < N → ∀ c : ℕ → ℝ, ∀ delta, |delta| ≤ B →
      ∀ kappa, kappa0 ≤ kappa →
        |curve N (fun d => c d + bump D delta d) kappa - curve N c kappa| ≤ eps := by
  obtain ⟨m, hm⟩ := exists_nat_ge (4 * B / (eps * kappa0 ^ 2))
  refine ⟨m + 1, fun N D hD hDN c delta hdelta kappa hk => ?_⟩
  have hD1 : 1 ≤ D := le_trans (Nat.le_add_left 1 m) hD
  have hDpos : (0 : ℝ) < D := by exact_mod_cast hD1
  have hmD : (m : ℝ) ≤ D := by exact_mod_cast le_trans (Nat.le_succ m) hD
  have hfloor : B ≤ eps * Real.exp (kappa0 * D) / D := by
    have h1 : kappa0 ^ 2 * D / 4 ≤ Real.exp (kappa0 * D) / D := exp_div_ge hk0 D hD1
    have h2 : 4 * B / (eps * kappa0 ^ 2) ≤ (D : ℝ) := le_trans hm hmD
    have hden : 0 < eps * kappa0 ^ 2 := by positivity
    rw [div_le_iff₀ hden] at h2
    have : eps * Real.exp (kappa0 * D) / D = eps * (Real.exp (kappa0 * D) / D) := by ring
    rw [this]
    nlinarith
  exact lag_unconstrained_of_bounded hD1 hDN hfloor c delta hdelta kappa hk

/-! ## 2. Numerical bounds on the exponential -/

lemma exp_pow_le (n : ℕ) : (2.7182818283 : ℝ) ^ n ≤ Real.exp n := by
  have he : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
  calc (2.7182818283 : ℝ) ^ n ≤ (Real.exp 1) ^ n := pow_le_pow_left₀ (by norm_num) he.le n
    _ = Real.exp n := by rw [← Real.exp_nat_mul]; ring_nf

lemma exp_ten_ge : (20000 : ℝ) ≤ Real.exp 10 := by
  have h := exp_pow_le 10
  norm_num at h ⊢
  linarith

lemma exp_fifteen_ge : (300000 : ℝ) ≤ Real.exp 15 := by
  have h := exp_pow_le 15
  norm_num at h ⊢
  linarith

lemma exp_nineteen_ge : (19000 : ℝ) ≤ Real.exp 19 := by
  have h := exp_pow_le 19
  norm_num at h ⊢
  linarith

/-! ## 3. The worked case: twenty residues, `10⁻³ kT`, `κ₀ = 1` -/

/-- **A two-charge-unit blind spot at lag 10.**  For a twenty-residue region read to `10⁻³ kT`
from conditions `κ ≥ 1`, the lag-10 correlation can be moved by more than two charge units
without moving any measurement by as much as the resolution. -/
theorem demo_lag10_floor (c : ℕ → ℝ) :
    ∃ c' : ℕ → ℝ, (∀ d, d ≠ 10 → c' d = c d) ∧ 2 ≤ c' 10 - c 10 ∧
      ∀ kappa, (1 : ℝ) ≤ kappa → |curve 20 c' kappa - curve 20 c kappa| ≤ 1 / 1000 := by
  obtain ⟨c', hkeep, hgap, hinv⟩ :=
    resolution_floor (N := 20) (D := 10) (eps := 1 / 1000) (kappa0 := 1) (by norm_num)
      (by norm_num) (by norm_num) c
  refine ⟨c', hkeep, ?_, hinv⟩
  rw [hgap]
  have h : Real.exp ((1 : ℝ) * (10 : ℕ)) = Real.exp 10 := by norm_num
  rw [h]
  have := exp_ten_ge
  norm_num
  linarith

/-- **Lag 15 is completely unconstrained.**  Charge autocorrelations of a twenty-residue region
of unit charges never exceed `20` in absolute value; at resolution `10⁻³ kT` and conditions
`κ ≥ 1`, *every* value in that range fits the data. -/
theorem demo_lag15_unconstrained (c : ℕ → ℝ) (delta : ℝ) (hdelta : |delta| ≤ 20)
    {kappa : ℝ} (hk : (1 : ℝ) ≤ kappa) :
    |curve 20 (fun d => c d + bump 15 delta d) kappa - curve 20 c kappa| ≤ 1 / 1000 := by
  refine lag_unconstrained_of_bounded (N := 20) (D := 15) (eps := 1 / 1000) (kappa0 := 1)
    (by norm_num) (by norm_num) ?_ c delta hdelta kappa hk
  have h : Real.exp ((1 : ℝ) * (15 : ℕ)) = Real.exp 15 := by norm_num
  rw [h]
  have := exp_fifteen_ge
  norm_num
  linarith

/-- **The end-to-end contact is worth 19 kT in the unscreened model.** -/
theorem demo_endpoint_zero_salt :
    Salt.energy 20 0 (qEnds 20) - Salt.energy 20 0 qOne = 19 := by
  rw [endpoint_pair_gap_zero_salt (by norm_num)]
  norm_num

/-- **And it is invisible at physiological salt.**  At every condition `κ ≥ 1` the same two
sequences differ by less than `10⁻³ kT`: four orders of magnitude of signal are destroyed by the
screening, and the titration cannot report the contact at all. -/
theorem demo_endpoint_invisible {kappa : ℝ} (hk : (1 : ℝ) ≤ kappa) :
    |Salt.energy 20 kappa (qEnds 20) - Salt.energy 20 kappa qOne| ≤ 1 / 1000 := by
  refine endpoint_pair_invisible (N := 20) (kappa0 := 1) (by norm_num) ?_ hk
  have h19 : ((20 : ℕ) : ℝ) - 1 = 19 := by norm_num
  rw [h19]
  have hexp : Real.exp (-((1 : ℝ) * 19)) = (Real.exp 19)⁻¹ := by
    rw [← Real.exp_neg]; norm_num
  rw [hexp]
  have h := exp_nineteen_ge
  have hpos : (0 : ℝ) < Real.exp 19 := Real.exp_pos 19
  rw [mul_inv_le_iff₀ hpos]
  linarith

/-- **The worked resolution record.**  A twenty-residue charged region, an energy resolution of
`10⁻³ kT`, and a titration whose lowest condition has a Debye length of one residue spacing:
(1) the lag-10 correlation is undetermined over at least two charge units; (2) the lag-15
correlation is undetermined over its entire physical range; (3) a contact between the two ends
of the region, worth `19 kT` unscreened, is invisible at every accessible condition. -/
theorem physiological_resolution_report :
    (∀ c : ℕ → ℝ, ∃ c' : ℕ → ℝ, (∀ d, d ≠ 10 → c' d = c d) ∧ 2 ≤ c' 10 - c 10 ∧
        ∀ kappa, (1 : ℝ) ≤ kappa → |curve 20 c' kappa - curve 20 c kappa| ≤ 1 / 1000) ∧
    (∀ (c : ℕ → ℝ) (delta : ℝ), |delta| ≤ 20 → ∀ kappa, (1 : ℝ) ≤ kappa →
        |curve 20 (fun d => c d + bump 15 delta d) kappa - curve 20 c kappa| ≤ 1 / 1000) ∧
    (Salt.energy 20 0 (qEnds 20) - Salt.energy 20 0 qOne = 19 ∧
      ∀ kappa, (1 : ℝ) ≤ kappa →
        |Salt.energy 20 kappa (qEnds 20) - Salt.energy 20 kappa qOne| ≤ 1 / 1000) :=
  ⟨demo_lag10_floor, fun c delta hd _ hk => demo_lag15_unconstrained c delta hd hk,
    demo_endpoint_zero_salt, fun _ hk => demo_endpoint_invisible hk⟩

end ResolutionDesign
end IDR
