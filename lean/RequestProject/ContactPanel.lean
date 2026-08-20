/-
# Part CXXXVI  The positive complement: a separation-resolved probe panel is stably invertible

Parts CXXXIV–CXXXV showed that the salt titration is an exponentially ill-conditioned way of
reading the charge correlations of a disordered region: the lag-`D` correlation is undetermined
below `eps·e^{κ₀D}/D`, so beyond a handful of lags nothing is identified whatever the design.
That is a statement about *one* observable, not about the region.  This part exhibits an
observable of the same pairwise model class whose inversion is *uniformly stable*, and thereby
turns the limitative results into a positive design rule.

The observable is a crosslinking-style panel: a reagent whose reach is `D` reports the total
charge correlation out to separation `D`,

    S(D) = ∑_{d=1}^{D} C(d),

which is exactly the pairwise energy `Pattern.pairEnergy` of Part LXXIII with the cumulative
kernel `cumKernel D` (`panelReading_eq_partial_sum`).  Then:

* `panel_inverts` — the inversion is the first difference, `C(D) = S(D) − S(D−1)`: no matrix, no
  conditioning, no extrapolation;
* `panel_stability` — **the stability theorem.**  If two regions' panel readings agree to `eps`,
  their charge correlations agree to `2 eps` at *every* lag.  The constant is `2`: independent of
  the lag, of the length of the region and of any solution condition.  Compare the titration,
  where the corresponding constant is `e^{κ₀D}/D`;
* `panel_identifies` — exact readings identify the autocorrelation exactly, hence (Part LXXIII)
  the pairwise energetics under every kernel and every ionic strength;
* `panel_resolves_lag15` with `titration_leaves_lag15_free` — the two observables side by side on
  the worked case of Part CXXXV: at a resolution of `10⁻³`, the panel pins the lag-15 correlation
  of a twenty-residue region to `±2·10⁻³`, while the salt titration leaves it free over its whole
  physical range `[−20, 20]`;
* `probe_choice_law` — the design rule that follows.

So the correct conclusion of Parts CXXXIV–CXXXV is not that long-range correlations in a
disordered region are unknowable, but that they must be read from a *separation-resolved*
observable.  Screening is an exponential filter on sequence separation, and no amount of salt
data undoes it; a probe whose kernel is a step in separation has no such filter, and inverts with
an absolute constant.
-/
import Mathlib
import RequestProject.ChargePatterning
import RequestProject.ResolutionDesign

set_option autoImplicit false

namespace IDR
namespace ContactPanel

open Finset

/-! ## 1. The panel and its inversion -/

/-- The cumulative kernel of a reagent whose reach is `D`: it couples every pair of residues at
separation at most `D`. -/
def cumKernel (D : ℕ) : ℕ → ℝ := fun e => if e ≤ D then 1 else 0

/-- The reading of the reagent of reach `D` on the sequence `q`. -/
def panelReading (N : ℕ) (q : ℕ → ℝ) (D : ℕ) : ℝ := Pattern.pairEnergy N (cumKernel D) q

/-- **What the panel measures.**  The reagent of reach `D` reports the total charge correlation
out to separation `D`. -/
theorem panelReading_eq_partial_sum {N D : ℕ} (hD : D < N) (q : ℕ → ℝ) :
    panelReading N q D = ∑ d ∈ Ico 1 (D + 1), Pattern.autocorr N q d := by
  rw [panelReading, Pattern.pairEnergy_eq_sum_autocorr]
  have hfil : (Ico 1 N).filter (fun d => d ≤ D) = Ico 1 (D + 1) := by
    ext d
    simp only [Finset.mem_filter, Finset.mem_Ico]
    omega
  rw [← hfil, Finset.sum_filter]
  refine Finset.sum_congr rfl fun d _ => ?_
  by_cases h : d ≤ D <;> simp [cumKernel, h]

/-- **The inversion is a first difference.**  The lag-`D` correlation is the increment of the
panel between reach `D − 1` and reach `D`. -/
theorem panel_inverts {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N) (q : ℕ → ℝ) :
    panelReading N q D - panelReading N q (D - 1) = Pattern.autocorr N q D := by
  have hD' : D - 1 < N := by omega
  rw [panelReading_eq_partial_sum hDN, panelReading_eq_partial_sum hD']
  have hsucc : D - 1 + 1 = D := by omega
  rw [hsucc, Finset.sum_Ico_succ_top hD1]
  ring

/-! ## 2. Stability -/

/-- **The stability theorem.**  If the panel readings of two regions agree to within `eps`, then
their charge correlations agree to within `2 eps` at every lag.  The constant is absolute: it
does not grow with the lag, with the length of the region, or with the ionic strength — in sharp
contrast with the exponential floor `eps·e^{κ₀D}/D` of the salt titration (Part CXXXIV). -/
theorem panel_stability {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N) {eps : ℝ} {q q' : ℕ → ℝ}
    (h : ∀ j, j < N → |panelReading N q j - panelReading N q' j| ≤ eps) :
    |Pattern.autocorr N q D - Pattern.autocorr N q' D| ≤ 2 * eps := by
  have hD' : D - 1 < N := by omega
  have hq := panel_inverts hD1 hDN q
  have hq' := panel_inverts hD1 hDN q'
  have hrw : Pattern.autocorr N q D - Pattern.autocorr N q' D
      = (panelReading N q D - panelReading N q' D)
        - (panelReading N q (D - 1) - panelReading N q' (D - 1)) := by
    rw [← hq, ← hq']; ring
  rw [hrw]
  calc |(panelReading N q D - panelReading N q' D)
          - (panelReading N q (D - 1) - panelReading N q' (D - 1))|
      ≤ |panelReading N q D - panelReading N q' D|
        + |panelReading N q (D - 1) - panelReading N q' (D - 1)| := abs_sub _ _
    _ ≤ eps + eps := add_le_add (h D hDN) (h (D - 1) hD')
    _ = 2 * eps := by ring

/-- **Exact readings identify the region exactly.**  Two sequences with identical panel readings
have identical charge autocorrelations, hence — by the completeness theorem of Part LXXIII —
identical pairwise energies under every separation kernel and every ionic strength. -/
theorem panel_identifies {N : ℕ} {q q' : ℕ → ℝ}
    (h : ∀ j, j < N → panelReading N q j = panelReading N q' j) :
    (∀ d, 1 ≤ d → d < N → Pattern.autocorr N q d = Pattern.autocorr N q' d) ∧
      ∀ w : ℕ → ℝ, Pattern.pairEnergy N w q = Pattern.pairEnergy N w q' := by
  have hcorr : ∀ d, 1 ≤ d → d < N → Pattern.autocorr N q d = Pattern.autocorr N q' d := by
    intro d hd1 hdN
    have := panel_stability hd1 hdN (eps := 0) (q := q) (q' := q')
      (fun j hj => by rw [h j hj]; simp)
    have h0 : |Pattern.autocorr N q d - Pattern.autocorr N q' d| ≤ 0 := by
      simpa using this
    have := abs_nonpos_iff.mp h0
    linarith [sub_eq_zero.mp this]
  exact ⟨hcorr, (Pattern.autocorr_eq_iff_pairEnergy_eq N q q').mp hcorr⟩

/-! ## 3. The two observables side by side, on the worked case of Part CXXXV -/

/-- With a panel read to `10⁻³`, the lag-15 correlation of a twenty-residue region is pinned to
`±2·10⁻³`. -/
theorem panel_resolves_lag15 {q q' : ℕ → ℝ}
    (h : ∀ j, j < 20 → |panelReading 20 q j - panelReading 20 q' j| ≤ 1 / 1000) :
    |Pattern.autocorr 20 q 15 - Pattern.autocorr 20 q' 15| ≤ 2 / 1000 := by
  have := panel_stability (N := 20) (D := 15) (by norm_num) (by norm_num) h
  linarith

/-- With a salt titration read to the same `10⁻³`, from conditions `κ ≥ 1`, the same coefficient
is free over its whole physical range: this is `demo_lag15_unconstrained` of Part CXXXV. -/
theorem titration_leaves_lag15_free (c : ℕ → ℝ) (delta : ℝ) (hdelta : |delta| ≤ 20)
    {kappa : ℝ} (hk : (1 : ℝ) ≤ kappa) :
    |Titration.curve 20 (fun d => c d + Resolution.bump 15 delta d) kappa
      - Titration.curve 20 c kappa| ≤ 1 / 1000 :=
  ResolutionDesign.demo_lag15_unconstrained c delta hdelta hk

/-! ## 4. The panel has a limitation of its own: its reach -/

/-- The panel reading of a correlation profile: the cumulative correlation out to reach `D`. -/
def panelCurve (c : ℕ → ℝ) (D : ℕ) : ℝ := ∑ d ∈ Ico 1 (D + 1), c d

lemma panelReading_eq_panelCurve {N D : ℕ} (hD : D < N) (q : ℕ → ℝ) :
    panelReading N q D = panelCurve (Pattern.autocorr N q) D :=
  panelReading_eq_partial_sum hD q

/-- **Reach is the panel's own horizon.**  A panel whose longest reagent reaches `R` says nothing
about any lag beyond `R`: the lag-`D₀` correlation can be moved by an arbitrary amount without
changing any reading.  The panel's stability constant is absolute, but it buys information only
out to the separations its reagents actually span. -/
theorem panel_reach_blind {R D₀ : ℕ} (hR : R < D₀) (c : ℕ → ℝ) (delta : ℝ) :
    ∀ D, D ≤ R → panelCurve (fun d => c d + Resolution.bump D₀ delta d) D = panelCurve c D := by
  intro D hD
  rw [panelCurve, panelCurve, Finset.sum_add_distrib, add_eq_left]
  refine Finset.sum_eq_zero fun d hd => ?_
  rw [Finset.mem_Ico] at hd
  have : d ≠ D₀ := by omega
  simp [Resolution.bump, this]

/-- **What it takes to constrain a lag at all.**  Against a lag-`D₀` discrepancy, a panel of
reach `R < D₀` and a titration whose conditions all lie at or above `log(D₀|δ|/eps)/D₀` are both
blind: every reading and every titration point is reproduced to within the resolution.  To learn
about separation `D₀` one needs either a reagent that spans it, or an ionic strength low enough
that the Debye length does. -/
theorem blind_to_lag_without_reach_or_low_salt {N R D₀ : ℕ} (hD1 : 1 ≤ D₀) (hD0N : D₀ < N)
    (hR : R < D₀) {eps : ℝ} (heps : 0 < eps) (c : ℕ → ℝ) (delta : ℝ) :
    (∀ D, D ≤ R → panelCurve (fun d => c d + Resolution.bump D₀ delta d) D = panelCurve c D) ∧
      (∀ kappa, Real.log ((D₀ : ℝ) * |delta| / eps) / D₀ ≤ kappa →
        |Titration.curve N (fun d => c d + Resolution.bump D₀ delta d) kappa
          - Titration.curve N c kappa| ≤ eps) := by
  refine ⟨panel_reach_blind hR c delta, fun kappa hkappa => ?_⟩
  by_contra hcon
  push_neg at hcon
  exact absurd (Resolution.separating_condition_below_threshold hD1 hD0N heps c hcon)
    (not_lt.mpr hkappa)

/-- **The probe-choice law.**  For the charge-correlation part of a model of a disordered region:
(1) a separation-resolved panel inverts by a first difference; (2) its inversion is `2`-Lipschitz,
uniformly in the lag, the length and the solution condition; (3) exact readings determine the
pairwise energetics under every kernel; and (4) on the worked twenty-residue case at resolution
`10⁻³` the panel pins the lag-15 correlation to `±2·10⁻³` where the salt titration leaves it free
over `[−20, 20]`.  Long-range charge correlations of a disordered region are identifiable — but
not from a screened energy. -/
theorem probe_choice_law :
    (∀ (N D : ℕ), 1 ≤ D → D < N → ∀ q : ℕ → ℝ,
        panelReading N q D - panelReading N q (D - 1) = Pattern.autocorr N q D) ∧
    (∀ (N D : ℕ), 1 ≤ D → D < N → ∀ (eps : ℝ) (q q' : ℕ → ℝ),
        (∀ j, j < N → |panelReading N q j - panelReading N q' j| ≤ eps) →
          |Pattern.autocorr N q D - Pattern.autocorr N q' D| ≤ 2 * eps) ∧
    (∀ (N : ℕ) (q q' : ℕ → ℝ), (∀ j, j < N → panelReading N q j = panelReading N q' j) →
        ∀ w : ℕ → ℝ, Pattern.pairEnergy N w q = Pattern.pairEnergy N w q') ∧
    ((∀ q q' : ℕ → ℝ, (∀ j, j < 20 → |panelReading 20 q j - panelReading 20 q' j| ≤ 1 / 1000) →
        |Pattern.autocorr 20 q 15 - Pattern.autocorr 20 q' 15| ≤ 2 / 1000) ∧
      (∀ (c : ℕ → ℝ) (delta : ℝ), |delta| ≤ 20 → ∀ kappa : ℝ, 1 ≤ kappa →
        |Titration.curve 20 (fun d => c d + Resolution.bump 15 delta d) kappa
          - Titration.curve 20 c kappa| ≤ 1 / 1000)) :=
  ⟨fun _ _ hD1 hDN q => panel_inverts hD1 hDN q,
    fun _ _ hD1 hDN _ _ _ h => panel_stability hD1 hDN h,
    fun _ _ _ h => (panel_identifies h).2,
    fun _ _ h => panel_resolves_lag15 h,
    fun c delta hd _ hk => titration_leaves_lag15_free c delta hd hk⟩

end ContactPanel
end IDR
