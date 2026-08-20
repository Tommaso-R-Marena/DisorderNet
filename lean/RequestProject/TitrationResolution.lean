/-
# Part CXXXIV  The salt titration is exponentially ill-conditioned

Part CXXIX (`SaltTitration.lean`) proved the clean identifiability statement: the screened
energy of a charged disordered region is the Dirichlet series

    E(κ) = ∑_{d=1}^{N−1} d · e^{−κ d} · C(d)

in the charge autocorrelations `C(d)`, and a titration over infinitely many ionic strengths
determines `C` exactly, hence determines the pairwise energetics under *every* solution
condition and *every* separation kernel.  Part CXXXIII (`TitrationDesign.lean`) counted the
conditions: `N − 1` of them, one per lag.

Both statements are noiseless.  This part is the quantitative correction, and it is severe.
Each lag enters the observable curve through the factor `d·e^{−κd}`, which is *exponentially
small in the lag* at any fixed ionic strength.  Therefore:

* `single_lag_gap` — perturbing the correlation profile at one lag `D` shifts the whole
  titration curve by exactly `D e^{−κD} δ`, at every condition;
* `single_lag_invisible` — so a perturbation as large as `δ = eps·e^{κ₀D}/D` is invisible, to
  resolution `eps`, at *every* condition `κ ≥ κ₀`: not just at the conditions one happened to
  measure, but on the entire accessible window;
* `resolution_floor` — hence the finest resolvable lag-`D` correlation is bounded below by
  `eps·e^{κ₀D}/D`, growing exponentially in `D`.  The `N − 1` conditions of Part CXXXIII are
  necessary but very far from sufficient: they must also reach far enough down in salt;
* `separating_condition_below_threshold` — a condition that actually separates two profiles
  differing by `δ` at lag `D` must satisfy `κ < log(D|δ|/eps)/D`, i.e. the Debye screening
  length must be of the order of the separation `D` itself, up to a logarithm;
* `detectable_at_low_salt` — and that bound is tight: any condition at or below the threshold
  does separate them.  Low salt is not a preference, it is the only regime with information;
* `no_uniform_profile_recovery` — over regions of unbounded length there is no uniform recovery
  at all: for any target accuracy `M` there is a length and a pair of profiles that are
  `eps`-indistinguishable at every `κ ≥ κ₀` yet differ by more than `M` at some lag;
* `endpoint_pair_invisible` / `endpoint_pair_gap_zero_salt` — the concrete instance, on genuine
  unit-charge sequences.  A charge at each end of the region and a charge at one end only have
  titration curves differing by exactly `(N−1)e^{−κ(N−1)}`: a contact between the two ends of a
  twenty-residue region is a large signal at zero salt and utterly invisible at any realistic
  ionic strength.
* `titration_resolution_law` — the four conclusions in one statement.

Design consequence, complementing Part CXXXIII: the number of salt conditions bounds *how many*
correlations are identified; the lowest ionic strength reached bounds *which* ones.  A model of
a charged disordered region should therefore report, with each fitted long-range correlation,
the resolution floor `eps·e^{κ₀D}/D` implied by the lowest condition of its calibration set --
and treat everything below that floor as unconstrained by the data, not as measured zero.
-/
import Mathlib
import RequestProject.ChargePatterning
import RequestProject.SaltCrossover
import RequestProject.SaltTitration

set_option autoImplicit false

namespace IDR
namespace Resolution

open Finset Titration

/-! ## 1. The titration curve as a functional of the correlation profile -/

-- The titration curve `Titration.curve N c κ = ∑_d d·e^{−κd}·c d` of a correlation profile `c`,
-- and its identification with the measured energy (`Titration.energy_eq_curve`), are those of
-- Part CXXIX.  This part asks how well that map can be inverted in the presence of noise.

/-- A perturbation of the correlation profile concentrated at a single lag. -/
def bump (D : ℕ) (delta : ℝ) : ℕ → ℝ := fun d => if d = D then delta else 0

lemma curve_bump {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N) (delta kappa : ℝ) :
    curve N (bump D delta) kappa = (D : ℝ) * Real.exp (-(kappa * D)) * delta := by
  rw [curve, Finset.sum_eq_single D]
  · rw [bump, if_pos rfl]
  · intro b _ hb
    rw [bump, if_neg hb, mul_zero]
  · intro h
    exact absurd (Finset.mem_Ico.mpr ⟨hD1, hDN⟩) h

/-- **The single-lag gap.**  Perturbing the correlation profile at lag `D` by `δ` shifts the
titration curve by exactly `D e^{−κD} δ`, at every ionic strength. -/
theorem single_lag_gap {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N) (c : ℕ → ℝ) (delta kappa : ℝ) :
    curve N (fun d => c d + bump D delta d) kappa - curve N c kappa
      = (D : ℝ) * Real.exp (-(kappa * D)) * delta := by
  have h : curve N (fun d => c d + bump D delta d) kappa
      = curve N c kappa + curve N (bump D delta) kappa := by
    rw [curve, curve, curve, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun d _ => by ring
  rw [h, curve_bump hD1 hDN, add_sub_cancel_left]

/-! ## 2. The resolution floor -/

/-- **A whole window of blindness.**  If the correlation at lag `D` is perturbed by no more than
`eps·e^{κ₀D}/D`, the titration curve moves by at most `eps` at *every* ionic strength `κ ≥ κ₀`.
No measurement in that window, however many conditions and however cleverly placed, can see the
perturbation. -/
theorem single_lag_invisible {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N)
    {eps kappa0 kappa delta : ℝ}
    (hdelta : |delta| ≤ eps * Real.exp (kappa0 * D) / D) (hk : kappa0 ≤ kappa) (c : ℕ → ℝ) :
    |curve N (fun d => c d + bump D delta d) kappa - curve N c kappa| ≤ eps := by
  have hDpos : (0 : ℝ) < D := by exact_mod_cast hD1
  rw [single_lag_gap hD1 hDN c delta kappa]
  rw [abs_mul, abs_mul, abs_of_pos hDpos, abs_of_pos (Real.exp_pos _)]
  have h1 : Real.exp (-(kappa * D)) ≤ Real.exp (-(kappa0 * D)) :=
    Real.exp_le_exp.mpr (by nlinarith)
  have h2 : Real.exp (-(kappa0 * D)) * Real.exp (kappa0 * D) = 1 := by
    rw [← Real.exp_add]; simp
  have hstep : (D : ℝ) * Real.exp (-(kappa * D)) * |delta|
      ≤ (D : ℝ) * Real.exp (-(kappa0 * D)) * (eps * Real.exp (kappa0 * D) / D) := by
    gcongr
  have hfin : (D : ℝ) * Real.exp (-(kappa0 * D)) * (eps * Real.exp (kappa0 * D) / D)
      = eps * (Real.exp (-(kappa0 * D)) * Real.exp (kappa0 * D)) := by
    field_simp
  rw [hfin, h2, mul_one] at hstep
  exact hstep

/-- **The resolution floor.**  At every lag `D` and every profile `c` there is a competing
profile differing from `c` only at that lag, by the exponentially large amount
`eps·e^{κ₀D}/D`, whose titration curve is within the resolution `eps` of that of `c` at every
accessible ionic strength.  The lag-`D` correlation is therefore not determined to better than
`eps·e^{κ₀D}/D` by any titration confined to `κ ≥ κ₀`. -/
theorem resolution_floor {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N) {eps kappa0 : ℝ} (heps : 0 ≤ eps)
    (c : ℕ → ℝ) :
    ∃ c' : ℕ → ℝ, (∀ d, d ≠ D → c' d = c d) ∧
      c' D - c D = eps * Real.exp (kappa0 * D) / D ∧
      ∀ kappa, kappa0 ≤ kappa → |curve N c' kappa - curve N c kappa| ≤ eps := by
  have hDpos : (0 : ℝ) < D := by exact_mod_cast hD1
  set delta : ℝ := eps * Real.exp (kappa0 * D) / D with hdel
  have hdelta_nonneg : 0 ≤ delta := by
    apply div_nonneg _ hDpos.le
    exact mul_nonneg heps (Real.exp_pos _).le
  refine ⟨fun d => c d + bump D delta d, ?_, ?_, ?_⟩
  · intro d hd; simp [bump, hd]
  · simp [bump]
  · intro kappa hkappa
    exact single_lag_invisible hD1 hDN (by rw [abs_of_nonneg hdelta_nonneg]) hkappa c

/-- **Where a separating condition must lie.**  If some ionic strength does resolve a lag-`D`
discrepancy `δ` at resolution `eps`, then `κ < log(D|δ|/eps)/D`: the Debye screening length must
be comparable to the separation `D` itself, up to a logarithmic factor. -/
theorem separating_condition_below_threshold {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N)
    {eps kappa delta : ℝ} (heps : 0 < eps) (c : ℕ → ℝ)
    (hsep : eps < |curve N (fun d => c d + bump D delta d) kappa - curve N c kappa|) :
    kappa < Real.log ((D : ℝ) * |delta| / eps) / D := by
  have hDpos : (0 : ℝ) < D := by exact_mod_cast hD1
  rw [single_lag_gap hD1 hDN c delta kappa, abs_mul, abs_mul, abs_of_pos hDpos,
    abs_of_pos (Real.exp_pos _)] at hsep
  have hdpos : 0 < |delta| := by
    rcases (abs_nonneg delta).lt_or_eq with h | h
    · exact h
    · exfalso; rw [← h] at hsep; simp at hsep; linarith
  have hE : eps / ((D : ℝ) * |delta|) < Real.exp (-(kappa * D)) := by
    rw [div_lt_iff₀ (by positivity)]
    nlinarith [Real.exp_pos (-(kappa * D))]
  have hlogE : Real.log (eps / ((D : ℝ) * |delta|)) < -(kappa * D) := by
    have := Real.log_lt_log (by positivity) hE
    rwa [Real.log_exp] at this
  have hinv : eps / ((D : ℝ) * |delta|) = ((D : ℝ) * |delta| / eps)⁻¹ := (inv_div _ _).symm
  rw [hinv, Real.log_inv] at hlogE
  rw [lt_div_iff₀ hDpos]
  linarith

/-- **And low salt does resolve it.**  Conversely, any condition at or below the threshold
`log(D|δ|/eps)/D` separates the two profiles at resolution `eps`.  The bound above is tight. -/
theorem detectable_at_low_salt {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N)
    {eps kappa delta : ℝ} (heps : 0 < eps) (hdelta : delta ≠ 0)
    (hk : kappa ≤ Real.log ((D : ℝ) * |delta| / eps) / D) (c : ℕ → ℝ) :
    eps ≤ |curve N (fun d => c d + bump D delta d) kappa - curve N c kappa| := by
  have hDpos : (0 : ℝ) < D := by exact_mod_cast hD1
  have hdpos : 0 < |delta| := abs_pos.mpr hdelta
  have hpos : 0 < (D : ℝ) * |delta| / eps := by positivity
  rw [single_lag_gap hD1 hDN c delta kappa, abs_mul, abs_mul, abs_of_pos hDpos,
    abs_of_pos (Real.exp_pos _)]
  have hk' : kappa * D ≤ Real.log ((D : ℝ) * |delta| / eps) := by
    rw [le_div_iff₀ hDpos] at hk; linarith
  have hexp : eps / ((D : ℝ) * |delta|) ≤ Real.exp (-(kappa * D)) := by
    have h1 : Real.exp (-Real.log ((D : ℝ) * |delta| / eps)) ≤ Real.exp (-(kappa * D)) :=
      Real.exp_le_exp.mpr (by linarith)
    have h2 : Real.exp (-Real.log ((D : ℝ) * |delta| / eps)) = eps / ((D : ℝ) * |delta|) := by
      rw [Real.exp_neg, Real.exp_log hpos]
      field_simp
    rwa [h2] at h1
  have := mul_le_mul_of_nonneg_left hexp (by positivity : (0 : ℝ) ≤ (D : ℝ) * |delta|)
  calc eps = (D : ℝ) * |delta| * (eps / ((D : ℝ) * |delta|)) := by field_simp
    _ ≤ (D : ℝ) * |delta| * Real.exp (-(kappa * D)) := this
    _ = (D : ℝ) * Real.exp (-(kappa * D)) * |delta| := by ring

/-! ## 3. No uniform recovery over regions of growing length -/

lemma exp_div_ge {t : ℝ} (ht : 0 < t) (D : ℕ) (hD : 1 ≤ D) :
    t ^ 2 * D / 4 ≤ Real.exp (t * D) / D := by
  have hDpos : (0 : ℝ) < D := by exact_mod_cast hD
  have h1 : 1 + t * D / 2 ≤ Real.exp (t * D / 2) := Real.add_one_le_exp _ |>.trans_eq' (by ring)
  have h2 : Real.exp (t * D / 2) * Real.exp (t * D / 2) = Real.exp (t * D) := by
    rw [← Real.exp_add]; ring_nf
  have h3 : (t * D / 2) ^ 2 ≤ Real.exp (t * D) := by
    rw [← h2]
    have hnn : (0 : ℝ) ≤ t * D / 2 := by positivity
    nlinarith
  rw [le_div_iff₀ hDpos]
  nlinarith

/-- **No uniform recovery.**  Fix any resolution `eps > 0` and any lowest accessible ionic
strength `κ₀ > 0`.  Then for every accuracy target `M` there is a region length and a pair of
correlation profiles that are indistinguishable to `eps` at every accessible condition yet
differ by more than `M` at one lag.  Long-range correlations in long disordered regions are not
recoverable from a salt titration at all. -/
theorem no_uniform_profile_recovery {eps kappa0 : ℝ} (heps : 0 < eps) (hk0 : 0 < kappa0)
    (M : ℝ) :
    ∃ (N D : ℕ) (c c' : ℕ → ℝ), 1 ≤ D ∧ D < N ∧
      (∀ kappa, kappa0 ≤ kappa → |curve N c' kappa - curve N c kappa| ≤ eps) ∧
      M ≤ c' D - c D := by
  obtain ⟨m, hm⟩ := exists_nat_ge (4 * M / (eps * kappa0 ^ 2))
  set D : ℕ := m + 1 with hD
  have hD1 : 1 ≤ D := Nat.le_add_left 1 m
  have hDpos : (0 : ℝ) < D := by exact_mod_cast hD1
  have hmD : (m : ℝ) ≤ D := by rw [hD]; push_cast; linarith
  have hbig : M ≤ eps * (Real.exp (kappa0 * D) / D) := by
    have h1 : kappa0 ^ 2 * D / 4 ≤ Real.exp (kappa0 * D) / D := exp_div_ge hk0 D hD1
    have h2 : 4 * M / (eps * kappa0 ^ 2) ≤ (D : ℝ) := le_trans hm hmD
    have hden : 0 < eps * kappa0 ^ 2 := by positivity
    rw [div_le_iff₀ hden] at h2
    nlinarith
  obtain ⟨c', hkeep, hgap, hinv⟩ :=
    resolution_floor (N := D + 1) hD1 (Nat.lt_succ_self D) heps.le (fun _ => (0 : ℝ))
  refine ⟨D + 1, D, fun _ => (0 : ℝ), c', hD1, Nat.lt_succ_self D, hinv, ?_⟩
  rw [hgap]
  have : eps * Real.exp (kappa0 * D) / D = eps * (Real.exp (kappa0 * D) / D) := by ring
  rw [this]
  exact hbig

/-! ## 4. The concrete instance: a contact between the two ends of the region -/

/-- A unit charge at each end of an `N`-residue region. -/
def qEnds (N : ℕ) : ℕ → ℝ := fun i => if i = 0 ∨ i = N - 1 then 1 else 0

-- A unit charge at one end only is `Titration.qOne`, from Part CXXIX.

lemma autocorr_qOne {d : ℕ} (hd : 1 ≤ d) (N : ℕ) : Pattern.autocorr N qOne d = 0 := by
  rw [Pattern.autocorr]
  apply Finset.sum_eq_zero
  intro i _
  rcases Nat.eq_zero_or_pos i with h | h
  · subst h
    have h0 : qOne (0 + d) = 0 := by
      simp only [qOne, zero_add]
      rw [if_neg]
      omega
    rw [h0, mul_zero]
  · have h0 : qOne i = 0 := by
      simp only [qOne]
      rw [if_neg]
      omega
    rw [h0, zero_mul]

lemma autocorr_qEnds_top {N : ℕ} (hN : 2 ≤ N) :
    Pattern.autocorr N (qEnds N) (N - 1) = 1 := by
  rw [Pattern.autocorr]
  have h : N - (N - 1) = 1 := by omega
  rw [h, Finset.range_one, Finset.sum_singleton]
  simp [qEnds]

lemma autocorr_qEnds_mid {N d : ℕ} (hN : 2 ≤ N) (hd1 : 1 ≤ d) (hd : d < N - 1) :
    Pattern.autocorr N (qEnds N) d = 0 := by
  rw [Pattern.autocorr]
  apply Finset.sum_eq_zero
  intro i hi
  rw [Finset.mem_range] at hi
  rcases Nat.eq_zero_or_pos i with h | h
  · subst h
    have : qEnds N (0 + d) = 0 := by
      simp only [qEnds, zero_add]
      rw [if_neg]
      omega
    rw [this, mul_zero]
  · have : qEnds N i = 0 := by
      simp only [qEnds]
      rw [if_neg]
      omega
    rw [this, zero_mul]

/-- **The end-to-end contact.**  The two sequences differ in their titration curves by exactly
`(N−1)e^{−κ(N−1)}` at every ionic strength. -/
theorem endpoint_pair_gap {N : ℕ} (hN : 2 ≤ N) (kappa : ℝ) :
    Salt.energy N kappa (qEnds N) - Salt.energy N kappa qOne
      = ((N : ℝ) - 1) * Real.exp (-(kappa * ((N : ℝ) - 1))) := by
  have hcast : ((N - 1 : ℕ) : ℝ) = (N : ℝ) - 1 := by
    have : 1 ≤ N := by omega
    push_cast [Nat.cast_sub this]; ring
  rw [energy_eq_curve, energy_eq_curve, curve, curve, ← Finset.sum_sub_distrib]
  rw [Finset.sum_eq_single (N - 1)]
  · rw [autocorr_qEnds_top hN, autocorr_qOne (by omega) N, hcast]; ring
  · intro d hd hne
    rw [Finset.mem_Ico] at hd
    rw [autocorr_qEnds_mid hN hd.1 (by omega), autocorr_qOne hd.1 N]
    ring
  · intro h
    exact absurd (Finset.mem_Ico.mpr ⟨by omega, by omega⟩) h

/-- At zero salt the contact is a signal of size `N − 1`: large, and growing with the length of
the region. -/
theorem endpoint_pair_gap_zero_salt {N : ℕ} (hN : 2 ≤ N) :
    Salt.energy N 0 (qEnds N) - Salt.energy N 0 qOne = (N : ℝ) - 1 := by
  rw [endpoint_pair_gap hN]
  simp

/-- **And it is invisible at any realistic ionic strength.**  Once the lowest accessible
condition satisfies `(N−1)e^{−κ₀(N−1)} ≤ eps` — i.e. once the Debye length is shorter than the
region by a mere logarithmic factor — the two sequences are indistinguishable to resolution
`eps` at every condition of the titration, in spite of the zero-salt signal of size `N − 1`. -/
theorem endpoint_pair_invisible {N : ℕ} (hN : 2 ≤ N) {eps kappa0 : ℝ}
    (hres : ((N : ℝ) - 1) * Real.exp (-(kappa0 * ((N : ℝ) - 1))) ≤ eps)
    {kappa : ℝ} (hk : kappa0 ≤ kappa) :
    |Salt.energy N kappa (qEnds N) - Salt.energy N kappa qOne| ≤ eps := by
  have hNpos : (0 : ℝ) ≤ (N : ℝ) - 1 := by
    have : (2 : ℝ) ≤ (N : ℝ) := by exact_mod_cast hN
    linarith
  rw [endpoint_pair_gap hN, abs_mul, abs_of_nonneg hNpos, abs_of_pos (Real.exp_pos _)]
  have h1 : Real.exp (-(kappa * ((N : ℝ) - 1))) ≤ Real.exp (-(kappa0 * ((N : ℝ) - 1))) :=
    Real.exp_le_exp.mpr (by nlinarith)
  nlinarith [Real.exp_pos (-(kappa * ((N : ℝ) - 1)))]

/-! ## 5. The law -/

/-- **The titration resolution law.**  For a region of length `N`, a titration confined to ionic
strengths `κ ≥ κ₀` and read at energy resolution `eps`:

1. the lag-`D` correlation is undetermined below the exponentially growing floor
   `eps·e^{κ₀D}/D`;
2. a condition separating a lag-`D` discrepancy `δ` must lie below `log(D|δ|/eps)/D`, and every
   condition below that threshold does separate it;
3. concretely, a contact between the two ends of the region — a zero-salt signal of size
   `N − 1` — is invisible whenever `(N−1)e^{−κ₀(N−1)} ≤ eps`.

Together with Part CXXXIII (`N − 1` conditions are needed) this fixes both axes of the
experimental design: how many conditions, and how low they must go. -/
theorem titration_resolution_law {N D : ℕ} (hD1 : 1 ≤ D) (hDN : D < N) {eps kappa0 : ℝ}
    (heps : 0 < eps) (c : ℕ → ℝ) :
    (∃ c' : ℕ → ℝ, (∀ d, d ≠ D → c' d = c d) ∧
        c' D - c D = eps * Real.exp (kappa0 * D) / D ∧
        ∀ kappa, kappa0 ≤ kappa → |curve N c' kappa - curve N c kappa| ≤ eps) ∧
    (∀ delta kappa, eps < |curve N (fun d => c d + bump D delta d) kappa - curve N c kappa| →
        kappa < Real.log ((D : ℝ) * |delta| / eps) / D) ∧
    (∀ delta kappa, delta ≠ 0 → kappa ≤ Real.log ((D : ℝ) * |delta| / eps) / D →
        eps ≤ |curve N (fun d => c d + bump D delta d) kappa - curve N c kappa|) ∧
    (2 ≤ N → ((N : ℝ) - 1) * Real.exp (-(kappa0 * ((N : ℝ) - 1))) ≤ eps →
        ∀ kappa, kappa0 ≤ kappa →
          |Salt.energy N kappa (qEnds N) - Salt.energy N kappa qOne| ≤ eps) := by
  refine ⟨resolution_floor hD1 hDN heps.le c, ?_, ?_, ?_⟩
  · intro delta kappa h
    exact separating_condition_below_threshold hD1 hDN heps c h
  · intro delta kappa hd hk
    exact detectable_at_low_salt hD1 hDN heps hd hk c
  · intro hN hres kappa hk
    exact endpoint_pair_invisible hN hres hk

end Resolution
end IDR
