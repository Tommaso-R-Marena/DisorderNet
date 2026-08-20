/-
# Part LXXXII  Heterogeneous kinetics: why an ensemble decays non-exponentially

A disordered region is not one species reacting at one rate.  Proteolysis, degradation, chemical
modification and labelling all proceed from whichever conformations expose the relevant site, so
what decays in the tube is a *mixture*: weights `w j` over the conformational substates and a
first-order rate `k j` for each.  In the slow-exchange limit -- interconversion slower than the
chemistry, which is exactly the regime in which substates are resolvable at all -- the surviving
fraction is

`surv w k t = sum_j w j exp (-(k j t))`,

a sum of exponentials, and the quantity a kineticist reports is the apparent rate
`apparentRate w k t = flux / surv`, the instantaneous logarithmic slope of that curve.

* `surv_zero`, `surv_pos`, `surv_antitone` -- the decay curve is a genuine survival function.
* `surv_ge_exp_mean` -- **heterogeneity always looks like stability.**  For every `t >= 0` the
  mixture survives at least as well as a homogeneous population decaying at the mean rate:
  `exp (-(mean k) t) <= surv t`.  A population with a broad rate distribution therefore *appears*
  more resistant than its own average chemistry, with no protective mechanism whatsoever.
* `apparentRate_zero` -- at `t = 0` the apparent rate is exactly the population mean rate: the
  initial slope, and only the initial slope, reports the average.
* `apparentRate_antitone` -- **the apparent rate falls with time**, monotonically, for every
  mixture: the fragile conformations are consumed first and the surviving population is
  progressively enriched in the slow ones.  This is proved by a Chebyshev-type symmetrisation of
  the double sum, not by differentiating, so it holds exactly and at all times.
* `apparentRate_ge_min`, `apparentRate_le_max` -- and it stays between the slowest and the fastest
  substate rate; `surv_ge_slowest` says the long-time tail is carried by the slowest one.
* `surv_log_convex` -- the survival curve is log-convex; equivalently `log surv` is convex, so a
  two-point fit of a single exponential always underestimates the early rate and overestimates the
  late one.
* `twoState_*` -- an explicit instance: half the population at rate `1`, half at rate `1/100`.  The
  initial apparent rate is `101/200`, and by `t = 10` the apparent rate has fallen below `1/10` --
  a fivefold change in the measured "rate constant" with no change in the sample.

The moral for a model of a disordered region is that a single fitted rate constant is not a
property of the region: it is a property of the ensemble *and* of the time window of the
measurement.  Reporting kinetics of a disordered region therefore requires reporting a rate
*distribution*, exactly as reporting its structure requires reporting a conformational
distribution.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace Hetero

open Finset

variable {m : ℕ}

/-- The surviving fraction of a heterogeneous population: substate weights `w`, first-order rates
`k`, no interconversion on the timescale of the chemistry. -/
noncomputable def surv (w k : Fin m → ℝ) (t : ℝ) : ℝ := ∑ j, w j * Real.exp (-(k j * t))

/-- The instantaneous reaction flux (minus the derivative of the surviving fraction). -/
noncomputable def flux (w k : Fin m → ℝ) (t : ℝ) : ℝ :=
  ∑ j, w j * k j * Real.exp (-(k j * t))

/-- The apparent (instantaneous) rate constant: the logarithmic slope of the decay curve, which is
what a single-exponential fit over a short window reports. -/
noncomputable def apparentRate (w k : Fin m → ℝ) (t : ℝ) : ℝ := flux w k t / surv w k t

variable {w k : Fin m → ℝ}

@[simp] lemma surv_zero (hws : ∑ j, w j = 1) : surv w k 0 = 1 := by
  simp [surv, hws]

lemma surv_pos (hw : ∀ j, 0 ≤ w j) (hws : ∑ j, w j = 1) (t : ℝ) : 0 < surv w k t := by
  obtain ⟨j₀, -, hj₀⟩ : ∃ j ∈ (Finset.univ : Finset (Fin m)), 0 < w j := by
    by_contra hcon
    push_neg at hcon
    have : ∑ j, w j = 0 :=
      Finset.sum_eq_zero fun j hj => le_antisymm (hcon j hj) (hw j)
    rw [hws] at this
    exact absurd this one_ne_zero
  refine Finset.sum_pos' (fun j _ => mul_nonneg (hw j) (Real.exp_pos _).le)
    ⟨j₀, Finset.mem_univ j₀, mul_pos hj₀ (Real.exp_pos _)⟩

lemma surv_antitone (hw : ∀ j, 0 ≤ w j) (hk : ∀ j, 0 ≤ k j) {t₁ t₂ : ℝ} (h : t₁ ≤ t₂) :
    surv w k t₂ ≤ surv w k t₁ :=
  Finset.sum_le_sum fun j _ =>
    mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 (by nlinarith [hk j])) (hw j)

/-- **Heterogeneity looks like stability.**  A mixture always survives at least as well as a
homogeneous population decaying at the mean rate. -/
theorem surv_ge_exp_mean (hw : ∀ j, 0 ≤ w j) (hws : ∑ j, w j = 1) (t : ℝ) :
    Real.exp (-((∑ j, w j * k j) * t)) ≤ surv w k t := by
  set K := ∑ j, w j * k j with hK
  have hpt : ∀ j : Fin m, Real.exp (-(K * t)) * (1 + (K - k j) * t)
      ≤ Real.exp (-(k j * t)) := by
    intro j
    have h1 : (K - k j) * t + 1 ≤ Real.exp ((K - k j) * t) := Real.add_one_le_exp _
    have h2 : Real.exp (-(K * t)) * Real.exp ((K - k j) * t) = Real.exp (-(k j * t)) := by
      rw [← Real.exp_add]
      ring_nf
    calc Real.exp (-(K * t)) * (1 + (K - k j) * t)
        ≤ Real.exp (-(K * t)) * Real.exp ((K - k j) * t) := by
          refine mul_le_mul_of_nonneg_left ?_ (Real.exp_pos _).le
          linarith
      _ = Real.exp (-(k j * t)) := h2
  have hsum : ∑ j, w j * (Real.exp (-(K * t)) * (1 + (K - k j) * t)) ≤ surv w k t :=
    Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hpt j) (hw j)
  have hleft : ∑ j, w j * (Real.exp (-(K * t)) * (1 + (K - k j) * t))
      = Real.exp (-(K * t)) := by
    have hexpand : ∀ j : Fin m, w j * (Real.exp (-(K * t)) * (1 + (K - k j) * t))
        = Real.exp (-(K * t)) * (w j * (1 + K * t)) - Real.exp (-(K * t)) * t * (w j * k j) := by
      intro j; ring
    rw [Finset.sum_congr rfl fun j _ => hexpand j, Finset.sum_sub_distrib, ← Finset.mul_sum,
      ← Finset.mul_sum, ← Finset.sum_mul, hws, ← hK]
    ring
  rw [hleft] at hsum
  exact hsum

/-- At time zero the apparent rate is exactly the population mean rate. -/
theorem apparentRate_zero (hws : ∑ j, w j = 1) :
    apparentRate w k 0 = ∑ j, w j * k j := by
  have h1 : surv w k 0 = 1 := surv_zero hws
  have h2 : flux w k 0 = ∑ j, w j * k j := by
    simp [flux]
  rw [apparentRate, h1, h2, div_one]

/-- **The apparent rate falls with time.**  The fastest substates are consumed first, so the
surviving population is progressively enriched in the slow ones and the measured rate constant
decreases -- for every mixture, at all times. -/
theorem apparentRate_antitone (hw : ∀ j, 0 ≤ w j) (hws : ∑ j, w j = 1) {t₁ t₂ : ℝ} (h : t₁ ≤ t₂) :
    apparentRate w k t₂ ≤ apparentRate w k t₁ := by
  have hs₁ : 0 < surv w k t₁ := surv_pos hw hws t₁
  have hs₂ : 0 < surv w k t₂ := surv_pos hw hws t₂
  have hkey : flux w k t₂ * surv w k t₁ ≤ flux w k t₁ * surv w k t₂ := by
    rw [← sub_nonneg]
    have hexp : flux w k t₁ * surv w k t₂ - flux w k t₂ * surv w k t₁
        = ∑ i, ∑ j, ((w i * k i * Real.exp (-(k i * t₁))) * (w j * Real.exp (-(k j * t₂)))
            - (w i * k i * Real.exp (-(k i * t₂))) * (w j * Real.exp (-(k j * t₁)))) := by
      rw [flux, flux, surv, surv, Finset.sum_mul_sum, Finset.sum_mul_sum, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ => (Finset.sum_sub_distrib _ _).symm
    rw [hexp]
    set G : Fin m → Fin m → ℝ := fun i j =>
      (w i * k i * Real.exp (-(k i * t₁))) * (w j * Real.exp (-(k j * t₂)))
        - (w i * k i * Real.exp (-(k i * t₂))) * (w j * Real.exp (-(k j * t₁))) with hG
    have hpair : ∀ i j, 0 ≤ G i j + G j i := by
      intro i j
      have hfac : G i j + G j i
          = w i * w j * ((k i - k j) *
              (Real.exp (-(k i * t₁) + -(k j * t₂)) - Real.exp (-(k i * t₂) + -(k j * t₁)))) := by
        simp only [hG, Real.exp_add]
        ring
      rw [hfac]
      refine mul_nonneg (mul_nonneg (hw i) (hw j)) ?_
      rcases le_total (k j) (k i) with hkij | hkij
      · have hle : -(k i * t₂) + -(k j * t₁) ≤ -(k i * t₁) + -(k j * t₂) := by nlinarith
        have hdiff : 0 ≤ Real.exp (-(k i * t₁) + -(k j * t₂))
            - Real.exp (-(k i * t₂) + -(k j * t₁)) := sub_nonneg.2 (Real.exp_le_exp.2 hle)
        exact mul_nonneg (by linarith) hdiff
      · have hle : -(k i * t₁) + -(k j * t₂) ≤ -(k i * t₂) + -(k j * t₁) := by nlinarith
        have hdiff : Real.exp (-(k i * t₁) + -(k j * t₂))
            - Real.exp (-(k i * t₂) + -(k j * t₁)) ≤ 0 := sub_nonpos.2 (Real.exp_le_exp.2 hle)
        have hk' : k i - k j ≤ 0 := by linarith
        nlinarith
    have hswap : ∑ i, ∑ j, G i j = ∑ i, ∑ j, G j i := Finset.sum_comm
    have hsum : ∑ i, ∑ j, (G i j + G j i) = (∑ i, ∑ j, G i j) + ∑ i, ∑ j, G j i := by
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun i _ => Finset.sum_add_distrib
    have hnn : 0 ≤ ∑ i, ∑ j, (G i j + G j i) :=
      Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => hpair i j
    rw [hsum, ← hswap] at hnn
    linarith
  rw [apparentRate, apparentRate, div_le_div_iff₀ hs₂ hs₁]
  exact hkey

lemma apparentRate_ge_min (hw : ∀ j, 0 ≤ w j) (hws : ∑ j, w j = 1) {c : ℝ} (hc : ∀ j, c ≤ k j)
    (t : ℝ) : c ≤ apparentRate w k t := by
  have hs : 0 < surv w k t := surv_pos hw hws t
  rw [apparentRate, le_div_iff₀ hs]
  have : c * surv w k t = ∑ j, w j * c * Real.exp (-(k j * t)) := by
    rw [surv, Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [this]
  exact Finset.sum_le_sum fun j _ =>
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left (hc j) (hw j)) (Real.exp_pos _).le

lemma apparentRate_le_max (hw : ∀ j, 0 ≤ w j) (hws : ∑ j, w j = 1) {c : ℝ} (hc : ∀ j, k j ≤ c)
    (t : ℝ) : apparentRate w k t ≤ c := by
  have hs : 0 < surv w k t := surv_pos hw hws t
  rw [apparentRate, div_le_iff₀ hs]
  have : c * surv w k t = ∑ j, w j * c * Real.exp (-(k j * t)) := by
    rw [surv, Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [this]
  exact Finset.sum_le_sum fun j _ =>
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left (hc j) (hw j)) (Real.exp_pos _).le

/-- The long-time tail is carried by the slowest substate: the survival curve is bounded below by
the contribution of any single one. -/
lemma surv_ge_slowest (hw : ∀ j, 0 ≤ w j) (j : Fin m) (t : ℝ) :
    w j * Real.exp (-(k j * t)) ≤ surv w k t :=
  Finset.single_le_sum (f := fun i => w i * Real.exp (-(k i * t)))
    (fun i _ => mul_nonneg (hw i) (Real.exp_pos _).le) (Finset.mem_univ j)

/-- **The survival curve is log-convex**, so no single exponential can match it on two time
points and in between. -/
theorem surv_log_convex (hw : ∀ j, 0 ≤ w j) (hws : ∑ j, w j = 1) (t₁ t₂ : ℝ) {a : ℝ}
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    surv w k (a * t₁ + (1 - a) * t₂) ≤ surv w k t₁ ^ a * surv w k t₂ ^ (1 - a) := by
  have hs₁ : 0 < surv w k t₁ := surv_pos hw hws t₁
  have hs₂ : 0 < surv w k t₂ := surv_pos hw hws t₂
  have ha1' : 0 ≤ 1 - a := by linarith
  have hAB : 0 < surv w k t₁ ^ a * surv w k t₂ ^ (1 - a) :=
    mul_pos (Real.rpow_pos_of_pos hs₁ a) (Real.rpow_pos_of_pos hs₂ (1 - a))
  have hterm : ∀ j : Fin m, w j * Real.exp (-(k j * (a * t₁ + (1 - a) * t₂)))
      ≤ surv w k t₁ ^ a * surv w k t₂ ^ (1 - a) *
        (a / surv w k t₁ * (w j * Real.exp (-(k j * t₁)))
          + (1 - a) / surv w k t₂ * (w j * Real.exp (-(k j * t₂)))) := by
    intro j
    have hsplit : Real.exp (-(k j * (a * t₁ + (1 - a) * t₂)))
        = Real.exp (-(k j * t₁)) ^ a * Real.exp (-(k j * t₂)) ^ (1 - a) := by
      rw [Real.rpow_def_of_pos (Real.exp_pos _), Real.rpow_def_of_pos (Real.exp_pos _),
        Real.log_exp, Real.log_exp, ← Real.exp_add]
      ring_nf
    have hgm : (Real.exp (-(k j * t₁)) / surv w k t₁) ^ a
          * (Real.exp (-(k j * t₂)) / surv w k t₂) ^ (1 - a)
        ≤ a * (Real.exp (-(k j * t₁)) / surv w k t₁)
          + (1 - a) * (Real.exp (-(k j * t₂)) / surv w k t₂) :=
      Real.geom_mean_le_arith_mean2_weighted ha0 ha1'
        (div_pos (Real.exp_pos _) hs₁).le (div_pos (Real.exp_pos _) hs₂).le (by ring)
    have hfac : Real.exp (-(k j * t₁)) ^ a * Real.exp (-(k j * t₂)) ^ (1 - a)
        = surv w k t₁ ^ a * surv w k t₂ ^ (1 - a) *
          ((Real.exp (-(k j * t₁)) / surv w k t₁) ^ a
            * (Real.exp (-(k j * t₂)) / surv w k t₂) ^ (1 - a)) := by
      rw [Real.div_rpow (Real.exp_pos _).le hs₁.le, Real.div_rpow (Real.exp_pos _).le hs₂.le]
      field_simp
    rw [hsplit, hfac]
    calc w j * (surv w k t₁ ^ a * surv w k t₂ ^ (1 - a) *
          ((Real.exp (-(k j * t₁)) / surv w k t₁) ^ a
            * (Real.exp (-(k j * t₂)) / surv w k t₂) ^ (1 - a)))
        ≤ w j * (surv w k t₁ ^ a * surv w k t₂ ^ (1 - a) *
            (a * (Real.exp (-(k j * t₁)) / surv w k t₁)
              + (1 - a) * (Real.exp (-(k j * t₂)) / surv w k t₂))) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hgm hAB.le) (hw j)
      _ = surv w k t₁ ^ a * surv w k t₂ ^ (1 - a) *
            (a / surv w k t₁ * (w j * Real.exp (-(k j * t₁)))
              + (1 - a) / surv w k t₂ * (w j * Real.exp (-(k j * t₂)))) := by
          field_simp
  have hsum : surv w k (a * t₁ + (1 - a) * t₂)
      ≤ ∑ j, surv w k t₁ ^ a * surv w k t₂ ^ (1 - a) *
          (a / surv w k t₁ * (w j * Real.exp (-(k j * t₁)))
            + (1 - a) / surv w k t₂ * (w j * Real.exp (-(k j * t₂)))) :=
    Finset.sum_le_sum fun j _ => hterm j
  have hcollapse : ∑ j, surv w k t₁ ^ a * surv w k t₂ ^ (1 - a) *
        (a / surv w k t₁ * (w j * Real.exp (-(k j * t₁)))
          + (1 - a) / surv w k t₂ * (w j * Real.exp (-(k j * t₂))))
      = surv w k t₁ ^ a * surv w k t₂ ^ (1 - a) := by
    rw [← Finset.mul_sum, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    have h1 : ∑ j, w j * Real.exp (-(k j * t₁)) = surv w k t₁ := rfl
    have h2 : ∑ j, w j * Real.exp (-(k j * t₂)) = surv w k t₂ := rfl
    rw [h1, h2]
    field_simp
    ring
  rw [hcollapse] at hsum
  exact hsum

/-! ### An explicit two-state instance -/

/-- Half the population reacts at rate `1`, half at rate `1/100`. -/
noncomputable def twoW : Fin 2 → ℝ := ![1 / 2, 1 / 2]

/-- The two substate rates. -/
noncomputable def twoK : Fin 2 → ℝ := ![1, 1 / 100]

lemma twoW_nonneg (j : Fin 2) : 0 ≤ twoW j := by fin_cases j <;> norm_num [twoW]

lemma twoW_sum : ∑ j, twoW j = 1 := by simp [twoW, Fin.sum_univ_two]; norm_num

/-- The initial apparent rate is the population mean, `101/200`. -/
theorem twoState_rate_zero : apparentRate twoW twoK 0 = 101 / 200 := by
  rw [apparentRate_zero twoW_sum]
  simp [twoW, twoK, Fin.sum_univ_two]
  norm_num

/-- By `t = 10` the apparent rate has fallen below `1/10`: the measured "rate constant" of the same
sample has changed fivefold, purely through enrichment of the slow substate. -/
theorem twoState_rate_ten : apparentRate twoW twoK 10 < 1 / 10 := by
  have hs : 0 < surv twoW twoK 10 := surv_pos twoW_nonneg twoW_sum 10
  have hsurv : surv twoW twoK 10 = 1 / 2 * Real.exp (-10) + 1 / 2 * Real.exp (-(1 / 10)) := by
    simp [surv, twoW, twoK, Fin.sum_univ_two]
    norm_num
  have hflux : flux twoW twoK 10
      = 1 / 2 * Real.exp (-10) + 1 / 200 * Real.exp (-(1 / 10)) := by
    simp [flux, twoW, twoK, Fin.sum_univ_two]
    norm_num
  rw [apparentRate, div_lt_iff₀ hs, hflux, hsurv]
  -- the slow substate dominates: `exp (-1/10)` exceeds `10 * exp (-10)`
  have h1 : Real.exp (-10 : ℝ) * Real.exp (99 / 10) = Real.exp (-(1 / 10)) := by
    rw [← Real.exp_add]
    norm_num
  have h2 : (10 : ℝ) < Real.exp (99 / 10) := by
    have := Real.add_one_le_exp (99 / 10 : ℝ)
    linarith
  have h3 : 10 * Real.exp (-10 : ℝ) < Real.exp (-(1 / 10)) := by
    rw [← h1]
    nlinarith [Real.exp_pos (-10 : ℝ)]
  linarith

end Hetero

end IDR
