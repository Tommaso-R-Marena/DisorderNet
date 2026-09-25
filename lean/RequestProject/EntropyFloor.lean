/-
# Part CL  A measured heterogeneity puts a floor under the conformational entropy

The certificates of Parts CXLVI–CXLIX bound *populations*.  The quantity that decides whether a
disordered region pays for order — the entropic price of folding upon binding — is the
conformational entropy of its ensemble, and no population bound alone gives a floor for it: a mean
distance on its own is consistent with a single conformation sitting exactly at the mean, whose
entropy is zero.  A measured *spread* is different, and this part turns it into a hard lower bound.

* `entropy_ge_two_block` — **entropy is bounded below by any two-block split.**  If a set of
  conformations carries weight `P`, then the Shannon entropy of the ensemble is at least
  `2P(1 − P)` nats.  Proved from `log t ≤ t − 1` alone: within a block every weight is at most the
  block mass, so `−log w ≥ −log P ≥ 1 − P`.

* `mass_above_mean_lower`, `mass_below_mean_lower` — **a variance forces mass on both sides of the
  mean.**  With all distances in `[0, B]`, the population above the mean and the population below
  it are each at least `Var/(2B²)`.  The proof is the elementary chain
  `Var ≤ B·⟨|r − ⟨r⟩|⟩ = 2B·⟨(r − ⟨r⟩)₊⟩ ≤ 2B²·pop(r > ⟨r⟩)`.

* `entropy_floor_from_variance` — **the floor.**  Combining the two, the conformational entropy of
  a region with distance variance `Var` and contour bound `B` is at least

  `Var² / (2B⁴)`  nats.

  It is positive exactly when the ensemble is heterogeneous, it uses no model of the chain, and it
  is computed from two measurable numbers.

* `ordering_cost_lower_bound` — the thermodynamic reading: a binding event that renders the region
  fully ordered destroys all of that entropy, so it costs at least `k_B T · Var²/(2B⁴)` in free
  energy, whatever the mechanism.  This is the assumption-free part of the "entropic penalty of
  folding upon binding" that a model of a disordered region must reproduce.

* `entropy_floor_positive` — the floor is strictly positive as soon as the measured variance is,
  so any evidence of conformational spread is quantitative evidence of conformational entropy.

Design consequence: a model of a disordered region that reports an ensemble narrower than the
measured variance is not merely imprecise, it violates a bound; and one that predicts an ordering
free energy below `k_B T·Var²/(2B⁴)` is inconsistent with its own ensemble.
-/
import Mathlib
import RequestProject.PopulationCertificate

set_option autoImplicit false

namespace IDR
namespace EntropyFloor

open Finset IDR.PopulationCertificate

variable {N : ℕ}

/-- Shannon entropy of the ensemble weights, in nats. -/
noncomputable def ent (w : Fin N → ℝ) : ℝ := ∑ j, -(w j * Real.log (w j))

/-- Population of conformations strictly below a threshold. -/
noncomputable def popLt (w x : Fin N → ℝ) (a : ℝ) : ℝ := ∑ j, if x j < a then w j else 0

/-- Within a block, each weight contributes at least `w j * (1 − P)` to the entropy, where `P` is
the mass of the block. -/
theorem block_term_bound {wj P : ℝ} (hwj : 0 ≤ wj) (hjP : wj ≤ P) :
    wj * (1 - P) ≤ -(wj * Real.log wj) := by
  rcases eq_or_lt_of_le hwj with h0 | h0
  · rw [← h0]
    simp
  · have hP : 0 < P := lt_of_lt_of_le h0 hjP
    have hlog : Real.log wj ≤ Real.log P := Real.log_le_log h0 hjP
    have hPlog : Real.log P ≤ P - 1 := Real.log_le_sub_one_of_pos hP
    nlinarith [hlog, hPlog, h0]

/-- **Entropy is bounded below by any two-block split.**  If the conformations satisfying a
predicate carry total weight `P`, the ensemble entropy is at least `2P(1 − P)` nats. -/
theorem entropy_ge_two_block {w : Fin N → ℝ} (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1)
    (S : Finset (Fin N)) :
    2 * (∑ j ∈ S, w j) * (1 - ∑ j ∈ S, w j) ≤ ent w := by
  set P := ∑ j ∈ S, w j with hP
  have hPc : ∑ j ∈ Sᶜ, w j = 1 - P := by
    have := Finset.sum_add_sum_compl S w
    rw [hsum] at this
    linarith [this]
  have hin : ∀ j ∈ S, w j ≤ P :=
    fun j hj => Finset.single_le_sum (f := w) (fun i _ => hw i) hj
  have hout : ∀ j ∈ Sᶜ, w j ≤ 1 - P := by
    intro j hj
    rw [← hPc]
    exact Finset.single_le_sum (f := w) (fun i _ => hw i) hj
  have h1 : P * (1 - P) ≤ ∑ j ∈ S, -(w j * Real.log (w j)) := by
    calc P * (1 - P) = ∑ j ∈ S, w j * (1 - P) := by rw [← Finset.sum_mul, ← hP]
      _ ≤ ∑ j ∈ S, -(w j * Real.log (w j)) :=
          Finset.sum_le_sum fun j hj => block_term_bound (hw j) (hin j hj)
  have h2 : (1 - P) * P ≤ ∑ j ∈ Sᶜ, -(w j * Real.log (w j)) := by
    calc (1 - P) * P = (1 - P) * (1 - (1 - P)) := by ring
      _ = ∑ j ∈ Sᶜ, w j * (1 - (1 - P)) := by rw [← Finset.sum_mul, hPc]
      _ ≤ ∑ j ∈ Sᶜ, -(w j * Real.log (w j)) :=
          Finset.sum_le_sum fun j hj => block_term_bound (hw j) (hout j hj)
  have hsplit : ent w = (∑ j ∈ S, -(w j * Real.log (w j)))
      + ∑ j ∈ Sᶜ, -(w j * Real.log (w j)) := by
    rw [ent, ← Finset.sum_add_sum_compl S]
  rw [hsplit]
  linarith

/-! ### A variance forces mass on both sides of the mean -/

/-- The mean of a bounded observable lies in the range of the bound. -/
theorem wmean_mem {w x : Fin N → ℝ} {B : ℝ} (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1)
    (hlo : ∀ j, 0 ≤ x j) (hhi : ∀ j, x j ≤ B) :
    0 ≤ wmean w x ∧ wmean w x ≤ B := by
  constructor
  · exact Finset.sum_nonneg fun j _ => mul_nonneg (hw j) (hlo j)
  · calc wmean w x ≤ ∑ j, w j * B :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hhi j) (hw j)
      _ = B := by rw [← Finset.sum_mul, hsum, one_mul]

/-- **A variance forces mass strictly above the mean.**  With distances in `[0, B]`, the population
above the ensemble mean is at least `Var/(2B²)`. -/
theorem mass_above_mean_lower {w x : Fin N → ℝ} {B : ℝ} (hB : 0 < B) (hw : ∀ j, 0 ≤ w j)
    (hsum : ∑ j, w j = 1) (hlo : ∀ j, 0 ≤ x j) (hhi : ∀ j, x j ≤ B) :
    wvar w x / (2 * B ^ 2) ≤ popGt w x (wmean w x) := by
  set mu := wmean w x with hmu
  obtain ⟨hmu0, hmuB⟩ := wmean_mem hw hsum hlo hhi
  rw [← hmu] at hmu0 hmuB
  set Apos : ℝ := ∑ j, w j * max (x j - mu) 0 with hApos
  set Aneg : ℝ := ∑ j, w j * max (mu - x j) 0 with hAneg
  have hzero : ∑ j, w j * (x j - mu) = 0 := by
    have hexp : ∀ j ∈ (Finset.univ : Finset (Fin N)), w j * (x j - mu) = w j * x j - mu * w j :=
      fun j _ => by ring
    rw [Finset.sum_congr rfl hexp, Finset.sum_sub_distrib, ← Finset.mul_sum, hsum, ← wmean, ← hmu]
    ring
  have hdiff : Apos - Aneg = 0 := by
    have hrepr : Apos - Aneg = ∑ j, w j * (x j - mu) := by
      rw [hApos, hAneg, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun j _ => ?_
      rcases le_total (x j) mu with h | h
      · rw [max_eq_right (by linarith), max_eq_left (by linarith)]
        ring
      · rw [max_eq_left (by linarith), max_eq_right (by linarith)]
        ring
    rw [hrepr, hzero]
  have habs : wvar w x ≤ B * (Apos + Aneg) := by
    have hpt : ∀ j ∈ (Finset.univ : Finset (Fin N)), w j * (x j - mu) ^ 2
        ≤ B * (w j * max (x j - mu) 0 + w j * max (mu - x j) 0) := by
      intro j _
      rcases le_total (x j) mu with h | h
      · rw [max_eq_right (by linarith), max_eq_left (by linarith)]
        have hd : 0 ≤ mu - x j := by linarith
        have hle : mu - x j ≤ B := by linarith [hlo j, hmuB]
        nlinarith [mul_nonneg (hw j) (mul_nonneg hd (by linarith : (0:ℝ) ≤ B - (mu - x j)))]
      · rw [max_eq_left (by linarith), max_eq_right (by linarith)]
        have hd : 0 ≤ x j - mu := by linarith
        have hle : x j - mu ≤ B := by linarith [hmu0, hhi j]
        nlinarith [mul_nonneg (hw j) (mul_nonneg hd (by linarith : (0:ℝ) ≤ B - (x j - mu)))]
    calc wvar w x ≤ ∑ j, B * (w j * max (x j - mu) 0 + w j * max (mu - x j) 0) := by
          rw [wvar, ← hmu]
          exact Finset.sum_le_sum hpt
      _ = B * (Apos + Aneg) := by
          rw [hApos, hAneg, ← Finset.sum_add_distrib, Finset.mul_sum]
  have hAB : Apos ≤ B * popGt w x mu := by
    rw [hApos, popGt, Finset.mul_sum]
    refine Finset.sum_le_sum fun j _ => ?_
    by_cases h : mu < x j
    · rw [if_pos h, max_eq_left (by linarith)]
      have : x j - mu ≤ B := by linarith [hmu0, hhi j]
      nlinarith [hw j]
    · push_neg at h
      rw [if_neg (not_lt.mpr h), max_eq_right (by linarith), mul_zero, mul_zero]
  have hApos_eq : Aneg = Apos := by linarith
  have hfinal : wvar w x ≤ 2 * B ^ 2 * popGt w x mu := by
    have : wvar w x ≤ B * (2 * Apos) := by rw [hApos_eq] at habs; linarith
    nlinarith [hAB, hB]
  rw [div_le_iff₀ (by positivity)]
  linarith

/-- **A variance forces mass strictly below the mean** as well, by the mirror argument. -/
theorem mass_below_mean_lower {w x : Fin N → ℝ} {B : ℝ} (hB : 0 < B) (hw : ∀ j, 0 ≤ w j)
    (hsum : ∑ j, w j = 1) (hlo : ∀ j, 0 ≤ x j) (hhi : ∀ j, x j ≤ B) :
    wvar w x / (2 * B ^ 2) ≤ popLt w x (wmean w x) := by
  -- apply the previous bound to the reflected observable `y = B − x`
  set y : Fin N → ℝ := fun j => B - x j with hy
  have hylo : ∀ j, 0 ≤ y j := fun j => by simp only [hy]; linarith [hhi j]
  have hyhi : ∀ j, y j ≤ B := fun j => by simp only [hy]; linarith [hlo j]
  have hmeany : wmean w y = B - wmean w x := by
    rw [wmean, wmean, hy]
    have hexp : ∀ j ∈ (Finset.univ : Finset (Fin N)), w j * (B - x j) = B * w j - w j * x j :=
      fun j _ => by ring
    rw [Finset.sum_congr rfl hexp, Finset.sum_sub_distrib, ← Finset.mul_sum, hsum, mul_one]
  have hvary : wvar w y = wvar w x := by
    rw [wvar, wvar, hmeany]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [hy]
    ring_nf
  have hmain := mass_above_mean_lower (x := y) hB hw hsum hylo hyhi
  rw [hvary, hmeany] at hmain
  have hconv : popGt w y (B - wmean w x) = popLt w x (wmean w x) := by
    rw [popGt, popLt]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [hy]
    by_cases h : x j < wmean w x
    · rw [if_pos (by linarith), if_pos h]
    · push_neg at h
      rw [if_neg (by push_neg; linarith), if_neg (not_lt.mpr h)]
  rwa [hconv] at hmain

/-! ### The entropy floor -/

/-- **The conformational entropy floor.**  A region whose internal distances lie in `[0, B]` and
have ensemble variance `Var` has conformational entropy at least `Var²/(2B⁴)` nats. -/
theorem entropy_floor_from_variance {w x : Fin N → ℝ} {B : ℝ} (hB : 0 < B) (hw : ∀ j, 0 ≤ w j)
    (hsum : ∑ j, w j = 1) (hlo : ∀ j, 0 ≤ x j) (hhi : ∀ j, x j ≤ B) :
    wvar w x ^ 2 / (2 * B ^ 4) ≤ ent w := by
  classical
  set mu := wmean w x with hmu
  set S : Finset (Fin N) := Finset.univ.filter (fun j => mu < x j) with hS
  set T : Finset (Fin N) := Finset.univ.filter (fun j => x j < mu) with hT
  have hSsum : ∑ j ∈ S, w j = popGt w x mu := by
    rw [popGt, hS, Finset.sum_filter]
  have hTsum : ∑ j ∈ T, w j = popLt w x mu := by
    rw [popLt, hT, Finset.sum_filter]
  have hab := mass_above_mean_lower (x := x) hB hw hsum hlo hhi
  have hbe := mass_below_mean_lower (x := x) hB hw hsum hlo hhi
  rw [← hmu] at hab hbe
  have hdisj : Disjoint S T := by
    rw [Finset.disjoint_left]
    intro j hjS hjT
    rw [hS, Finset.mem_filter] at hjS
    rw [hT, Finset.mem_filter] at hjT
    linarith [hjS.2, hjT.2]
  have hle1 : ∑ j ∈ S, w j + ∑ j ∈ T, w j ≤ 1 := by
    rw [← Finset.sum_union hdisj, ← hsum]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun j _ _ => hw j
  have hblock := entropy_ge_two_block hw hsum S
  have hv0 : 0 ≤ wvar w x := wvar_nonneg hw
  have hq : (0:ℝ) < 2 * B ^ 2 := by positivity
  have hp : wvar w x / (2 * B ^ 2) ≤ ∑ j ∈ S, w j := by rw [hSsum]; exact hab
  have hq' : ∑ j ∈ S, w j ≤ 1 - wvar w x / (2 * B ^ 2) := by
    rw [hTsum] at hle1
    rw [hSsum]
    linarith
  have hkey : wvar w x ^ 2 / (2 * B ^ 4)
      ≤ 2 * (∑ j ∈ S, w j) * (1 - ∑ j ∈ S, w j) := by
    have hpp : 0 ≤ wvar w x / (2 * B ^ 2) := div_nonneg hv0 hq.le
    have h1 : wvar w x / (2 * B ^ 2) ≤ ∑ j ∈ S, w j := hp
    have h2 : wvar w x / (2 * B ^ 2) ≤ 1 - ∑ j ∈ S, w j := by linarith
    have hmul : (wvar w x / (2 * B ^ 2)) * (wvar w x / (2 * B ^ 2))
        ≤ (∑ j ∈ S, w j) * (1 - ∑ j ∈ S, w j) := by
      apply mul_le_mul h1 h2 hpp (le_trans hpp h1)
    have hsq : (wvar w x / (2 * B ^ 2)) * (wvar w x / (2 * B ^ 2))
        = wvar w x ^ 2 / (4 * B ^ 4) := by
      field_simp
      ring
    rw [hsq] at hmul
    have hhalf : wvar w x ^ 2 / (2 * B ^ 4) = 2 * (wvar w x ^ 2 / (4 * B ^ 4)) := by ring
    rw [hhalf]
    linarith
  linarith

/-- The floor is strictly positive as soon as the measured spread is. -/
theorem entropy_floor_positive {w x : Fin N → ℝ} {B : ℝ} (hB : 0 < B) (hw : ∀ j, 0 ≤ w j)
    (hsum : ∑ j, w j = 1) (hlo : ∀ j, 0 ≤ x j) (hhi : ∀ j, x j ≤ B) (hv : 0 < wvar w x) :
    0 < ent w := by
  have hfloor := entropy_floor_from_variance hB hw hsum hlo hhi
  have : 0 < wvar w x ^ 2 / (2 * B ^ 4) := by positivity
  linarith

/-- **The thermodynamic reading.**  A binding event that renders the region fully ordered — a
single conformation, zero conformational entropy — destroys at least `Var²/(2B⁴)` nats, so it costs
at least `k_B T · Var²/(2B⁴)` of free energy, whatever the mechanism. -/
theorem ordering_cost_lower_bound {w x : Fin N → ℝ} {B kT : ℝ} (hB : 0 < B) (hkT : 0 ≤ kT)
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) (hlo : ∀ j, 0 ≤ x j) (hhi : ∀ j, x j ≤ B) :
    kT * (wvar w x ^ 2 / (2 * B ^ 4)) ≤ kT * ent w :=
  mul_le_mul_of_nonneg_left (entropy_floor_from_variance hB hw hsum hlo hhi) hkT

end EntropyFloor
end IDR
