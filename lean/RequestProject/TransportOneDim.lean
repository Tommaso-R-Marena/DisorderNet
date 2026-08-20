/-
# The exact transport distance along a measured coordinate: histograms, not means

`RequestProject.TransportDuality` turns a discrepancy in the *mean* of a Lipschitz
observable into a lower bound on structural error.  That certificate is sharp for two-state
ensembles, but it is blind in a way that matters for disordered regions: a candidate
ensemble can reproduce the measured mean radius of gyration, or the measured mean FRET
efficiency, *exactly* and still be badly wrong, because the disordered region is broad and
often bimodal while the model is narrow.

This file removes that blindness by solving the transport problem on the observable line
exactly.  For two ensembles supported on a common ordered grid of descriptor values
`t 0 ≤ t 1 ≤ …` with weights `p` and `q`,

  `transportCost |·-·| = ∑ₖ (t (k+1) - t k) · |cumW p (k+1) - cumW q (k+1)|`,

the `L¹` distance between the two cumulative distribution functions.  Both inequalities come
from one decomposition: the cost of *any* plan is the sum over the gaps of the grid of the
gap length times the mass the plan drags across that gap, and the marginal constraints force
the *net* mass crossing gap `k` to be the cumulative difference there.  Any plan therefore
pays at least `|cumulative difference|` per gap, and the monotone (quantile) plan pays
exactly that.

Consequences proved here:

* `transportCost_line_eq_cdfL1` -- the exact one-dimensional transport distance.
* `mean_gap_le_cdfL1` -- the histogram certificate dominates the mean certificate.
* `bimodal_vs_unimodal_certificate` -- and strictly so: an explicit compact/expanded
  two-state region and a single-state model with *identical* mean descriptor are one full
  grid unit apart, so the mean-based certificate returns `0` where the histogram
  certificate returns the true distance.
* `structural_certificate_from_histograms` -- combined with the data-processing inequality
  of `RequestProject.TransportProcessing`, a measured histogram of any `L`-Lipschitz
  descriptor certifies a structural transport error of at least `cdfL1 / L`, in ångströms,
  with no modelling assumption.

The design consequence: an ensemble model of a disordered region must be fitted and
falsified against *distributions* of the measured descriptors, and the `L¹` distance between
cumulative distributions is not a heuristic score -- it is exactly the transport distance the
structural theory demands, computable from the experimental histogram.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Transport
import RequestProject.TransportGeometry
import RequestProject.TransportProcessing
import RequestProject.TransportTotalVariation

namespace IDR

open Finset
open scoped Classical

/-! ## The observable line -/

/-- The metric on the observable line: the gap between two descriptor values. -/
def lineCost (a b : ℝ) : ℝ := |a - b|

lemma lineCost_nonneg (a b : ℝ) : 0 ≤ lineCost a b := abs_nonneg _

lemma lineCost_self (a : ℝ) : lineCost a a = 0 := by simp [lineCost]

lemma lineCost_comm (a b : ℝ) : lineCost a b = lineCost b a := abs_sub_comm _ _

lemma lineCost_triangle (a b c : ℝ) : lineCost a c ≤ lineCost a b + lineCost b c := by
  simpa [lineCost] using abs_sub_le a b c

/-- Cumulative weight: the population on grid sites `0, …, k-1`. -/
def cumW (p : ℕ → ℝ) (k : ℕ) : ℝ := ∑ i ∈ Finset.range k, p i

@[simp] lemma cumW_zero (p : ℕ → ℝ) : cumW p 0 = 0 := by simp [cumW]

lemma cumW_succ (p : ℕ → ℝ) (k : ℕ) : cumW p (k + 1) = cumW p k + p k :=
  Finset.sum_range_succ _ _

lemma cumW_mono {p : ℕ → ℝ} (hp : ∀ i, 0 ≤ p i) : Monotone (cumW p) := by
  intro a b hab
  simp only [cumW]
  exact Finset.sum_le_sum_of_subset_of_nonneg (by simpa using hab) (fun i _ _ => hp i)

lemma cumW_nonneg {p : ℕ → ℝ} (hp : ∀ i, 0 ≤ p i) (k : ℕ) : 0 ≤ cumW p k := by
  simpa using cumW_mono hp (Nat.zero_le k)

lemma cumW_le_one {p : ℕ → ℝ} (hp : ∀ i, 0 ≤ p i) {n : ℕ}
    (hs : ∑ i ∈ Finset.range n, p i = 1) {k : ℕ} (hk : k ≤ n) : cumW p k ≤ 1 := by
  have h := cumW_mono hp hk
  rwa [show cumW p n = 1 from hs] at h

/-- An ensemble of descriptor values living on a common ordered grid: site `i` carries
population `p i`.  Reading an experimental histogram into the theory is exactly this. -/
noncomputable def gridEns (n : ℕ) (t : ℕ → ℝ) (p : ℕ → ℝ)
    (hp : ∀ i, 0 ≤ p i) (hs : ∑ i ∈ Finset.range n, p i = 1) : Ens ℝ where
  card := n
  pt := fun i => t (i : ℕ)
  w := fun i => p (i : ℕ)
  w_nonneg := fun i => hp _
  w_sum := by
    rw [Fin.sum_univ_eq_sum_range (fun i => p i) n]
    exact hs

/-- The `L¹` distance between the two cumulative distribution functions on the grid. -/
def cdfL1 (n : ℕ) (t : ℕ → ℝ) (p q : ℕ → ℝ) : ℝ :=
  ∑ k ∈ Finset.range n, (t (k + 1) - t k) * |cumW p (k + 1) - cumW q (k + 1)|

lemma cdfL1_nonneg {n : ℕ} {t : ℕ → ℝ} (ht : Monotone t) (p q : ℕ → ℝ) :
    0 ≤ cdfL1 n t p q :=
  Finset.sum_nonneg fun k _ =>
    mul_nonneg (by simpa using sub_nonneg.2 (ht (Nat.le_succ k))) (abs_nonneg _)

/-! ## The flow across a gap of the grid -/

/-- The mass a plan `g` moves rightwards across the gap between sites `k` and `k+1`. -/
def rightFlow (n : ℕ) (g : ℕ → ℕ → ℝ) (k : ℕ) : ℝ :=
  ∑ i ∈ Finset.range (k + 1), ∑ j ∈ Finset.Ico (k + 1) n, g i j

/-- The mass a plan `g` moves leftwards across the gap between sites `k` and `k+1`. -/
def leftFlow (n : ℕ) (g : ℕ → ℕ → ℝ) (k : ℕ) : ℝ :=
  ∑ i ∈ Finset.Ico (k + 1) n, ∑ j ∈ Finset.range (k + 1), g i j

lemma rightFlow_nonneg {n : ℕ} {g : ℕ → ℕ → ℝ} (hg : ∀ i j, 0 ≤ g i j) (k : ℕ) :
    0 ≤ rightFlow n g k :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => hg _ _

lemma leftFlow_nonneg {n : ℕ} {g : ℕ → ℕ → ℝ} (hg : ∀ i j, 0 ≤ g i j) (k : ℕ) :
    0 ≤ leftFlow n g k :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => hg _ _

lemma sum_block_right (n k : ℕ) (hk : k < n) (g : ℕ → ℕ → ℝ) :
    ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n, (if i ≤ k ∧ k < j then g i j else 0)
      = rightFlow n g k := by
  have hinner : ∀ i : ℕ, ∑ j ∈ Finset.range n, (if i ≤ k ∧ k < j then g i j else 0)
      = if i ≤ k then ∑ j ∈ Finset.Ico (k + 1) n, g i j else 0 := by
    intro i
    by_cases hi : i ≤ k
    · simp only [hi, true_and, if_true]
      rw [← Finset.sum_filter]
      congr 1
      ext j
      simp [Finset.mem_Ico, Finset.mem_range]
      omega
    · simp [hi]
  rw [Finset.sum_congr rfl (fun i _ => hinner i), ← Finset.sum_filter, rightFlow]
  congr 1
  ext i
  simp [Finset.mem_range]
  omega

lemma sum_block_left (n k : ℕ) (hk : k < n) (g : ℕ → ℕ → ℝ) :
    ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n, (if j ≤ k ∧ k < i then g i j else 0)
      = leftFlow n g k := by
  rw [Finset.sum_comm]
  rw [sum_block_right n k hk (fun a b => g b a), rightFlow, leftFlow, Finset.sum_comm]

/-- The gap sum of a monotone grid telescopes to the distance between two of its sites. -/
lemma gap_sum (n : ℕ) {t : ℕ → ℝ} {a b : ℕ} (hab : a ≤ b) (hbn : b ≤ n) :
    ∑ k ∈ Finset.range n, (if a ≤ k ∧ k < b then (t (k + 1) - t k) else 0) = t b - t a := by
  rw [← Finset.sum_filter]
  have hset : (Finset.range n).filter (fun k => a ≤ k ∧ k < b) = Finset.Ico a b := by
    ext k; simp [Finset.mem_Ico, Finset.mem_range]; omega
  rw [hset, Finset.sum_Ico_eq_sub _ hab, Finset.sum_range_sub t, Finset.sum_range_sub t]
  ring

lemma gap_sum_zero (n : ℕ) {t : ℕ → ℝ} {a b : ℕ} (hba : b ≤ a) :
    ∑ k ∈ Finset.range n, (if a ≤ k ∧ k < b then (t (k + 1) - t k) else 0) = 0 := by
  refine Finset.sum_eq_zero fun k _ => ?_
  have h : ¬ (a ≤ k ∧ k < b) := by omega
  simp [h]

/-- The distance between two grid sites, resolved into the elementary gaps that separate
them. -/
lemma weighted_abs_eq_gap_sum (n : ℕ) {t : ℕ → ℝ} (ht : Monotone t) {i j : ℕ} (hi : i < n)
    (hj : j < n) (x : ℝ) :
    x * |t i - t j| = ∑ k ∈ Finset.range n,
      ((if i ≤ k ∧ k < j then x else 0) + (if j ≤ k ∧ k < i then x else 0))
        * (t (k + 1) - t k) := by
  have expand : ∀ a b : ℕ,
      ∑ k ∈ Finset.range n, (if a ≤ k ∧ k < b then x else 0) * (t (k + 1) - t k)
        = x * ∑ k ∈ Finset.range n, (if a ≤ k ∧ k < b then (t (k + 1) - t k) else 0) := by
    intro a b
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    by_cases h : a ≤ k ∧ k < b <;> simp [h]
  simp only [add_mul]
  rw [Finset.sum_add_distrib, expand, expand]
  rcases le_total i j with h | h
  · rw [gap_sum n h (le_of_lt hj), gap_sum_zero n h, abs_of_nonpos (by linarith [ht h])]
    ring
  · rw [gap_sum n h (le_of_lt hi), gap_sum_zero n h, abs_of_nonneg (by linarith [ht h])]
    ring

/-- **Cost = gap length times mass dragged across the gap, summed over the gaps.**  The cost
of any plan on the observable line decomposes over the elementary gaps of the grid. -/
lemma sum_cost_eq_sum_flows (n : ℕ) {t : ℕ → ℝ} (ht : Monotone t) (g : ℕ → ℕ → ℝ) :
    ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n, g i j * |t i - t j|
      = ∑ k ∈ Finset.range n, (t (k + 1) - t k) * (rightFlow n g k + leftFlow n g k) := by
  have step1 : ∀ i ∈ Finset.range n, ∀ j ∈ Finset.range n,
      g i j * |t i - t j| = ∑ k ∈ Finset.range n,
        ((if i ≤ k ∧ k < j then g i j else 0) + (if j ≤ k ∧ k < i then g i j else 0))
          * (t (k + 1) - t k) := fun i hi j hj =>
    weighted_abs_eq_gap_sum n ht (Finset.mem_range.1 hi) (Finset.mem_range.1 hj) (g i j)
  rw [Finset.sum_congr rfl (fun i hi => Finset.sum_congr rfl (fun j hj => step1 i hi j hj))]
  rw [Finset.sum_congr rfl (fun i _ => Finset.sum_comm (s := Finset.range n) (t := Finset.range n)
      (f := fun j k => ((if i ≤ k ∧ k < j then g i j else 0)
        + (if j ≤ k ∧ k < i then g i j else 0)) * (t (k + 1) - t k))), Finset.sum_comm]
  refine Finset.sum_congr rfl fun k hk => ?_
  have hkn : k < n := Finset.mem_range.1 hk
  have hpull : ∀ i, ∑ j ∈ Finset.range n,
      ((if i ≤ k ∧ k < j then g i j else 0) + (if j ≤ k ∧ k < i then g i j else 0))
        * (t (k + 1) - t k)
      = (∑ j ∈ Finset.range n, ((if i ≤ k ∧ k < j then g i j else 0)
          + (if j ≤ k ∧ k < i then g i j else 0))) * (t (k + 1) - t k) :=
    fun i => (Finset.sum_mul _ _ _).symm
  rw [Finset.sum_congr rfl (fun i _ => hpull i), ← Finset.sum_mul]
  have hsplit : ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n,
      ((if i ≤ k ∧ k < j then g i j else 0) + (if j ≤ k ∧ k < i then g i j else 0))
      = rightFlow n g k + leftFlow n g k := by
    rw [Finset.sum_congr rfl (fun i _ => Finset.sum_add_distrib), Finset.sum_add_distrib,
      sum_block_right n k hkn g, sum_block_left n k hkn g]
  rw [hsplit]
  ring

/-- **The marginals fix the net flow.**  Whatever plan is used, the mass crossing the gap
after site `k` rightwards minus the mass crossing it leftwards is the cumulative
difference there. -/
lemma flow_net (n : ℕ) (g : ℕ → ℕ → ℝ) (p q : ℕ → ℝ)
    (hrow : ∀ i ∈ Finset.range n, ∑ j ∈ Finset.range n, g i j = p i)
    (hcol : ∀ j ∈ Finset.range n, ∑ i ∈ Finset.range n, g i j = q j)
    {k : ℕ} (hk : k + 1 ≤ n) :
    rightFlow n g k - leftFlow n g k = cumW p (k + 1) - cumW q (k + 1) := by
  have hsub : Finset.range (k + 1) ⊆ Finset.range n := by simpa using hk
  have hP : cumW p (k + 1)
      = ∑ i ∈ Finset.range (k + 1), ∑ j ∈ Finset.range (k + 1), g i j + rightFlow n g k := by
    rw [cumW, Finset.sum_congr rfl (fun i hi => (hrow i (hsub hi)).symm), rightFlow,
      ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => (Finset.sum_range_add_sum_Ico _ hk).symm
  have hQ : cumW q (k + 1)
      = ∑ j ∈ Finset.range (k + 1), ∑ i ∈ Finset.range (k + 1), g i j + leftFlow n g k := by
    rw [cumW, Finset.sum_congr rfl (fun j hj => (hcol j (hsub hj)).symm), leftFlow,
      Finset.sum_comm (s := Finset.Ico (k + 1) n) (t := Finset.range (k + 1))
        (f := fun i j => g i j), ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => (Finset.sum_range_add_sum_Ico _ hk).symm
  rw [hP, hQ, Finset.sum_comm (s := Finset.range (k + 1)) (t := Finset.range (k + 1))
    (f := fun i j => g i j)]
  ring

/-! ## The monotone (quantile) plan -/

/-- The monotone transport plan: match the two populations through their quantiles.  Site
`i` of the source occupies the interval `[cumW p i, cumW p (i+1))` of quantile space and site
`j` of the target the interval `[cumW q j, cumW q (j+1))`; the plan moves the mass in their
overlap. -/
noncomputable def monoPlan (p q : ℕ → ℝ) (i j : ℕ) : ℝ :=
  max 0 (min (cumW p (i + 1)) (cumW q (j + 1)) - max (cumW p i) (cumW q j))

lemma monoPlan_nonneg (p q : ℕ → ℝ) (i j : ℕ) : 0 ≤ monoPlan p q i j := le_max_left _ _

lemma monoPlan_comm (p q : ℕ → ℝ) (i j : ℕ) : monoPlan p q i j = monoPlan q p j i := by
  unfold monoPlan
  rw [min_comm (cumW p (i + 1)) (cumW q (j + 1)), max_comm (cumW p i) (cumW q j)]

/-- The clamping identity behind every telescoping computation with the monotone plan. -/
lemma clamp_sub {a b u v : ℝ} (hab : a ≤ b) (huv : u ≤ v) :
    min b (max a v) - min b (max a u) = max 0 (min b v - max a u) := by
  simp only [min_def, max_def]
  split_ifs <;> linarith

lemma sub_min_max {a b B : ℝ} (hab : a ≤ b) : b - min b (max a B) = max b B - max a B := by
  simp only [min_def, max_def]
  split_ifs <;> linarith

lemma max_sub_right (A B : ℝ) : max A B - B = max 0 (A - B) := by
  simp only [max_def]
  split_ifs <;> linarith

lemma max_add_max_neg (x : ℝ) : max 0 x + max 0 (-x) = |x| := by
  rcases le_total 0 x with h | h
  · rw [max_eq_right h, max_eq_left (by linarith), abs_of_nonneg h]; ring
  · rw [max_eq_left h, max_eq_right (by linarith), abs_of_nonpos h]; ring

lemma monoPlan_telescope {p q : ℕ → ℝ} (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i) (i j : ℕ) :
    monoPlan p q i j
      = min (cumW p (i + 1)) (max (cumW p i) (cumW q (j + 1)))
        - min (cumW p (i + 1)) (max (cumW p i) (cumW q j)) :=
  (clamp_sub (cumW_mono hp (Nat.le_succ i)) (cumW_mono hq (Nat.le_succ j))).symm

/-- The monotone plan has the source histogram as its first marginal. -/
lemma monoPlan_row {p q : ℕ → ℝ} (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i) {n : ℕ}
    (hps : ∑ i ∈ Finset.range n, p i = 1) (hqs : ∑ i ∈ Finset.range n, q i = 1)
    {i : ℕ} (hi : i + 1 ≤ n) :
    ∑ j ∈ Finset.range n, monoPlan p q i j = p i := by
  rw [Finset.sum_congr rfl (fun j _ => monoPlan_telescope hp hq i j),
    Finset.sum_range_sub (fun j => min (cumW p (i + 1)) (max (cumW p i) (cumW q j))) n]
  have h1 : cumW q n = 1 := hqs
  have hple : cumW p (i + 1) ≤ 1 := cumW_le_one hp hps hi
  have hpa : 0 ≤ cumW p i := cumW_nonneg hp i
  have hab : cumW p i ≤ cumW p (i + 1) := cumW_mono hp (Nat.le_succ i)
  rw [h1, cumW_zero, max_eq_right (le_trans hab hple), max_eq_left hpa, min_eq_left hple,
    min_eq_right hab, cumW_succ]
  ring

/-- The monotone plan has the target histogram as its second marginal. -/
lemma monoPlan_col {p q : ℕ → ℝ} (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i) {n : ℕ}
    (hps : ∑ i ∈ Finset.range n, p i = 1) (hqs : ∑ i ∈ Finset.range n, q i = 1)
    {j : ℕ} (hj : j + 1 ≤ n) :
    ∑ i ∈ Finset.range n, monoPlan p q i j = q j := by
  rw [Finset.sum_congr rfl (fun i _ => monoPlan_comm p q i j)]
  exact monoPlan_row hq hp hqs hps hj

lemma monoPlan_rightFlow_inner {p q : ℕ → ℝ} (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i) {n : ℕ}
    (hqs : ∑ i ∈ Finset.range n, q i = 1) {i k : ℕ} (hple : cumW p (i + 1) ≤ 1)
    (hk : k + 1 ≤ n) :
    ∑ j ∈ Finset.Ico (k + 1) n, monoPlan p q i j
      = max (cumW p (i + 1)) (cumW q (k + 1)) - max (cumW p i) (cumW q (k + 1)) := by
  rw [Finset.sum_congr rfl (fun j _ => monoPlan_telescope hp hq i j),
    Finset.sum_Ico_eq_sub _ hk,
    Finset.sum_range_sub (fun j => min (cumW p (i + 1)) (max (cumW p i) (cumW q j))) n,
    Finset.sum_range_sub (fun j => min (cumW p (i + 1)) (max (cumW p i) (cumW q j))) (k + 1)]
  have h1 : cumW q n = 1 := hqs
  have hpa : 0 ≤ cumW p i := cumW_nonneg hp i
  have hab : cumW p i ≤ cumW p (i + 1) := cumW_mono hp (Nat.le_succ i)
  rw [h1, cumW_zero, max_eq_right (le_trans hab hple), max_eq_left hpa, min_eq_left hple,
    min_eq_right hab]
  have := sub_min_max (a := cumW p i) (b := cumW p (i + 1)) (B := cumW q (k + 1)) hab
  linarith

/-- Under the monotone plan the mass crossing a gap rightwards is exactly the positive part
of the cumulative difference: nothing is moved that does not have to be. -/
lemma monoPlan_rightFlow {p q : ℕ → ℝ} (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i) {n : ℕ}
    (hps : ∑ i ∈ Finset.range n, p i = 1) (hqs : ∑ i ∈ Finset.range n, q i = 1)
    {k : ℕ} (hk : k + 1 ≤ n) :
    rightFlow n (monoPlan p q) k = max 0 (cumW p (k + 1) - cumW q (k + 1)) := by
  rw [rightFlow]
  have hstep : ∀ i ∈ Finset.range (k + 1), ∑ j ∈ Finset.Ico (k + 1) n, monoPlan p q i j
      = max (cumW p (i + 1)) (cumW q (k + 1)) - max (cumW p i) (cumW q (k + 1)) := by
    intro i hi
    have hi' : i + 1 ≤ n := by have := Finset.mem_range.1 hi; omega
    exact monoPlan_rightFlow_inner hp hq hqs (cumW_le_one hp hps hi') hk
  rw [Finset.sum_congr rfl hstep,
    Finset.sum_range_sub (fun i => max (cumW p i) (cumW q (k + 1))) (k + 1), cumW_zero,
    max_eq_right (cumW_nonneg hq (k + 1))]
  exact max_sub_right _ _

/-- And leftwards it is the negative part. -/
lemma monoPlan_leftFlow {p q : ℕ → ℝ} (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i) {n : ℕ}
    (hps : ∑ i ∈ Finset.range n, p i = 1) (hqs : ∑ i ∈ Finset.range n, q i = 1)
    {k : ℕ} (hk : k + 1 ≤ n) :
    leftFlow n (monoPlan p q) k = max 0 (cumW q (k + 1) - cumW p (k + 1)) := by
  have hcomm : leftFlow n (monoPlan p q) k = rightFlow n (monoPlan q p) k := by
    rw [leftFlow, rightFlow, Finset.sum_comm]
    exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => monoPlan_comm p q i j
  rw [hcomm, monoPlan_rightFlow hq hp hqs hps hk]

/-! ## The exact one-dimensional transport distance -/

/-- A transport plan on `Fin n × Fin n`, read as a plan on the grid indices. -/
noncomputable def planToNat (n : ℕ) (gam : Fin n → Fin n → ℝ) : ℕ → ℕ → ℝ :=
  fun a b => if ha : a < n then (if hb : b < n then gam ⟨a, ha⟩ ⟨b, hb⟩ else 0) else 0

lemma planToNat_apply {n : ℕ} (gam : Fin n → Fin n → ℝ) (i j : Fin n) :
    planToNat n gam (i : ℕ) (j : ℕ) = gam i j := by
  simp [planToNat, i.isLt, j.isLt]

lemma planToNat_nonneg {n : ℕ} {gam : Fin n → Fin n → ℝ} (h : ∀ i j, 0 ≤ gam i j) (a b : ℕ) :
    0 ≤ planToNat n gam a b := by
  unfold planToNat
  split_ifs
  · exact h _ _
  · exact le_refl 0
  · exact le_refl 0

lemma sum_fin_sq_eq_sum_range (n : ℕ) (t : ℕ → ℝ) (g : ℕ → ℕ → ℝ) :
    ∑ i : Fin n, ∑ j : Fin n, g (i : ℕ) (j : ℕ) * |t (i : ℕ) - t (j : ℕ)|
      = ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n, g i j * |t i - t j| := by
  rw [Fin.sum_univ_eq_sum_range
    (fun i => ∑ j : Fin n, g i (j : ℕ) * |t i - t (j : ℕ)|) n]
  exact Finset.sum_congr rfl fun i _ =>
    Fin.sum_univ_eq_sum_range (fun j => g i j * |t i - t j|) n

/-- **The transport distance along a measured coordinate is the `L¹` distance between the
cumulative distributions.**  Both bounds come from the gap decomposition: every plan pays at
least the net cumulative discrepancy at each gap, and the monotone (quantile) plan pays
exactly that.  This is the exact solution of the transport problem the falsification theory
needs, in the one situation where an experiment really does deliver a full distribution. -/
theorem transportCost_line_eq_cdfL1 (n : ℕ) {t : ℕ → ℝ} (ht : Monotone t) {p q : ℕ → ℝ}
    (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hps : ∑ i ∈ Finset.range n, p i = 1) (hqs : ∑ i ∈ Finset.range n, q i = 1) :
    transportCost lineCost (gridEns n t p hp hps) (gridEns n t q hq hqs) = cdfL1 n t p q := by
  refine le_antisymm ?_ ?_
  · -- the monotone plan attains the value
    have hcoup : IsCoupling (gridEns n t p hp hps) (gridEns n t q hq hqs)
        (fun (i : Fin n) (j : Fin n) => monoPlan p q (i : ℕ) (j : ℕ)) := by
      refine ⟨fun i j => monoPlan_nonneg p q _ _, fun i => ?_, fun j => ?_⟩
      · show ∑ j : Fin n, monoPlan p q (i : ℕ) (j : ℕ) = p (i : ℕ)
        rw [Fin.sum_univ_eq_sum_range (fun j => monoPlan p q (i : ℕ) j) n]
        exact monoPlan_row hp hq hps hqs i.isLt
      · show ∑ i : Fin n, monoPlan p q (i : ℕ) (j : ℕ) = q (j : ℕ)
        rw [Fin.sum_univ_eq_sum_range (fun i => monoPlan p q i (j : ℕ)) n]
        exact monoPlan_col hp hq hps hqs j.isLt
    refine (transportCost_le_of_coupling lineCost_nonneg hcoup).trans_eq ?_
    have hcost : planCost (gridEns n t p hp hps) (gridEns n t q hq hqs) lineCost
        (fun (i : Fin n) (j : Fin n) => monoPlan p q (i : ℕ) (j : ℕ))
        = ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n, monoPlan p q i j * |t i - t j| :=
      sum_fin_sq_eq_sum_range n t (monoPlan p q)
    rw [hcost, sum_cost_eq_sum_flows n ht (monoPlan p q), cdfL1]
    refine Finset.sum_congr rfl fun k hk => ?_
    have hk' : k + 1 ≤ n := Finset.mem_range.1 hk
    rw [monoPlan_rightFlow hp hq hps hqs hk', monoPlan_leftFlow hp hq hps hqs hk']
    rw [show cumW q (k + 1) - cumW p (k + 1) = -(cumW p (k + 1) - cumW q (k + 1)) by ring,
      max_add_max_neg]
  · -- and no plan can do better
    refine le_csInf (transportCost_set_nonempty _ _ _) ?_
    rintro r ⟨gam, hgam, rfl⟩
    have hcost : planCost (gridEns n t p hp hps) (gridEns n t q hq hqs) lineCost gam
        = ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n,
            planToNat n gam i j * |t i - t j| := by
      rw [← sum_fin_sq_eq_sum_range n t (planToNat n gam)]
      exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by
        rw [planToNat_apply]
        rfl
    have hnn : ∀ a b, 0 ≤ planToNat n gam a b := planToNat_nonneg hgam.nonneg
    have hrow : ∀ i ∈ Finset.range n, ∑ j ∈ Finset.range n, planToNat n gam i j = p i := by
      intro i hi
      have hin : i < n := Finset.mem_range.1 hi
      rw [← Fin.sum_univ_eq_sum_range (fun j => planToNat n gam i j) n]
      have : ∀ j : Fin n, planToNat n gam i (j : ℕ) = gam ⟨i, hin⟩ j := by
        intro j
        have := planToNat_apply gam ⟨i, hin⟩ j
        simpa using this
      rw [Finset.sum_congr rfl (fun j _ => this j)]
      exact hgam.row ⟨i, hin⟩
    have hcol : ∀ j ∈ Finset.range n, ∑ i ∈ Finset.range n, planToNat n gam i j = q j := by
      intro j hj
      have hjn : j < n := Finset.mem_range.1 hj
      rw [← Fin.sum_univ_eq_sum_range (fun i => planToNat n gam i j) n]
      have : ∀ i : Fin n, planToNat n gam (i : ℕ) j = gam i ⟨j, hjn⟩ := by
        intro i
        have := planToNat_apply gam i ⟨j, hjn⟩
        simpa using this
      rw [Finset.sum_congr rfl (fun i _ => this i)]
      exact hgam.col ⟨j, hjn⟩
    rw [hcost, sum_cost_eq_sum_flows n ht (planToNat n gam), cdfL1]
    refine Finset.sum_le_sum fun k hk => ?_
    have hk' : k + 1 ≤ n := Finset.mem_range.1 hk
    have hgap : 0 ≤ t (k + 1) - t k := by simpa using sub_nonneg.2 (ht (Nat.le_succ k))
    refine mul_le_mul_of_nonneg_left ?_ hgap
    have hnet := flow_net n (planToNat n gam) p q hrow hcol hk'
    have hR := rightFlow_nonneg (n := n) hnn k
    have hL := leftFlow_nonneg (n := n) hnn k
    rcases abs_cases (cumW p (k + 1) - cumW q (k + 1)) with ⟨h1, _⟩ | ⟨h1, _⟩ <;> rw [h1] <;>
      linarith

/-! ## The histogram certificate dominates the mean certificate -/

lemma gridEns_mean (n : ℕ) (t : ℕ → ℝ) (p : ℕ → ℝ) (hp : ∀ i, 0 ≤ p i)
    (hs : ∑ i ∈ Finset.range n, p i = 1) :
    (gridEns n t p hp hs).expect id = ∑ i ∈ Finset.range n, p i * t i := by
  simp only [Ens.expect, gridEns, id_eq]
  exact Fin.sum_univ_eq_sum_range (fun i => p i * t i) n

/-- **The histogram certificate is never weaker than the mean certificate.**  The `L¹`
distance between the cumulative distributions bounds the discrepancy of the means, so every
falsification the mean can achieve the histogram achieves too. -/
theorem mean_gap_le_cdfL1 (n : ℕ) {t : ℕ → ℝ} (ht : Monotone t) {p q : ℕ → ℝ}
    (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hps : ∑ i ∈ Finset.range n, p i = 1) (hqs : ∑ i ∈ Finset.range n, q i = 1) :
    |∑ i ∈ Finset.range n, p i * t i - ∑ i ∈ Finset.range n, q i * t i| ≤ cdfL1 n t p q := by
  have h := expect_diff_le_of_lipschitz (c := lineCost) lineCost_nonneg (L := 1) (f := id)
    (fun x y => by simp [lineCost]) (gridEns n t p hp hps) (gridEns n t q hq hqs)
  rwa [gridEns_mean, gridEns_mean, transportCost_line_eq_cdfL1 n ht hp hq hps hqs,
    one_mul] at h

/-! ## And strictly stronger: matching the mean proves nothing -/

/-- The measured region: a compact state at descriptor value `0` and an expanded state at
value `2`, equally populated -- the standard picture of a disordered region in exchange
between a collapsed and an extended form. -/
noncomputable def bimodalW : ℕ → ℝ :=
  fun i => if i = 0 then 1 / 2 else if i = 2 then 1 / 2 else 0

/-- The candidate model: a single narrow state at the *measured mean* descriptor value `1`.
It reproduces the mean exactly. -/
noncomputable def unimodalW : ℕ → ℝ := fun i => if i = 1 then 1 else 0

lemma bimodalW_nonneg (i : ℕ) : 0 ≤ bimodalW i := by
  unfold bimodalW; split_ifs <;> norm_num

lemma unimodalW_nonneg (i : ℕ) : 0 ≤ unimodalW i := by
  unfold unimodalW; split_ifs <;> norm_num

lemma bimodalW_sum : ∑ i ∈ Finset.range 3, bimodalW i = 1 := by
  norm_num [Finset.sum_range_succ, bimodalW]

lemma unimodalW_sum : ∑ i ∈ Finset.range 3, unimodalW i = 1 := by
  norm_num [Finset.sum_range_succ, unimodalW]

/-- **Matching the mean proves nothing.**  A single-state model placed at the measured mean
descriptor value reproduces that mean exactly -- so the mean-based falsification certificate
returns `0` -- while its exact transport distance to the two-state region is one full grid
unit.  Only the distribution sees the error. -/
theorem bimodal_vs_unimodal_certificate :
    (∑ i ∈ Finset.range 3, bimodalW i * (i : ℝ))
        = ∑ i ∈ Finset.range 3, unimodalW i * (i : ℝ) ∧
      cdfL1 3 (fun k => (k : ℝ)) bimodalW unimodalW = 1 := by
  constructor
  · norm_num [Finset.sum_range_succ, bimodalW, unimodalW]
  · norm_num [cdfL1, cumW, Finset.sum_range_succ, bimodalW, unimodalW]

/-- The same statement read as a transport distance between the two ensembles. -/
theorem bimodal_vs_unimodal_transport :
    transportCost lineCost
        (gridEns 3 (fun k => (k : ℝ)) bimodalW bimodalW_nonneg bimodalW_sum)
        (gridEns 3 (fun k => (k : ℝ)) unimodalW unimodalW_nonneg unimodalW_sum) = 1 := by
  rw [transportCost_line_eq_cdfL1 3 (fun a b hab => by exact_mod_cast Nat.cast_le.2 hab)
    bimodalW_nonneg unimodalW_nonneg bimodalW_sum unimodalW_sum]
  exact bimodal_vs_unimodal_certificate.2

/-! ## From the measured histogram to ångströms of structural error -/

/-- **The histogram certificate in structural units.**  If the descriptor `h` is
`L`-Lipschitz with respect to the structural metric `c` -- radius of gyration, an
inter-residue distance, a FRET efficiency -- and the two ensembles push forward to the grid
histograms `p` and `q`, then the structural transport distance is at least the `L¹` distance
between the cumulative histograms divided by `L`.  No modelling assumption enters: this is a
measured number times a geometric constant. -/
theorem structural_certificate_from_histograms {X : Type*} {c : X → X → ℝ}
    (hc : ∀ x y, 0 ≤ c x y) {L : ℝ} (hL : 0 < L) {h : X → ℝ}
    (hlip : ∀ x y, |h x - h y| ≤ L * c x y) (E F : Ens X)
    (n : ℕ) {t : ℕ → ℝ} (ht : Monotone t) {p q : ℕ → ℝ}
    (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hps : ∑ i ∈ Finset.range n, p i = 1) (hqs : ∑ i ∈ Finset.range n, q i = 1)
    (hE : (E.map h).Same (gridEns n t p hp hps))
    (hF : (F.map h).Same (gridEns n t q hq hqs)) :
    cdfL1 n t p q / L ≤ transportCost c E F := by
  have hmap : transportCost lineCost (E.map h) (F.map h) ≤ L * transportCost c E F :=
    transportCost_map_le hc lineCost_nonneg (fun x x' => by simpa [lineCost] using hlip x x') E F
  have heq : transportCost lineCost (E.map h) (F.map h) = cdfL1 n t p q := by
    rw [transportCost_congr_same lineCost_nonneg lineCost_self lineCost_triangle hE hF]
    exact transportCost_line_eq_cdfL1 n ht hp hq hps hqs
  rw [heq] at hmap
  rw [div_le_iff₀ hL]
  linarith

end IDR
