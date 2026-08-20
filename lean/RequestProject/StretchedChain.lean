/-
# Part CXVII  The exact force–extension law of a stretched disordered chain

Part CXVI solved the partially directed self-avoiding chain at zero force: the number of
conformations is `λ ^ n` up to a constant with `λ = 1 + √2`, so the entropy per residue is
exactly `log (1 + √2)`.  Single-molecule experiments do not observe a chain at zero force; they
pull on it.  This file solves the same chain *under tension*, exactly.

A stretching force `f` conjugate to the extension along the preferred axis weights every east
bond by `u = exp (β f)` and leaves the vertical bonds unweighted.  The partition function is the
honest sum over conformations

  `Zp u n p = ∑ w, u ^ (number of east bonds of w)`,

`p` recording which bond the chain is continued from.  Everything below is proved from that
definition.

* `Zp_succ_east`, `Zp_succ_north`, `Zp_succ_south` — the exact transfer relations, and
  `Zp_north_eq_south` the up/down symmetry.
* `lam_quadratic` — the growth constant `λ(u)` is the positive root of `λ² = (1 + u) λ + u`,
  and `Zp_bounds` brackets the partition function between two constant multiples of `λ ^ n`.
* `free_energy_per_residue` — the free energy per residue is exactly `log λ(u)`, for every
  force.  `free_energy_zero_force` recovers `log (1 + √2)` at `f = 0`, and `lam_strictMono`
  says the free energy is strictly increasing in the force.
* `hasDerivAt_lam` — `λ` is differentiable, with an explicit derivative, so the extension per
  residue `extension u = u λ'(u) / λ(u)` is well defined; `extension_one`: at zero force the
  chain is extended by *exactly* one half of its contour length; `extension_pos`,
  `extension_lt_one`: the extension is always between zero and full stretching; and
  `extension_tendsto_one` : it saturates at full stretching as the force grows.
* `mean_east_eq` — the thermodynamic identity behind that name at finite length: the logarithmic
  derivative of the partition function in `log u` is exactly the mean number of east bonds, i.e.
  the mean extension of the chain.

The interpretation for a model of a disordered region: a stretched disordered chain has a
non-degenerate force–extension law that is fixed by the conformational entropy alone, with no
free parameters.  At zero force half the bonds already point along the pulling axis, and the
approach to full extension is smooth: any model of a disordered region that predicts a
single extension, or that treats the extension as a fitted parameter rather than as the
derivative of a free energy, is making a claim the statistical mechanics already settles.
-/
import Mathlib
import RequestProject.PartiallyDirected

namespace IDR.SAW
namespace PD

open scoped BigOperators

/-! ## The partition function of the stretched chain -/

/-- The number of east bonds of a conformation: its extension along the pulling axis. -/
def eastCount (w : List Letter) : ℕ := w.count 0

/-- The partition function of an `n`-bond partially directed chain continued from a bond `p`,
with fugacity `u = exp (β f)` per east bond. -/
noncomputable def Zp (u : ℝ) (n : ℕ) (p : Letter) : ℝ :=
  ∑ w ∈ wordsFrom n p, u ^ eastCount w

@[simp] lemma Zp_zero (u : ℝ) (p : Letter) : Zp u 0 p = 1 := by
  simp [Zp, wordsFrom, eastCount]

/-- The weight of a single bond. -/
noncomputable def wt (u : ℝ) (a : Letter) : ℝ := if a = 0 then u else 1

lemma Zp_succ (u : ℝ) (n : ℕ) (p : Letter) :
    Zp u (n + 1) p = ∑ a ∈ Finset.univ.filter (fun a => ok p a), wt u a * Zp u n a := by
  rw [Zp, wordsFrom, Finset.sum_biUnion]
  · refine Finset.sum_congr rfl fun a _ => ?_
    rw [Finset.sum_image (fun x _ y _ h => by simpa using h)]
    rw [Zp, Finset.mul_sum]
    refine Finset.sum_congr rfl fun v _ => ?_
    have hcount : eastCount (a :: v) = (if a = 0 then 1 else 0) + eastCount v := by
      unfold eastCount
      rw [List.count_cons]
      by_cases h : a = 0 <;> simp [h, Nat.add_comm]
    rw [hcount, pow_add, wt]
    by_cases h : a = 0 <;> simp [h]
  · intro a _ b _ hab
    simp only [Finset.disjoint_left, Finset.mem_image]
    rintro w ⟨x, _, rfl⟩ ⟨y, _, hEq⟩
    apply hab
    have := congrArg List.head? hEq
    simpa using this.symm

lemma Zp_succ_east (u : ℝ) (n : ℕ) :
    Zp u (n + 1) 0 = u * Zp u n 0 + Zp u n 1 + Zp u n 2 := by
  have h : (Finset.univ.filter fun a : Letter => ok 0 a) = Finset.univ := by decide
  rw [Zp_succ, h, Fin.sum_univ_three]
  simp [wt]

lemma Zp_succ_north (u : ℝ) (n : ℕ) : Zp u (n + 1) 1 = u * Zp u n 0 + Zp u n 1 := by
  have h : (Finset.univ.filter fun a : Letter => ok 1 a) = {0, 1} := by decide
  rw [Zp_succ, h, Finset.sum_pair (by decide : (0 : Letter) ≠ 1)]
  simp [wt]

lemma Zp_succ_south (u : ℝ) (n : ℕ) : Zp u (n + 1) 2 = u * Zp u n 0 + Zp u n 2 := by
  have h : (Finset.univ.filter fun a : Letter => ok 2 a) = {0, 2} := by decide
  rw [Zp_succ, h, Finset.sum_pair (by decide : (0 : Letter) ≠ 2)]
  simp [wt]

/-- Up/down symmetry of the stretched chain. -/
lemma Zp_north_eq_south (u : ℝ) (n : ℕ) : Zp u n 1 = Zp u n 2 := by
  induction n with
  | zero => simp
  | succ n ih => rw [Zp_succ_north, Zp_succ_south, ih]

/-! ## The growth constant -/

/-- The discriminant of the transfer matrix. -/
noncomputable def disc (u : ℝ) : ℝ := u ^ 2 + 6 * u + 1

/-- The growth constant of the stretched chain: the positive root of `λ² = (1 + u) λ + u`. -/
noncomputable def lamF (u : ℝ) : ℝ := ((1 + u) + Real.sqrt (disc u)) / 2

lemma disc_pos {u : ℝ} (hu : 0 < u) : 0 < disc u := by
  unfold disc; nlinarith

lemma sqrt_disc_sq {u : ℝ} (hu : 0 < u) : Real.sqrt (disc u) * Real.sqrt (disc u) = disc u :=
  Real.mul_self_sqrt (le_of_lt (disc_pos hu))

lemma sqrt_disc_gt {u : ℝ} (hu : 0 < u) : 1 + u < Real.sqrt (disc u) := by
  have h1 : (0:ℝ) < 1 + u := by linarith
  have h2 : (1 + u) ^ 2 < disc u := by unfold disc; nlinarith
  nlinarith [sqrt_disc_sq hu, Real.sqrt_nonneg (disc u)]

lemma lamF_gt_one {u : ℝ} (hu : 0 < u) : 1 + u < lamF u := by
  have := sqrt_disc_gt hu
  unfold lamF
  linarith

lemma lamF_pos {u : ℝ} (hu : 0 < u) : 0 < lamF u := by
  have := lamF_gt_one hu; linarith

/-- **The characteristic equation of the transfer matrix.** -/
lemma lam_quadratic {u : ℝ} (hu : 0 < u) : lamF u * lamF u = (1 + u) * lamF u + u := by
  have h : Real.sqrt (disc u) * Real.sqrt (disc u) = u ^ 2 + 6 * u + 1 := by
    rw [sqrt_disc_sq hu]; rfl
  unfold lamF
  nlinarith [h]

lemma lamF_strictMonoOn {u v : ℝ} (hu : 0 < u) (huv : u < v) : lamF u < lamF v := by
  have h1 : Real.sqrt (disc u) < Real.sqrt (disc v) := by
    refine Real.sqrt_lt_sqrt (le_of_lt (disc_pos hu)) ?_
    unfold disc; nlinarith
  unfold lamF
  linarith

/-- At zero force the growth constant is the connective constant of the partially directed
chain, `1 + √2`. -/
lemma lamF_one : lamF 1 = 1 + Real.sqrt 2 := by
  have h : disc 1 = 8 := by norm_num [disc]
  have h8 : Real.sqrt 8 = 2 * Real.sqrt 2 := by
    rw [show (8:ℝ) = 2 ^ 2 * 2 by norm_num, Real.sqrt_mul (by positivity), Real.sqrt_sq (by norm_num)]
  rw [lamF, h, h8]
  ring

/-! ## The partition function grows like `λ ^ n` -/

lemma Zp_bounds {u : ℝ} (hu : 0 < u) (n : ℕ) :
    (min (1 / (lamF u - 1)) (1 / u)) * ((lamF u - 1) * lamF u ^ n) ≤ Zp u n 0 ∧
      Zp u n 0 ≤ (max (1 / (lamF u - 1)) (1 / u)) * ((lamF u - 1) * lamF u ^ n) ∧
      (min (1 / (lamF u - 1)) (1 / u)) * (u * lamF u ^ n) ≤ Zp u n 1 ∧
      Zp u n 1 ≤ (max (1 / (lamF u - 1)) (1 / u)) * (u * lamF u ^ n) := by
  set L := lamF u with hL
  have hL1 : 1 + u < L := lamF_gt_one hu
  have hLpos : 0 < L := by linarith
  have hLm : 0 < L - 1 := by linarith
  have hquad : L * L = (1 + u) * L + u := lam_quadratic hu
  set c := min (1 / (L - 1)) (1 / u) with hc
  set C := max (1 / (L - 1)) (1 / u) with hC
  have hcpos : 0 < c := lt_min (by positivity) (by positivity)
  have hc1 : c * (L - 1) ≤ 1 := by
    have : c ≤ 1 / (L - 1) := min_le_left _ _
    calc c * (L - 1) ≤ (1 / (L - 1)) * (L - 1) := by nlinarith
      _ = 1 := by field_simp
  have hc2 : c * u ≤ 1 := by
    have : c ≤ 1 / u := min_le_right _ _
    calc c * u ≤ (1 / u) * u := by nlinarith
      _ = 1 := by field_simp
  have hC1 : 1 ≤ C * (L - 1) := by
    have : 1 / (L - 1) ≤ C := le_max_left _ _
    calc (1:ℝ) = (1 / (L - 1)) * (L - 1) := by field_simp
      _ ≤ C * (L - 1) := by nlinarith
  have hC2 : 1 ≤ C * u := by
    have : 1 / u ≤ C := le_max_right _ _
    calc (1:ℝ) = (1 / u) * u := by field_simp
      _ ≤ C * u := by nlinarith
  induction n with
  | zero =>
      refine ⟨?_, ?_, ?_, ?_⟩ <;> simp <;> nlinarith
  | succ n ih =>
      obtain ⟨h1, h2, h3, h4⟩ := ih
      have hpow : (0:ℝ) < L ^ n := pow_pos hLpos n
      have hsucc : L ^ (n + 1) = L ^ n * L := by ring
      have hsouth := Zp_north_eq_south u n
      refine ⟨?_, ?_, ?_, ?_⟩
      · rw [Zp_succ_east, ← hsouth, hsucc]
        have hu1 : u * (c * ((L - 1) * L ^ n)) ≤ u * Zp u n 0 :=
          mul_le_mul_of_nonneg_left h1 hu.le
        have key : c * ((L - 1) * (L ^ n * L))
            = u * (c * ((L - 1) * L ^ n)) + 2 * (c * (u * L ^ n)) := by
          linear_combination (c * L ^ n) * hquad
        linarith
      · rw [Zp_succ_east, ← hsouth, hsucc]
        have hu2 : u * Zp u n 0 ≤ u * (C * ((L - 1) * L ^ n)) :=
          mul_le_mul_of_nonneg_left h2 hu.le
        have key : C * ((L - 1) * (L ^ n * L))
            = u * (C * ((L - 1) * L ^ n)) + 2 * (C * (u * L ^ n)) := by
          linear_combination (C * L ^ n) * hquad
        linarith
      · rw [Zp_succ_north, hsucc]
        have hu1 : u * (c * ((L - 1) * L ^ n)) ≤ u * Zp u n 0 :=
          mul_le_mul_of_nonneg_left h1 hu.le
        have key : c * (u * (L ^ n * L)) = u * (c * ((L - 1) * L ^ n)) + c * (u * L ^ n) := by
          ring
        linarith
      · rw [Zp_succ_north, hsucc]
        have hu2 : u * Zp u n 0 ≤ u * (C * ((L - 1) * L ^ n)) :=
          mul_le_mul_of_nonneg_left h2 hu.le
        have key : C * (u * (L ^ n * L)) = u * (C * ((L - 1) * L ^ n)) + C * (u * L ^ n) := by
          ring
        linarith

/-- A two-sided exponential bracket forces the logarithmic growth rate. -/
lemma tendsto_log_div_of_bounds {a b L : ℝ} (ha : 0 < a) (hb : 0 < b) (hL : 0 < L)
    {Z : ℕ → ℝ} (h1 : ∀ n, a * L ^ n ≤ Z n) (h2 : ∀ n, Z n ≤ b * L ^ n) :
    Filter.Tendsto (fun n : ℕ => Real.log (Z n) / n) Filter.atTop (nhds (Real.log L)) := by
  have hlow : ∀ n : ℕ, 0 < n → Real.log a / n + Real.log L ≤ Real.log (Z n) / n := by
    intro n hn
    have hnpos : (0:ℝ) < n := by exact_mod_cast hn
    have hpos : 0 < a * L ^ n := by positivity
    have hlog := Real.log_le_log hpos (h1 n)
    rw [Real.log_mul (ne_of_gt ha) (ne_of_gt (pow_pos hL n)), Real.log_pow] at hlog
    rw [le_div_iff₀ hnpos, add_mul, div_mul_cancel₀ _ (ne_of_gt hnpos)]
    linarith
  have hhigh : ∀ n : ℕ, 0 < n → Real.log (Z n) / n ≤ Real.log b / n + Real.log L := by
    intro n hn
    have hnpos : (0:ℝ) < n := by exact_mod_cast hn
    have hZpos : 0 < Z n := lt_of_lt_of_le (by positivity) (h1 n)
    have hlog := Real.log_le_log hZpos (h2 n)
    rw [Real.log_mul (ne_of_gt hb) (ne_of_gt (pow_pos hL n)), Real.log_pow] at hlog
    rw [div_le_iff₀ hnpos, add_mul, div_mul_cancel₀ _ (ne_of_gt hnpos)]
    linarith
  have hA : Filter.Tendsto (fun n : ℕ => Real.log a / n + Real.log L) Filter.atTop
      (nhds (Real.log L)) := by
    have := tendsto_const_div_atTop_nhds_zero_nat (Real.log a)
    simpa using this.add tendsto_const_nhds
  have hB : Filter.Tendsto (fun n : ℕ => Real.log b / n + Real.log L) Filter.atTop
      (nhds (Real.log L)) := by
    have := tendsto_const_div_atTop_nhds_zero_nat (Real.log b)
    simpa using this.add tendsto_const_nhds
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' hA hB ?_ ?_
  · filter_upwards [Filter.eventually_gt_atTop 0] with n hn using hlow n hn
  · filter_upwards [Filter.eventually_gt_atTop 0] with n hn using hhigh n hn

/-- **The exact free energy per residue of the stretched chain**, for every force: `log λ(u)`
with `u = exp (β f)`. -/
theorem free_energy_per_residue {u : ℝ} (hu : 0 < u) :
    Filter.Tendsto (fun n : ℕ => Real.log (Zp u n 0) / n) Filter.atTop
      (nhds (Real.log (lamF u))) := by
  have hL1 : 1 + u < lamF u := lamF_gt_one hu
  have hLm : 0 < lamF u - 1 := by linarith
  have hcpos : 0 < min (1 / (lamF u - 1)) (1 / u) := lt_min (by positivity) (by positivity)
  have hCpos : 0 < max (1 / (lamF u - 1)) (1 / u) := lt_of_lt_of_le hcpos (min_le_max)
  refine tendsto_log_div_of_bounds (a := min (1 / (lamF u - 1)) (1 / u) * (lamF u - 1))
    (b := max (1 / (lamF u - 1)) (1 / u) * (lamF u - 1)) (by positivity) (by positivity)
    (lamF_pos hu) ?_ ?_
  · intro n
    have := (Zp_bounds hu n).1
    calc min (1 / (lamF u - 1)) (1 / u) * (lamF u - 1) * lamF u ^ n
        = min (1 / (lamF u - 1)) (1 / u) * ((lamF u - 1) * lamF u ^ n) := by ring
      _ ≤ Zp u n 0 := this
  · intro n
    have := (Zp_bounds hu n).2.1
    calc Zp u n 0 ≤ max (1 / (lamF u - 1)) (1 / u) * ((lamF u - 1) * lamF u ^ n) := this
      _ = max (1 / (lamF u - 1)) (1 / u) * (lamF u - 1) * lamF u ^ n := by ring

/-- At zero force the free energy per residue is the zero-force entropy of Part CXVI. -/
theorem free_energy_zero_force :
    Filter.Tendsto (fun n : ℕ => Real.log (Zp 1 n 0) / n) Filter.atTop
      (nhds (Real.log (1 + Real.sqrt 2))) := by
  have h := free_energy_per_residue (u := 1) (by norm_num)
  rwa [lamF_one] at h

/-- The free energy per residue is strictly increasing in the force. -/
theorem lamF_strictMono {u v : ℝ} (hu : 0 < u) (huv : u < v) :
    Real.log (lamF u) < Real.log (lamF v) :=
  Real.log_lt_log (lamF_pos hu) (lamF_strictMonoOn hu huv)

/-! ## The force–extension law -/

lemma hasDerivAt_disc (u : ℝ) : HasDerivAt disc (2 * u + 6) u := by
  have h : HasDerivAt (fun x : ℝ => x ^ 2 + 6 * x + 1) (2 * u ^ 1 + 6 * 1) u :=
    ((hasDerivAt_pow 2 u).add ((hasDerivAt_id u).const_mul 6)).add_const 1
  have hfun : disc = fun x : ℝ => x ^ 2 + 6 * x + 1 := rfl
  rw [hfun]
  convert h using 1
  ring

/-- The growth constant is differentiable in the fugacity, with an explicit derivative. -/
theorem hasDerivAt_lam {u : ℝ} (hu : 0 < u) :
    HasDerivAt lamF (1 / 2 + (u + 3) / (2 * Real.sqrt (disc u))) u := by
  have hd : disc u ≠ 0 := ne_of_gt (disc_pos hu)
  have hsq : HasDerivAt (fun x => Real.sqrt (disc x))
      ((2 * u + 6) / (2 * Real.sqrt (disc u))) u :=
    (hasDerivAt_disc u).sqrt hd
  have h1 : HasDerivAt (fun x : ℝ => (1 + x) + Real.sqrt (disc x))
      (1 + (2 * u + 6) / (2 * Real.sqrt (disc u))) u :=
    ((hasDerivAt_id u).const_add 1).add hsq
  have h2 := h1.div_const 2
  refine h2.congr_deriv ?_
  have hs : 0 < Real.sqrt (disc u) := Real.sqrt_pos.2 (disc_pos hu)
  field_simp
  ring

/-- The mean extension per residue at fugacity `u`: the logarithmic derivative of the growth
constant, `u λ'(u) / λ(u)`. -/
noncomputable def extension (u : ℝ) : ℝ :=
  u * (1 / 2 + (u + 3) / (2 * Real.sqrt (disc u))) / lamF u

/-- **At zero force the chain is already extended by exactly half its contour length.** -/
theorem extension_one : extension 1 = 1 / 2 := by
  have h8 : Real.sqrt (disc 1) = 2 * Real.sqrt 2 := by
    have : disc 1 = 8 := by norm_num [disc]
    rw [this, show (8:ℝ) = 2 ^ 2 * 2 by norm_num, Real.sqrt_mul (by positivity),
      Real.sqrt_sq (by norm_num)]
  have hs : Real.sqrt 2 > 0 := Real.sqrt_pos.2 (by norm_num)
  have hs2 : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  rw [extension, h8, lamF_one]
  field_simp
  nlinarith [hs2, hs]

lemma sqrt_disc_ge {u : ℝ} (hu : 0 < u) : u ≤ Real.sqrt (disc u) := by
  have h : u ^ 2 ≤ disc u := by unfold disc; nlinarith
  nlinarith [sqrt_disc_sq hu, Real.sqrt_nonneg (disc u)]

lemma sqrt_disc_le {u : ℝ} (hu : 0 < u) : Real.sqrt (disc u) ≤ u + 3 := by
  have h : disc u ≤ (u + 3) ^ 2 := by unfold disc; nlinarith
  nlinarith [sqrt_disc_sq hu, Real.sqrt_nonneg (disc u)]

/-- The force–extension law in closed form. -/
lemma extension_eq {u : ℝ} (hu : 0 < u) :
    extension u
      = u * (Real.sqrt (disc u) + u + 3) /
          (Real.sqrt (disc u) * (1 + u + Real.sqrt (disc u))) := by
  have hs : 0 < Real.sqrt (disc u) := Real.sqrt_pos.2 (disc_pos hu)
  have hL : 0 < lamF u := lamF_pos hu
  have hLval : lamF u = ((1 + u) + Real.sqrt (disc u)) / 2 := rfl
  rw [extension, hLval]
  field_simp
  ring

/-- The slack left at fugacity `u`: the chain is short of full stretching by exactly this. -/
lemma one_sub_extension {u : ℝ} (hu : 0 < u) :
    1 - extension u
      = (Real.sqrt (disc u) + 3 * u + 1) /
          (Real.sqrt (disc u) * (1 + u + Real.sqrt (disc u))) := by
  have hs : 0 < Real.sqrt (disc u) := Real.sqrt_pos.2 (disc_pos hu)
  have hsq : Real.sqrt (disc u) * Real.sqrt (disc u) = u ^ 2 + 6 * u + 1 := by
    rw [sqrt_disc_sq hu]; rfl
  rw [extension_eq hu]
  field_simp
  nlinarith [hsq]

/-- The extension is strictly positive at every positive force fugacity. -/
theorem extension_pos {u : ℝ} (hu : 0 < u) : 0 < extension u := by
  have hs : 0 < Real.sqrt (disc u) := Real.sqrt_pos.2 (disc_pos hu)
  rw [extension_eq hu]
  positivity

/-- **The chain never reaches full stretching**: at every finite force some slack remains. -/
theorem extension_lt_one {u : ℝ} (hu : 0 < u) : extension u < 1 := by
  have hs : 0 < Real.sqrt (disc u) := Real.sqrt_pos.2 (disc_pos hu)
  have h : 0 < 1 - extension u := by
    rw [one_sub_extension hu]
    positivity
  linarith

/-- **Saturation.**  As the force grows the chain approaches full stretching, with slack at
most `4 / u`. -/
theorem one_sub_extension_le {u : ℝ} (hu : 1 ≤ u) : 1 - extension u ≤ 4 / u := by
  have hu0 : 0 < u := lt_of_lt_of_le one_pos hu
  have hs : 0 < Real.sqrt (disc u) := Real.sqrt_pos.2 (disc_pos hu0)
  have hge := sqrt_disc_ge hu0
  have hle := sqrt_disc_le hu0
  rw [one_sub_extension hu0, div_le_div_iff₀ (by positivity) hu0]
  nlinarith [hge, hle, hu0, hu]

theorem extension_tendsto_one : Filter.Tendsto extension Filter.atTop (nhds 1) := by
  have hzero : Filter.Tendsto (fun u : ℝ => 1 - extension u) Filter.atTop (nhds 0) := by
    have hlow : ∀ᶠ u : ℝ in Filter.atTop, (0:ℝ) ≤ 1 - extension u := by
      filter_upwards [Filter.eventually_gt_atTop (0:ℝ)] with u hu
      linarith [extension_lt_one hu]
    have hhigh : ∀ᶠ u : ℝ in Filter.atTop, 1 - extension u ≤ 4 / u := by
      filter_upwards [Filter.eventually_ge_atTop (1:ℝ)] with u hu
      exact one_sub_extension_le hu
    have h4 : Filter.Tendsto (fun u : ℝ => 4 / u) Filter.atTop (nhds 0) := by
      simpa [div_eq_mul_inv] using tendsto_inv_atTop_zero.const_mul (4:ℝ)
    exact tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds h4 hlow hhigh
  simpa using hzero.const_sub (1:ℝ)

/-! ## The thermodynamic identity at finite length -/

lemma hasDerivAt_Zp (u : ℝ) (n : ℕ) (p : Letter) :
    HasDerivAt (fun v => Zp v n p)
      (∑ w ∈ wordsFrom n p, (eastCount w : ℝ) * u ^ (eastCount w - 1)) u := by
  have h : ∀ w ∈ wordsFrom n p, HasDerivAt (fun v : ℝ => v ^ eastCount w)
      ((eastCount w : ℝ) * u ^ (eastCount w - 1)) u := by
    intro w _
    simpa using hasDerivAt_pow (eastCount w) u
  have hsum := HasDerivAt.sum h
  rw [Finset.sum_fn] at hsum
  simpa [Zp] using hsum

/-- **Extension is conjugate to force.**  At every chain length, the derivative of the
partition function in the fugacity returns the mean number of east bonds — the mean extension
of the chain along the pulling axis. -/
theorem mean_east_eq (u : ℝ) (n : ℕ) :
    u * deriv (fun v => Zp v n 0) u / Zp u n 0
      = (∑ w ∈ wordsFrom n 0, (eastCount w : ℝ) * u ^ eastCount w) / Zp u n 0 := by
  have hderiv := (hasDerivAt_Zp u n 0).deriv
  rw [hderiv]
  congr 1
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun w _ => ?_
  rcases Nat.eq_zero_or_pos (eastCount w) with h | h
  · simp [h]
  · rw [← mul_assoc, mul_comm u ((eastCount w : ℝ)), mul_assoc, ← pow_succ']
    congr 2
    omega

end PD
end IDR.SAW
