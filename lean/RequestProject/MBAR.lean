/-
# Part CV  The estimator behind the windows: MBAR/WHAM self-consistency

Part LXVI proves what umbrella-sampling window data determines, and says explicitly what it does
not treat: "the WHAM/MBAR self-consistency iteration and its convergence" are "treated elsewhere
or not at all".  Every published free-energy profile from window data is produced by that
iteration, so what it determines is part of what a model of a disordered region can be held to.
This file proves the four facts that matter.

The equations, in the exponentiated variables `z i = exp(−f i)` with sample weights `mu`, sample
counts `N` and Boltzmann factors `W i w = exp(−u i w)`, are the fixed points of

    (T z) i  =  ∑_w  mu w · W i w / ( ∑_j N j · W j w / z j ) .

* `mbarMap_smul` — **the equations are exactly scale invariant**, so the free energies are
  determined only up to an additive constant.  This is not a defect of the estimator: it is the
  statement that only free energy *differences* exist, and `freeEnergy_diff_eq` turns it into the
  differences being well defined.
* `mbar_consistent` — **the estimator is consistent.**  In the infinite-sampling limit, where the
  sample weight is the mixture density that the pooled windows actually realise, the true
  partition functions are a fixed point of the equations, exactly.  Nothing about the iteration is
  vacuous, and the target it aims at is the right one.
* `mbar_unique_up_to_scale` — **and the fixed point is unique up to that scale.**  With positive
  overlap (every window giving every sample nonzero weight) any two positive solutions are
  proportional, so the iteration cannot converge to a spurious second answer and the free energy
  differences it reports are a function of the data alone.  The proof is the maximal-ratio
  argument: the coordinate where `z/y` is largest cannot be strictly largest.
* `mbar_iter_bracket` — **the iteration is non-expansive in the ratio bracket**: if the current
  guess lies between `a·y` and `b·y` for the solution `y`, so does its update.  A bracket that
  starts finite stays finite; the iteration cannot diverge, and the reported uncertainty of a
  profile can be propagated through it.

Together with Part LXVI this closes the umbrella-sampling chain: the window data determines the
populations, the estimator's fixed point is the truth, it is unique, and the iteration towards it
is stable.  What is still not proved here is a *rate* of convergence, which depends on the overlap
in a way that a worst-case bound would render useless.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset

namespace IDR.MBAR

variable {I Ω : Type*} [Fintype I] [Fintype Ω]

/-- The mixture denominator of the MBAR equations at the current free-energy guess `z`. -/
noncomputable def denom (N : I → ℝ) (W : I → Ω → ℝ) (z : I → ℝ) (w : Ω) : ℝ :=
  ∑ j, N j * W j w / z j

/-- The MBAR/WHAM self-consistency map, in the variables `z i = exp(−f i)`. -/
noncomputable def mbarMap (N : I → ℝ) (W : I → Ω → ℝ) (mu : Ω → ℝ) (z : I → ℝ) : I → ℝ :=
  fun i => ∑ w, mu w * (W i w / denom N W z w)

/-- A positive solution of the self-consistency equations. -/
def IsSolution (N : I → ℝ) (W : I → Ω → ℝ) (mu : Ω → ℝ) (z : I → ℝ) : Prop :=
  (∀ i, 0 < z i) ∧ mbarMap N W mu z = z

omit [Fintype Ω] in
lemma denom_pos [Nonempty I] {N : I → ℝ} {W : I → Ω → ℝ} {z : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hz : ∀ i, 0 < z i) (w : Ω) :
    0 < denom N W z w :=
  Finset.sum_pos (fun j _ => div_pos (mul_pos (hN j) (hW j w)) (hz j)) Finset.univ_nonempty

omit [Fintype Ω] in
/-- Lowering the guess raises the mixture denominator. -/
lemma denom_anti {N : I → ℝ} {W : I → Ω → ℝ} {z z' : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hz : ∀ i, 0 < z i) (hle : ∀ i, z i ≤ z' i)
    (w : Ω) : denom N W z' w ≤ denom N W z w :=
  Finset.sum_le_sum fun j _ =>
    div_le_div_of_nonneg_left (mul_pos (hN j) (hW j w)).le (hz j) (hle j)

/-! ## Scale invariance: only free energy differences exist -/

omit [Fintype Ω] in
lemma denom_smul {N : I → ℝ} {W : I → Ω → ℝ} {z : I → ℝ} {c : ℝ} (hc : c ≠ 0) (w : Ω) :
    denom N W (fun i => c * z i) w = denom N W z w / c := by
  unfold denom
  rw [Finset.sum_div]
  refine Finset.sum_congr rfl fun j _ => ?_
  field_simp

/-- **The self-consistency equations are exactly scale invariant** — equivalently, invariant under
adding a constant to every free energy. -/
theorem mbarMap_smul {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ} {z : I → ℝ} {c : ℝ} (hc : 0 < c) :
    mbarMap N W mu (fun i => c * z i) = fun i => c * mbarMap N W mu z i := by
  funext i
  unfold mbarMap
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun w _ => ?_
  rw [denom_smul hc.ne' w]
  field_simp

/-- Scaling a solution gives a solution: the additive constant in the free energies is free. -/
theorem IsSolution.smul {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ} {z : I → ℝ} {c : ℝ} (hc : 0 < c)
    (h : IsSolution N W mu z) : IsSolution N W mu (fun i => c * z i) := by
  refine ⟨fun i => mul_pos hc (h.1 i), ?_⟩
  rw [mbarMap_smul (N := N) (W := W) (mu := mu) (z := z) hc]
  funext i
  rw [congrFun h.2 i]

/-- Free energies from the exponentiated variables. -/
noncomputable def freeEnergy (z : I → ℝ) : I → ℝ := fun i => -Real.log (z i)

omit [Fintype I] in
/-- **Free energy differences are invariant under the scale freedom**: they are what the data
determines. -/
theorem freeEnergy_diff_eq {z : I → ℝ} {c : ℝ} (hc : 0 < c) (hz : ∀ i, 0 < z i) (i j : I) :
    freeEnergy (fun k => c * z k) i - freeEnergy (fun k => c * z k) j
      = freeEnergy z i - freeEnergy z j := by
  unfold freeEnergy
  rw [Real.log_mul hc.ne' (hz i).ne', Real.log_mul hc.ne' (hz j).ne']
  ring

/-! ## Consistency: the truth is a fixed point -/

/-- **The estimator is consistent.**  With the sample weight equal to the mixture density the
pooled windows realise, the vector of true partition functions solves the self-consistency
equations exactly. -/
theorem mbar_consistent [Nonempty I] [Nonempty Ω] {N : I → ℝ} {W : I → Ω → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) :
    mbarMap N W (denom N W (fun i => ∑ w, W i w)) (fun i => ∑ w, W i w)
      = fun i => ∑ w, W i w := by
  have hZ : ∀ i, 0 < ∑ w, W i w := fun i =>
    Finset.sum_pos (fun w _ => hW i w) Finset.univ_nonempty
  funext i
  unfold mbarMap
  refine Finset.sum_congr rfl fun w _ => ?_
  have hd : denom N W (fun i => ∑ w, W i w) w ≠ 0 := (denom_pos hN hW hZ w).ne'
  field_simp

/-! ## Uniqueness up to the scale -/

/-- Monotonicity of the map in the current guess. -/
lemma mbarMap_mono [Nonempty I] {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ} {z z' : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hmu : ∀ w, 0 ≤ mu w)
    (hz : ∀ i, 0 < z i) (hz' : ∀ i, 0 < z' i) (hle : ∀ i, z i ≤ z' i) (i : I) :
    mbarMap N W mu z i ≤ mbarMap N W mu z' i := by
  refine Finset.sum_le_sum fun w _ => mul_le_mul_of_nonneg_left ?_ (hmu w)
  exact div_le_div_of_nonneg_left (hW i w).le (denom_pos hN hW hz' w)
    (denom_anti hN hW hz hle w)

/-- **The positive solution of the self-consistency equations is unique up to the scale
freedom.** -/
theorem mbar_unique_up_to_scale [Nonempty I] {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hmu : ∀ w, 0 < mu w)
    {z y : I → ℝ} (hzs : IsSolution N W mu z) (hys : IsSolution N W mu y) :
    ∃ c : ℝ, 0 < c ∧ ∀ i, z i = c * y i := by
  obtain ⟨hzpos, hzfix⟩ := hzs
  obtain ⟨hypos, hyfix⟩ := hys
  obtain ⟨i₀, -, hmax⟩ :=
    Finset.exists_max_image (Finset.univ : Finset I) (fun i => z i / y i) Finset.univ_nonempty
  set lam := z i₀ / y i₀ with hlam
  have hlampos : 0 < lam := div_pos (hzpos i₀) (hypos i₀)
  have hlypos : ∀ i, 0 < lam * y i := fun i => mul_pos hlampos (hypos i)
  have hbound : ∀ j, z j ≤ lam * y j := by
    intro j
    have hj := hmax j (Finset.mem_univ j)
    rw [hlam, div_le_div_iff₀ (hypos j) (hypos i₀)] at hj
    rw [hlam, div_mul_eq_mul_div, le_div_iff₀ (hypos i₀)]
    linarith [hj]
  have heq₀ : z i₀ = lam * y i₀ := by
    rw [hlam, div_mul_cancel₀ _ (hypos i₀).ne']
  have hly : IsSolution N W mu (fun i => lam * y i) := IsSolution.smul hlampos ⟨hypos, hyfix⟩
  -- termwise comparison at the maximising coordinate
  have hterm : ∀ w ∈ (Finset.univ : Finset Ω),
      mu w * (W i₀ w / denom N W z w) ≤ mu w * (W i₀ w / denom N W (fun i => lam * y i) w) := by
    intro w _
    refine mul_le_mul_of_nonneg_left ?_ (hmu w).le
    exact div_le_div_of_nonneg_left (hW i₀ w).le (denom_pos hN hW hlypos w)
      (denom_anti hN hW hzpos hbound w)
  have hsum : ∑ w, mu w * (W i₀ w / denom N W z w)
      = ∑ w, mu w * (W i₀ w / denom N W (fun i => lam * y i) w) := by
    have h1 : mbarMap N W mu (fun i => lam * y i) i₀ = lam * y i₀ := congrFun hly.2 i₀
    have h2 : mbarMap N W mu z i₀ = z i₀ := congrFun hzfix i₀
    unfold mbarMap at h1 h2
    rw [h1, h2, heq₀]
  have hpt := (Finset.sum_eq_sum_iff_of_le hterm).mp hsum
  refine ⟨lam, hlampos, fun j => ?_⟩
  -- the denominators must agree at every sample
  have hdeneq : ∀ w, denom N W z w = denom N W (fun i => lam * y i) w := by
    intro w
    have hw := hpt w (Finset.mem_univ w)
    have h3 : W i₀ w / denom N W z w = W i₀ w / denom N W (fun i => lam * y i) w :=
      mul_left_cancel₀ (hmu w).ne' hw
    rw [div_eq_div_iff (denom_pos hN hW hzpos w).ne' (denom_pos hN hW hlypos w).ne'] at h3
    exact (mul_left_cancel₀ (hW i₀ w).ne' h3).symm
  obtain ⟨w₀⟩ : Nonempty Ω := by
    by_contra hcon
    have hE : IsEmpty Ω := not_nonempty_iff.mp hcon
    have h2 : mbarMap N W mu z i₀ = 0 := by
      unfold mbarMap
      simp
    rw [congrFun hzfix i₀] at h2
    exact (hzpos i₀).ne' h2
  -- and inside the denominator, termwise
  have hle : ∀ k ∈ (Finset.univ : Finset I),
      N k * W k w₀ / (lam * y k) ≤ N k * W k w₀ / z k := fun k _ =>
    div_le_div_of_nonneg_left (mul_pos (hN k) (hW k w₀)).le (hzpos k) (hbound k)
  have hsum2 : ∑ k, N k * W k w₀ / (lam * y k) = ∑ k, N k * W k w₀ / z k :=
    (hdeneq w₀).symm
  have hj := (Finset.sum_eq_sum_iff_of_le hle).mp hsum2 j (Finset.mem_univ j)
  rw [div_eq_div_iff (hlypos j).ne' (hzpos j).ne'] at hj
  exact mul_left_cancel₀ (mul_pos (hN j) (hW j w₀)).ne' hj

/-! ## Stability of the iteration -/

/-- **The iteration is non-expansive in the ratio bracket.**  If the current guess lies between
`a·y` and `b·y` for a solution `y`, then so does its update. -/
theorem mbar_iter_bracket [Nonempty I] {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hmu : ∀ w, 0 ≤ mu w)
    {y z : I → ℝ} (hy : IsSolution N W mu y) (hz : ∀ i, 0 < z i)
    {a b : ℝ} (ha : 0 < a) (hb : 0 < b)
    (hlo : ∀ i, a * y i ≤ z i) (hhi : ∀ i, z i ≤ b * y i) :
    ∀ i, a * y i ≤ mbarMap N W mu z i ∧ mbarMap N W mu z i ≤ b * y i := by
  intro i
  have hyp := hy.1
  have hlow : mbarMap N W mu (fun k => a * y k) i ≤ mbarMap N W mu z i :=
    mbarMap_mono hN hW hmu (fun k => mul_pos ha (hyp k)) hz hlo i
  have hhigh : mbarMap N W mu z i ≤ mbarMap N W mu (fun k => b * y k) i :=
    mbarMap_mono hN hW hmu hz (fun k => mul_pos hb (hyp k)) hhi i
  have hfa : mbarMap N W mu (fun k => a * y k) i = a * y i :=
    congrFun (IsSolution.smul ha hy).2 i
  have hfb : mbarMap N W mu (fun k => b * y k) i = b * y i :=
    congrFun (IsSolution.smul hb hy).2 i
  exact ⟨by rw [← hfa]; exact hlow, by rw [← hfb]; exact hhigh⟩

end IDR.MBAR
