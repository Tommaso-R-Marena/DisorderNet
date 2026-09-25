import RequestProject.Physics.PartialCalculus

/-!
# Part CXLI — Forces from many-body potentials

Molecular dynamics of a protein needs the *gradient* of the potential energy, not merely
its value.  This file computes those gradients honestly, in continuous three-dimensional
space, for a general radial pair potential (and hence for Lennard-Jones, screened Coulomb,
and any other smooth two-body term), and derives from them the two conservation laws that
any physically admissible force field must satisfy:

* `total_force_zero` — the forces sum to zero (Newton's third law / momentum conservation),
* `total_torque_zero` — the torques sum to zero (angular-momentum conservation).

We also record the smoothness of the energy away from atomic collisions and the exact
equilibrium bond length and well depth of the Lennard-Jones potential.
-/

noncomputable section

namespace RequestProject.Physics

open scoped RealInnerProductSpace
open Real Finset

variable {N : ℕ}

/-! ### Translation of partial derivatives -/

lemma HasPDerivAt.comp_sub {n : ℕ} {u : Sp n → ℝ} {i : Fin n} {d : ℝ} {x z : Sp n}
    (h : HasPDerivAt u i d (x - z)) : HasPDerivAt (fun y => u (y - z)) i d x := by
  have hfun : (fun t : ℝ => u ((x + t • EuclideanSpace.single i (1 : ℝ)) - z))
      = fun t : ℝ => u ((x - z) + t • EuclideanSpace.single i (1 : ℝ)) := by
    funext t
    congr 1
    abel
  simpa [HasPDerivAt, hfun] using h

/-- The `i`-th partial derivative of `y ↦ ‖y - z‖` away from `z`. -/
lemma hasPDerivAt_dist {n : ℕ} {x z : Sp n} (hxz : x ≠ z) (i : Fin n) :
    HasPDerivAt (fun y : Sp n => ‖y - z‖) i ((x - z) i / ‖x - z‖) x := by
  have hne : x - z ≠ 0 := sub_ne_zero.mpr hxz
  exact HasPDerivAt.comp_sub (hasPDerivAt_norm hne i)

/-! ### The pairwise-additive energy and its gradient -/

/-- The pairwise-additive energy of a configuration, for a radial pair potential `V`. -/
def pairEnergy (V : ℝ → ℝ) (x : Fin N → Sp 3) : ℝ :=
  (1 / 2) * ∑ i, ∑ j ∈ Finset.univ.erase i, V ‖x i - x j‖

/-- The part of the energy that does not involve atom `k`. -/
def pairEnergyRest (V : ℝ → ℝ) (x : Fin N → Sp 3) (k : Fin N) : ℝ :=
  (1 / 2) * ∑ i ∈ Finset.univ.erase k, ∑ j ∈ (Finset.univ.erase i).erase k, V ‖x i - x j‖

/-- Moving atom `k` only changes the terms that involve atom `k`. -/
theorem pairEnergy_update (V : ℝ → ℝ) (x : Fin N → Sp 3) (k : Fin N) (y : Sp 3) :
    pairEnergy V (Function.update x k y)
      = (∑ j ∈ Finset.univ.erase k, V ‖y - x j‖) + pairEnergyRest V x k := by
  classical
  unfold pairEnergy pairEnergyRest
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ k)]
  have hk : ∀ j ∈ Finset.univ.erase k,
      V ‖Function.update x k y k - Function.update x k y j‖ = V ‖y - x j‖ := by
    intro j hj
    have hjk : j ≠ k := Finset.ne_of_mem_erase hj
    simp [Function.update_apply, hjk]
  have hinner : ∀ i ∈ Finset.univ.erase k,
      ∑ j ∈ Finset.univ.erase i, V ‖Function.update x k y i - Function.update x k y j‖
        = V ‖y - x i‖ + ∑ j ∈ (Finset.univ.erase i).erase k, V ‖x i - x j‖ := by
    intro i hi
    have hik : i ≠ k := Finset.ne_of_mem_erase hi
    have hmem : k ∈ Finset.univ.erase i := Finset.mem_erase.mpr ⟨fun h => hik h.symm, Finset.mem_univ k⟩
    rw [← Finset.add_sum_erase _ _ hmem]
    congr 1
    · simp [Function.update_apply, hik, norm_sub_rev]
    · refine Finset.sum_congr rfl fun j hj => ?_
      have hjk : j ≠ k := Finset.ne_of_mem_erase hj
      simp [Function.update_apply, hik, hjk]
  rw [Finset.sum_congr rfl hinner, Finset.sum_congr rfl hk, Finset.sum_add_distrib]
  ring

/-- The force on atom `k`: minus the gradient of the energy with respect to its position. -/
def force (V : ℝ → ℝ) (x : Fin N → Sp 3) (k : Fin N) : Sp 3 :=
  -grad (fun y => pairEnergy V (Function.update x k y)) (x k)

/-- **The force law, in components.**  For a smooth radial pair potential the force on atom
`k` is the sum of the central two-body forces exerted by the other atoms. -/
theorem force_apply {V V1 : ℝ → ℝ} (hV : ∀ s, 0 < s → HasDerivAt V (V1 s) s)
    {x : Fin N → Sp 3} (hx : Function.Injective x) (k : Fin N) (a : Fin 3) :
    force V x k a
      = -∑ j ∈ Finset.univ.erase k, V1 ‖x k - x j‖ * ((x k - x j) a / ‖x k - x j‖) := by
  classical
  have hfun : (fun y => pairEnergy V (Function.update x k y))
      = fun y => (∑ j ∈ Finset.univ.erase k, V ‖y - x j‖) + pairEnergyRest V x k := by
    funext y; exact pairEnergy_update V x k y
  have hterm : ∀ j ∈ Finset.univ.erase k,
      HasPDerivAt (fun y : Sp 3 => V ‖y - x j‖) a
        (V1 ‖x k - x j‖ * ((x k - x j) a / ‖x k - x j‖)) (x k) := by
    intro j hj
    have hjk : j ≠ k := Finset.ne_of_mem_erase hj
    have hne : x k ≠ x j := fun h => hjk (hx h).symm
    have hpos : 0 < ‖x k - x j‖ := by
      have : x k - x j ≠ 0 := sub_ne_zero.mpr hne
      exact norm_pos_iff.mpr this
    have hdist := hasPDerivAt_dist (z := x j) hne a
    exact HasPDerivAt.comp (hV _ hpos) hdist
  have hsum : HasPDerivAt (fun y : Sp 3 => (∑ j ∈ Finset.univ.erase k, V ‖y - x j‖)
      + pairEnergyRest V x k) a
      (∑ j ∈ Finset.univ.erase k, V1 ‖x k - x j‖ * ((x k - x j) a / ‖x k - x j‖)) (x k) := by
    have h1 := HasPDerivAt.sum (i := a) (x := x k)
      (U := fun j (y : Sp 3) => V ‖y - x j‖)
      (D := fun j => V1 ‖x k - x j‖ * ((x k - x j) a / ‖x k - x j‖)) hterm
    simpa using h1.add (HasPDerivAt.const (i := a) (x := x k) (pairEnergyRest V x k))
  have : pderiv a (fun y => pairEnergy V (Function.update x k y)) (x k)
      = ∑ j ∈ Finset.univ.erase k, V1 ‖x k - x j‖ * ((x k - x j) a / ‖x k - x j‖) := by
    rw [hfun]; exact hsum.pderiv_eq
  simp [force, this]

/-- **The force law, in vector form.** -/
theorem force_eq {V V1 : ℝ → ℝ} (hV : ∀ s, 0 < s → HasDerivAt V (V1 s) s)
    {x : Fin N → Sp 3} (hx : Function.Injective x) (k : Fin N) :
    force V x k
      = -∑ j ∈ Finset.univ.erase k, (V1 ‖x k - x j‖ / ‖x k - x j‖) • (x k - x j) := by
  ext a
  rw [force_apply hV hx k a]
  simp [Finset.sum_apply, mul_div_assoc, mul_comm, mul_left_comm, mul_assoc, div_mul_eq_mul_div]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- **Newton's third law**: in a two-atom system the forces are equal and opposite. -/
theorem newton_third_law {V V1 : ℝ → ℝ} (hV : ∀ s, 0 < s → HasDerivAt V (V1 s) s)
    {x : Fin 2 → Sp 3} (hx : Function.Injective x) :
    force V x 0 = -force V x 1 := by
  classical
  rw [force_eq hV hx 0, force_eq hV hx 1]
  have h0 : (Finset.univ.erase (0 : Fin 2)) = {1} := by decide
  have h1 : (Finset.univ.erase (1 : Fin 2)) = {0} := by decide
  rw [h0, h1]
  simp only [Finset.sum_singleton, neg_neg]
  rw [norm_sub_rev (x 1) (x 0)]
  have : x 1 - x 0 = -(x 0 - x 1) := by abel
  rw [this]
  module

/-- **Momentum conservation**: the total force of a pairwise-additive potential vanishes. -/
theorem total_force_zero {V V1 : ℝ → ℝ} (hV : ∀ s, 0 < s → HasDerivAt V (V1 s) s)
    {x : Fin N → Sp 3} (hx : Function.Injective x) :
    ∑ k, force V x k = 0 := by
  classical
  have hforce : ∀ k, force V x k
      = -∑ j ∈ Finset.univ.erase k, (V1 ‖x k - x j‖ / ‖x k - x j‖) • (x k - x j) :=
    fun k => force_eq hV hx k
  set F : Fin N → Fin N → Sp 3 := fun k j =>
    if j = k then 0 else (V1 ‖x k - x j‖ / ‖x k - x j‖) • (x k - x j) with hF
  have hrew : ∀ k, force V x k = -∑ j, F k j := by
    intro k
    rw [hforce k]
    congr 1
    have hsplit : ∑ j, F k j = ∑ j ∈ Finset.univ.erase k, F k j := by
      refine (Finset.sum_subset (Finset.subset_univ _) ?_).symm
      intro j _ hj
      have hjk : j = k := by
        by_contra hne
        exact hj (Finset.mem_erase.mpr ⟨hne, Finset.mem_univ j⟩)
      simp [hF, hjk]
    rw [hsplit]
    refine Finset.sum_congr rfl fun j hj => ?_
    have hjk : j ≠ k := Finset.ne_of_mem_erase hj
    simp [hF, hjk]
  have hanti : ∀ k j, F k j = -F j k := by
    intro k j
    by_cases h : j = k
    · subst h; simp [hF]
    · have h' : k ≠ j := fun hh => h hh.symm
      have hnorm : ‖x k - x j‖ = ‖x j - x k‖ := norm_sub_rev _ _
      simp only [hF, if_neg h, if_neg h', hnorm]
      have : x k - x j = -(x j - x k) := by abel
      rw [this]
      module
  have hS : ∑ k, ∑ j, F k j = 0 := by
    have h1 : ∑ k, ∑ j, F k j = ∑ j, ∑ k, F k j := Finset.sum_comm
    have h2 : ∑ j, ∑ k, F k j = -∑ j, ∑ k, F j k := by
      rw [← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [← Finset.sum_neg_distrib]
      exact Finset.sum_congr rfl fun k _ => hanti k j
    have hself : ∑ k, ∑ j, F k j = -∑ k, ∑ j, F k j := h1.trans h2
    have htwo : (2 : ℝ) • (∑ k, ∑ j, F k j) = 0 := by
      rw [two_smul]
      nth_rewrite 2 [hself]
      simp
    have := smul_eq_zero.mp htwo
    rcases this with h | h
    · norm_num at h
    · exact h
  calc ∑ k, force V x k = ∑ k, -∑ j, F k j := by
        exact Finset.sum_congr rfl fun k _ => hrew k
    _ = -∑ k, ∑ j, F k j := by rw [Finset.sum_neg_distrib]
    _ = 0 := by rw [hS]; simp

/-! ### Cross products and torque -/

/-- The cross product of three-dimensional vectors. -/
def cross (u v : Sp 3) : Sp 3 :=
  (WithLp.equiv 2 (Fin 3 → ℝ)).symm
    ![u 1 * v 2 - u 2 * v 1, u 2 * v 0 - u 0 * v 2, u 0 * v 1 - u 1 * v 0]

@[simp] lemma cross_self (u : Sp 3) : cross u u = 0 := by
  ext i; fin_cases i <;> simp [cross] <;> ring

lemma cross_antisymm (u v : Sp 3) : cross u v = -cross v u := by
  ext i; fin_cases i <;> simp [cross] <;> ring

lemma cross_smul_right (u v : Sp 3) (c : ℝ) : cross u (c • v) = c • cross u v := by
  ext i; fin_cases i <;> simp [cross] <;> ring

lemma cross_sub_right (u v w : Sp 3) : cross u (v - w) = cross u v - cross u w := by
  ext i; fin_cases i <;> simp [cross] <;> ring

lemma cross_neg_right (u v : Sp 3) : cross u (-v) = -cross u v := by
  ext i; fin_cases i <;> simp [cross] <;> ring

lemma cross_sum_right {ι : Type*} (s : Finset ι) (u : Sp 3) (f : ι → Sp 3) :
    cross u (∑ j ∈ s, f j) = ∑ j ∈ s, cross u (f j) := by
  classical
  induction s using Finset.induction with
  | empty => ext i; fin_cases i <;> simp [cross]
  | insert b s hb ih =>
      rw [Finset.sum_insert hb, Finset.sum_insert hb, ← ih]
      ext i; fin_cases i <;> simp [cross] <;> ring

/-- **Angular-momentum conservation**: the total torque of a central pairwise potential
about the origin vanishes. -/
theorem total_torque_zero {V V1 : ℝ → ℝ} (hV : ∀ s, 0 < s → HasDerivAt V (V1 s) s)
    {x : Fin N → Sp 3} (hx : Function.Injective x) :
    ∑ k, cross (x k) (force V x k) = 0 := by
  classical
  set T : Fin N → Fin N → Sp 3 := fun k j =>
    if j = k then 0 else (V1 ‖x k - x j‖ / ‖x k - x j‖) • cross (x k) (x j) with hT
  have hterm : ∀ k, cross (x k) (force V x k) = ∑ j, T k j := by
    intro k
    rw [force_eq hV hx k, cross_neg_right, cross_sum_right]
    rw [← Finset.sum_neg_distrib]
    have hsplit : ∑ j, T k j = ∑ j ∈ Finset.univ.erase k, T k j := by
      refine (Finset.sum_subset (Finset.subset_univ _) ?_).symm
      intro j _ hj
      have hjk : j = k := by
        by_contra hne
        exact hj (Finset.mem_erase.mpr ⟨hne, Finset.mem_univ j⟩)
      simp [hT, hjk]
    rw [hsplit]
    refine Finset.sum_congr rfl fun j hj => ?_
    have hjk : j ≠ k := Finset.ne_of_mem_erase hj
    simp only [hT, if_neg hjk, cross_smul_right, cross_sub_right, cross_self]
    module
  have hanti : ∀ k j, T k j = -T j k := by
    intro k j
    by_cases h : j = k
    · subst h; simp [hT]
    · have h' : k ≠ j := fun hh => h hh.symm
      have hnorm : ‖x k - x j‖ = ‖x j - x k‖ := norm_sub_rev _ _
      simp only [hT, if_neg h, if_neg h', hnorm]
      rw [cross_antisymm (x k) (x j)]
      module
  have hS : ∑ k, ∑ j, T k j = 0 := by
    have h1 : ∑ k, ∑ j, T k j = ∑ j, ∑ k, T k j := Finset.sum_comm
    have h2 : ∑ j, ∑ k, T k j = -∑ k, ∑ j, T k j := by
      rw [← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [← Finset.sum_neg_distrib]
      exact Finset.sum_congr rfl fun k _ => hanti k j
    have hself : ∑ k, ∑ j, T k j = -∑ k, ∑ j, T k j := h1.trans h2
    have htwo : (2 : ℝ) • (∑ k, ∑ j, T k j) = 0 := by
      rw [two_smul]
      nth_rewrite 2 [hself]
      simp
    rcases smul_eq_zero.mp htwo with h | h
    · norm_num at h
    · exact h
  calc ∑ k, cross (x k) (force V x k) = ∑ k, ∑ j, T k j :=
        Finset.sum_congr rfl fun k _ => hterm k
    _ = 0 := hS

/-! ### Smoothness away from collisions -/

/-- The pair energy is smooth in the position of any atom, as long as the atoms are
distinct. -/
theorem contDiffAt_pairEnergy {V : ℝ → ℝ} (hV : ContDiff ℝ (⊤ : ℕ∞) V)
    {x : Fin N → Sp 3} (hx : Function.Injective x) (k : Fin N) :
    ContDiffAt ℝ (⊤ : ℕ∞) (fun y => pairEnergy V (Function.update x k y)) (x k) := by
  classical
  have hfun : (fun y => pairEnergy V (Function.update x k y))
      = fun y => (∑ j ∈ Finset.univ.erase k, V ‖y - x j‖) + pairEnergyRest V x k := by
    funext y; exact pairEnergy_update V x k y
  rw [hfun]
  refine ContDiffAt.add ?_ contDiffAt_const
  refine ContDiffAt.sum fun j hj => ?_
  have hjk : j ≠ k := Finset.ne_of_mem_erase hj
  have hne : x k - x j ≠ 0 := sub_ne_zero.mpr fun h => hjk (hx h).symm
  have hsub : ContDiffAt ℝ (⊤ : ℕ∞) (fun y : Sp 3 => y - x j) (x k) :=
    (contDiff_id.sub contDiff_const).contDiffAt
  exact hV.contDiffAt.comp (x k) (hsub.norm ℝ (by simpa using hne))

/-! ### The Lennard-Jones potential -/

/-- The Lennard-Jones 12-6 potential. -/
def lennardJones (eps sigma r : ℝ) : ℝ :=
  4 * eps * ((sigma / r) ^ 12 - (sigma / r) ^ 6)

/-- Derivative of the Lennard-Jones potential. -/
theorem hasDerivAt_lennardJones (eps sigma : ℝ) {r : ℝ} (hr : 0 < r) :
    HasDerivAt (lennardJones eps sigma)
      (4 * eps * (-12 * sigma ^ 12 / r ^ 13 + 6 * sigma ^ 6 / r ^ 7)) r := by
  have h12 : HasDerivAt (fun s : ℝ => (sigma / s) ^ 12) (-12 * sigma ^ 12 / r ^ 13) r := by
    have hd : HasDerivAt (fun s : ℝ => sigma / s) (-(sigma / r ^ 2)) r := by
      have := (hasDerivAt_inv hr.ne').const_mul sigma
      simpa [div_eq_mul_inv, mul_comm, mul_neg, neg_div] using this
    have hrne : r ≠ 0 := hr.ne'
    have h := hd.pow 12
    convert h using 1
    norm_num
    field_simp
  have h6 : HasDerivAt (fun s : ℝ => (sigma / s) ^ 6) (-6 * sigma ^ 6 / r ^ 7) r := by
    have hd : HasDerivAt (fun s : ℝ => sigma / s) (-(sigma / r ^ 2)) r := by
      have := (hasDerivAt_inv hr.ne').const_mul sigma
      simpa [div_eq_mul_inv, mul_comm, mul_neg, neg_div] using this
    have hrne : r ≠ 0 := hr.ne'
    have h := hd.pow 6
    convert h using 1
    norm_num
    field_simp
  have := (h12.sub h6).const_mul (4 * eps)
  convert this using 1
  ring

lemma two_pow_sixth_pow_six : ((2 : ℝ) ^ ((1 : ℝ) / 6)) ^ (6 : ℕ) = 2 := by
  rw [← Real.rpow_natCast ((2 : ℝ) ^ ((1 : ℝ) / 6)) 6, ← Real.rpow_mul (by norm_num)]
  norm_num

/-- **The Lennard-Jones equilibrium bond length** is exactly `2^{1/6} σ`: there the force
vanishes. -/
theorem lennardJones_force_zero {eps sigma : ℝ} (hsig : 0 < sigma) :
    4 * eps * (-12 * sigma ^ 12 / ((2 : ℝ) ^ ((1 : ℝ) / 6) * sigma) ^ 13
      + 6 * sigma ^ 6 / ((2 : ℝ) ^ ((1 : ℝ) / 6) * sigma) ^ 7) = 0 := by
  have ha : (0 : ℝ) < (2 : ℝ) ^ ((1 : ℝ) / 6) := Real.rpow_pos_of_pos (by norm_num) _
  have ha6 := two_pow_sixth_pow_six
  set a := (2 : ℝ) ^ ((1 : ℝ) / 6) with hadef
  have hane : a ≠ 0 := ha.ne'
  have hsne : sigma ≠ 0 := hsig.ne'
  have key : -12 * sigma ^ 12 / (a * sigma) ^ 13 + 6 * sigma ^ 6 / (a * sigma) ^ 7 = 0 := by
    field_simp
    nlinarith [ha6, sq_nonneg (a ^ 6), pow_pos ha 7, pow_pos hsig 7]
  rw [key]
  ring

/-- At the equilibrium separation the Lennard-Jones energy is exactly `-ε`. -/
theorem lennardJones_min_value {eps sigma : ℝ} (hsig : 0 < sigma) :
    lennardJones eps sigma ((2 : ℝ) ^ ((1 : ℝ) / 6) * sigma) = -eps := by
  have ha : (0 : ℝ) < (2 : ℝ) ^ ((1 : ℝ) / 6) := Real.rpow_pos_of_pos (by norm_num) _
  have ha6 := two_pow_sixth_pow_six
  set a := (2 : ℝ) ^ ((1 : ℝ) / 6) with hadef
  have hane : a ≠ 0 := ha.ne'
  have hsne : sigma ≠ 0 := hsig.ne'
  unfold lennardJones
  have hx : sigma / (a * sigma) = 1 / a := by field_simp
  rw [hx]
  have h6 : (1 / a) ^ 6 = 1 / 2 := by
    rw [div_pow, one_pow, ha6]
  have h12 : (1 / a) ^ 12 = 1 / 4 := by
    have : (1 / a) ^ 12 = ((1 / a) ^ 6) ^ 2 := by ring
    rw [this, h6]; norm_num
  rw [h12, h6]
  ring

end RequestProject.Physics
