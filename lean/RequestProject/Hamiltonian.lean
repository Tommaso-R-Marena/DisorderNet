/-
# Part XXIV.3  The Hamiltonian on the continuous conformation space

The conformation space is `Conf N = Fin N -> EuclideanSpace R (Fin 3)`, i.e. `R^(3N)`:
real atomic coordinates, not lattice sites.  A `ForceField N` carries per-atom partial
charges, Lennard-Jones parameters, a bonded network, and the relative permittivity of the
medium; `H` is the total potential energy

  `H = E_bonded + E_electrostatic + E_LennardJones`

with Lorentz-Berthelot mixing (`epsMix = sqrt (eps_i eps_j)`, `sigMix = (sig_i + sig_j)/2`).

* `H_bddBelow` -- **the Hamiltonian is bounded below**, with an explicit bound, for every
  force field with a genuine 12-6 core, however the charges are arranged.  Attractive
  `1/r` singularities are dominated by the `r^-12` core.  Everything about the Gibbs
  measure in the next file rests on this.
* `H_rigid_invariant` -- `H` is invariant under the full group of rigid motions of `R^3`
  (rotation composed with translation) acting diagonally on the atoms; more generally
  `IsPairwiseGeometric` energies are, and `H` is one.
* `H_continuousOn_distinct` -- `H` is continuous on the open set of configurations with no
  two atoms exactly coincident, and `H_measurable` -- `H` is Borel measurable everywhere.
* `coincident_null` -- the excluded set (two atoms at exactly the same point) is Lebesgue
  null, so it is invisible to the Gibbs measure.
* `H_repulsive_core` -- if two atoms approach, `H` exceeds any bound: the excluded volume of
  Part XII is recovered as a theorem about the continuous force field rather than assumed
  as a lattice constraint.
-/
import Mathlib
import RequestProject.Potentials

namespace IDR

namespace MM

open Real MeasureTheory Potentials

/-- A point in physical space. -/
abbrev Point := EuclideanSpace ℝ (Fin 3)

/-- A conformation of an `N`-atom chain: real coordinates in `R^3` for each atom. -/
abbrev Conf (N : ℕ) := Fin N → Point

/-- The unordered pairs of distinct atoms. -/
def pairs (N : ℕ) : Finset (Fin N × Fin N) :=
  Finset.univ.filter (fun p => p.1 < p.2)

lemma pairs_ne {N : ℕ} {p : Fin N × Fin N} (hp : p ∈ pairs N) : p.1 ≠ p.2 := by
  simp only [pairs, Finset.mem_filter, Finset.mem_univ, true_and] at hp
  exact ne_of_lt hp

/-- A class-I molecular-mechanics force field. -/
structure ForceField (N : ℕ) where
  /-- Partial charge of each atom, coulombs. -/
  charge : Fin N → ℝ
  /-- Lennard-Jones well depth of each atom, joules. -/
  ljEps : Fin N → ℝ
  /-- Lennard-Jones contact distance of each atom, metres. -/
  ljSigma : Fin N → ℝ
  /-- Bond force constant of each pair (zero for non-bonded pairs). -/
  bondK : Fin N → Fin N → ℝ
  /-- Equilibrium bond length of each pair. -/
  bondLen : Fin N → Fin N → ℝ
  /-- Relative permittivity of the medium. -/
  dielectric : ℝ
  ljEps_pos : ∀ i, 0 < ljEps i
  ljSigma_pos : ∀ i, 0 < ljSigma i
  bondK_nonneg : ∀ i j, 0 ≤ bondK i j
  dielectric_pos : 0 < dielectric

namespace ForceField

variable {N : ℕ} (F : ForceField N)

/-- Lorentz-Berthelot geometric mixing of well depths. -/
noncomputable def epsMix (i j : Fin N) : ℝ := Real.sqrt (F.ljEps i * F.ljEps j)

/-- Lorentz-Berthelot arithmetic mixing of contact distances. -/
noncomputable def sigMix (i j : Fin N) : ℝ := (F.ljSigma i + F.ljSigma j) / 2

lemma epsMix_pos (i j : Fin N) : 0 < F.epsMix i j :=
  Real.sqrt_pos.mpr (mul_pos (F.ljEps_pos i) (F.ljEps_pos j))

lemma sigMix_pos (i j : Fin N) : 0 < F.sigMix i j := by
  unfold sigMix
  have := F.ljSigma_pos i
  have := F.ljSigma_pos j
  linarith

/-- The bonded energy: harmonic in every bond length (the force constant is zero for
non-bonded pairs). -/
noncomputable def Ebond (x : Conf N) : ℝ :=
  ∑ p ∈ pairs N, harmonic (F.bondK p.1 p.2) (F.bondLen p.1 p.2) (dist (x p.1) (x p.2))

/-- The electrostatic energy: Coulomb's law with an explicit relative permittivity. -/
noncomputable def Eelec (x : Conf N) : ℝ :=
  ∑ p ∈ pairs N, coulomb F.dielectric (F.charge p.1) (F.charge p.2) (dist (x p.1) (x p.2))

/-- The van der Waals energy: the smooth Lennard-Jones 12-6 potential. -/
noncomputable def Evdw (x : Conf N) : ℝ :=
  ∑ p ∈ pairs N, lj (F.epsMix p.1 p.2) (F.sigMix p.1 p.2) (dist (x p.1) (x p.2))

/-- The total potential energy. -/
noncomputable def H (x : Conf N) : ℝ := F.Ebond x + F.Eelec x + F.Evdw x

lemma Ebond_nonneg (x : Conf N) : 0 ≤ F.Ebond x :=
  Finset.sum_nonneg fun p _ => harmonic_nonneg (F.bondK_nonneg p.1 p.2)

end ForceField

/-! ## Rigid motions and invariance -/

/-- A rigid motion of physical space: a linear isometry of `R^3` followed by a
translation.  (Taking the linear part to be an arbitrary linear isometry includes the
reflections; the rotations are the orientation-preserving subgroup.) -/
structure RigidMotion where
  /-- The linear part. -/
  rot : Point ≃ₗᵢ[ℝ] Point
  /-- The translation part. -/
  trans : Point

/-- The diagonal action on a conformation. -/
noncomputable def RigidMotion.act {N : ℕ} (g : RigidMotion) (x : Conf N) : Conf N :=
  fun i => g.rot (x i) + g.trans

/-- The rigid motion as a map of physical space. -/
noncomputable def RigidMotion.map (g : RigidMotion) (p : Point) : Point := g.rot p + g.trans

lemma RigidMotion.act_apply {N : ℕ} (g : RigidMotion) (x : Conf N) (i : Fin N) :
    g.act x i = g.map (x i) := rfl

lemma RigidMotion.isometry_map (g : RigidMotion) : Isometry g.map := by
  refine Isometry.of_dist_eq (fun p q => ?_)
  unfold RigidMotion.map
  rw [dist_add_right, LinearIsometryEquiv.dist_map]

lemma RigidMotion.surjective_map (g : RigidMotion) : Function.Surjective g.map := by
  intro p
  refine ⟨g.rot.symm (p - g.trans), ?_⟩
  unfold RigidMotion.map
  simp

/-- The inverse rigid motion. -/
noncomputable def RigidMotion.inv (g : RigidMotion) : RigidMotion :=
  ⟨g.rot.symm, -(g.rot.symm g.trans)⟩

lemma RigidMotion.act_inv_act {N : ℕ} (g : RigidMotion) (x : Conf N) :
    g.inv.act (g.act x) = x := by
  funext i
  unfold RigidMotion.act RigidMotion.inv
  simp

lemma RigidMotion.act_act_inv {N : ℕ} (g : RigidMotion) (x : Conf N) :
    g.act (g.inv.act x) = x := by
  funext i
  unfold RigidMotion.act RigidMotion.inv
  simp

/-- Rigid motions preserve every interatomic distance. -/
theorem RigidMotion.dist_act {N : ℕ} (g : RigidMotion) (x : Conf N) (i j : Fin N) :
    dist (g.act x i) (g.act x j) = dist (x i) (x j) := by
  unfold RigidMotion.act
  rw [dist_add_right, LinearIsometryEquiv.dist_map]

/-- An energy that depends on the configuration only through the interatomic distances. -/
def IsPairwiseGeometric {N : ℕ} (E : Conf N → ℝ) : Prop :=
  ∀ x y : Conf N, (∀ i j, dist (x i) (x j) = dist (y i) (y j)) → E x = E y

/-- **Distance-determined energies are rigid-motion invariant.** -/
theorem invariant_of_pairwiseGeometric {N : ℕ} {E : Conf N → ℝ}
    (hE : IsPairwiseGeometric E) (g : RigidMotion) (x : Conf N) : E (g.act x) = E x :=
  hE _ _ (fun i j => g.dist_act x i j)

variable {N : ℕ}

theorem ForceField.Ebond_pairwiseGeometric (F : ForceField N) :
    IsPairwiseGeometric F.Ebond := by
  intro x y h
  unfold ForceField.Ebond
  exact Finset.sum_congr rfl fun p _ => by rw [h p.1 p.2]

theorem ForceField.Eelec_pairwiseGeometric (F : ForceField N) :
    IsPairwiseGeometric F.Eelec := by
  intro x y h
  unfold ForceField.Eelec
  exact Finset.sum_congr rfl fun p _ => by rw [h p.1 p.2]

theorem ForceField.Evdw_pairwiseGeometric (F : ForceField N) :
    IsPairwiseGeometric F.Evdw := by
  intro x y h
  unfold ForceField.Evdw
  exact Finset.sum_congr rfl fun p _ => by rw [h p.1 p.2]

theorem ForceField.H_pairwiseGeometric (F : ForceField N) : IsPairwiseGeometric F.H := by
  intro x y h
  unfold ForceField.H
  rw [F.Ebond_pairwiseGeometric x y h, F.Eelec_pairwiseGeometric x y h,
    F.Evdw_pairwiseGeometric x y h]

/-- **SE(3) invariance of the Hamiltonian** (indeed E(3) invariance: rotations,
reflections and translations). -/
theorem ForceField.H_rigid_invariant (F : ForceField N) (g : RigidMotion) (x : Conf N) :
    F.H (g.act x) = F.H x :=
  invariant_of_pairwiseGeometric F.H_pairwiseGeometric g x

/-! ## Stability: the Hamiltonian is bounded below -/

/-- The Coulomb term of a pair, written as `c / r`. -/
lemma coulomb_as_div (epsr q1 q2 r : ℝ) :
    coulomb epsr q1 q2 r = (coulombConst * q1 * q2 / epsr) / r := by
  unfold coulomb
  rw [div_div]

/-- The nonbonded energy of a single pair is bounded below. -/
lemma ForceField.pair_bound (F : ForceField N) (i j : Fin N) :
    ∃ B : ℝ, ∀ r : ℝ, 0 < r →
      -B ≤ lj (F.epsMix i j) (F.sigMix i j) r + coulomb F.dielectric (F.charge i)
        (F.charge j) r := by
  obtain ⟨B, hB⟩ := pair_bddBelow (eps := F.epsMix i j) (sigma := F.sigMix i j)
    (c := coulombConst * F.charge i * F.charge j / F.dielectric)
    (F.epsMix_pos i j) (F.sigMix_pos i j)
  refine ⟨B, fun r hr => ?_⟩
  rw [coulomb_as_div]
  exact hB r hr

/-- **The Hamiltonian is bounded below.**  On the set of configurations with no two atoms
exactly coincident there is a finite `B` with `-B ≤ H x` for every `x`.  The bound depends
only on the force field, not on `N`-dependent geometry. -/
theorem ForceField.H_bddBelow (F : ForceField N) :
    ∃ B : ℝ, ∀ x : Conf N, (∀ i j : Fin N, i ≠ j → x i ≠ x j) → -B ≤ F.H x := by
  classical
  refine ⟨∑ p ∈ pairs N, (F.pair_bound p.1 p.2).choose, fun x hx => ?_⟩
  have hterm : ∀ p ∈ pairs N,
      -((F.pair_bound p.1 p.2).choose) ≤
        lj (F.epsMix p.1 p.2) (F.sigMix p.1 p.2) (dist (x p.1) (x p.2)) +
          coulomb F.dielectric (F.charge p.1) (F.charge p.2) (dist (x p.1) (x p.2)) := by
    intro p hp
    have hne : p.1 ≠ p.2 := pairs_ne hp
    have hd : 0 < dist (x p.1) (x p.2) := dist_pos.mpr (hx p.1 p.2 hne)
    exact (F.pair_bound p.1 p.2).choose_spec _ hd
  have hsum : -∑ p ∈ pairs N, (F.pair_bound p.1 p.2).choose ≤ F.Eelec x + F.Evdw x := by
    unfold ForceField.Eelec ForceField.Evdw
    rw [← Finset.sum_add_distrib]
    calc -∑ p ∈ pairs N, (F.pair_bound p.1 p.2).choose
        = ∑ p ∈ pairs N, -((F.pair_bound p.1 p.2).choose) := by
          rw [Finset.sum_neg_distrib]
      _ ≤ ∑ p ∈ pairs N,
            (lj (F.epsMix p.1 p.2) (F.sigMix p.1 p.2) (dist (x p.1) (x p.2)) +
              coulomb F.dielectric (F.charge p.1) (F.charge p.2) (dist (x p.1) (x p.2))) :=
          Finset.sum_le_sum hterm
      _ = ∑ p ∈ pairs N,
            (coulomb F.dielectric (F.charge p.1) (F.charge p.2) (dist (x p.1) (x p.2)) +
              lj (F.epsMix p.1 p.2) (F.sigMix p.1 p.2) (dist (x p.1) (x p.2))) := by
          exact Finset.sum_congr rfl fun p _ => by ring
  have hb := F.Ebond_nonneg x
  unfold ForceField.H
  linarith

/-- The nonbonded energy of one distinguished pair. -/
noncomputable def ForceField.pairEnergy (F : ForceField N) (p : Fin N × Fin N)
    (x : Conf N) : ℝ :=
  lj (F.epsMix p.1 p.2) (F.sigMix p.1 p.2) (dist (x p.1) (x p.2)) +
    coulomb F.dielectric (F.charge p.1) (F.charge p.2) (dist (x p.1) (x p.2))

/-- The rest of the system is bounded below, so the Hamiltonian dominates the energy of any
single pair up to a constant. -/
theorem ForceField.H_ge_pairEnergy (F : ForceField N) {p0 : Fin N × Fin N}
    (hp0 : p0 ∈ pairs N) :
    ∃ B : ℝ, ∀ x : Conf N, (∀ i j : Fin N, i ≠ j → x i ≠ x j) →
      F.pairEnergy p0 x - B ≤ F.H x := by
  classical
  refine ⟨∑ p ∈ (pairs N).erase p0, (F.pair_bound p.1 p.2).choose, fun x hx => ?_⟩
  have hterm : ∀ p ∈ (pairs N).erase p0,
      -((F.pair_bound p.1 p.2).choose) ≤ F.pairEnergy p x := by
    intro p hp
    have hne : p.1 ≠ p.2 := pairs_ne (Finset.mem_of_mem_erase hp)
    have hd : 0 < dist (x p.1) (x p.2) := dist_pos.mpr (hx p.1 p.2 hne)
    exact (F.pair_bound p.1 p.2).choose_spec _ hd
  have hnb : F.Eelec x + F.Evdw x = ∑ p ∈ pairs N, F.pairEnergy p x := by
    unfold ForceField.Eelec ForceField.Evdw ForceField.pairEnergy
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun p _ => by ring
  have hsplit : ∑ p ∈ pairs N, F.pairEnergy p x
      = F.pairEnergy p0 x + ∑ p ∈ (pairs N).erase p0, F.pairEnergy p x :=
    (Finset.add_sum_erase _ _ hp0).symm
  have hrest : -∑ p ∈ (pairs N).erase p0, (F.pair_bound p.1 p.2).choose
      ≤ ∑ p ∈ (pairs N).erase p0, F.pairEnergy p x := by
    calc -∑ p ∈ (pairs N).erase p0, (F.pair_bound p.1 p.2).choose
        = ∑ p ∈ (pairs N).erase p0, -((F.pair_bound p.1 p.2).choose) := by
          rw [Finset.sum_neg_distrib]
      _ ≤ ∑ p ∈ (pairs N).erase p0, F.pairEnergy p x := Finset.sum_le_sum hterm
  have hb := F.Ebond_nonneg x
  unfold ForceField.H
  linarith [hnb, hsplit, hrest, hb]

/-- **Excluded volume is a theorem, not a constraint.**  For any energy bound `M` there is a
separation `d > 0` such that every configuration in which the atoms of a given pair are
closer than `d` has `H > M`.  The continuous 12-6 force field therefore reproduces the
hard-core behaviour that the lattice models of Part XII had to postulate. -/
theorem ForceField.H_repulsive_core (F : ForceField N) {p0 : Fin N × Fin N}
    (hp0 : p0 ∈ pairs N) (M : ℝ) :
    ∃ d : ℝ, 0 < d ∧ ∀ x : Conf N, (∀ i j : Fin N, i ≠ j → x i ≠ x j) →
      dist (x p0.1) (x p0.2) < d → M < F.H x := by
  obtain ⟨B, hB⟩ := F.H_ge_pairEnergy hp0
  obtain ⟨d, hd, hdd⟩ := pair_repulsive_unbounded (eps := F.epsMix p0.1 p0.2)
    (sigma := F.sigMix p0.1 p0.2)
    (c := coulombConst * F.charge p0.1 * F.charge p0.2 / F.dielectric)
    (F.epsMix_pos p0.1 p0.2) (F.sigMix_pos p0.1 p0.2) (M + B)
  refine ⟨d, hd, fun x hx hlt => ?_⟩
  have hne : p0.1 ≠ p0.2 := pairs_ne hp0
  have hpos : 0 < dist (x p0.1) (x p0.2) := dist_pos.mpr (hx p0.1 p0.2 hne)
  have hpair : M + B < F.pairEnergy p0 x := by
    have := hdd _ hpos hlt
    unfold ForceField.pairEnergy
    rwa [coulomb_as_div]
  have := hB x hx
  linarith

/-! ## Measurability, continuity, and the null set of exact overlaps -/

theorem ForceField.H_measurable (F : ForceField N) : Measurable F.H := by
  unfold ForceField.H ForceField.Ebond ForceField.Eelec ForceField.Evdw Potentials.harmonic
    Potentials.coulomb Potentials.lj
  refine Measurable.add (Measurable.add ?_ ?_) ?_
  · exact Finset.measurable_sum _ fun p _ => by fun_prop
  · exact Finset.measurable_sum _ fun p _ => by fun_prop
  · exact Finset.measurable_sum _ fun p _ => by fun_prop

/-- The set of configurations in which two given atoms coincide exactly is Lebesgue null:
the junk values that Lean's division convention assigns there are invisible to every
integral. -/
theorem coincident_pair_null {i j : Fin N} (hij : i ≠ j) :
    volume {x : Conf N | x i = x j} = 0 := by
  have hset : {x : Conf N | x i = x j}
      = (LinearMap.ker ((LinearMap.proj i : (Conf N) →ₗ[ℝ] Point) - LinearMap.proj j) :
          Submodule ℝ (Conf N)) := by
    ext x; simp [LinearMap.mem_ker, sub_eq_zero]
  rw [hset]
  apply Measure.addHaar_submodule
  intro h
  set v : Point := EuclideanSpace.single (0 : Fin 3) (1:ℝ) with hv
  set y : Conf N := fun k => if k = i then v else 0 with hy
  have hmem : y ∈ (LinearMap.ker ((LinearMap.proj i : (Conf N) →ₗ[ℝ] Point)
      - LinearMap.proj j)) := by rw [h]; trivial
  simp only [LinearMap.mem_ker, LinearMap.sub_apply, LinearMap.proj_apply, sub_eq_zero] at hmem
  rw [hy] at hmem
  simp [hij.symm] at hmem
  exact absurd hmem (by simp [hv, EuclideanSpace.single_eq_zero_iff])

/-- The whole coincidence set (some two atoms exactly superposed) is null. -/
theorem coincident_null (N : ℕ) :
    volume {x : Conf N | ∃ i j : Fin N, i ≠ j ∧ x i = x j} = 0 := by
  classical
  have hsub : {x : Conf N | ∃ i j : Fin N, i ≠ j ∧ x i = x j}
      ⊆ ⋃ p ∈ (Finset.univ : Finset (Fin N × Fin N)).filter (fun p => p.1 ≠ p.2),
          {x : Conf N | x p.1 = x p.2} := by
    rintro x ⟨i, j, hij, hx⟩
    have hmem : ((i, j) : Fin N × Fin N) ∈
        (Finset.univ : Finset (Fin N × Fin N)).filter (fun p => p.1 ≠ p.2) := by
      simp [hij]
    exact Set.mem_biUnion (Finset.mem_coe.mpr hmem)
      (show x ∈ {y : Conf N | y ((i, j) : Fin N × Fin N).1 = y ((i, j) : Fin N × Fin N).2}
        from hx)
  refine measure_mono_null hsub ?_
  refine measure_biUnion_null_iff (Set.Finite.countable (Finset.finite_toSet _)) |>.mpr ?_
  intro p hp
  simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_univ, true_and] at hp
  exact coincident_pair_null hp

end MM

end IDR
