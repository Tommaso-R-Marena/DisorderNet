/-
# Part LI.1  Electronic polarisability: the self-consistent field, and its three-body term

The molecular model of Part XXIV is *non-polarisable*: every atom carries a fixed charge, so
the energy is a sum of terms attached to pairs of atoms.  `RequestProject.ManyBody` prices the
solvent-averaged non-additivity; this file prices the electronic one, which is the assumption a
reviewer names first, and it does so exactly rather than by a caricature.

The physical model is the standard point-polarisable one.  Site `i` carries an isotropic
polarisability `a i` and feels an external field `E i` plus the field of every other induced
dipole through a coupling matrix `T`.  The induced dipoles are then *not* given by a formula;
they are the solution of the self-consistent (linear) system

  `mu i = a i * (E i + ∑ j, T i j * mu j)`   (`SelfConsistent`),

and the interaction energy is `-(1/2) ∑ i, mu i * E i` (`energy`).

* `selfConsistent_unique` -- under the standard damping condition
  `|a i| * ∑ j |T i j| ≤ c < 1` (`Damped`: the polarisation catastrophe is excluded) the system
  has **at most one** solution, and
* `selfConsistent_exists` -- exactly one: injectivity of `I - aT` on a finite-dimensional space
  gives surjectivity, so the induced dipoles exist for every external field.  Adding
  polarisability is therefore a well-posed repair, not a hand-wave.
* `clusterDipole_selfConsistent`, `energy_uniform_eq` -- the symmetric cluster of `m` equally
  polarisable sites in a uniform field is solved in closed form:
  `mu = a / (1 - (m-1) a t)` and `U = -(m/2) * mu`.
* `threeBody_eq` -- **the exact three-body term.**  The inclusion--exclusion residue of the
  three-site cluster after every one- and two-body term has been fitted is

  `U₃ - 3U₂ + 3U₁ = -3 a³ t² / ((1 - 2at)(1 - at))`,

  strictly negative (cooperative) whenever `a t ≠ 0` in the damped regime
  (`threeBody_neg`), and `O(t²)`: it is second order in the coupling, which is why a
  pairwise-fitted force field can be accurate and still structurally wrong.
* `polarisable_not_pairwise_additive` -- consequently **no** assignment of one-body and
  two-body energies whatsoever reproduces the cluster energies of a polarisable model: the
  obstruction is representability, not parameterisation.  The witness is `a = 1`, `t = 1/4`,
  where the residue is exactly `-1/2`.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace Polarisability

open Finset

variable {n : ℕ}

/-- **The self-consistent field condition.**  `mu i` is the dipole induced at site `i` by the
external field `E i` together with the fields of all the other induced dipoles, transmitted by
the coupling matrix `T`. -/
def SelfConsistent (a : Fin n → ℝ) (T : Fin n → Fin n → ℝ) (E mu : Fin n → ℝ) : Prop :=
  ∀ i, mu i = a i * (E i + ∑ j, T i j * mu j)

/-- **The damping (no polarisation catastrophe) condition.**  Each row of the response operator
`a·T` has `ℓ¹` norm at most `c < 1`. -/
def Damped (a : Fin n → ℝ) (T : Fin n → Fin n → ℝ) (c : ℝ) : Prop :=
  c < 1 ∧ ∀ i, |a i| * ∑ j, |T i j| ≤ c

/-- The polarisation energy of a set of induced dipoles in an external field. -/
noncomputable def energy (E mu : Fin n → ℝ) : ℝ := -(1 / 2) * ∑ i, mu i * E i

/-- **Uniqueness of the induced dipoles.**  A damped polarisable model has at most one
self-consistent solution for a given external field. -/
theorem selfConsistent_unique {a : Fin n → ℝ} {T : Fin n → Fin n → ℝ} {c : ℝ}
    (hd : Damped a T c) {E mu nu : Fin n → ℝ}
    (hmu : SelfConsistent a T E mu) (hnu : SelfConsistent a T E nu) : mu = nu := by
  obtain ⟨hc, hrow⟩ := hd
  rcases Nat.eq_zero_or_pos n with hn | hn
  · subst hn; funext i; exact absurd i.2 (by omega)
  have hne : (Finset.univ : Finset (Fin n)).Nonempty := by
    refine ⟨⟨0, hn⟩, Finset.mem_univ _⟩
  obtain ⟨i0, -, hi0⟩ := Finset.exists_max_image Finset.univ (fun i => |mu i - nu i|) hne
  set M := |mu i0 - nu i0| with hM
  have hMnn : 0 ≤ M := abs_nonneg _
  have key : M ≤ c * M := by
    have hdiff : mu i0 - nu i0 = a i0 * ∑ j, T i0 j * (mu j - nu j) := by
      have h1 := hmu i0
      have h2 := hnu i0
      have : ∑ j, T i0 j * (mu j - nu j)
          = (∑ j, T i0 j * mu j) - ∑ j, T i0 j * nu j := by
        rw [← Finset.sum_sub_distrib]; exact Finset.sum_congr rfl (fun j _ => by ring)
      rw [this, h1, h2]; ring
    have hbound : |∑ j, T i0 j * (mu j - nu j)| ≤ (∑ j, |T i0 j|) * M := by
      calc |∑ j, T i0 j * (mu j - nu j)| ≤ ∑ j, |T i0 j * (mu j - nu j)| :=
            Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ j, |T i0 j| * M := by
            refine Finset.sum_le_sum (fun j _ => ?_)
            rw [abs_mul]
            exact mul_le_mul_of_nonneg_left (hi0 j (Finset.mem_univ j)) (abs_nonneg _)
        _ = (∑ j, |T i0 j|) * M := by rw [Finset.sum_mul]
    calc M = |a i0| * |∑ j, T i0 j * (mu j - nu j)| := by rw [hM, hdiff, abs_mul]
      _ ≤ |a i0| * ((∑ j, |T i0 j|) * M) :=
          mul_le_mul_of_nonneg_left hbound (abs_nonneg _)
      _ = (|a i0| * ∑ j, |T i0 j|) * M := by ring
      _ ≤ c * M := mul_le_mul_of_nonneg_right (hrow i0) hMnn
  have hM0 : M = 0 := le_antisymm (by nlinarith) hMnn
  funext i
  have : |mu i - nu i| ≤ 0 := hM0 ▸ hi0 i (Finset.mem_univ i)
  have := abs_nonpos_iff.mp this
  linarith [sub_eq_zero.mp this]

/-- The linear operator `I - a·T` whose kernel controls the self-consistent system. -/
def scfMap (a : Fin n → ℝ) (T : Fin n → Fin n → ℝ) : (Fin n → ℝ) →ₗ[ℝ] (Fin n → ℝ) where
  toFun d := fun i => d i - a i * ∑ j, T i j * d j
  map_add' u v := by
    funext i
    simp only [Pi.add_apply]
    have : ∑ j, T i j * (u j + v j) = (∑ j, T i j * u j) + ∑ j, T i j * v j := by
      rw [← Finset.sum_add_distrib]; exact Finset.sum_congr rfl (fun j _ => by ring)
    rw [this]; ring
  map_smul' r v := by
    funext i
    simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
    have : ∑ j, T i j * (r * v j) = r * ∑ j, T i j * v j := by
      rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun j _ => by ring)
    rw [this]; ring

/-- **Existence of the induced dipoles.**  In a damped polarisable model the self-consistent
system has a solution for every external field. -/
theorem selfConsistent_exists {a : Fin n → ℝ} {T : Fin n → Fin n → ℝ} {c : ℝ}
    (hd : Damped a T c) (E : Fin n → ℝ) : ∃ mu : Fin n → ℝ, SelfConsistent a T E mu := by
  have hinj : Function.Injective (scfMap a T) := by
    rw [← LinearMap.ker_eq_bot]
    rw [Submodule.eq_bot_iff]
    intro d hd0
    have hd0' : ∀ i, d i - a i * ∑ j, T i j * d j = 0 := by
      intro i
      have : (scfMap a T) d = 0 := hd0
      exact congrFun this i
    have h1 : SelfConsistent a T (fun _ => 0) d := by
      intro i
      have := hd0' i
      simp only [zero_add]
      linarith
    have h2 : SelfConsistent a T (fun _ => 0) (fun _ => (0 : ℝ)) := by
      intro i; simp
    exact selfConsistent_unique hd h1 h2
  have hsurj : Function.Surjective (scfMap a T) :=
    LinearMap.injective_iff_surjective.mp hinj
  obtain ⟨mu, hmu⟩ := hsurj (fun i => a i * E i)
  refine ⟨mu, fun i => ?_⟩
  have := congrFun hmu i
  simp only [scfMap, LinearMap.coe_mk, AddHom.coe_mk] at this
  have h := this
  nlinarith [h]

/-! ### The symmetric cluster, solved in closed form -/

/-- Every site equally polarisable. -/
def uniformPol (m : ℕ) (a : ℝ) : Fin m → ℝ := fun _ => a

/-- Every pair of distinct sites equally coupled. -/
def uniformCoupling (m : ℕ) (t : ℝ) : Fin m → Fin m → ℝ := fun i j => if i = j then 0 else t

/-- A uniform external field of unit strength. -/
def uniformField (m : ℕ) : Fin m → ℝ := fun _ => 1

/-- The self-consistent dipole of each site of a symmetric `m`-site cluster. -/
noncomputable def clusterDipole (m : ℕ) (a t : ℝ) : ℝ := a / (1 - ((m : ℝ) - 1) * a * t)

/-- The polarisation energy of the symmetric `m`-site cluster. -/
noncomputable def clusterEnergy (m : ℕ) (a t : ℝ) : ℝ := -(m : ℝ) / 2 * clusterDipole m a t

/-- The row sum of the uniform coupling matrix is `(m-1) t`. -/
lemma sum_uniformCoupling (m : ℕ) (t : ℝ) (i : Fin m) (v : ℝ) :
    ∑ j, uniformCoupling m t i j * v = ((m : ℝ) - 1) * t * v := by
  classical
  have hstep : ∀ j : Fin m,
      uniformCoupling m t i j * v = t * v - (if i = j then t * v else 0) := by
    intro j; by_cases h : i = j <;> simp [uniformCoupling, h]
  rw [Finset.sum_congr rfl (fun j _ => hstep j), Finset.sum_sub_distrib, Finset.sum_const,
    Finset.sum_ite_eq]
  simp
  ring

/-- **The closed-form solution.**  The constant dipole `a / (1 - (m-1)at)` is self-consistent
for the symmetric cluster. -/
theorem clusterDipole_selfConsistent (m : ℕ) (a t : ℝ) (h : 1 - ((m : ℝ) - 1) * a * t ≠ 0) :
    SelfConsistent (uniformPol m a) (uniformCoupling m t) (uniformField m)
      (fun _ => clusterDipole m a t) := by
  intro i
  simp only [uniformPol, uniformField]
  rw [sum_uniformCoupling]
  simp only [clusterDipole]
  set D : ℝ := 1 - ((m : ℝ) - 1) * a * t with hDdef
  have hD : D ≠ 0 := h
  have hexp : ((m : ℝ) - 1) * a * t = 1 - D := by rw [hDdef]; ring
  field_simp
  nlinarith [hexp]

/-- The energy of the closed-form solution. -/
theorem energy_uniform_eq (m : ℕ) (a t : ℝ) :
    energy (uniformField m) (fun _ => clusterDipole m a t) = clusterEnergy m a t := by
  simp only [energy, uniformField, clusterEnergy, mul_one, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

/-! ### The three-body residue -/

/-- The inclusion--exclusion residue of the three-site cluster: what is left of its energy
after the best possible one-body and two-body terms have been subtracted. -/
noncomputable def threeBody (a t : ℝ) : ℝ :=
  clusterEnergy 3 a t - 3 * clusterEnergy 2 a t + 3 * clusterEnergy 1 a t

/-- **The exact three-body term of a polarisable cluster.** -/
theorem threeBody_eq (a t : ℝ) (h2 : 1 - 2 * a * t ≠ 0) (h1 : 1 - a * t ≠ 0) :
    threeBody a t = -3 * a ^ 3 * t ^ 2 / ((1 - 2 * a * t) * (1 - a * t)) := by
  simp only [threeBody, clusterEnergy, clusterDipole]
  norm_num
  field_simp
  ring

/-- **The three-body term is cooperative and second order in the coupling.**  In the damped
regime it is strictly negative whenever the coupling is nonzero. -/
theorem threeBody_neg {a t : ℝ} (ha : 0 < a) (ht : 0 < t) (hd : a * t < 1 / 2) :
    threeBody a t < 0 := by
  have h2 : 1 - 2 * a * t ≠ 0 := by nlinarith
  have h1 : 1 - a * t ≠ 0 := by nlinarith
  rw [threeBody_eq a t h2 h1]
  have hp2 : 0 < 1 - 2 * a * t := by nlinarith
  have hp1 : 0 < 1 - a * t := by nlinarith
  have hnum : 0 < 3 * a ^ 3 * t ^ 2 := by positivity
  have : 0 < (1 - 2 * a * t) * (1 - a * t) := mul_pos hp2 hp1
  rw [div_neg_iff]
  right
  constructor <;> nlinarith

/-- The witness used below: at `a = 1`, `t = 1/4` the residue is exactly `-1/2`. -/
theorem threeBody_witness : threeBody 1 (1 / 4) = -(1 / 2) := by
  rw [threeBody_eq 1 (1/4) (by norm_num) (by norm_num)]
  norm_num

/-! ### No pairwise decomposition exists -/

/-- The energy that a polarisable model assigns to the sub-cluster carried by a set `S` of the
three sites: by symmetry it depends only on how many sites `S` contains. -/
noncomputable def clusterEnergyOn (a t : ℝ) (S : Finset (Fin 3)) : ℝ :=
  clusterEnergy S.card a t

/-- A one-body plus two-body ("pairwise additive") energy model on three sites. -/
def IsPairEnergy (U : Finset (Fin 3) → ℝ) : Prop :=
  ∃ f : Fin 3 → ℝ, ∃ g : Fin 3 → Fin 3 → ℝ,
    ∀ S : Finset (Fin 3), U S = (∑ i ∈ S, f i) + ∑ i ∈ S, ∑ j ∈ S, g i j

/-- Any pairwise-additive energy has vanishing inclusion--exclusion residue on the triple. -/
theorem residue_eq_zero_of_pairEnergy {U : Finset (Fin 3) → ℝ} (h : IsPairEnergy U) :
    U {0, 1, 2} - (U {0, 1} + U {0, 2} + U {1, 2}) + (U {0} + U {1} + U {2}) = 0 := by
  obtain ⟨f, g, hU⟩ := h
  rw [hU {0,1,2}, hU {0,1}, hU {0,2}, hU {1,2}, hU {0}, hU {1}, hU {2}]
  rw [show ({0,1,2} : Finset (Fin 3)) = Finset.univ by decide]
  simp only [Fin.sum_univ_three, Finset.sum_pair (by decide : (0:Fin 3) ≠ 1),
    Finset.sum_pair (by decide : (0:Fin 3) ≠ 2), Finset.sum_pair (by decide : (1:Fin 3) ≠ 2),
    Finset.sum_singleton]
  ring

/-- **A polarisable model is not a pairwise model.**  No assignment of one-body and two-body
energies reproduces the cluster energies of the polarisable model, for any nonzero coupling in
the damped regime: the residue is `-3a³t²/((1-2at)(1-at)) ≠ 0`. -/
theorem polarisable_not_pairwise_additive {a t : ℝ} (ha : 0 < a) (ht : 0 < t)
    (hd : a * t < 1 / 2) : ¬ IsPairEnergy (clusterEnergyOn a t) := by
  intro h
  have hz := residue_eq_zero_of_pairEnergy h
  have hcards : clusterEnergyOn a t {0, 1, 2} = clusterEnergy 3 a t ∧
      clusterEnergyOn a t {0, 1} = clusterEnergy 2 a t ∧
      clusterEnergyOn a t {0, 2} = clusterEnergy 2 a t ∧
      clusterEnergyOn a t {1, 2} = clusterEnergy 2 a t ∧
      clusterEnergyOn a t {0} = clusterEnergy 1 a t ∧
      clusterEnergyOn a t {1} = clusterEnergy 1 a t ∧
      clusterEnergyOn a t {2} = clusterEnergy 1 a t := by
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      simp [clusterEnergyOn, show ({0,1,2} : Finset (Fin 3)).card = 3 by decide,
        show ({0,1} : Finset (Fin 3)).card = 2 by decide,
        show ({0,2} : Finset (Fin 3)).card = 2 by decide,
        show ({1,2} : Finset (Fin 3)).card = 2 by decide]
  obtain ⟨h3, h12, h02, h13, h1, h2', h3'⟩ := hcards
  rw [h3, h12, h02, h13, h1, h2', h3'] at hz
  have : threeBody a t = 0 := by simp only [threeBody]; linarith
  exact absurd this (ne_of_lt (threeBody_neg ha ht hd))

end Polarisability

end IDR
