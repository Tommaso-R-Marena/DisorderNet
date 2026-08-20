/-
# Part LXXXVI  A cryo-EM map is an occupancy, not a structure

Single-particle reconstruction averages many copies of the molecule.  What the reconstruction
returns at a voxel is therefore not "the atom is here" but "this fraction of the particles put the
atom here": the map is the *push-forward of the conformational ensemble onto one-atom occupancies*.
This file makes that statement precise on a finite voxel grid and draws the consequences that
matter for modelling a disordered region.

* `dens` — the map of atom `a`: `dens E a x` is the population of conformations placing `a` in
  voxel `x`.  `dens_nonneg`, `sum_dens_eq_one`, `dens_le_one`: it is a probability distribution
  over voxels, and no value in a map can exceed the value a single rigid structure would give.
* `one_le_card_support_mul_peak`, `card_support_ge_inv_peak` — **heterogeneity is readable off the
  map**: if no voxel of the map exceeds `p`, the atom occupies at least `1/p` distinct voxels.  A
  weak map is a count of conformations, not a poorly resolved single conformation.
* `card_above_mul_threshold_le_one`, `card_above_le_inv_threshold` — the dual statement for the
  contour level a map is displayed at: at most `1/t` voxels per atom can survive thresholding at
  `t`.
* `uniformSpread_dens`, `uniform_spread_invisible`, `invisible_mass` — hence the disappearance of
  disordered regions from maps is a theorem, not an artefact: an atom spread uniformly over more
  than `1/t` voxels has *no* voxel above the contour level, so the displayed map contains none of
  it, while the population it carries is the whole of it.  "No density" is a population statement.
* `expect_from_map`, `coordMean_from_map`, `expect_eq_of_dens_eq` — the positive half: every
  single-atom average is a linear read-out of the map and is therefore determined by it.  Mean
  position and positional variance are recoverable exactly.
* `map_blind_to_correlation` — and nothing more is.  Two explicit two-residue ensembles have
  *identical* maps for both atoms while one has the residues always in contact and the other never.
  Joint conformational information is destroyed by the average, so a map cannot referee between
  ensembles that agree residue by residue.
* `sum_classDens` — three-dimensional classification does not repair this by itself: for *any*
  partition of the particles into classes, the class maps sum back to the total map.  Reproducing
  the consensus map is therefore no evidence that a classification is correct.
* `resid_pos_of_injective` — and a `K`-class reconstruction remains a `K`-point model: if the
  particles realise `n > K` distinct positions of an atom, every assignment of one structure per
  class leaves a strictly positive mean squared error, however the classes are chosen.

Design consequence: the deliverable a map supports is a per-atom occupancy distribution together
with the single-atom averages computed from it, and a statement of how many conformations the peak
height forces; a set of coordinates, or a class count chosen for convenience, claims more than the
data contain.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

open Finset

namespace Cryo

/-- A conformational ensemble as seen by a single-particle experiment: `n` conformations with
populations `w`, each placing every atom `a` in a voxel `pos j a`. -/
structure Ens (n : ℕ) (A X : Type*) where
  /-- Population of conformation `j`. -/
  w : Fin n → ℝ
  /-- Populations are nonnegative. -/
  hw : ∀ j, 0 ≤ w j
  /-- Populations sum to one. -/
  hsum : ∑ j, w j = 1
  /-- Voxel occupied by atom `a` in conformation `j`. -/
  pos : Fin n → A → X

variable {n : ℕ} {A X : Type*}

/-- The reconstructed map of atom `a`: the population of conformations placing `a` at `x`. -/
def dens [DecidableEq X] (E : Ens n A X) (a : A) (x : X) : ℝ :=
  ∑ j, if E.pos j a = x then E.w j else 0

/-! ## The map is an occupancy distribution -/

theorem dens_nonneg [DecidableEq X] (E : Ens n A X) (a : A) (x : X) : 0 ≤ dens E a x := by
  refine sum_nonneg fun j _ => ?_
  by_cases h : E.pos j a = x <;> simp [h, E.hw j]

theorem sum_dens_eq_one [Fintype X] [DecidableEq X] (E : Ens n A X) (a : A) :
    ∑ x, dens E a x = 1 := by
  simp only [dens]
  rw [Finset.sum_comm]
  simpa using E.hsum

theorem dens_le_one [Fintype X] [DecidableEq X] (E : Ens n A X) (a : A) (x : X) :
    dens E a x ≤ 1 := by
  have h := sum_dens_eq_one E a
  have : dens E a x ≤ ∑ y, dens E a y :=
    Finset.single_le_sum (f := fun y => dens E a y) (fun y _ => dens_nonneg E a y)
      (mem_univ x)
  simpa [h] using this

open scoped Classical in
/-- The set of voxels the atom actually visits. -/
noncomputable def support [Fintype X] [DecidableEq X] (E : Ens n A X) (a : A) : Finset X :=
  univ.filter fun x => dens E a x ≠ 0

theorem sum_dens_support [Fintype X] [DecidableEq X] (E : Ens n A X) (a : A) :
    ∑ x ∈ support E a, dens E a x = 1 := by
  rw [← sum_dens_eq_one E a]
  refine Finset.sum_subset (Finset.subset_univ _) ?_
  intro x _ hx
  simpa [support] using hx

/-! ## Peak height counts conformations -/

/-- **The peak height is a conformation count.**  If no voxel of the map of atom `a` rises above
`p`, then the atom occupies at least `1/p` distinct voxels. -/
theorem one_le_card_support_mul_peak [Fintype X] [DecidableEq X] (E : Ens n A X) (a : A)
    {p : ℝ} (hp : ∀ x, dens E a x ≤ p) : 1 ≤ (support E a).card * p := by
  have h1 : ∑ x ∈ support E a, dens E a x ≤ ∑ _x ∈ support E a, p :=
    Finset.sum_le_sum fun x _ => hp x
  rw [sum_dens_support] at h1
  simpa [mul_comm] using h1

theorem card_support_ge_inv_peak [Fintype X] [DecidableEq X] (E : Ens n A X) (a : A)
    {p : ℝ} (hp0 : 0 < p) (hp : ∀ x, dens E a x ≤ p) :
    1 / p ≤ (support E a).card := by
  rw [div_le_iff₀ hp0]
  simpa using one_le_card_support_mul_peak E a hp

/-- **The contour level bounds what can be displayed.**  At most `1/t` voxels of a single atom's
map can exceed the level `t`. -/
theorem card_above_mul_threshold_le_one [Fintype X] [DecidableEq X] (E : Ens n A X) (a : A)
    (t : ℝ) : ((univ.filter fun x => t ≤ dens E a x).card : ℝ) * t ≤ 1 := by
  set S : Finset X := univ.filter fun x => t ≤ dens E a x with hS
  have h1 : ∑ _x ∈ S, t ≤ ∑ x ∈ S, dens E a x := by
    refine Finset.sum_le_sum fun x hx => ?_
    simpa [hS] using (mem_filter.mp hx).2
  have h2 : ∑ x ∈ S, dens E a x ≤ ∑ x, dens E a x :=
    Finset.sum_le_sum_of_subset_of_nonneg (subset_univ _)
      (fun x _ _ => dens_nonneg E a x)
  rw [sum_dens_eq_one] at h2
  have := h1.trans h2
  simpa [mul_comm] using this

theorem card_above_le_inv_threshold [Fintype X] [DecidableEq X] (E : Ens n A X) (a : A)
    {t : ℝ} (ht : 0 < t) :
    ((univ.filter fun x => t ≤ dens E a x).card : ℝ) ≤ 1 / t := by
  rw [le_div_iff₀ ht]
  exact card_above_mul_threshold_le_one E a t

/-! ## Why disordered regions vanish from maps -/

/-- An atom spread uniformly over `m` distinct voxels: conformation `j` places it at voxel `j`. -/
noncomputable def uniformSpread (m : ℕ) (hm : 0 < m) : Ens m Unit (Fin m) where
  w := fun _ => 1 / m
  hw := fun _ => by positivity
  hsum := by
    have hm' : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hm.ne'
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
  pos := fun j _ => j

@[simp] theorem uniformSpread_dens (m : ℕ) (hm : 0 < m) (x : Fin m) :
    dens (uniformSpread m hm) () x = 1 / m := by
  simp [dens, uniformSpread]

/-- **The disappearance of a disordered region is a theorem.**  If the atom is spread over more
than `1/t` voxels, no voxel of its map reaches the contour level `t`, so the displayed map contains
none of it. -/
theorem uniform_spread_invisible (m : ℕ) (hm : 0 < m) {t : ℝ} (ht : 1 / m < t) :
    ∀ x, dens (uniformSpread m hm) () x < t := by
  intro x
  simpa using ht

/-- …and yet the whole population is there: the voxels below the contour level carry all of it. -/
theorem invisible_mass (m : ℕ) (hm : 0 < m) {t : ℝ} (ht : 1 / m < t) :
    ∑ x ∈ univ.filter fun x => dens (uniformSpread m hm) () x < t,
      dens (uniformSpread m hm) () x = 1 := by
  have hfilter : (univ.filter fun x => dens (uniformSpread m hm) () x < t) = univ :=
    Finset.filter_true_of_mem fun x _ => uniform_spread_invisible m hm ht x
  rw [hfilter, sum_dens_eq_one]

/-! ## What the map determines: single-atom averages -/

/-- **Every single-atom average is a linear read-out of the map.** -/
theorem expect_from_map [Fintype X] [DecidableEq X] (E : Ens n A X) (a : A) (f : X → ℝ) :
    ∑ j, E.w j * f (E.pos j a) = ∑ x, f x * dens E a x := by
  have : ∀ x : X, f x * dens E a x = ∑ j, if E.pos j a = x then E.w j * f (E.pos j a) else 0 := by
    intro x
    rw [dens, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    by_cases h : E.pos j a = x <;> simp [h, mul_comm]
  rw [Finset.sum_congr rfl fun x _ => this x, Finset.sum_comm]
  simp

/-- The mean position of an atom along a coordinate `c` is computed from the map. -/
theorem coordMean_from_map [Fintype X] [DecidableEq X] (E : Ens n A X) (a : A) (c : X → ℝ) :
    ∑ j, E.w j * c (E.pos j a) = ∑ x, c x * dens E a x :=
  expect_from_map E a c

/-- Two ensembles with the same map agree on every single-atom average. -/
theorem expect_eq_of_dens_eq [Fintype X] [DecidableEq X] {n' : ℕ} (E : Ens n A X)
    (E' : Ens n' A X) (a : A) (f : X → ℝ) (h : ∀ x, dens E a x = dens E' a x) :
    ∑ j, E.w j * f (E.pos j a) = ∑ j, E'.w j * f (E'.pos j a) := by
  rw [expect_from_map, expect_from_map]
  exact Finset.sum_congr rfl fun x _ => by rw [h x]

/-! ## What the map does not determine: correlations -/

/-- Two residues always in the same voxel. -/
noncomputable def contactEns : Ens 2 (Fin 2) (Fin 2) where
  w := fun _ => 1 / 2
  hw := fun _ => by norm_num
  hsum := by norm_num
  pos := fun j _ => j

/-- The same two residues, never in the same voxel. -/
noncomputable def antiEns : Ens 2 (Fin 2) (Fin 2) where
  w := fun _ => 1 / 2
  hw := fun _ => by norm_num
  hsum := by norm_num
  pos := fun j a => j + a

/-- Population of conformations in which the two residues share a voxel. -/
def contactFreq (E : Ens n (Fin 2) (Fin 2)) : ℝ :=
  ∑ j, if E.pos j 0 = E.pos j 1 then E.w j else 0

/-- **A map cannot see correlations.**  Two ensembles with identical maps for *both* residues, one
of which has the residues always in contact and the other never.  Averaging over particles destroys
the joint distribution, so no reconstruction, at any resolution or signal-to-noise, can referee
between ensembles that agree residue by residue. -/
theorem map_blind_to_correlation :
    (∀ (a x : Fin 2), dens contactEns a x = dens antiEns a x) ∧
      contactFreq contactEns = 1 ∧ contactFreq antiEns = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · intro a x
    fin_cases a <;> fin_cases x <;>
      norm_num [dens, contactEns, antiEns, Fin.sum_univ_two]
    all_goals decide
  · norm_num [contactFreq, contactEns, Fin.sum_univ_two]
  · norm_num [contactFreq, antiEns, Fin.sum_univ_two]

/-! ## Classification does not repair it -/

/-- The (unnormalised) map of class `k` under a classification `kap` of the particles. -/
def classDens [DecidableEq X] {K : ℕ} (E : Ens n A X) (kap : Fin n → Fin K) (k : Fin K)
    (a : A) (x : X) : ℝ :=
  ∑ j ∈ univ.filter fun j => kap j = k, if E.pos j a = x then E.w j else 0

/-- **Any classification reproduces the consensus map.**  The class maps sum to the total map for
*every* partition of the particles, so agreement with the consensus map is no evidence that a
classification is correct. -/
theorem sum_classDens [DecidableEq X] {K : ℕ} (E : Ens n A X) (kap : Fin n → Fin K) (a : A)
    (x : X) : ∑ k, classDens E kap k a x = dens E a x := by
  simp only [dens, classDens]
  exact Finset.sum_fiberwise (s := univ) (g := kap)
    (f := fun j => if E.pos j a = x then E.w j else 0)

/-- The mean squared error of a `K`-structure model: class `k` is represented by the single
coordinate value `mu k`. -/
def resid (E : Ens n A X) (a : A) (c : X → ℝ) {K : ℕ} (kap : Fin n → Fin K) (mu : Fin K → ℝ) : ℝ :=
  ∑ j, E.w j * (c (E.pos j a) - mu (kap j)) ^ 2

/-- **A `K`-class reconstruction is a `K`-point model.**  If the particles realise `n > K` distinct
coordinate values of the atom, then for every classification into `K` classes and every choice of
one structure per class the mean squared error is strictly positive. -/
theorem resid_pos_of_injective (E : Ens n A X) (a : A) (c : X → ℝ)
    (hpos : ∀ j, 0 < E.w j) (hinj : Function.Injective fun j => c (E.pos j a))
    {K : ℕ} (hK : K < n) (kap : Fin n → Fin K) (mu : Fin K → ℝ) :
    0 < resid E a c kap mu := by
  obtain ⟨j₁, -, j₂, -, hne, hkap⟩ :=
    Finset.exists_ne_map_eq_of_card_lt_of_maps_to (s := (univ : Finset (Fin n)))
      (t := (univ : Finset (Fin K))) (by simpa using hK) (fun j _ => mem_univ (kap j))
  have hcne : c (E.pos j₁ a) ≠ c (E.pos j₂ a) := fun h => hne (hinj h)
  have hone : c (E.pos j₁ a) - mu (kap j₁) ≠ 0 ∨ c (E.pos j₂ a) - mu (kap j₂) ≠ 0 := by
    by_contra hcon
    push_neg at hcon
    obtain ⟨h1, h2⟩ := hcon
    apply hcne
    have e1 : c (E.pos j₁ a) = mu (kap j₁) := by linarith [sub_eq_zero.mp h1]
    have e2 : c (E.pos j₂ a) = mu (kap j₂) := by linarith [sub_eq_zero.mp h2]
    rw [e1, e2, hkap]
  have hsq : ∀ (j : Fin n), c (E.pos j a) - mu (kap j) ≠ 0 →
      0 < E.w j * (c (E.pos j a) - mu (kap j)) ^ 2 := by
    intro j hj
    exact mul_pos (hpos j) (lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hj)))
  refine Finset.sum_pos' (fun j _ => mul_nonneg (E.hw j) (sq_nonneg _)) ?_
  rcases hone with h | h
  · exact ⟨j₁, mem_univ _, hsq j₁ h⟩
  · exact ⟨j₂, mem_univ _, hsq j₂ h⟩

end Cryo

end IDR
