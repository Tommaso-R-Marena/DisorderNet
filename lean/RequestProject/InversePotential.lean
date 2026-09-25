/-
# Part LXIV.1  Structure-based coarse-graining: the inverse problem

Part LX shows that integrating out the solvent produces a potential of mean force which is
exact, not pairwise, and temperature dependent.  This file treats the inverse of that
operation, which is what coarse-grained models of disordered regions are actually built by:
*fit* a potential so that the model reproduces a measured structural statistic -- a radial
distribution function, a set of contact frequencies, a distance histogram -- and then use the
fitted potential elsewhere.  Iterative Boltzmann inversion, force matching and relative-entropy
coarse-graining are all instances.  Three questions are answered exactly.

Setting.  A finite configuration space `Fin N`, a finite family of structural features
`n : Fin m → Fin N → ℝ` (read: the pair counts in each distance bin), a parameter vector
`theta` giving the energy `E(x) = Σ_a theta a · n a x`, and the Boltzmann distribution `gibbs`
at inverse temperature `b`.

* `gibbs_pos`, `gibbs_sum_one` -- the model is a probability distribution on the library.
* `gibbs_unique_of_meanFeature_eq` -- **the inverse problem is well posed** (Henderson's
  uniqueness theorem, in the discrete setting): two parameter vectors whose Boltzmann
  distributions have the *same* mean features have the *same* Boltzmann distribution.  The proof
  is the symmetrised relative entropy: `KL(p‖p') + KL(p'‖p) = b·⟨theta' − theta, ⟨n⟩_p − ⟨n⟩_p'⟩`,
  which the hypothesis makes zero, and Gibbs' inequality then forces `p = p'`.  Matching the
  structure determines the model, so a fitted coarse-grained potential is not arbitrary.
* `pair_potentials_blind_to_three_body` -- **but the model it determines is blind to
  higher-order structure.**  On three spins, the parity ensemble (uniform on the four
  configurations with `s₁s₂s₃ = +1`) has *all three* pair correlations equal to zero, exactly
  like the uniform ensemble, and triple correlation `1`.  By the uniqueness theorem, every
  pair-potential model matching those pair correlations *is* the uniform ensemble, whose triple
  correlation is `0`.  Fitting the pair structure exactly therefore gets the three-body
  structure maximally wrong, and no choice of pair potential repairs it.
* `inverse_potential_temperature_dependent` -- **and the fitted potential is a free energy.**
  In the smallest two-state model, the potential that reproduces the target feature value `1/3`
  at inverse temperature `1` is `log 2`, and the one that reproduces the *same* target at
  inverse temperature `2` is `(log 2)/2`, not `log 2`.  The fit is a state-point statement:
  transferring the potential to another temperature changes the structure it reproduces.

Together with Part LX, these bracket the coarse-graining operation from both sides: the forward
map (integrate out) leaves an object that is not a pairwise, transferable potential, and the
inverse map (fit to structure) returns a unique but equally non-transferable one, blind by
construction to everything beyond the statistics it was fitted to.
-/
import Mathlib
import RequestProject.DisorderedRegions

set_option autoImplicit false

namespace IDR

namespace Inverse

open Finset

variable {N m : ℕ}

/-- The energy of a configuration under a linear (e.g. pairwise) potential model. -/
noncomputable def energy (n : Fin m → Fin N → ℝ) (theta : Fin m → ℝ) (x : Fin N) : ℝ :=
  ∑ a, theta a * n a x

/-- The partition function. -/
noncomputable def part (n : Fin m → Fin N → ℝ) (theta : Fin m → ℝ) (b : ℝ) : ℝ :=
  ∑ x, Real.exp (-b * energy n theta x)

/-- The Boltzmann distribution of the model. -/
noncomputable def gibbs (n : Fin m → Fin N → ℝ) (theta : Fin m → ℝ) (b : ℝ) (x : Fin N) : ℝ :=
  Real.exp (-b * energy n theta x) / part n theta b

/-- The mean value of feature `a` in a distribution: the model's structural output. -/
noncomputable def meanFeature (n : Fin m → Fin N → ℝ) (p : Fin N → ℝ) (a : Fin m) : ℝ :=
  ∑ x, p x * n a x

lemma part_pos [NeZero N] (n : Fin m → Fin N → ℝ) (theta : Fin m → ℝ) (b : ℝ) :
    0 < part n theta b := by
  refine Finset.sum_pos (fun x _ => Real.exp_pos _) ?_
  exact Finset.univ_nonempty

lemma gibbs_pos [NeZero N] (n : Fin m → Fin N → ℝ) (theta : Fin m → ℝ) (b : ℝ) (x : Fin N) :
    0 < gibbs n theta b x :=
  div_pos (Real.exp_pos _) (part_pos n theta b)

lemma gibbs_sum_one [NeZero N] (n : Fin m → Fin N → ℝ) (theta : Fin m → ℝ) (b : ℝ) :
    ∑ x, gibbs n theta b x = 1 := by
  simp only [gibbs, ← Finset.sum_div]
  exact div_self (ne_of_gt (part_pos n theta b))

/-- The mean energy is the parameter vector paired with the mean features. -/
lemma meanEnergy_eq (n : Fin m → Fin N → ℝ) (theta : Fin m → ℝ) (p : Fin N → ℝ) :
    ∑ x, p x * energy n theta x = ∑ a, theta a * meanFeature n p a := by
  simp only [energy, meanFeature, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun x _ => by ring

/-! ## Henderson uniqueness -/

/-- **The inverse problem is well posed.**  Two linear potentials whose Boltzmann distributions
have the same mean features have the same Boltzmann distribution. -/
theorem gibbs_unique_of_meanFeature_eq [NeZero N] (n : Fin m → Fin N → ℝ)
    (theta theta' : Fin m → ℝ) (b : ℝ)
    (h : ∀ a, meanFeature n (gibbs n theta b) a = meanFeature n (gibbs n theta' b) a) :
    gibbs n theta b = gibbs n theta' b := by
  have hkl : ∀ (u v : Fin m → ℝ), klDiv (gibbs n u b) (gibbs n v b)
      = (-b * (∑ a, u a * meanFeature n (gibbs n u b) a)
          + b * (∑ a, v a * meanFeature n (gibbs n u b) a))
        + Real.log (part n v b / part n u b) := by
    intro u v
    have hZu : part n u b ≠ 0 := ne_of_gt (part_pos n u b)
    have hZv : part n v b ≠ 0 := ne_of_gt (part_pos n v b)
    have hlog' : ∀ x, Real.log (gibbs n u b x / gibbs n v b x)
        = (-b * energy n u x + b * energy n v x)
          + Real.log (part n v b / part n u b) := by
      intro x
      have hA : Real.exp (-b * energy n u x) ≠ 0 := ne_of_gt (Real.exp_pos _)
      have hC : Real.exp (-b * energy n v x) ≠ 0 := ne_of_gt (Real.exp_pos _)
      have hx : gibbs n u b x / gibbs n v b x
          = (Real.exp (-b * energy n u x) / Real.exp (-b * energy n v x))
            * (part n v b / part n u b) := by
        simp only [gibbs]
        field_simp
      rw [hx, ← Real.exp_sub,
        show -b * energy n u x - -b * energy n v x
          = -b * energy n u x + b * energy n v x from by ring,
        Real.log_mul (ne_of_gt (Real.exp_pos _))
          (ne_of_gt (div_pos (part_pos n v b) (part_pos n u b))), Real.log_exp]
    rw [klDiv, Finset.sum_congr rfl (fun x _ => by rw [hlog' x])]
    have expand : ∀ x : Fin N, gibbs n u b x
        * ((-b * energy n u x + b * energy n v x) + Real.log (part n v b / part n u b))
        = (-b * (gibbs n u b x * energy n u x) + b * (gibbs n u b x * energy n v x))
          + Real.log (part n v b / part n u b) * gibbs n u b x := by
      intro x; ring
    rw [Finset.sum_congr rfl (fun x _ => expand x), Finset.sum_add_distrib, ← Finset.mul_sum,
      gibbs_sum_one, mul_one, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
      meanEnergy_eq, meanEnergy_eq]
  have hMF : ∀ (u : Fin m → ℝ),
      (∑ a, u a * meanFeature n (gibbs n theta b) a)
        = ∑ a, u a * meanFeature n (gibbs n theta' b) a :=
    fun u => Finset.sum_congr rfl fun a _ => by rw [h a]
  have hZ : 0 < part n theta b := part_pos n theta b
  have hZ' : 0 < part n theta' b := part_pos n theta' b
  have hlogs : Real.log (part n theta' b / part n theta b)
      + Real.log (part n theta b / part n theta' b) = 0 := by
    rw [← Real.log_mul (ne_of_gt (div_pos hZ' hZ)) (ne_of_gt (div_pos hZ hZ')),
      div_mul_div_comm, mul_comm (part n theta' b) (part n theta b), div_self
        (ne_of_gt (mul_pos hZ hZ')), Real.log_one]
  have hsum : klDiv (gibbs n theta b) (gibbs n theta' b)
      + klDiv (gibbs n theta' b) (gibbs n theta b) = 0 := by
    rw [hkl theta theta', hkl theta' theta, ← hMF theta, ← hMF theta']
    linarith [hlogs]
  have h1 : 0 ≤ klDiv (gibbs n theta b) (gibbs n theta' b) :=
    klDiv_nonneg (fun x => (gibbs_pos n theta b x).le) (gibbs_sum_one n theta b)
      (fun x => gibbs_pos n theta' b x) (gibbs_sum_one n theta' b)
  have h2 : 0 ≤ klDiv (gibbs n theta' b) (gibbs n theta b) :=
    klDiv_nonneg (fun x => (gibbs_pos n theta' b x).le) (gibbs_sum_one n theta' b)
      (fun x => gibbs_pos n theta b x) (gibbs_sum_one n theta b)
  have hzero : klDiv (gibbs n theta b) (gibbs n theta' b) = 0 := by linarith
  exact (klDiv_eq_zero_iff (fun x => (gibbs_pos n theta b x).le) (gibbs_sum_one n theta b)
    (fun x => gibbs_pos n theta' b x) (gibbs_sum_one n theta' b)).mp hzero

/-- At zero potential the model is the uniform distribution. -/
lemma gibbs_zero [NeZero N] (n : Fin m → Fin N → ℝ) (b : ℝ) (x : Fin N) :
    gibbs n (fun _ => 0) b x = 1 / (N : ℝ) := by
  have he : ∀ y : Fin N, Real.exp (-b * energy n (fun _ => 0) y) = 1 := by
    intro y
    simp [energy]
  rw [gibbs, part, he x, Finset.sum_congr rfl (fun y _ => he y)]
  simp

/-! ## Three spins: the pair structure does not carry the triple structure -/

/-- The three spins of a configuration, read off the binary digits of its index. -/
def spin (i : ℕ) (x : Fin 8) : ℝ := if (x.val / 2 ^ i) % 2 = 0 then 1 else -1

/-- The three pair correlations `s₁s₂`, `s₁s₃`, `s₂s₃` on the eight spin configurations. -/
def pairFeat (a : Fin 3) (x : Fin 8) : ℝ :=
  if a.val = 0 then spin 0 x * spin 1 x
  else if a.val = 1 then spin 0 x * spin 2 x
  else spin 1 x * spin 2 x

/-- The triple correlation `s₁s₂s₃`. -/
def tripleFeat (x : Fin 8) : ℝ := spin 0 x * spin 1 x * spin 2 x

/-- The parity ensemble: uniform on the four configurations with `s₁s₂s₃ = +1`. -/
noncomputable def parityEns : Fin 8 → ℝ := fun x => if tripleFeat x = 1 then 1/4 else 0

lemma parityEns_sum : ∑ x, parityEns x = 1 := by
  norm_num [parityEns, tripleFeat, spin, Fin.sum_univ_eight]

lemma parityEns_pairFeat (a : Fin 3) : meanFeature pairFeat parityEns a = 0 := by
  fin_cases a <;>
    norm_num [meanFeature, parityEns, tripleFeat, pairFeat, spin, Fin.sum_univ_eight]

lemma parityEns_tripleFeat : ∑ x, parityEns x * tripleFeat x = 1 := by
  norm_num [parityEns, tripleFeat, spin, Fin.sum_univ_eight]

lemma uniform_pairFeat (a : Fin 3) :
    meanFeature pairFeat (fun _ : Fin 8 => (1 : ℝ) / 8) a = 0 := by
  fin_cases a <;> norm_num [meanFeature, pairFeat, spin, Fin.sum_univ_eight]

lemma uniform_tripleFeat : ∑ x : Fin 8, ((1 : ℝ) / 8) * tripleFeat x = 0 := by
  norm_num [tripleFeat, spin, Fin.sum_univ_eight]

/-- The zero potential does match the pair structure of the parity ensemble, so the hypothesis
of the next theorem is satisfiable and the statement is not vacuous. -/
theorem zero_matches_pair_structure (b : ℝ) (a : Fin 3) :
    meanFeature pairFeat (gibbs pairFeat (fun _ => 0) b) a = meanFeature pairFeat parityEns a := by
  have huni : gibbs pairFeat (fun _ => 0) b = fun _ : Fin 8 => (1 : ℝ) / 8 := by
    funext x
    rw [gibbs_zero]
    norm_num
  rw [huni, uniform_pairFeat a, parityEns_pairFeat a]

/-- **A pair potential that reproduces the pair structure gets the three-body structure
maximally wrong.**  Every pairwise Boltzmann model whose pair correlations match those of the
parity ensemble has triple correlation `0`, while the parity ensemble has triple correlation
`1`. -/
theorem pair_potentials_blind_to_three_body (b : ℝ) (theta : Fin 3 → ℝ)
    (h : ∀ a, meanFeature pairFeat (gibbs pairFeat theta b) a
      = meanFeature pairFeat parityEns a) :
    (∑ x, gibbs pairFeat theta b x * tripleFeat x) = 0 ∧
      (∑ x, parityEns x * tripleFeat x) = 1 := by
  have hzero : ∀ a, meanFeature pairFeat (gibbs pairFeat theta b) a
      = meanFeature pairFeat (gibbs pairFeat (fun _ => 0) b) a := by
    intro a
    rw [h a, parityEns_pairFeat a]
    have : gibbs pairFeat (fun _ => 0) b = fun _ : Fin 8 => (1 : ℝ) / 8 := by
      funext x
      rw [gibbs_zero]
      norm_num
    rw [this, uniform_pairFeat a]
  have huniq := gibbs_unique_of_meanFeature_eq pairFeat theta (fun _ => 0) b hzero
  refine ⟨?_, parityEns_tripleFeat⟩
  rw [huniq]
  have : gibbs pairFeat (fun _ => 0) b = fun _ : Fin 8 => (1 : ℝ) / 8 := by
    funext x
    rw [gibbs_zero]
    norm_num
  rw [this]
  exact uniform_tripleFeat

/-! ## The fitted potential is a free energy -/

/-- A single feature on two configurations: the "contact" indicator. -/
def twoFeat : Fin 1 → Fin 2 → ℝ := fun _ x => (x.val : ℝ)

lemma twoFeat_mean (theta : Fin 1 → ℝ) (b : ℝ) :
    meanFeature twoFeat (gibbs twoFeat theta b) 0
      = Real.exp (-b * theta 0) / (1 + Real.exp (-b * theta 0)) := by
  have he0 : energy twoFeat theta 0 = 0 := by
    simp [energy, twoFeat]
  have he1 : energy twoFeat theta 1 = theta 0 := by
    simp [energy, twoFeat]
  have hpart : part twoFeat theta b = 1 + Real.exp (-b * theta 0) := by
    rw [part, Fin.sum_univ_two, he0, he1]
    norm_num
  simp only [meanFeature, gibbs, Fin.sum_univ_two, hpart, he0, he1, twoFeat]
  norm_num

/-- **The fitted potential is temperature dependent.**  The potential reproducing the feature
value `1/3` at inverse temperature `1` is `log 2`; the one reproducing the same value at inverse
temperature `2` is `(log 2)/2`.  A potential fitted to structure at one state point does not
reproduce that structure at another. -/
theorem inverse_potential_temperature_dependent :
    meanFeature twoFeat (gibbs twoFeat (fun _ => Real.log 2) 1) 0 = 1/3 ∧
    meanFeature twoFeat (gibbs twoFeat (fun _ => Real.log 2 / 2) 2) 0 = 1/3 ∧
    meanFeature twoFeat (gibbs twoFeat (fun _ => Real.log 2) 2) 0 ≠ 1/3 := by
  have hlog2 : Real.exp (-Real.log 2) = 1/2 := by
    rw [Real.exp_neg, Real.exp_log (by norm_num : (0:ℝ) < 2)]
    norm_num
  refine ⟨?_, ?_, ?_⟩
  · rw [twoFeat_mean]
    norm_num
    rw [hlog2]
    norm_num
  · rw [twoFeat_mean]
    have : -(2:ℝ) * (Real.log 2 / 2) = -Real.log 2 := by ring
    rw [this, hlog2]
    norm_num
  · rw [twoFeat_mean]
    have h2 : (-2 : ℝ) * Real.log 2 = -Real.log 4 := by
      rw [show (4:ℝ) = 2^2 by norm_num, Real.log_pow]
      push_cast
      ring
    have hexp : Real.exp ((-2 : ℝ) * Real.log 2) = 1/4 := by
      rw [h2, Real.exp_neg, Real.exp_log (by norm_num : (0:ℝ) < 4)]
      norm_num
    rw [hexp]
    norm_num

end Inverse

end IDR
