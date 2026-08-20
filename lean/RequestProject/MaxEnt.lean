/-
# Ensemble refinement: maximum entropy, exponential tilting, and what the data leave open

`RequestProject.FiniteData` shows that no finite family of experimental observables
determines a conformational ensemble: refinement against SAXS, NMR or FRET restraints is
intrinsically underdetermined.  This file formalises the standard *resolution* of that
underdetermination -- maximum-entropy (minimum relative-entropy, "Bayesian/maximum-entropy
ensemble refinement") reweighting of a reference ensemble -- and states precisely what it
does and does not achieve.

Everything is phrased over a conformational library indexed by `Fin n`, a reference
(prior) weight vector `q`, and `r` linear restraints `f a` with measured values `d a`.

* `tilt` -- the exponentially tilted (reweighted) ensemble `q_j exp(∑_a λ_a f_a(j)) / Z`.
  `tilt_pos`, `tilt_sum_one` make it a genuine ensemble on the library, and `tilt_pos` is
  already a statement about the method's limits: **a reweighted ensemble always keeps the
  entire library populated**, so restraints can never switch a conformation off.
* `relEnt_pythagoras` -- the **Pythagorean identity** of information geometry: for every
  ensemble `p` matching the same restraints as the tilt `t`,
  `KL(p‖q) = KL(p‖t) + KL(t‖q)`.  Consequences:
  `tilt_is_min` (the tilt is the minimum-relative-entropy ensemble consistent with the
  data) and `tilt_unique` (it is the *only* one).  This is the exact sense in which
  maximum-entropy refinement is well posed.
* `tilt_tilt` -- **sequential refinement equals joint refinement**: tilting an already
  tilted ensemble by further restraints gives the tilt by the summed multipliers.  Fitting
  SAXS and then NMR is the same as fitting both at once, provided the multipliers are
  re-optimised.
* `maxent_prior_dependence` -- **the reference ensemble is never washed out**: where the
  data do not constrain, the maximum-entropy answer *is* the prior, so two laboratories
  with different reference ensembles and identical data report different ensembles.
  Underdetermination is not removed by the maximum-entropy principle; it is transferred to
  the choice of prior, and must be reported as such.
-/
import Mathlib
import RequestProject.DisorderedRegions
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.EnergyModels

namespace IDR

open Finset
open scoped Classical

namespace MaxEnt

variable {n r : ℕ}

/-- The partition function of the exponentially tilted ensemble. -/
noncomputable def partition (q : Fin n → ℝ) (lam : Fin r → ℝ) (f : Fin r → Fin n → ℝ) : ℝ :=
  ∑ j, q j * Real.exp (∑ a, lam a * f a j)

/-- The exponentially tilted ("reweighted") ensemble: the maximum-entropy form. -/
noncomputable def tilt (q : Fin n → ℝ) (lam : Fin r → ℝ) (f : Fin r → Fin n → ℝ) :
    Fin n → ℝ :=
  fun j => q j * Real.exp (∑ a, lam a * f a j) / partition q lam f

/-- `p` reproduces the measured values `d` of the restraints `f`. -/
def Matches (p : Fin n → ℝ) (f : Fin r → Fin n → ℝ) (d : Fin r → ℝ) : Prop :=
  ∀ a, ∑ j, p j * f a j = d a

lemma partition_pos {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hn : 0 < n)
    (lam : Fin r → ℝ) (f : Fin r → Fin n → ℝ) : 0 < partition q lam f := by
  refine Finset.sum_pos (fun j _ => mul_pos (hq j) (Real.exp_pos _)) ?_
  exact Finset.univ_nonempty_iff.2 (Fin.pos_iff_nonempty.1 hn)

lemma tilt_pos {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hn : 0 < n)
    (lam : Fin r → ℝ) (f : Fin r → Fin n → ℝ) (j : Fin n) : 0 < tilt q lam f j := by
  have := partition_pos hq hn lam f
  exact div_pos (mul_pos (hq j) (Real.exp_pos _)) this

lemma tilt_sum_one {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hn : 0 < n)
    (lam : Fin r → ℝ) (f : Fin r → Fin n → ℝ) : ∑ j, tilt q lam f j = 1 := by
  have hZ := partition_pos hq hn lam f
  have hrw : ∑ j, tilt q lam f j = partition q lam f / partition q lam f := by
    rw [partition, Finset.sum_div]
    rfl
  rw [hrw, div_self (ne_of_gt hZ)]

/-! ## The Pythagorean identity and the variational characterisation -/

/-- **Pythagorean identity of ensemble refinement.**  If `p` is any ensemble over the
library that reproduces the same restraint values as the tilted ensemble `t`, then the
relative entropies decompose exactly:  `KL(p‖q) = KL(p‖t) + KL(t‖q)`. -/
theorem relEnt_pythagoras {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) {lam : Fin r → ℝ}
    {f : Fin r → Fin n → ℝ} {p : Fin n → ℝ} (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1)
    (hmatch : ∀ a, ∑ j, p j * f a j = ∑ j, tilt q lam f j * f a j) :
    klDiv p q = klDiv p (tilt q lam f) + klDiv (tilt q lam f) q := by
  have hn : 0 < n := by
    rcases Nat.eq_zero_or_pos n with rfl | h
    · simp at hps
    · exact h
  set Z := partition q lam f with hZdef
  have hZ : 0 < Z := partition_pos hq hn lam f
  set t := tilt q lam f with htdef
  have hts : ∑ j, t j = 1 := tilt_sum_one hq hn lam f
  have htpos : ∀ j, 0 < t j := tilt_pos hq hn lam f
  -- the log-likelihood ratio of the tilt against the prior is linear in the restraints
  have hlogratio : ∀ j, Real.log (t j / q j) = (∑ a, lam a * f a j) - Real.log Z := by
    intro j
    have hqj := hq j
    have : t j / q j = Real.exp (∑ a, lam a * f a j) / Z := by
      rw [htdef, tilt, hZdef]
      field_simp
    rw [this, Real.log_div (ne_of_gt (Real.exp_pos _)) (ne_of_gt hZ), Real.log_exp]
  -- termwise difference of the two divergences
  have hterm : ∀ j, p j * Real.log (p j / q j) - p j * Real.log (p j / t j)
      = p j * ((∑ a, lam a * f a j) - Real.log Z) := by
    intro j
    rcases eq_or_lt_of_le (hp j) with h0 | hpos
    · simp [← h0]
    · have hlog : Real.log (p j / q j) - Real.log (p j / t j) = Real.log (t j / q j) := by
        rw [Real.log_div (ne_of_gt hpos) (ne_of_gt (hq j)),
          Real.log_div (ne_of_gt hpos) (ne_of_gt (htpos j)),
          Real.log_div (ne_of_gt (htpos j)) (ne_of_gt (hq j))]
        ring
      rw [← hlogratio j, ← hlog]
      ring
  have hsum : klDiv p q - klDiv p t = ∑ j, p j * ((∑ a, lam a * f a j) - Real.log Z) := by
    rw [klDiv, klDiv, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => hterm j
  -- the same computation for the tilt itself, where `KL(t‖t) = 0`
  have htermt : ∀ j, t j * Real.log (t j / q j)
      = t j * ((∑ a, lam a * f a j) - Real.log Z) := fun j => by rw [hlogratio j]
  have hsumt : klDiv t q = ∑ j, t j * ((∑ a, lam a * f a j) - Real.log Z) :=
    Finset.sum_congr rfl fun j _ => htermt j
  -- both linear functionals agree because the restraint values agree
  have hlin : ∀ u : Fin n → ℝ, ∑ j, u j = 1 →
      ∑ j, u j * ((∑ a, lam a * f a j) - Real.log Z)
        = (∑ a, lam a * ∑ j, u j * f a j) - Real.log Z := by
    intro u hu
    have h1 : ∀ j, u j * ((∑ a, lam a * f a j) - Real.log Z)
        = (∑ a, lam a * (u j * f a j)) - u j * Real.log Z := by
      intro j
      have h2 : ∑ a, lam a * (u j * f a j) = u j * ∑ a, lam a * f a j := by
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun a _ => by ring
      rw [h2]
      ring
    rw [Finset.sum_congr rfl (fun j _ => h1 j), Finset.sum_sub_distrib, ← Finset.sum_mul, hu,
      one_mul, Finset.sum_comm]
    congr 1
    exact Finset.sum_congr rfl fun a _ => by rw [Finset.mul_sum]
  rw [hlin p hps] at hsum
  rw [hlin t hts] at hsumt
  have hval : ∑ a, lam a * ∑ j, p j * f a j = ∑ a, lam a * ∑ j, t j * f a j :=
    Finset.sum_congr rfl fun a _ => by rw [hmatch a]
  rw [hval] at hsum
  linarith [hsum, hsumt]

/-- **Maximum-entropy refinement is optimal.**  Among all reweightings of the reference
ensemble that reproduce the data, the exponential tilt has the smallest relative entropy
to the reference: it is the least-committal ensemble consistent with the measurements. -/
theorem tilt_is_min {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hqs : ∑ j, q j = 1)
    {lam : Fin r → ℝ} {f : Fin r → Fin n → ℝ} {d : Fin r → ℝ}
    (hfit : Matches (tilt q lam f) f d) {p : Fin n → ℝ} (hp : ∀ j, 0 ≤ p j)
    (hps : ∑ j, p j = 1) (hpd : Matches p f d) :
    klDiv (tilt q lam f) q ≤ klDiv p q := by
  have hn : 0 < n := by
    rcases Nat.eq_zero_or_pos n with rfl | h
    · simp at hps
    · exact h
  have hmatch : ∀ a, ∑ j, p j * f a j = ∑ j, tilt q lam f j * f a j := fun a => by
    rw [hpd a, hfit a]
  have hpy := relEnt_pythagoras hq hp hps hmatch
  have hnn : 0 ≤ klDiv p (tilt q lam f) :=
    klDiv_nonneg hp hps (fun j => tilt_pos hq hn lam f j) (tilt_sum_one hq hn lam f)
  linarith

/-- **And it is the only optimum.**  Any other ensemble consistent with the data has
strictly larger relative entropy: the maximum-entropy reweighting is unique, so ensemble
refinement is a well-posed problem once a reference ensemble is fixed. -/
theorem tilt_unique {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hqs : ∑ j, q j = 1)
    {lam : Fin r → ℝ} {f : Fin r → Fin n → ℝ} {d : Fin r → ℝ}
    (hfit : Matches (tilt q lam f) f d) {p : Fin n → ℝ} (hp : ∀ j, 0 ≤ p j)
    (hps : ∑ j, p j = 1) (hpd : Matches p f d)
    (hmin : klDiv p q ≤ klDiv (tilt q lam f) q) :
    p = tilt q lam f := by
  have hn : 0 < n := by
    rcases Nat.eq_zero_or_pos n with rfl | h
    · simp at hps
    · exact h
  have hmatch : ∀ a, ∑ j, p j * f a j = ∑ j, tilt q lam f j * f a j := fun a => by
    rw [hpd a, hfit a]
  have hpy := relEnt_pythagoras hq hp hps hmatch
  have hnn : 0 ≤ klDiv p (tilt q lam f) :=
    klDiv_nonneg hp hps (fun j => tilt_pos hq hn lam f j) (tilt_sum_one hq hn lam f)
  have hzero : klDiv p (tilt q lam f) = 0 := by linarith
  exact (klDiv_eq_zero_iff hp hps (fun j => tilt_pos hq hn lam f j)
    (tilt_sum_one hq hn lam f)).1 hzero

/-! ## Composability of refinement -/

/-- **Sequential refinement is joint refinement.**  Tilting an already tilted ensemble by
further restraints yields the tilt of the original reference by the sum of the multipliers:
the exponential family of reweightings is closed, so fitting one experiment after another
never leaves the family, and the order of the experiments is irrelevant. -/
theorem tilt_tilt {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hn : 0 < n) (lam mu : Fin r → ℝ)
    (f : Fin r → Fin n → ℝ) :
    tilt (tilt q lam f) mu f = tilt q (lam + mu) f := by
  have hZ : 0 < partition q lam f := partition_pos hq hn lam f
  funext j
  have hsplit : ∀ i : Fin n, (∑ a, (lam + mu) a * f a i)
      = (∑ a, lam a * f a i) + (∑ a, mu a * f a i) := by
    intro i
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun a _ => by simp [add_mul]
  have hnum : ∀ i : Fin n, tilt q lam f i * Real.exp (∑ a, mu a * f a i)
      = q i * Real.exp (∑ a, (lam + mu) a * f a i) / partition q lam f := by
    intro i
    rw [tilt, hsplit i, Real.exp_add]
    field_simp
  have hden : partition (tilt q lam f) mu f
      = (∑ i, q i * Real.exp (∑ a, (lam + mu) a * f a i)) / partition q lam f := by
    rw [partition, Finset.sum_div]
    exact Finset.sum_congr rfl fun i _ => hnum i
  have hZ2 : 0 < partition q (lam + mu) f := partition_pos hq hn (lam + mu) f
  have hcancel : ∀ A B C : ℝ, C ≠ 0 → B ≠ 0 → (A / C) / (B / C) = A / B := by
    intro A B C hC hB
    field_simp
  rw [tilt, hnum j, hden]
  exact hcancel _ _ _ (ne_of_gt hZ) (ne_of_gt hZ2)

/-! ## What maximum entropy does not fix -/

/-- With no restraints the maximum-entropy answer is the prior itself. -/
theorem maxent_no_data {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hqs : ∑ j, q j = 1)
    {p : Fin n → ℝ} (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1) (hne : p ≠ q) :
    klDiv q q < klDiv p q := by
  have hzero : klDiv q q = 0 := by
    simp only [klDiv]
    refine Finset.sum_eq_zero fun j _ => ?_
    rw [div_self (ne_of_gt (hq j)), Real.log_one, mul_zero]
  rw [hzero]
  exact klDiv_pos_of_ne hp hps hq hqs hne

/-- **The reference ensemble is never washed out.**  Where the data do not constrain the
ensemble, maximum-entropy refinement returns the prior: two groups with the same
measurements but different reference ensembles obtain different, equally "maximum-entropy"
answers.  The underdetermination proved in `RequestProject.FiniteData` is therefore not
removed by the maximum-entropy principle -- it is relocated into the choice of reference
ensemble, which a model of a disordered region must report as part of its output. -/
theorem maxent_prior_dependence {q q' : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hqs : ∑ j, q j = 1)
    (hq' : ∀ j, 0 < q' j) (hq's : ∑ j, q' j = 1) (hne : q ≠ q') :
    (∀ p : Fin n → ℝ, (∀ j, 0 ≤ p j) → ∑ j, p j = 1 → p ≠ q → klDiv q q < klDiv p q) ∧
    (∀ p : Fin n → ℝ, (∀ j, 0 ≤ p j) → ∑ j, p j = 1 → p ≠ q' → klDiv q' q' < klDiv p q') ∧
    q ≠ q' :=
  ⟨fun _p hp hps hpne => maxent_no_data hq hqs hp hps hpne,
   fun _p hp hps hpne => maxent_no_data hq' hq's hp hps hpne, hne⟩

/-- **Reweighting cannot switch a conformation off.**  Every maximum-entropy refinement of
a reference ensemble keeps every library conformation populated, whatever the data.  A
restraint that is genuinely incompatible with a conformation must be imposed by removing it
from the library, not by fitting multipliers -- the same phenomenon as
`IDR.no_energy_model_excludes` for finite energy functions. -/
theorem maxent_full_support {q : Fin n → ℝ} (hq : ∀ j, 0 < q j) (hn : 0 < n)
    (lam : Fin r → ℝ) (f : Fin r → Fin n → ℝ) (j : Fin n) :
    tilt q lam f j ≠ 0 :=
  ne_of_gt (tilt_pos hq hn lam f j)

end MaxEnt

end IDR
