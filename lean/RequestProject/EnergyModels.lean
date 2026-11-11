/-
# Energy-based and factorised models: what the physics form of the model buys, and what it
# does not

A natural reaction to the negative results of `RequestProject.Statistics` is: *do not
predict a structure, predict an energy function, and let statistical mechanics produce the
ensemble.*  This file examines that proposal, and the closely related idea of predicting a
disordered region *piecewise* (one module, domain or residue at a time) and combining the
answers.

## Energy-based models: correct in form, but with two hard limits

* `exists_energy_representation` -- **the form is right**: every ensemble with strictly
  positive weights over a conformational library is exactly the Boltzmann ensemble of some
  energy function at any chosen temperature.  Nothing is lost by parametrising a model by
  an energy instead of by weights.
* `energy_unique_up_to_const` -- but the energy is **identifiable only up to an additive
  constant**, and only on the library: absolute energies are not observable, so a model
  can only ever be trained on, and evaluated by, energy *differences*.
* `energyEns_prob_pos` and `no_energy_model_excludes` -- and at any finite temperature and
  any finite energy, **every conformation in the library keeps a positive population**.
  Hard constraints (excluded volume, chain connectivity, an obligate contact) are therefore
  not expressible as finite energies: they must be built into the *library* -- the support
  of the model -- which is a modelling decision the energy function cannot make.  The
  choice of what conformations the model can produce is prior to, and not learnable from,
  the energy.

## Factorised models: the coupling is exactly what they throw away

* `Ens.prod`, `prod_factorised` -- predicting two parts of a chain independently and
  pairing the answers gives an ensemble in which the parts are statistically independent.
* `separable_energy_prod` -- and an energy that is a *sum of independent contributions*
  (a one-body or per-residue energy, a per-residue propensity or "disorder score") produces
  exactly such a product ensemble, at every temperature.
* `no_factorised_captures_corrPair` -- but a correlated target is then unreachable.  Since
  coupling between segments is what makes a disordered region function -- avidity, allostery,
  coupled folding and binding -- a per-residue or per-module model is not a coarse
  approximation of the truth, it is in a class that excludes it.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics

namespace IDR

open Finset
open scoped Classical

variable {X Y : Type*}

/-! ## Boltzmann ensembles over a conformational library -/

lemma energy_partition_pos {m : ℕ} (hm : 0 < m) (lib : Fin m → X) (beta : ℝ) (U : X → ℝ) :
    0 < ∑ i, Real.exp (-(beta * U (lib i))) := by
  have hne : (Finset.univ : Finset (Fin m)).Nonempty := by
    rw [Finset.univ_nonempty_iff]
    exact Fin.pos_iff_nonempty.1 hm
  exact Finset.sum_pos (fun i _ => Real.exp_pos _) hne

/-- The Boltzmann ensemble over the conformational library `lib` for the energy function
`U` at inverse temperature `beta`: the output of an energy-based model. -/
noncomputable def energyEns {m : ℕ} (hm : 0 < m) (lib : Fin m → X) (beta : ℝ) (U : X → ℝ) :
    Ens X where
  card := m
  pt := lib
  w := fun j => Real.exp (-(beta * U (lib j))) / ∑ i, Real.exp (-(beta * U (lib i)))
  w_nonneg := fun j =>
    div_nonneg (Real.exp_pos _).le (energy_partition_pos hm lib beta U).le
  w_sum := by
    rw [← Finset.sum_div]
    exact div_self (ne_of_gt (energy_partition_pos hm lib beta U))

lemma energyEns_w {m : ℕ} (hm : 0 < m) (lib : Fin m → X) (beta : ℝ) (U : X → ℝ) (j : Fin m) :
    (energyEns hm lib beta U).w j
      = Real.exp (-(beta * U (lib j))) / ∑ i, Real.exp (-(beta * U (lib i))) := rfl

/-- **Finite energies cannot exclude anything.**  Every conformation of the library has a
strictly positive population, at every temperature and for every energy function. -/
theorem energyEns_prob_pos {m : ℕ} (hm : 0 < m) (lib : Fin m → X) (beta : ℝ) (U : X → ℝ)
    (j : Fin m) : 0 < (energyEns hm lib beta U).prob (lib j) := by
  refine ((energyEns hm lib beta U).prob_pos_iff (lib j)).2 ⟨j, ?_, rfl⟩
  rw [energyEns_w]
  exact div_pos (Real.exp_pos _) (energy_partition_pos hm lib beta U)

/-- Consequently a hard constraint -- a conformation that must have zero population -- is
never reproduced by an energy model whose library contains it.  Whatever the model must
forbid has to be excluded from the support, not penalised in the energy. -/
theorem no_energy_model_excludes {m : ℕ} (hm : 0 < m) (lib : Fin m → X) (beta : ℝ)
    (U : X → ℝ) {T : Ens X} {j : Fin m} (hT : T.prob (lib j) = 0) :
    ¬ (energyEns hm lib beta U).Same T := by
  intro h
  have := energyEns_prob_pos hm lib beta U j
  rw [Ens.prob_eq_of_same h, hT] at this
  exact lt_irrefl 0 this

/-- **Every ensemble is an energy model.**  Any ensemble whose weights are strictly
positive is exactly the Boltzmann ensemble of the energy `U = -(1/beta) log w` over its own
library, at whatever inverse temperature `beta ≠ 0` one likes.  Parametrising by an energy
loses nothing -- and, because the temperature is free, an energy model has no more
information in it than the weights it produces. -/
theorem exists_energy_representation (E : Ens X) (hpos : 0 < E.card)
    (hinj : Function.Injective E.pt) (hw : ∀ j, 0 < E.w j) {beta : ℝ} (hbeta : beta ≠ 0) :
    ∃ U : X → ℝ, (energyEns hpos E.pt beta U).Same E := by
  classical
  refine ⟨fun x => -(Real.log (E.prob x)) / beta, ?_⟩
  have hexp : ∀ j, Real.exp (-(beta * (-(Real.log (E.prob (E.pt j))) / beta))) = E.w j := by
    intro j
    rw [Ens.prob_pt_of_injective E hinj j]
    rw [show -(beta * (-(Real.log (E.w j)) / beta)) = Real.log (E.w j) by field_simp]
    exact Real.exp_log (hw j)
  have hZ : ∑ i, Real.exp (-(beta * (-(Real.log (E.prob (E.pt i))) / beta))) = 1 := by
    rw [Finset.sum_congr rfl fun i _ => hexp i]
    exact E.w_sum
  intro f
  simp only [Ens.expect, energyEns, hZ, div_one]
  exact Finset.sum_congr rfl fun j _ => by rw [hexp j]

/-- **The energy is identifiable only up to an additive constant.**  Two energy functions
give the same ensemble exactly when they differ by a constant on the library: absolute
energies are unobservable, so an energy-based model can be trained only on differences. -/
theorem energy_unique_up_to_const {m : ℕ} (hm : 0 < m) (lib : Fin m → X)
    (hlib : Function.Injective lib) {beta : ℝ} (hbeta : beta ≠ 0) (U V : X → ℝ)
    (h : (energyEns hm lib beta U).Same (energyEns hm lib beta V)) :
    ∃ c : ℝ, ∀ j, U (lib j) = V (lib j) + c := by
  set Zu : ℝ := ∑ i, Real.exp (-(beta * U (lib i))) with hZu
  set Zv : ℝ := ∑ i, Real.exp (-(beta * V (lib i))) with hZv
  have hZup : 0 < Zu := energy_partition_pos hm lib beta U
  have hZvp : 0 < Zv := energy_partition_pos hm lib beta V
  have hweq : ∀ j : Fin m,
      Real.exp (-(beta * U (lib j))) / Zu = Real.exp (-(beta * V (lib j))) / Zv := by
    intro j
    have h1 := Ens.prob_pt_of_injective (energyEns hm lib beta U) hlib j
    have h2 := Ens.prob_pt_of_injective (energyEns hm lib beta V) hlib j
    have h3 : (energyEns hm lib beta U).prob (lib j) = (energyEns hm lib beta V).prob (lib j) :=
      Ens.prob_eq_of_same h _
    have h1' : (energyEns hm lib beta U).prob (lib j) = (energyEns hm lib beta U).w j := h1
    have h2' : (energyEns hm lib beta V).prob (lib j) = (energyEns hm lib beta V).w j := h2
    rw [h1', h2'] at h3
    simpa [energyEns_w, hZu, hZv] using h3
  refine ⟨-(Real.log (Zu / Zv)) / beta, fun j => ?_⟩
  have hkey : Real.exp (-(beta * U (lib j)))
      = Real.exp (-(beta * V (lib j))) * (Zu / Zv) := by
    have := hweq j
    field_simp at this ⊢
    linarith
  have hlog : -(beta * U (lib j)) = -(beta * V (lib j)) + Real.log (Zu / Zv) := by
    have hpos : 0 < Zu / Zv := div_pos hZup hZvp
    have := congrArg Real.log hkey
    rwa [Real.log_exp, Real.log_mul (Real.exp_ne_zero _) (ne_of_gt hpos), Real.log_exp] at this
  have : beta * U (lib j) = beta * (V (lib j) + -(Real.log (Zu / Zv)) / beta) := by
    field_simp
    linarith
  exact mul_left_cancel₀ hbeta this

/-! ## Product ensembles and factorised models -/

/-- The independent (product) combination of an ensemble for one part of the chain with an
ensemble for another: what "predict the parts separately and pair the answers" produces. -/
noncomputable def Ens.prod (E : Ens X) (F : Ens Y) : Ens (X × Y) where
  card := E.card * F.card
  pt := fun k => (E.pt (finProdFinEquiv.symm k).1, F.pt (finProdFinEquiv.symm k).2)
  w := fun k => E.w (finProdFinEquiv.symm k).1 * F.w (finProdFinEquiv.symm k).2
  w_nonneg := fun k => mul_nonneg (E.w_nonneg _) (F.w_nonneg _)
  w_sum := by
    rw [← Equiv.sum_comp finProdFinEquiv
      (fun k => E.w (finProdFinEquiv.symm k).1 * F.w (finProdFinEquiv.symm k).2)]
    simp only [Equiv.symm_apply_apply]
    rw [Fintype.sum_prod_type, ← Finset.sum_mul_sum]
    rw [E.w_sum, F.w_sum]
    ring

lemma Ens.expect_prod (E : Ens X) (F : Ens Y) (h : X × Y → ℝ) :
    (E.prod F).expect h = ∑ j, ∑ k, E.w j * F.w k * h (E.pt j, F.pt k) := by
  simp only [Ens.expect, Ens.prod]
  rw [← Equiv.sum_comp finProdFinEquiv
    (fun k => (E.w (finProdFinEquiv.symm k).1 * F.w (finProdFinEquiv.symm k).2) *
      h (E.pt (finProdFinEquiv.symm k).1, F.pt (finProdFinEquiv.symm k).2))]
  simp only [Equiv.symm_apply_apply]
  rw [Fintype.sum_prod_type]

/-- A model is *factorised* over the two parts when they are statistically independent in
its output: every product observable averages as a product. -/
def Factorised (M : Ens (X × Y)) : Prop :=
  ∀ (f : X → ℝ) (g : Y → ℝ),
    M.expect (fun p => f p.1 * g p.2)
      = M.expect (fun p => f p.1) * M.expect (fun p => g p.2)

/-- Predicting the two parts separately gives a factorised model. -/
theorem prod_factorised (E : Ens X) (F : Ens Y) : Factorised (E.prod F) := by
  intro f g
  rw [Ens.expect_prod, Ens.expect_prod, Ens.expect_prod]
  have h1 : ∑ j, ∑ k, E.w j * F.w k * (f (E.pt j) * g (F.pt k))
      = (∑ j, E.w j * f (E.pt j)) * ∑ k, F.w k * g (F.pt k) := by
    rw [Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => by ring
  have h2 : ∑ j, ∑ k, E.w j * F.w k * f (E.pt j) = ∑ j, E.w j * f (E.pt j) := by
    refine Finset.sum_congr rfl fun j _ => ?_
    calc ∑ k, E.w j * F.w k * f (E.pt j)
        = (E.w j * f (E.pt j)) * ∑ k, F.w k := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun k _ => by ring
      _ = E.w j * f (E.pt j) := by rw [F.w_sum, mul_one]
  have h3 : ∑ j, ∑ k, E.w j * F.w k * g (F.pt k) = ∑ k, F.w k * g (F.pt k) := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun k _ => ?_
    calc ∑ j, E.w j * F.w k * g (F.pt k)
        = (F.w k * g (F.pt k)) * ∑ j, E.w j := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun j _ => by ring
      _ = F.w k * g (F.pt k) := by rw [E.w_sum, mul_one]
  rw [h1, h2, h3]

/-- Being factorised is an observational property, so it transfers along `Same`. -/
lemma factorised_of_same {M N : Ens (X × Y)} (h : M.Same N) (hM : Factorised M) :
    Factorised N := by
  intro f g
  rw [← h (fun p => f p.1 * g p.2), ← h (fun p => f p.1), ← h (fun p => g p.2)]
  exact hM f g

/-- A separable energy -- a sum of independent per-part contributions, the formal content
of a per-residue propensity, a one-body potential or an additive "disorder score" -- gives
exactly the product of the two per-part Boltzmann ensembles, at every temperature. -/
theorem separable_energy_prod {m n : ℕ} (hm : 0 < m) (hn : 0 < n) (libX : Fin m → X)
    (libY : Fin n → Y) (beta : ℝ) (u : X → ℝ) (v : Y → ℝ) :
    Factorised
      (energyEns (Nat.mul_pos hm hn) (fun k => (libX (finProdFinEquiv.symm k).1,
          libY (finProdFinEquiv.symm k).2)) beta (fun p => u p.1 + v p.2)) := by
  classical
  set lib : Fin (m * n) → X × Y := fun k =>
    (libX (finProdFinEquiv.symm k).1, libY (finProdFinEquiv.symm k).2) with hlib
  set U : X × Y → ℝ := fun p => u p.1 + v p.2 with hU
  set Zu : ℝ := ∑ i, Real.exp (-(beta * u (libX i))) with hZu
  set Zv : ℝ := ∑ i, Real.exp (-(beta * v (libY i))) with hZv
  have hZup : 0 < Zu := energy_partition_pos hm libX beta u
  have hZvp : 0 < Zv := energy_partition_pos hn libY beta v
  -- the partition function factorises
  have hZ : ∑ i, Real.exp (-(beta * U (lib i))) = Zu * Zv := by
    rw [← Equiv.sum_comp finProdFinEquiv (fun i => Real.exp (-(beta * U (lib i))))]
    simp only [hlib, hU, Equiv.symm_apply_apply]
    rw [Fintype.sum_prod_type, hZu, hZv, Finset.sum_mul_sum]
    refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => ?_
    rw [← Real.exp_add]
    ring_nf
  -- hence the ensemble is the product of the two one-part Boltzmann ensembles
  have hsame :
      (energyEns (Nat.mul_pos hm hn) lib beta U).Same
        ((energyEns hm libX beta u).prod (energyEns hn libY beta v)) := by
    intro f
    rw [Ens.expect_prod]
    simp only [Ens.expect, energyEns, hZ]
    rw [← Equiv.sum_comp finProdFinEquiv (fun k =>
      Real.exp (-(beta * U (lib k))) / (Zu * Zv) * f (lib k))]
    simp only [hlib, hU, Equiv.symm_apply_apply]
    rw [Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => ?_
    rw [show -(beta * (u (libX j) + v (libY k)))
        = -(beta * u (libX j)) + -(beta * v (libY k)) by ring, Real.exp_add]
    rw [← hZu, ← hZv]
    field_simp
  exact factorised_of_same hsame.symm (prod_factorised _ _)

/-! ## Correlated targets are outside the factorised class -/

/-- A perfectly correlated two-part target: the two halves are either both `0` or both
`1`.  This is the minimal model of coupling between two segments of a disordered region --
or between a region and the partner it binds. -/
noncomputable def corrPair : Ens (ℝ × ℝ) :=
  unif (by norm_num) ![((0 : ℝ), (0 : ℝ)), ((1 : ℝ), (1 : ℝ))]

lemma corrPair_expect (f : ℝ × ℝ → ℝ) :
    corrPair.expect f = (f (0, 0) + f (1, 1)) / 2 := by
  rw [corrPair, unif_expect]
  norm_num [Fin.sum_univ_succ]

lemma corrPair_not_factorised : ¬ Factorised corrPair := by
  intro h
  have := h (fun x => x) (fun y => y)
  rw [corrPair_expect, corrPair_expect, corrPair_expect] at this
  norm_num at this

/-- **Factorised models cannot represent coupling.**  No model that treats the two parts
independently -- a per-residue predictor, a per-module predictor, a one-body energy, a
mean-field free energy -- reproduces a correlated target.  The failure is not a matter of
accuracy: correlation is identically zero in the whole model class. -/
theorem no_factorised_captures_corrPair (M : Ens (ℝ × ℝ)) (hM : Factorised M) :
    ¬ M.Same corrPair := fun h => corrPair_not_factorised (factorised_of_same h hM)

/-- In particular no product of two independently predicted per-part ensembles, and no
separable energy model, captures it. -/
theorem no_prod_captures_corrPair (E F : Ens ℝ) : ¬ (E.prod F).Same corrPair :=
  no_factorised_captures_corrPair _ (prod_factorised E F)

/-! ## What rescues factorisation: a latent variable -/

lemma expect_prod_dirac (x : X) (y : Y) (f : X × Y → ℝ) :
    ((Ens.dirac x).prod (Ens.dirac y)).expect f = f (x, y) := by
  rw [Ens.expect_prod]
  simp [Ens.dirac]

/-- **Conditional independence is enough; unconditional independence is not.**  Every
two-part ensemble, correlated or not, is a *mixture* of factorised ensembles.  So the way
to keep the computational convenience of predicting the parts separately is to make the
factorisation conditional on a latent state and to mix over it: the latent variable is
exactly the object that carries the correlation the factorisation throws away.  This is the
precise sense in which a latent-variable generative model is the right shape, and a
per-residue independent model is not. -/
theorem mixture_of_products_universal (M : Ens (X × Y)) (f : X × Y → ℝ) :
    M.expect f
      = ∑ j, M.w j * ((Ens.dirac (M.pt j).1).prod (Ens.dirac (M.pt j).2)).expect f := by
  have hj : ∀ j, ((Ens.dirac (M.pt j).1).prod (Ens.dirac (M.pt j).2)).expect f = f (M.pt j) :=
    fun j => expect_prod_dirac _ _ f
  rw [Finset.sum_congr rfl (fun j _ => by rw [hj j])]
  rfl

end IDR
