/-
# Part III capstone: the quantitative design laws

`RequestProject.Verdict` says what a model of an intrinsically disordered region must be
and what it cannot be.  This file collects the *quantitative* laws proved in
`RequestProject.Metric`, `RequestProject.MaxEnt` and `RequestProject.Invariance` into a
single statement, `quantitative_design_laws`, over a finite conformational library.

Read as a specification, the six clauses say:

1. **Measure error in population space.**  Uniform observational error is exactly the `ℓ¹`
   distance between predicted and true populations; there is no other natural loss.
2. **Size the model by the ensemble.**  Reaching accuracy `eps` on a target that populates
   `m` conformations with weight `≥ δ` requires at least `m - eps/δ` components, and this
   linear law is achieved.
3. **Size it by the entropy.**  An exactly correct model needs at least `exp H`
   components, `H` the conformational entropy of the target.
4. **Refine by maximum entropy, and report the prior.**  Minimum-relative-entropy
   reweighting of a reference ensemble is the unique ensemble consistent with the data, it
   composes across experiments, and where the data are silent it returns the prior --
   which must therefore be reported as part of the model.
5. **Pay for every invariance.**  Any two inputs the architecture conflates cost half the
   distance between their targets: bounded receptive fields and composition-only sequence
   features are quantitatively wrong.
6. **Couple the segments.**  Independent per-segment heads produce a product distribution
   and cannot represent inter-segment correlation.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.ModelNature
import RequestProject.EnergyModels
import RequestProject.Metric
import RequestProject.MaxEnt
import RequestProject.Invariance
import RequestProject.Verdict

namespace IDR

open Finset
open scoped Classical

/-- **The quantitative design laws for a model of an intrinsically disordered region.**
Each clause is an instance of a theorem proved in this development; together they turn the
qualitative verdict of `IDR.model_must_be` / `IDR.model_cannot_be` into numbers a designer
can use. -/
theorem quantitative_design_laws :
    -- (1) the loss is the population-space `ℓ¹` distance
    (∀ (N : ℕ) (eps : ℝ) (E F : Ens (Fin N)),
        ApproxSame eps E F ↔ Ens.ell1 E F ≤ eps) ∧
    -- (2) capacity must grow linearly with the number of populated conformations ...
    (∀ (N k m : ℕ) (M E : Ens (Fin N)) (g : Fin m → Fin N) (delta eps : ℝ),
        M.card ≤ k → Function.Injective g → 0 < delta → (∀ l, delta ≤ E.prob (g l)) →
        ApproxSame eps M E → (m : ℝ) - eps / delta ≤ k) ∧
    -- ... and the linear law is attained
    (∀ (N k m : ℕ) (hk : 0 < k) (hkm : k ≤ m) (g : Fin m → Fin N), Function.Injective g →
        Ens.ell1 (unif hk (fun l : Fin k => g (Fin.castLE hkm l)))
            (unif (lt_of_lt_of_le hk hkm) g) = 2 * ((m : ℝ) - k) / m) ∧
    -- (3) an exactly correct model carries at least `exp H` components
    (∀ (N : ℕ) (M E : Ens (Fin N)), M.Same E → 0 < M.card →
        Real.exp (entropy E) ≤ M.card) ∧
    -- (4) maximum-entropy refinement is the unique fit, composes, and returns the prior
    (∀ (n r : ℕ) (q : Fin n → ℝ) (lam : Fin r → ℝ) (f : Fin r → Fin n → ℝ)
        (d : Fin r → ℝ) (p : Fin n → ℝ),
        (∀ j, 0 < q j) → ∑ j, q j = 1 → MaxEnt.Matches (MaxEnt.tilt q lam f) f d →
        (∀ j, 0 ≤ p j) → ∑ j, p j = 1 → MaxEnt.Matches p f d →
        klDiv (MaxEnt.tilt q lam f) q ≤ klDiv p q ∧
          (klDiv p q ≤ klDiv (MaxEnt.tilt q lam f) q → p = MaxEnt.tilt q lam f)) ∧
    (∀ (n r : ℕ) (q : Fin n → ℝ) (lam mu : Fin r → ℝ) (f : Fin r → Fin n → ℝ),
        (∀ j, 0 < q j) → 0 < n →
        MaxEnt.tilt (MaxEnt.tilt q lam f) mu f = MaxEnt.tilt q (lam + mu) f) ∧
    (∀ (n : ℕ) (q p : Fin n → ℝ), (∀ j, 0 < q j) → ∑ j, q j = 1 → (∀ j, 0 ≤ p j) →
        ∑ j, p j = 1 → p ≠ q → klDiv q q < klDiv p q) ∧
    -- (5) every architectural invariance costs half the distance it conflates
    (∀ (N : ℕ) (I : Type) (A T : I → Ens (Fin N)) (i i' : I), (A i).Same (A i') →
        Ens.ell1 (T i) (T i') / 2
          ≤ max (Ens.ell1 (A i) (T i)) (Ens.ell1 (A i') (T i'))) ∧
    -- ... in particular a composition-only sequence model misplaces half the population
    (∀ (A : (Fin 4 → Bool) → Ens Bool) (T : (Fin 4 → Bool) → Ens Bool),
        (∀ s s', compo s = compo s' → (A s).Same (A s')) → (∀ s, T s = Ens.dirac (s 1)) →
        1 ≤ max (Ens.ell1 (A patA) (T patA)) (Ens.ell1 (A patB) (T patB))) ∧
    -- (6) two independent heads cannot represent a coupled pair of segments
    (∀ (P Q : Ens ℝ), ¬ (P.prod Q).Same corrPair) := by
  refine ⟨fun N eps E F => Ens.approxSame_iff_ell1_le eps E F,
    fun N k m M E g delta eps hM hg hdpos hd happ =>
      Ens.capacity_lower_bound hM hg hdpos hd happ,
    fun N k m hk hkm g hg => ell1_truncated_unif hk hkm hg,
    fun N M E h hM => card_ge_exp_entropy h hM,
    ?_, ?_, ?_,
    fun N I A T i i' h => invariance_cost_pair (A := A) (T := T) h,
    fun A T hA hT => composition_blind_error A hA T hT,
    fun P Q => no_prod_captures_corrPair P Q⟩
  · intro n r q lam f d p hq hqs hfit hp hps hpd
    exact ⟨MaxEnt.tilt_is_min hq hqs hfit hp hps hpd,
      fun hmin => MaxEnt.tilt_unique hq hqs hfit hp hps hpd hmin⟩
  · intro n r q lam mu f hq hn
    exact MaxEnt.tilt_tilt hq hn lam mu f
  · intro n q p hq hqs hp hps hne
    exact MaxEnt.maxent_no_data hq hqs hp hps hne

/-- **The sampling law.**  A model that presents its answer as `N` sampled structures --
molecular-dynamics snapshots, generated decoys, a structural library -- is a model of
capacity `N`, so the capacity laws apply verbatim: reproducing a target that populates `m`
conformations with weight at least `δ` to uniform accuracy `eps` needs
`N ≥ m - eps/δ` snapshots, and reproducing it exactly needs `N ≥ exp H`.  Sampling harder
is not an alternative to modelling the ensemble; it is the same requirement. -/
theorem sampling_law {N n m : ℕ} {Model E : Ens (Fin N)} (hN : Model.card ≤ n)
    {g : Fin m → Fin N} (hg : Function.Injective g) {delta eps : ℝ} (hdpos : 0 < delta)
    (hd : ∀ l, delta ≤ E.prob (g l)) (happ : ApproxSame eps Model E) :
    (m : ℝ) - eps / delta ≤ n :=
  Ens.capacity_lower_bound hN hg hdpos hd happ

/-- **The locality law**, stated separately because it quantifies over the sequence length,
the alphabet and the receptive field.  A predictor that cannot see position `j` misplaces
at least half of the conformational population of a region switched by residue `j`. -/
theorem locality_law {L : ℕ} {Alph X : Type*} [Fintype X] [DecidableEq X] [DecidableEq Alph]
    {S : Finset (Fin L)} {A : (Fin L → Alph) → Ens X} (hA : ReceptiveField S A)
    {j : Fin L} (hj : j ∉ S) (u v : X) (huv : u ≠ v) (a b : Alph) (hab : a ≠ b)
    (s : Fin L → Alph) (T : (Fin L → Alph) → Ens X)
    (hT : ∀ t : Fin L → Alph, T t = if t j = a then Ens.dirac u else Ens.dirac v) :
    1 ≤ max (Ens.ell1 (A (Function.update s j a)) (T (Function.update s j a)))
        (Ens.ell1 (A (Function.update s j b)) (T (Function.update s j b))) :=
  distal_switch_error hA hj u v huv a b hab s T hT

end IDR
