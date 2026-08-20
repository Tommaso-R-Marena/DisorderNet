/-
# Part XXVIII  Local restraints: scalar couplings and hydrogen exchange

Parts IX and IX.2 treated the global restraints (SAXS, PRE/NOE, smFRET) that dominate ensemble
modelling of disordered regions.  This part treats the two *local*, residue-resolved restraints
that carry most of the remaining experimental weight, and shows that each of them constrains
the ensemble far less than it is usually taken to.

* `RequestProject.Karplus` -- a three-bond scalar coupling averaged over an interconverting
  region is exactly `A⟨cos²θ⟩ + B⟨cos θ⟩ + C`.  It is therefore *two numbers* about the torsion
  distribution, no matter how many couplings of that torsion are measured, it is blind to the
  sign of the torsion, and the standard single-angle inversion is biased by exactly
  `A·Var(cos θ)`.  Explicitly: a five-basin torsion distribution's population of the `θ = π/2`
  basin can be anything in `[0, 1/2]` without changing any measured coupling.  Two basins, and
  only two, are identifiable from one coupling.
* `RequestProject.HX` -- hydrogen exchange averages *rates*, so the apparent protection free
  energy is `-RT log ⟨p_open⟩`, which is at most the mean local stability `⟨-RT log p_open⟩`,
  strictly so whenever the ensemble is heterogeneous, and is capped by any minority open state.
  Worse, the observable is not a functional of the equilibrium ensemble at all: two kinetic
  schemes with identical equilibrium populations exchange at different rates, so a model that
  outputs only a distribution cannot predict the experiment without a kinetic forward model.

`IDR.local_restraint_laws` bundles the five statements.  The design consequence is the same one
Part IX drew for the global data, sharpened: local restraints must enter as *forward-modelled*
ensemble averages with their exact nonlinear kernels, they must be accompanied by the
degeneracy they leave (a coupling fixes two moments, not a distribution), and hydrogen exchange
must either be restricted to a verified EX2 regime or modelled kinetically.
-/
import Mathlib
import RequestProject.Karplus
import RequestProject.HydrogenExchange

set_option autoImplicit false

namespace IDR

/-- **The design laws of local restraints.**

1. *A scalar coupling is two moments.*  For any Karplus parametrisation and any normalised
   torsion ensemble, the measured coupling is `A⟨cos²θ⟩ + B⟨cos θ⟩ + C`; consequently any two
   ensembles agreeing on those two moments agree on every coupling at once.
2. *The single-angle inversion is biased by the variance.*  `⟨J⟩` exceeds the coupling of the
   mean cosine by exactly `A·Var(cos θ)`, and the variance is strictly positive as soon as two
   populated conformers differ in `cos θ`.
3. *Torsion populations are not identifiable.*  There is an explicit five-basin ensemble whose
   `θ = π/2` population may be set to any value in `[0, 1/2]` with no change in any coupling;
   with only two basins and a discriminating coupling the population *is* determined.
4. *Hydrogen exchange reports below the mean stability.*  `-RT log⟨p_open⟩ ≤ ⟨-RT log p_open⟩`,
   strictly for a heterogeneous ensemble, and a conformer of weight `w` and openness `p` caps
   the apparent free energy at `-RT log p - RT log w`.
5. *Hydrogen exchange is not a functional of the equilibrium ensemble.*  Two kinetic schemes
   with the same opening equilibrium constant give different observed exchange rates; the EX2
   formula is exact only in the limit `k_int/k_cl → 0`, with the stated defect. -/
theorem local_restraint_laws :
    -- 1  a coupling sees only the first two moments of `cos θ`
    ((∀ (m : ℕ) (A B C : ℝ) (w θ : Fin m → ℝ), (∑ k, w k = 1) →
        Karplus.avgJ A B C w θ = A * Karplus.cosSq w θ + B * Karplus.cosMean w θ + C) ∧
      (∀ (m m' : ℕ) (w θ : Fin m → ℝ) (v phi : Fin m' → ℝ), (∑ k, w k = 1) → (∑ k, v k = 1) →
        Karplus.cosMean w θ = Karplus.cosMean v phi → Karplus.cosSq w θ = Karplus.cosSq v phi →
        ∀ A B C : ℝ, Karplus.avgJ A B C w θ = Karplus.avgJ A B C v phi) ∧
      (∀ (m : ℕ) (A B C : ℝ) (w θ : Fin m → ℝ),
        Karplus.avgJ A B C w (fun k => -(θ k)) = Karplus.avgJ A B C w θ)) ∧
    -- 2  the single-angle reading is biased by `A · Var(cos θ)`
    ((∀ (m : ℕ) (A B C : ℝ) (w θ : Fin m → ℝ), (∑ k, w k = 1) →
        Karplus.avgJ A B C w θ
          = Karplus.karplusOfCos A B C (Karplus.cosMean w θ) + A * Karplus.cosVar w θ) ∧
      (∀ (m : ℕ) (w θ : Fin m → ℝ), (∀ k, 0 ≤ w k) → (∑ k, w k = 1) →
        ∀ i j : Fin m, 0 < w i → 0 < w j → Real.cos (θ i) ≠ Real.cos (θ j) →
          0 < Karplus.cosVar w θ)) ∧
    -- 3  populations are unidentifiable in general, identifiable for two basins
    ((∀ u : ℝ, 0 ≤ u → u ≤ 1 / 2 →
        ∃ p : Fin 5 → ℝ, (∀ k, 0 ≤ p k) ∧ (∑ k, p k = 1) ∧
          (∀ A B C : ℝ, Karplus.avgJ A B C p Karplus.fiveAngles
            = Karplus.avgJ A B C Karplus.popA Karplus.fiveAngles) ∧ p 2 = u) ∧
      (∀ A B C c1 c2 p p' : ℝ,
        Karplus.karplusOfCos A B C c1 ≠ Karplus.karplusOfCos A B C c2 →
        p * Karplus.karplusOfCos A B C c1 + (1 - p) * Karplus.karplusOfCos A B C c2
          = p' * Karplus.karplusOfCos A B C c1 + (1 - p') * Karplus.karplusOfCos A B C c2 →
        p = p')) ∧
    -- 4  hydrogen exchange reports at or below the mean local stability, and is capped
    ((∀ (m : ℕ) (RT : ℝ), 0 < RT → ∀ w p : Fin m → ℝ, (∀ k, 0 ≤ w k) → (∑ k, w k = 1) →
        (∀ k, 0 < p k) → HX.deltaGapp RT w p ≤ HX.meanDeltaG RT w p) ∧
      (∀ RT : ℝ, 0 < RT →
        HX.deltaGapp RT (![1 / 2, 1 / 2] : Fin 2 → ℝ) (![1, 1 / 100] : Fin 2 → ℝ)
          < HX.meanDeltaG RT (![1 / 2, 1 / 2] : Fin 2 → ℝ) (![1, 1 / 100] : Fin 2 → ℝ)) ∧
      (∀ (m : ℕ) (RT : ℝ), 0 < RT → ∀ w p : Fin m → ℝ, (∀ i, 0 ≤ w i) → (∀ i, 0 < p i) →
        ∀ k : Fin m, 0 < w k →
          HX.deltaGapp RT w p ≤ -RT * Real.log (p k) - RT * Real.log (w k))) ∧
    -- 5  the equilibrium ensemble does not determine the exchange measurement
    ((∃ kop kcl kop' kcl' kint : ℝ, 0 < kop ∧ 0 < kcl ∧ 0 < kop' ∧ 0 < kcl' ∧ 0 < kint ∧
        HX.Kop kop kcl = HX.Kop kop' kcl' ∧ HX.kex kop kcl kint ≠ HX.kex kop' kcl' kint) ∧
      (∀ kop kcl kint : ℝ, 0 < kcl → 0 < kint →
        kint * HX.Kop kop kcl - HX.kex kop kcl kint
          = kint * HX.Kop kop kcl * (kint / (kcl + kint)))) :=
  ⟨⟨fun _ A B C _ _ hsum => Karplus.avgJ_eq A B C hsum,
      fun _ _ _ _ _ _ hw hv h1 h2 => Karplus.avgJ_congr_of_moments hw hv h1 h2,
      fun _ A B C w θ => Karplus.avgJ_reflect A B C w θ⟩,
    ⟨fun _ A B C _ _ hsum => Karplus.avgJ_eq_karplusOfCos_add_var A B C hsum,
      fun _ _ _ hw hsum _ _ hi hj hcos => Karplus.cosVar_pos_of_two hw hsum hi hj hcos⟩,
    ⟨fun _ hu0 hu1 => Karplus.karplus_any_population_consistent hu0 hu1,
      fun _ _ _ _ _ _ _ hne h => Karplus.twoBasin_identifiable hne h⟩,
    ⟨fun _ _ hRT _ _ hw hsum hp => HX.deltaGapp_le_mean hRT hw hsum hp,
      fun _ hRT => HX.deltaGapp_lt_mean_two hRT,
      fun _ _ hRT _ _ hw hp k hk => HX.deltaGapp_le_of_weight hRT hw hp k hk⟩,
    ⟨HX.equilibrium_does_not_determine_kex,
      fun _ _ _ hcl hint => HX.kex_eq_EX2_sub hcl hint⟩⟩

end IDR
