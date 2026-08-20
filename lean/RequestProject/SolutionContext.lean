/-
# Part CXXXII  Capstone: the solution context of a disordered-region model

Parts CXXIX–CXXXI added the laboratory to the exactly solved electrostatics of Parts CXX–CXXVIII.
This file states their conclusions as one theorem about the *interface* of a model of an
intrinsically disordered region: what the model must be told, what a measurement can tell it, and
what no measurement of this class ever will.

`solution_context_law` bundles, for a region of `N` residues:

1. **The read-out is the autocorrelation series.**  The measured screened energy at inverse
   Debye length `κ` is `∑_d d e^{−κd} C(d)` — the pairwise model contributes only through the
   `N − 1` charge autocorrelations.
2. **A salt series identifies exactly that, and hence everything in the class.**  Agreement of the
   energies at infinitely many ionic strengths forces agreement of all the autocorrelations, and
   with them the energy under *every* separation kernel — including the unreachable salt-free
   limit.
3. **One condition identifies nothing**, and a titration with fewer conditions than lags leaves a
   blind direction in correlation space.
4. **Above the Manning threshold the bare charge density is unobservable**, and the whole
   electrostatic budget is capped at `(b/lB)² · 4N/κ²` regardless of how charged the sequence is.
5. **The context variable is the ionic strength, not the salt concentration**: equal ionic
   strengths are exactly indistinguishable, while equal concentrations of a monovalent and a
   divalent buffer are not.

Taken with the capacity, context and ensemble laws of the earlier parts, this is the electrostatic
half of the specification: a model of a disordered region must accept the ionic strength (and the
solvent, through the Bjerrum length) as an explicit input, must be fitted across a series of such
conditions, must renormalise its charges before scoring, and must not report more electrostatic
detail than the `N − 1` autocorrelations — the only functionals of the sequence that this class of
measurement can ever constrain.
-/
import Mathlib
import RequestProject.SaltTitration
import RequestProject.Manning
import RequestProject.IonicStrength

set_option autoImplicit false

namespace IDR
namespace SolutionContext

open Finset

/-- **The solution-context law for a model of a disordered region.**  Five statements, proved in
Parts CXXIX–CXXXI, about what a salt series identifies, what condensation hides, and which
buffer variable the model must carry. -/
theorem solution_context_law {N : ℕ} (hN : 0 < N) {lB b : ℝ} (hlB : 0 < lB) (hb : 0 < b)
    {kappa : ℝ} (hk : 0 < kappa) {A c : ℝ} (hA : 0 < A) (hc : 0 < c) :
    -- 1. the read-out is the autocorrelation series
    (∀ (k : ℝ) (q : ℕ → ℝ), Salt.energy N k q
        = ∑ d ∈ Ico 1 N, (d : ℝ) * Real.exp (-(k * d)) * Pattern.autocorr N q d) ∧
    -- 2. an infinite salt series identifies the autocorrelations, hence every kernel
    (∀ (S : Set ℝ), S.Infinite → ∀ q q' : ℕ → ℝ,
        (∀ k ∈ S, Salt.energy N k q = Salt.energy N k q') →
          (∀ d, 1 ≤ d → d < N → Pattern.autocorr N q d = Pattern.autocorr N q' d) ∧
            ∀ w : ℕ → ℝ, Pattern.pairEnergy N w q = Pattern.pairEnergy N w q') ∧
    -- 3a. one condition identifies nothing
    (Salt.energy 3 kappa Titration.qOne = Salt.energy 3 kappa (Titration.qTuned kappa) ∧
        Pattern.autocorr 3 Titration.qOne 2 ≠ Pattern.autocorr 3 (Titration.qTuned kappa) 2) ∧
    -- 3b. fewer conditions than lags leave a blind direction
    (∀ (k : ℕ), k + 1 < N → ∀ kap : Fin k → ℝ,
        ∃ cc : ℕ → ℝ, (∃ d, 1 ≤ d ∧ d < N ∧ cc d ≠ 0) ∧ ∀ j, Titration.curve N cc (kap j) = 0) ∧
    -- 4. counterion condensation: saturation and the ceiling
    (∀ q : ℕ → ℝ, (∀ i, |q i| ≤ 1) →
        (∀ s s', b / lB ≤ s → b / lB ≤ s' →
            Manning.manningEnergy N kappa lB b s q = Manning.manningEnergy N kappa lB b s' q) ∧
          ∀ s, 0 ≤ s → |Manning.manningEnergy N kappa lB b s q|
            ≤ (b / lB) ^ 2 * (4 * N / kappa ^ 2)) ∧
    -- 5. ionic strength suffices; salt concentration does not
    ((∀ {n m : ℕ} (cc z : Fin n → ℝ) (cc' z' : Fin m → ℝ),
        Ionic.ionicStrength cc z = Ionic.ionicStrength cc' z' → ∀ q : ℕ → ℝ,
          Salt.energy N (Ionic.kappaOf A (Ionic.ionicStrength cc z)) q
            = Salt.energy N (Ionic.kappaOf A (Ionic.ionicStrength cc' z')) q) ∧
      ∀ pred : ℝ → ℝ,
        (Real.exp (-(A * Real.sqrt c)) - Real.exp (-(2 * (A * Real.sqrt c)))) / 2
          ≤ max |pred c - Salt.energy 2
                  (Ionic.kappaOf A (Ionic.ionicStrength ![c, c] ![1, -1])) (fun _ => 1)|
                |pred c - Salt.energy 2
                  (Ionic.kappaOf A (Ionic.ionicStrength ![c, c] ![2, -2])) (fun _ => 1)|) := by
  obtain ⟨h1, h2, h3, h4, _⟩ := Titration.salt_titration_law N hN
  refine ⟨h1, h2, h3 kappa hk.le, h4, fun q hq => ?_, ?_, ?_⟩
  · exact ⟨fun s s' hs hs' => Manning.manning_saturation hlB hb hs hs' N kappa q,
      fun s hs => Manning.manning_ceiling hk hlB hb hs hq⟩
  · intro n m cc z cc' z' h q
    exact Ionic.equal_ionicStrength_equal_prediction (A := A) h N q
  · exact fun pred => Ionic.no_concentration_blind_model hA hc pred

end SolutionContext
end IDR
