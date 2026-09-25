/-
# Part LVIII  Orientational NMR: residual dipolar couplings

Every observable treated so far is a distance, a rate, a population or a coupling constant.  None
is orientational.  The measurement that is -- and that is used on disordered regions precisely
because it survives conformational averaging -- is the residual dipolar coupling of a weakly
aligned sample.  `RequestProject.Rdc` brings it inside the development.

`IDR.rdc_laws` bundles six statements about `rdc A u = (3uᵀAu − tr A)/2` with `A` symmetric and
traceless:

1. *A bond vector and its reverse are indistinguishable*, in every medium.
2. *One medium never determines an orientation*: for an axially symmetric tensor the coupling is
   constant on a whole circle of unit directions, so the ambiguity is continuous.
3. *A vanishing coupling is what symmetry gives*: averaged over three orthogonal directions the
   coupling is zero for every alignment tensor.
4. *Population and order enter only through their product*: a fully ordered ensemble at an
   intermediate angle and a half-ordered ensemble at the pole give the same coupling exactly.
5. *All the media in the world determine at most the second moment* of the orientational
   distribution -- and two explicit ensembles with no conformer in common share it, hence share
   every residual dipolar coupling.
6. *What is determined is five numbers*: an alignment tensor decomposes explicitly on five basis
   tensors, and five suitably chosen bond directions recover it.

The reading for a model of a disordered region: RDCs are a genuine orientational constraint and
not a distance, and the object they constrain is one symmetric traceless second-moment tensor per
bond.  A model is compared with RDC data correctly only by predicting that tensor from the
ensemble -- and correctly reporting that the comparison cannot see anything else.
-/
import Mathlib
import RequestProject.Rdc

set_option autoImplicit false

namespace IDR

open IDR.Rdc

/-- **The laws of residual dipolar couplings.**

1. invariance under reversal of the bond vector;
2. a continuous level set: one medium does not determine an orientation;
3. the coupling of an orientationally symmetric set of directions vanishes;
4. population and order parameter enter only through their product;
5. any number of media determine at most the second moment, and the second moment does not
   determine the ensemble;
6. an alignment tensor is five numbers, and five bond directions determine it. -/
theorem rdc_laws :
    (∀ (A : Matrix (Fin 3) (Fin 3) ℝ) (u : Fin 3 → ℝ), rdc A (-u) = rdc A u) ∧
    (Alignment axialTensor ∧
      ∀ theta : ℝ, rdc axialTensor ![Real.cos theta, Real.sin theta, 0] = 3 / 2) ∧
    (∀ A : Matrix (Fin 3) (Fin 3) ℝ, Alignment A →
      (rdc A ![1, 0, 0] + rdc A ![0, 1, 0] + rdc A ![0, 0, 1]) / 3 = 0) ∧
    (meanRdc ![1/2, 1/2] ![orderedDir, orderedDir] axialTensor
      = meanRdc ![1/2, 1/2] ![![0, 0, 1], nullDir] axialTensor) ∧
    ((∀ (n m : ℕ) (w : Fin n → ℝ) (u : Fin n → Fin 3 → ℝ) (w' : Fin m → ℝ)
        (v : Fin m → Fin 3 → ℝ), (∑ k, w k) = (∑ k, w' k) →
        secondMoment w u = secondMoment w' v →
        ∀ A : Matrix (Fin 3) (Fin 3) ℝ, meanRdc w u A = meanRdc w' v A) ∧
      ((∀ k l : Fin 2, ensA k ≠ ensB l) ∧
        secondMoment ![1/2, 1/2] ensA = secondMoment ![1/2, 1/2] ensB ∧
        ∀ A : Matrix (Fin 3) (Fin 3) ℝ,
          meanRdc ![1/2, 1/2] ensA A = meanRdc ![1/2, 1/2] ensB A)) ∧
    ((∀ A : Matrix (Fin 3) (Fin 3) ℝ, Alignment A →
        A = A 0 0 • Matrix.diagonal ![1, 0, -1] + A 1 1 • Matrix.diagonal ![0, 1, -1]
          + A 0 1 • (Matrix.of ![![0, 1, 0], ![1, 0, 0], ![0, 0, 0]])
          + A 0 2 • (Matrix.of ![![0, 0, 1], ![0, 0, 0], ![1, 0, 0]])
          + A 1 2 • (Matrix.of ![![0, 0, 0], ![0, 0, 1], ![0, 1, 0]])) ∧
      (∀ A B : Matrix (Fin 3) (Fin 3) ℝ, Alignment A → Alignment B →
        rdc A ![1, 0, 0] = rdc B ![1, 0, 0] →
        rdc A ![0, 1, 0] = rdc B ![0, 1, 0] →
        rdc A ![rt, rt, 0] = rdc B ![rt, rt, 0] →
        rdc A ![rt, 0, rt] = rdc B ![rt, 0, rt] →
        rdc A ![0, rt, rt] = rdc B ![0, rt, rt] → A = B)) := by
  refine ⟨rdc_neg, ⟨axialTensor_alignment, rdc_level_set_circle⟩,
    fun A hA => rdc_axes_mean_zero hA, order_population_degenerate,
    ⟨fun n m w u w' v hwt hS A => meanRdc_eq_of_secondMoment_eq w u w' v hwt hS A,
      secondMoment_not_injective⟩,
    fun A hA => alignment_decomposition hA,
    fun A B hA hB h0 h1 h2 h3 h4 => rdc_five_directions_determine hA hB h0 h1 h2 h3 h4⟩

end IDR
