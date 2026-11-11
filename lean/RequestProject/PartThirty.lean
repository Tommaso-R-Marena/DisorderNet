/-
# Part XXX  The entropy price of ordering

Part XVI established the reciprocity between conformational populations and affinity, and Part X
established that a binding constant is not a function of a mean structure with error bars.  Part
XXX supplies the accounting they leave open: what a disordered region pays, in free energy, for
being disordered when it binds, and what that implies for how a binding model must be built and
scored.

`RequestProject.Selection` works in the standard finite-conformer partition-function formalism:
free-state populations `p`, a binding-competent subset `S`, per-conformer interaction `-eps k`,
and the bound-state conformational sum restricted to `S`.

* `deltaG_eq_selection` -- the conformational-selection decomposition
  `ΔG = -e + kT log (1/P_S)`;
* `selection_penalty_pos` -- the penalty is strictly positive unless the whole free ensemble is
  competent;
* `penalty_uniform_eq_log_card` -- for a uniform free ensemble binding through one conformer it
  is exactly `kT log m`, `kT` times the conformational entropy of the free state, so two models
  agreeing on the bound structure and disagreeing on the free-state breadth predict affinities
  differing by exactly that entropy;
* `deltaG_antitone_subset` -- a complex that tolerates more conformers binds at least as well:
  fuzziness is thermodynamically favoured, not a modelling defect;
* `deltaG_le_neg_mean` -- scoring a binder by the mean interaction energy over the ensemble is
  systematically conservative (Jensen for `exp`);
* `deltaG_le_of_conformer` -- and one competent conformer of population `p_k` guarantees
  `ΔG ≤ -eps_k + kT log (1/p_k)` whatever the rest of the ensemble does.

`IDR.selection_laws` bundles the six.  The design consequence: an affinity prediction for a
disordered region is a difference of two ensemble free energies, so a model must report the
free-state populations of the competent conformers -- not a bound pose, and not a mean structure.
-/
import Mathlib
import RequestProject.Selection

set_option autoImplicit false

namespace IDR

/-- **The design laws of coupled folding and binding.**

1. *Conformational selection is an exact decomposition*: with a uniform interaction over the
   competent set, `ΔG = -e + kT log (1/P_S)`.
2. *The entropy penalty is strictly positive* whenever the competent population is below one.
3. *It is the conformational entropy*: `kT log m` for a uniform free ensemble binding through a
   single conformer.
4. *Fuzziness pays it back*: enlarging the tolerated set can only strengthen binding.
5. *Mean interaction energies underestimate binding.*
6. *A single competent conformer already caps the binding free energy.* -/
theorem selection_laws :
    -- 1  the conformational-selection decomposition
    (∀ (m : ℕ) (beta e : ℝ), 0 < beta → ∀ (p eps : Fin m → ℝ) (S : Finset (Fin m)),
        (∀ k, 0 ≤ p k) → (∀ k ∈ S, eps k = e) → ∀ k0 ∈ S, 0 < p k0 →
        Selection.deltaG beta p eps S = -e + (1 / beta) * Real.log (1 / Selection.popOf p S)) ∧
    -- 2  the penalty is strictly positive
    (∀ beta P : ℝ, 0 < beta → 0 < P → P < 1 → 0 < (1 / beta) * Real.log (1 / P)) ∧
    -- 3  and equals the conformational entropy of a uniform free ensemble
    (∀ (beta : ℝ) (m : ℕ), 0 < m → ∀ k0 : Fin m,
        Selection.deltaG beta (fun _ => (1 : ℝ) / m) (fun _ => (0 : ℝ)) {k0}
          = (1 / beta) * Real.log m) ∧
    -- 4  a fuzzier complex binds at least as well
    (∀ (m : ℕ) (beta : ℝ), 0 < beta → ∀ (p eps : Fin m → ℝ) (S T : Finset (Fin m)), S ⊆ T →
        (∀ k, 0 ≤ p k) → ∀ k0 ∈ S, 0 < p k0 →
        Selection.deltaG beta p eps T ≤ Selection.deltaG beta p eps S) ∧
    -- 5  the mean interaction energy underestimates the binding free energy
    (∀ (m : ℕ) (beta : ℝ), 0 < beta → ∀ p eps : Fin m → ℝ, (∀ k, 0 ≤ p k) → (∑ k, p k = 1) →
        Selection.deltaG beta p eps Finset.univ ≤ -∑ k, p k * eps k) ∧
    -- 6  one competent conformer caps the binding free energy
    (∀ (m : ℕ) (beta : ℝ), 0 < beta → ∀ (p eps : Fin m → ℝ) (S : Finset (Fin m)),
        (∀ i, 0 ≤ p i) → ∀ k ∈ S, 0 < p k →
        Selection.deltaG beta p eps S ≤ -eps k + (1 / beta) * Real.log (1 / p k)) :=
  ⟨fun _ _ _ hbeta _ _ _ hp heps _ hk0 hpk => Selection.deltaG_eq_selection hbeta hp heps hk0 hpk,
    fun _ _ hbeta hP0 hP1 => Selection.selection_penalty_pos hbeta hP0 hP1,
    fun beta _ hm k0 => Selection.penalty_uniform_eq_log_card beta hm k0,
    fun _ _ hbeta _ _ _ _ hST hp _ hk0 hpk => Selection.deltaG_antitone_subset hbeta hST hp hk0 hpk,
    fun _ _ hbeta _ _ hp hsum => Selection.deltaG_le_neg_mean hbeta hp hsum,
    fun _ _ hbeta _ _ _ hp _ hk hpk => Selection.deltaG_le_of_conformer hbeta hp hk hpk⟩

end IDR
