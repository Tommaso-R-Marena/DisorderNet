/-
# Part LXXVII  What a radius of gyration reports, and what it cannot

`RequestProject.ChainGeometry` treats the most-quoted single number about a disordered region --
its radius of gyration -- as what it actually is: one linear functional of the matrix of squared
interbead distances.

`IDR.chain_geometry_laws` bundles five statements.

1. *The readout identity.*  `Rg^2 = (1 / 2 n^2) * sum_{i,j} |x_i - x_j|^2` exactly, in any real
   inner-product space.  Consequently (2) any two conformations with the same distance matrix have
   the same radius of gyration: the map from structure to size is many-to-one before any
   experimental error is considered.
2. *The two-module law.*  For a chain cut into two blocks -- the realistic folded-domain plus
   disordered-tail architecture -- the global second moment is exactly
   `n Rg^2 = n_A Rg_A^2 + n_B Rg_B^2 + (n_A n_B / n) d^2`.
3. *A usable inequality.*  Hence a measured global `Rg` caps the separation of the two module
   centroids: `d^2 <= (n^2 / (n_A n_B)) Rg^2`.  This is a genuine structural constraint extracted
   from a single scalar measurement.
4. *A hard degeneracy.*  Two explicit four-bead chains have the same global `Rg^2 = 2` while one
   holds compact modules `2` apart and the other has a fourfold expanded module with coincident
   centroids.  Expansion and separation trade off exactly, so the scalar cannot report either.
5. *The connectivity ceiling, and its sharpness.*  Bonded chains satisfy
   `Rg^2 <= b^2 (n^2 - 1)/12`, and the straight rod attains it.

The moral matches Parts LXIX-LXXII.  A size measurement is a licensed report -- linear,
recomputable, with a computable ceiling -- and it is also, provably, blind along an explicit
direction of model space.  Both halves belong in the statement of any model of a disordered region.
-/
import Mathlib
import RequestProject.ChainGeometry

set_option autoImplicit false

namespace IDR

open Finset IDR.ChainGeo

/-- **The chain-geometry laws.**

1. the radius of gyration is a linear readout of the squared-distance matrix, and is therefore
   blind to everything the distance matrix does not distinguish;
2. the two-module parallel-axis identity;
3. the resulting cap on inter-module separation from a measured global size;
4. an explicit pair of chains with equal global size but opposite architecture;
5. the connectivity ceiling `b^2 (n^2 - 1)/12`, attained by the rod. -/
theorem chain_geometry_laws :
    (∀ {E : Type} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (s : Finset ℕ) (x : ℕ → E),
        rg2 s x = pairSum s x / (2 * (s.card : ℝ) ^ 2)) ∧
    (∀ {E : Type} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (s : Finset ℕ) (x y : ℕ → E),
        (∀ i ∈ s, ∀ j ∈ s, ‖x i - x j‖ = ‖y i - y j‖) → rg2 s x = rg2 s y) ∧
    (∀ {E : Type} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (s t : Finset ℕ) (x : ℕ → E),
        Disjoint s t → 0 < s.card → 0 < t.card →
        ((s ∪ t).card : ℝ) * rg2 (s ∪ t) x
          = (s.card : ℝ) * rg2 s x + (t.card : ℝ) * rg2 t x
            + ((s.card : ℝ) * t.card / ((s.card : ℝ) + t.card)) * ‖cen s x - cen t x‖ ^ 2) ∧
    (∀ {E : Type} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (s t : Finset ℕ) (x : ℕ → E),
        Disjoint s t → 0 < s.card → 0 < t.card →
        ‖cen s x - cen t x‖ ^ 2
          ≤ (((s.card : ℝ) + t.card) ^ 2 / ((s.card : ℝ) * t.card)) * rg2 (s ∪ t) x) ∧
    (rg2 (range 4) chainSep = 2 ∧ rg2 (range 4) chainExp = 2 ∧
      rg2 {0, 1} chainSep = 1 ∧ rg2 {0, 1} chainExp = 4 ∧
      ‖cen {0, 1} chainSep - cen {2, 3} chainSep‖ = 2 ∧
      ‖cen ({0, 1} : Finset ℕ) chainExp - cen {2, 3} chainExp‖ = 0) ∧
    (∀ {E : Type} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (b : ℝ) (x : ℕ → E),
        (∀ k, ‖x (k + 1) - x k‖ ≤ b) → ∀ n : ℕ, 0 < n →
        rg2 (range n) x ≤ b ^ 2 * ((n : ℝ) ^ 2 - 1) / 12) ∧
    (∀ (b : ℝ), 0 ≤ b → ∀ n : ℕ, 0 < n →
        rg2 (range n) (rod b) = b ^ 2 * ((n : ℝ) ^ 2 - 1) / 12 ∧
          ∀ k, ‖rod b (k + 1) - rod b k‖ ≤ b) :=
  ⟨fun s x => rg2_eq_pairSum s x,
    fun s x y h => rg2_congr_of_dist_eq s x y h,
    fun s t x hst hs ht => rg2_union s t x hst hs ht,
    fun s t x hst hs ht => inter_module_dist_sq_le s t x hst hs ht,
    rg2_tradeoff,
    fun b x hb n hn => rg2_le_rod b x hb n hn,
    fun b hb n hn => rg2_rod_sharp b hb n hn⟩

end IDR
