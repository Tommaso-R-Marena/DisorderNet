/-
# Part LXXX  Crosslinking mass spectrometry: the yield is a population, not a distance

`RequestProject.Crosslink` reads a crosslink experiment as it actually is: the reported quantity
`freq E d r` is the *fraction of the ensemble* in which the two sites lie within the reach `r` of
the spacer arm — one value of the cumulative distribution of that site–site distance.

`IDR.crosslink_laws` bundles five statements.

1. *What a yield is.*  It lies in `[0,1]` and is nondecreasing in the spacer reach; it is an
   ensemble average, hence a linear restraint of exactly the kind Parts LXIX and LXX count.
2. *A partial yield is already a proof of disorder.*  A single structure predicts every yield to
   be `0` or `1`; so a sub-stoichiometric yield `0 < f < 1` refutes every single-structure model
   of the region, with no geometric analysis and no second crosslink.
3. *A consistency test on the data.*  Mutually exclusive crosslinks — no conformation satisfies
   two of them — have yields summing to at most one.  A data set violating that comes from no
   ensemble at all, so this is a falsifiable check on the measurement, not on the model.
4. *The multiplicity theorem, and an instance.*  `k` mutually exclusive crosslinks that are all
   observed force at least `k` conformations in every model reproducing the data.  Explicitly: a
   tail tip visiting three anchors `0, 10, 20` with a spacer reaching `3` yields three exclusive
   crosslinks at `1/3` each, so no model with fewer than three states can be right.  "The
   crosslinks are incompatible with a single structure" is thereby upgraded from a refutation to a
   quantitative lower bound on ensemble size.
5. *And the positive design statement.*  Because yields are values of one cumulative distribution,
   a series of crosslinkers with spacer reaches `0, h, 2h, …` recovers the mean site–site distance
   by a layer-cake sum, to within the spacing `h`.  One crosslinker reports one number; a
   spacer-length series reports a distribution.  This is what makes crosslinking an ensemble
   observable rather than a violated-restraint counter.
-/
import Mathlib
import RequestProject.Crosslink

set_option autoImplicit false

namespace IDR

open Finset IDR.Crosslink

/-- **The crosslinking laws.**

1. a yield is a number in `[0,1]`, nondecreasing in the spacer reach;
2. a sub-stoichiometric yield refutes every single-structure model;
3. the yields of mutually exclusive crosslinks sum to at most one;
4. `k` observed mutually exclusive crosslinks force `k` conformations in any model reproducing
   the data — instantiated by the three-anchor tail, which forces three;
5. a spacer-length series of mesh `h` measures the mean site–site distance to within `h`. -/
theorem crosslink_laws {X : Type*} :
    (∀ (E : Ens X) (d : X → ℝ) (r r' : ℝ), 0 ≤ freq E d r ∧ freq E d r ≤ 1 ∧
        (r ≤ r' → freq E d r ≤ freq E d r')) ∧
    (∀ (E : Ens X) (d : X → ℝ) (r : ℝ), 0 < freq E d r → freq E d r < 1 →
        ¬ E.Deterministic) ∧
    (∀ (E : Ens X) {ι : Type} [DecidableEq ι] (d : ι → X → ℝ) (r : ι → ℝ),
        Exclusive d r → ∀ s : Finset ι, ∑ i ∈ s, freq E (d i) (r i) ≤ 1) ∧
    ((∀ (k : ℕ) (E M : Ens X), M.Same E → ∀ (d : Fin k → X → ℝ) (r : Fin k → ℝ),
        Exclusive d r → (∀ i, 0 < freq E (d i) (r i)) → k ≤ M.card) ∧
      (∀ M : Ens ℝ, M.Same triad → 3 ≤ M.card)) ∧
    (∀ (E : Ens X) (d : X → ℝ) (h : ℝ), 0 < h → ∀ n : ℕ,
        (∀ j, 0 ≤ d (E.pt j) ∧ d (E.pt j) ≤ n * h) →
        |E.expect d - series E d h n| ≤ h) :=
  ⟨fun E d r _ => ⟨freq_nonneg E d r, freq_le_one E d r, fun hr => freq_mono E d hr⟩,
    fun _ _ _ h0 h1 => not_deterministic_of_fractional h0 h1,
    fun E _ _ d r hex s => sum_freq_le_one E d r hex s,
    ⟨fun _ _ _ hsame d r hex hobs => card_ge_of_exclusive_model hsame d r hex hobs,
      fun _ hsame => triad_card_ge hsame⟩,
    fun E d _ hh n hd => mean_distance_from_series E d hh n hd⟩

end IDR
