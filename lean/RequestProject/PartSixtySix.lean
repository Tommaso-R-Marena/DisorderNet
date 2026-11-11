/-
# Part LXVI  Umbrella sampling: overlap is exactly the condition

The free-energy profile of a disordered region -- along the radius of gyration, the end-to-end
distance, a contact coordinate -- is assembled from biased windows.  `RequestProject.Umbrella`
treats the assembly exactly on a finite conformation space and answers the question a referee
should ask of every reported profile: *which* features of it are determined by the data, and
which are artefacts of the recombination.

`IDR.umbrella_laws` bundles five statements:

1. *Within a window there is nothing left to argue about.*  Reweighting the biased histogram by
   `exp(V)` returns exactly the conditional distribution of the target on the window support.
2. *A window sees only shape.*  The data are invariant under rescaling the target, which is the
   precise reason WHAM and MBAR return free-energy offsets only up to one global constant.
3. *If no window straddles a split of the conformation space, the relative free energy across
   that split is completely undetermined*: for **every** positive ratio `r` there is a target
   with that ratio of populations reproducing every window histogram exactly.  Not merely noisy
   -- arbitrary.  This is the failure mode behind a profile whose two basins were sampled by
   disjoint sets of windows.
4. *And overlap is enough.*  If consecutive window supports meet and the windows cover the
   space, then two targets with the same window data are proportional, hence have the same
   populations and the same free-energy differences.  With (3) this makes connectivity of the
   window supports exactly the right condition -- necessary and sufficient.
5. *Overlap on paper is not overlap in the data.*  If the shared region carries probability `m`,
   a run of `N` frames misses it entirely with probability at least `1 - N m`, so runs shorter
   than `1/(2 m)` frames return, at least half the time, data whose recorded supports do not
   straddle -- and then (3) applies to what was actually recorded.  An explicit instance on three
   states shows the recorded supports `{0}` and `{2}` of two genuinely overlapping windows
   leaving the free energy of `{0,1}` against `{2}` free to take any value.

The practical reading: a reported free-energy difference between two regions of conformation
space is a *theorem about the data* only if some window carried population in both regions --
and carried it in the frames, not in the design.  Otherwise the number reported is the number
put in through the reference, exactly as the maximum-entropy analysis of Part III predicted for
underdetermined refinement.
-/
import Mathlib
import RequestProject.Umbrella

set_option autoImplicit false

namespace IDR

open IDR.Umbrella

/-- **The umbrella-sampling laws.**

1. reweighting a window by `exp(V)` returns the conditional distribution of the target on the
   window;
2. window data are invariant under rescaling the target;
3. with no straddling window, every population ratio across the split is consistent with the
   data;
4. with overlapping, covering windows the target is determined up to a positive constant, hence
   all populations and free-energy differences are determined;
5. a shared region of probability `m` is missed entirely by an `N`-frame run with probability at
   least `1 - N m`, and an explicit three-state instance shows the resulting data leaving a free
   energy completely free. -/
theorem umbrella_laws :
    (∀ (n : ℕ) (pi : Fin n → ℝ), (∀ x, 0 < pi x) → ∀ (V : Fin n → ℝ) (S : Finset (Fin n)),
        S.Nonempty → ∀ x ∈ S,
          winDist pi V S x * Real.exp (V x) / (∑ y ∈ S, winDist pi V S y * Real.exp (V y))
            = pi x / ∑ y ∈ S, pi y) ∧
    (∀ (n : ℕ) (pi V : Fin n → ℝ) (S : Finset (Fin n)) (c : ℝ), 0 < c →
        winDist (fun x => c * pi x) V S = winDist pi V S) ∧
    (∀ (n : ℕ) (pi : Fin n → ℝ), (∀ x, 0 < pi x) → ∀ (A : Finset (Fin n)), A.Nonempty →
        Aᶜ.Nonempty → ∀ (S : ℕ → Finset (Fin n)) (V : ℕ → Fin n → ℝ) (K : ℕ),
        (∀ k, k ≤ K → (S k ⊆ A ∨ Disjoint (S k) A)) → ∀ r : ℝ, 0 < r →
          ∃ q : Fin n → ℝ, (∀ x, 0 < q x) ∧
            (∀ k, k ≤ K → winDist q (V k) (S k) = winDist pi (V k) (S k)) ∧
            (∑ x ∈ A, q x) / (∑ x ∈ Aᶜ, q x) = r) ∧
    (∀ (n : ℕ) (pi q : Fin n → ℝ), (∀ x, 0 < pi x) → (∀ x, 0 < q x) →
        ∀ (S : ℕ → Finset (Fin n)) (V : ℕ → Fin n → ℝ) (K : ℕ),
        (∀ k, k ≤ K → (S k).Nonempty) → (∀ k, k < K → ((S k) ∩ (S (k + 1))).Nonempty) →
        (∀ x : Fin n, ∃ k, k ≤ K ∧ x ∈ S k) →
        (∀ k, k ≤ K → winDist q (V k) (S k) = winDist pi (V k) (S k)) →
          (∃ c, 0 < c ∧ ∀ x, q x = c * pi x) ∧
            ∀ A : Finset (Fin n),
              (∑ x ∈ A, q x) / (∑ x, q x) = (∑ x ∈ A, pi x) / (∑ x, pi x)) ∧
    ((∀ (m : ℝ), m ≤ 1 → ∀ N : ℕ, 1 - N * m ≤ (1 - m) ^ N) ∧
      (∀ (pi : Fin 3 → ℝ), (∀ x, 0 < pi x) → ∀ (V : ℕ → Fin 3 → ℝ) (r : ℝ), 0 < r →
        ∃ q : Fin 3 → ℝ, (∀ x, 0 < q x) ∧
          winDist q (V 0) {0} = winDist pi (V 0) {0} ∧
          winDist q (V 1) {2} = winDist pi (V 1) {2} ∧
          (∑ x ∈ ({0, 1} : Finset (Fin 3)), q x)
            / (∑ x ∈ ({0, 1} : Finset (Fin 3))ᶜ, q x) = r)) := by
  refine ⟨fun n pi hpi V S hS x hx => winDist_unbias hpi V hS hx,
    fun n pi V S c hc => winDist_smul V hc,
    fun n pi hpi A hA hAc S V K hstr r hr =>
      free_energy_unidentifiable hpi hA hAc S V K hstr hr,
    fun n pi q hpi hq S V K hns hov hcov hdata => ?_,
    ⟨fun m hm1 N => overlap_missed_ge hm1 N,
      fun pi hpi V r hr => thin_overlap_example hpi V hr⟩⟩
  obtain ⟨c, hc, hprop⟩ := umbrella_identifiable hpi hq S V K hns hov hcov hdata
  exact ⟨⟨c, hc, hprop⟩, fun A => population_determined hc hprop A⟩

end IDR
