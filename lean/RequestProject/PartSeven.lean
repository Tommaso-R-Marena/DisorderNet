/-
# Part VII capstone: the collective design laws

Parts I--VI model a disordered region one molecule at a time.  The biology that makes
intrinsically disordered regions interesting is often collective: multivalent disordered
proteins condense into dense phases, and their binding curves are switches rather than
graded isotherms.  Part VII adds that axis to the specification, and — this is the point for
a designer — proves that the collective behaviour is carried by exactly the information a
single-chain model does *not* contain.

`collective_design_laws` bundles seven clauses:

1. **Demixing is non-convexity.**  A convex free-energy density never phase separates, and
   every failure of convexity *is* a phase-separating composition.  The object a model must
   predict, if it is to predict condensation, is the *curvature* of the free-energy density
   in the concentration.
2. **The lever rule.**  Once the coexisting compositions are known the phase fractions are
   fixed; there is no further freedom to fit.
3. **Affine blindness.**  Adding any affine function of the concentration to the free-energy
   density changes no demixing statement.  The free energy of the isolated chain enters
   affinely, so **the single-chain ensemble — however exactly a model reproduces it —
   cannot decide whether a condensate forms.**
4. **A worked critical point.**  Flory--Huggins: convex, hence stable at every composition,
   for coupling `chi ≤ 2`; demixing at `chi = 4`.
5. **Cooperativity needs coupled sites.**  Independent motifs give the Langmuir isotherm and
   Hill coefficient exactly `1` at every valence, coupled motifs Hill coefficient exactly
   `n`; so for valence at least two no independent-site model reproduces the coupled
   response.  The `10%`-to-`90%` activity window shrinks from `81`-fold to `81^{1/n}`-fold
   and tends to `1`: multivalency is what makes the response a switch.
6. **The bulk-measurement law and its cost.**  A two-phase sample is a mass-weighted mixture
   of its phase ensembles; a single ensemble fitted to it is off by `(1-t)·d` on one phase
   and `t·d` on the other, and *any* single ensemble is off by at least `d/2` on one of
   them.  A model of a condensing protein must be conditional on the local concentration.
7. **Demixing is invisible to averaged data.**  A demixed sample and a homogeneous sample
   with the averaged populations are observationally identical, so the phase structure is a
   latent variable that only spatially resolved or single-molecule data can expose.

With `IDR.model_must_be` and `IDR.model_cannot_be` (Part II),
`IDR.quantitative_design_laws` (Part III), `IDR.physical_design_laws` (Part IV),
`IDR.statistical_design_laws` (Part V) and `IDR.precision_design_laws` (Part VI), this
completes the specification on a seventh axis: what the model must contain in order to say
anything about the collective behaviour of the disordered region.
-/
import Mathlib
import RequestProject.Condensate
import RequestProject.Valence
import RequestProject.TwoPhase

namespace IDR

open Set Filter Topology

/-- **The collective design laws for a model of an intrinsically disordered region.**
Each clause is an instance of a theorem proved in Part VII. -/
theorem collective_design_laws :
    -- (1) demixing is exactly the failure of convexity of the free-energy density
    (∀ (s : Set ℝ), Convex ℝ s → ∀ f : ℝ → ℝ,
        (∃ c, Phase.PhaseSeparates s f c) ↔ ¬ ConvexOn ℝ s f) ∧
    -- (2) the lever rule fixes the phase fractions
    (∀ c₁ c₂ c : ℝ, c₁ < c₂ → c ∈ Icc c₁ c₂ →
        ∃ t : ℝ, t ∈ Icc (0:ℝ) 1 ∧ t = (c₂ - c) / (c₂ - c₁) ∧ c = t * c₁ + (1 - t) * c₂) ∧
    -- (3) the phase diagram is blind to every affine (single-chain) contribution
    (∀ (s : Set ℝ) (f : ℝ → ℝ) (a b c : ℝ),
        Phase.PhaseSeparates s (fun x => f x + (a * x + b)) c ↔ Phase.PhaseSeparates s f c) ∧
    -- (4) Flory--Huggins: stable below the critical coupling, demixed above it
    ((∀ chi : ℝ, chi ≤ 2 → ∀ c, ¬ Phase.PhaseSeparates (Icc (0:ℝ) 1) (Phase.floryFE chi) c) ∧
      Phase.PhaseSeparates (Icc (0:ℝ) 1) (Phase.floryFE 4) (1/2)) ∧
    -- (5) cooperativity requires coupled motifs, and sharpens with valence
    ((∀ (n : ℕ) (x : ℝ), 0 ≤ x → 0 < n →
        Valence.occupancy (fun y => (1 + y) ^ n) x / n = Valence.fracOcc 1 x) ∧
      (∀ (n : ℕ) (u : ℝ), HasDerivAt (fun v => Real.log
          (Valence.fracOcc n (Real.exp v) / (1 - Valence.fracOcc n (Real.exp v)))) n u) ∧
      (∀ n : ℕ, 2 ≤ n → Valence.fracOcc n ≠ Valence.fracOcc 1) ∧
      Tendsto (fun n : ℕ => (81:ℝ) ^ ((n:ℝ)⁻¹)) atTop (𝓝 1)) ∧
    -- (6) the bulk-measurement law, and the cost of fitting one ensemble to two phases
    (∀ (X : Type) [Fintype X] (E F : Ens X) (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1),
        (∀ f : X → ℝ, (Ens.mix E F t ht0 ht1).expect f
            = t * E.expect f + (1 - t) * F.expect f) ∧
          Ens.ell1 (Ens.mix E F t ht0 ht1) E = (1 - t) * Ens.ell1 E F ∧
          Ens.ell1 (Ens.mix E F t ht0 ht1) F = t * Ens.ell1 E F ∧
          ∀ M : Ens X, Ens.ell1 E F / 2 ≤ max (Ens.ell1 M E) (Ens.ell1 M F)) ∧
    -- (7) and no averaged measurement can tell a demixed sample from a homogeneous one
    (∃ E F G : Ens Bool,
        (Ens.mix E F (1/2) (by norm_num) (by norm_num)).Same
          (Ens.mix G G (1/2) (by norm_num) (by norm_num)) ∧ ¬ E.Same G ∧ ¬ F.Same G) := by
  refine ⟨fun s hs f => Phase.exists_phaseSeparates_iff_not_convexOn hs f,
    fun c₁ c₂ c h hc => Phase.lever_rule h hc,
    fun s f a b c => Phase.phaseSeparates_add_affine f a b c,
    ⟨fun chi hchi c => Phase.no_demixing_of_weak_coupling hchi c, Phase.flory_demixes⟩,
    ⟨fun n x hx hn => Valence.perSite_occupancy_independent n hx hn,
      fun n u => Valence.hill_slope n u,
      fun n hn => Valence.no_cooperativity_from_independent_sites hn,
      Valence.ligand_window_tendsto_one⟩,
    ?_, bulk_cannot_detect_demixing⟩
  intro X _ E F t ht0 ht1
  exact ⟨fun f => Ens.expect_mix E F t ht0 ht1 f, Ens.ell1_mix_left E F ht0 ht1,
    Ens.ell1_mix_right E F ht0 ht1, fun M => Ens.no_single_ensemble_fits_both_phases M E F⟩

end IDR
