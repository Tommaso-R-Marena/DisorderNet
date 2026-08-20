/-
# Part XCIII  Four amendments for realism

Parts I–XCII build the disorder model around three idealisations that a structural biologist
would object to on sight.  This part removes them, and shows what survives.

1. **The native state is not the majority species.**  Many folded, functional proteins keep only
   20–30% of their population in the active conformation and shuttle between active and inactive
   forms.  `RequestProject.MarginalStability` replaces the majority criterion by an occupancy
   threshold `theta`: the thermodynamic gate becomes `kT·log(theta·(M−1)/(1−theta))`
   (`IDR.Marginal.occupancy_gap`), marginal stability buying exactly `kT·log((1−theta)/theta)` of
   slack — `1.1 kT` at 25%.  Everything else survives, including the compositional entropy floor
   (`IDR.Marginal.disordered_of_low_composition_entropy'`).  The two criteria genuinely differ:
   on a flat four-state landscape every conformation holds a quarter of the population, which is
   functional at `theta = 1/4` and not ordered at `1/2`.  And the equilibrium is switchable:
   stabilising the native state by `kT·log((1−theta)/theta)` makes it the majority species
   (`IDR.Marginal.shift_to_majority`), which is why marginal stability is a design feature and
   not a defect.

2. **Folding is driven by conditional probabilities.**  Hydrophobic–polar patterning and pairwise
   helix propensity are statements about `P(next | previous)`, not about composition.
   `RequestProject.ConditionalPropensity` proves the chain rule, the subadditivity of entropy,
   and the exact accounting: a pair model of a length-`n+1` region assigns `n·I` fewer nats than
   the composition-matched independent model, `I` the adjacent-residue mutual information
   (`IDR.CondSeq.block_entropy_deficit`).  For a blocky HP chain (switch probability `1/4`),
   `I = (3/4)log 3 − log 2 ≈ 0.131` nats per residue.

3. **Single-residue entropy is the wrong complexity screen.**  `ATATAT…` has the maximal
   composition entropy `log 2` and no complexity at all.  `RequestProject.KmerEntropy` proves the
   collapse: a period-`P` sequence shows at most `P` distinct `k`-mers, so its `k`-mer entropy
   never exceeds `log P` and its rate `H_k/k` falls below any threshold
   (`IDR.Kmer.alt_rate_eventually_small`); and composition is provably blind — `ATAT` and `AABB`
   have identical single-residue distributions and `2`-mer entropies `log 2` and `log 4`.

4. **A single mutation can spike `ΔΔG`.**  Breaking a salt bridge changes the energy by the whole
   contact depth in one substitution (`IDR.Spike.bridge_spike`), so the site-Lipschitz hypothesis
   is false (`IDR.Spike.bridge_not_siteLip`) and no additive model can fit the landscape to
   better than `|J|/4` (`IDR.Spike.additive_error_ge`).  `RequestProject.EpistaticSpike` rebuilds
   the mutational bound in a weighted Hamming metric, with a per-site cost `Smax` on a sparse set
   `Bad` of spike sites and `L` elsewhere, and shows the entropy floor survives with the
   correction `2·Smax·|Bad|`.

The capstone below combines amendments 1 and 4 — the case the previous parts do not cover — into
a single entropy floor valid at native occupancy `theta` *and* with epistatic spikes
(`disordered_of_low_entropy_marginal_and_spiky`), and then bundles one representative statement
from each of the four amendments (`four_realism_amendments`).

What is *not* claimed.  The pair model of amendment 2 is a first-order chain; real propensities
are longer-ranged, and the deficit `n·I` is a lower bound on what a richer conditional model
would remove.  The spike model of amendment 4 charges each spike site its worst case, so the
correction `2·Smax·|Bad|` is conservative.  Amendment 1 changes a threshold, not the physics: a
region whose population is spread over exponentially many conformations is disordered at every
`theta`.
-/
import Mathlib
import RequestProject.SequenceEntropyCore
import RequestProject.SequenceEntropyLimit
import RequestProject.MarginalStability
import RequestProject.ConditionalPropensity
import RequestProject.KmerEntropy
import RequestProject.EpistaticSpike

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR

namespace PartXCIII

open Finset
open SeqLimit
open scoped Classical

variable {N q M : ℕ}

/-! ## The composition step, isolated -/

/-- **Low compositional entropy means few minority residues.**  If the single-residue entropy of
a sequence is below `kappa·log 2` with `kappa ≤ 1/2`, some letter accounts for all but fewer than
`kappa·N` of its residues.  This is the only place the entropy hypothesis is used in any of the
floor theorems. -/
theorem exists_modal_letter_low_minority {kappa : ℝ} (hN : 0 < N) (hk : kappa ≤ 1 / 2)
    {s : Seq N q} (hlow : SeqEnt.H (comp s) < kappa * Real.log 2) :
    ∃ c : Fin q, (minorityCount s c : ℝ) < kappa * N := by
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hnn := comp_nonneg s
  have hsum := comp_sum_one hN s
  have hqpos : 0 < q := Fin.pos_iff_nonempty.2 ⟨s ⟨0, hN⟩⟩
  obtain ⟨c, -, hc⟩ := Finset.exists_max_image Finset.univ (comp s) ⟨⟨0, hqpos⟩,
    Finset.mem_univ _⟩
  refine ⟨c, ?_⟩
  set del : ℝ := ∑ a ∈ Finset.univ.erase c, comp s a with hdel
  have hdelval : del = (minorityCount s c : ℝ) / N := by rw [hdel, sum_comp_erase hN s c]
  have hdel0 : 0 ≤ del := by rw [hdelval]; positivity
  have hmin_entropy : -Real.log (comp s c) ≤ SeqEnt.H (comp s) :=
    SeqEnt.H_ge_neg_log_max hnn hsum (fun a => hc a (Finset.mem_univ a))
  have hlog2 : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  have hHlt : SeqEnt.H (comp s) < Real.log 2 := by nlinarith
  have hcpos : 1 / 2 < comp s c := by
    by_contra hcon
    push_neg at hcon
    have hcnn : 0 ≤ comp s c := hnn c
    rcases eq_or_lt_of_le hcnn with h0 | hpos
    · have hall : ∀ a, comp s a ≤ 0 := by
        intro a
        have hle := hc a (Finset.mem_univ a)
        rw [← h0] at hle
        exact hle
      have hnp : ∑ a, comp s a ≤ 0 := Finset.sum_nonpos fun a _ => hall a
      linarith
    · have hlt : Real.log (comp s c) ≤ Real.log (1/2) := Real.log_le_log hpos hcon
      have h12 : Real.log (1/2) = -Real.log 2 := by rw [one_div, Real.log_inv]
      rw [h12] at hlt
      linarith
  have hdelhalf : del < 1 / 2 := by
    have hsplit : comp s c + del = 1 := by
      rw [hdel, Finset.add_sum_erase _ _ (Finset.mem_univ c)]
      exact hsum
    linarith
  have hminor : del * Real.log 2 ≤ SeqEnt.H (comp s) :=
    SeqEnt.H_ge_minority hnn hsum hdel hdelhalf.le
  have hdelk : del < kappa := by
    have := lt_of_le_of_lt hminor hlow
    exact lt_of_mul_lt_mul_right (by linarith) hlog2.le
  rw [hdelval] at hdelk
  calc (minorityCount s c : ℝ) = ((minorityCount s c : ℝ) / N) * N := by field_simp
    _ < kappa * N := mul_lt_mul_of_pos_right hdelk hN0

/-! ## The two hardest amendments at once: marginal stability *and* epistatic spikes -/

/-- **Heterogeneity requirement at occupancy `theta`, with spike sites.** -/
theorem functional_needs_minority_with_spikes {beta theta L Smax S0 : ℝ} (hbeta : 0 < beta)
    (hM : 1 < M) (h0 : 0 < theta) (h1 : theta < 1) (hL : 0 ≤ L) (hS : 0 ≤ Smax)
    {Bad : Finset (Fin N)} {E : Seq N q → Fin M → ℝ} (hlip : Spike.SplitLip Bad L Smax E)
    (c : Fin q) (hflat : ∀ j j' : Fin M, E (fun _ => c) j - E (fun _ => c) j' ≤ S0)
    {s : Seq N q} (h : Marginal.Functional theta beta E s) :
    Marginal.gate theta beta M
      ≤ S0 + 2 * L * (minorityCount s c : ℝ) + 2 * Smax * (Bad.card : ℝ) := by
  have hne : (Finset.univ : Finset (Fin M)).Nonempty := ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  obtain ⟨jmax, -, hjmax⟩ := Finset.exists_max_image Finset.univ (E s) hne
  obtain ⟨jmin, -, hjmin⟩ := Finset.exists_min_image Finset.univ (E s) hne
  have hspread := Marginal.functional_needs_spread hbeta hM h0 h1
    (fun j => hjmax j (Finset.mem_univ j)) (fun j => hjmin j (Finset.mem_univ j)) h
  have hd : hdist s (fun _ => c) = minorityCount s c := rfl
  have hbound : ∀ j, |E s j - E (fun _ => c) j|
      ≤ L * (minorityCount s c : ℝ) + Smax * (Bad.card : ℝ) := by
    intro j
    have hw1 := Spike.splitLip_wdist hlip (hdist s (fun _ => c)) s (fun _ => c) rfl j
    have hw2 := Spike.wdist_le (Bad := Bad) (L := L) (Smax := Smax) hL hS s (fun _ => c)
    rw [hd] at hw2
    linarith
  have hA := (abs_le.1 (hbound jmax)).2
  have hB := (abs_le.1 (hbound jmin)).1
  have hC : E (fun _ => c) jmax - E (fun _ => c) jmin ≤ S0 := hflat jmax jmin
  linarith

/-- **The entropy floor, amended twice over.**  A low-complexity tract cannot hold even a
fraction `theta` of its population in one conformation — with the gate reduced by marginal
stability *and* the budget inflated by a sparse set of epistatic spike sites.  This is the
project's central obstruction, stated for the model a structural biologist would accept. -/
theorem disordered_of_low_entropy_marginal_and_spiky {beta theta L Smax S0 kappa : ℝ}
    (hbeta : 0 < beta) (hM : 1 < M) (hN : 0 < N) (h0 : 0 < theta) (h1 : theta < 1)
    (hL : 0 ≤ L) (hS : 0 ≤ Smax) {Bad : Finset (Fin N)} {E : Seq N q → Fin M → ℝ}
    (hlip : Spike.SplitLip Bad L Smax E)
    (hflat : ∀ (c : Fin q) (j j' : Fin M), E (fun _ => c) j - E (fun _ => c) j' ≤ S0)
    (hk : kappa ≤ 1 / 2)
    (hthr : S0 + 2 * L * (kappa * N) + 2 * Smax * (Bad.card : ℝ) < Marginal.gate theta beta M)
    {s : Seq N q} (hlow : SeqEnt.H (comp s) < kappa * Real.log 2) :
    ¬ Marginal.Functional theta beta E s := by
  intro hfun
  obtain ⟨c, hc⟩ := exists_modal_letter_low_minority hN hk hlow
  have hgap := functional_needs_minority_with_spikes hbeta hM h0 h1 hL hS hlip c (hflat c) hfun
  nlinarith [hgap, hc, hthr, hL]

/-! ## The four amendments, in one statement -/

/-- **Four realism amendments to the disorder model.**

1. *Marginal stability is not disorder.*  On a flat four-state landscape the native state holds a
   quarter of the population: functional at occupancy `1/4`, not ordered at `1/2`.  And a native
   state at occupancy `theta` becomes the majority species once it is stabilised by
   `kT·log((1−theta)/theta)`.
2. *Conditional propensities remove entropy.*  A blocky hydrophobic/polar chain has strictly
   positive adjacent-pair mutual information, and over `n` steps the pair model assigns exactly
   `n·I` fewer nats than the composition-matched independent model.
3. *Repeats defeat composition.*  `ATATAT…` has the maximal single-residue entropy `log 2` and a
   `k`-mer entropy that never exceeds `log 2` for any window width.
4. *One mutation can cost everything.*  A salt bridge of depth `J` yields a single substitution
   with `ΔΔG = |J|`, and every additive model misfits the cycle by at least `|J|/4`. -/
theorem four_realism_amendments
    (r n k m : ℕ) (hr : 0 < r) (hm : 0 < m)
    (bta cst : ℝ) (s0 : Seq 4 2)
    {a R beta theta ddG : ℝ} (ha : 0 < a) (hR : 0 < R) (hbeta : 0 < beta)
    (h0 : 0 < theta) (h1 : theta < 1) (hocc : theta ≤ Marginal.occ a R)
    (hddG : Real.log ((1 - theta) / theta) / beta ≤ ddG)
    {J : ℝ} {f : Seq 4 2 → ℝ} (hf : Spike.Additive f) :
    (Marginal.Functional (1/4) bta (fun (_ : Seq 4 2) (_ : Fin 4) => cst) s0
      ∧ ¬ Ordered bta (fun (_ : Seq 4 2) (_ : Fin 4) => cst) s0)
    ∧ 1 / 2 ≤ Marginal.occ (a * Real.exp (beta * ddG)) R
    ∧ (0 < CondSeq.mutualInfo (CondSeq.dimer CondSeq.hpP (CondSeq.hpT (1/4)))
        ∧ ((n : ℝ) + 1) * SeqEnt.H CondSeq.hpP - CondSeq.blockH CondSeq.hpP (CondSeq.hpT (1/4)) n
            = n * ((3/4) * Real.log 3 - Real.log 2))
    ∧ (Kmer.kmerEnt Kmer.alt 1 (2 * r) = Real.log 2 ∧ Kmer.kmerEnt Kmer.alt k m ≤ Real.log 2)
    ∧ (∃ (t : Seq 4 2) (i : Fin 4) (c : Fin 2),
          |Spike.bridge J 0 1 0 1 (Function.update t i c) - Spike.bridge J 0 1 0 1 t| = |J|)
    ∧ (∃ t : Seq 4 2, |J| / 4 ≤ |f t - Spike.bridge J 0 1 0 1 t|) := by
  refine ⟨Marginal.flat_four_functional_not_ordered bta cst s0,
    Marginal.shift_to_majority ha hR hbeta h0 h1 hocc hddG,
    ⟨CondSeq.hp_quarter_mutualInfo_pos, CondSeq.hp_block_deficit n⟩,
    ⟨Kmer.alt_single_entropy hr, Kmer.alt_kmerEnt_le_log_two k hm⟩,
    Spike.bridge_spike (by decide) (by decide),
    Spike.additive_error_ge hf s0 (by decide) (by decide)⟩

end PartXCIII

end IDR
