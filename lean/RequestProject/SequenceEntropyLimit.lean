/-
# Is there a maximum information entropy of a sequence beyond which a region must be disordered?

This file answers that question inside an explicit, physically standard model of a region of a
protein: `q` letters, `N` residues, `M` accessible conformations, a sequence-dependent energy
`E s j`, and the Boltzmann distribution at inverse temperature `beta`.  A region is called
**ordered** when a single conformation carries at least half of the equilibrium population, and
**disordered** otherwise -- the definition already used throughout this project.

The answer has three parts, and they do not all point the same way.

## 1.  The thermodynamic gate (`ordered_native_le`, `ordered_needs_spread`)

Order requires the native conformation to beat *all* the others, so the sequence must generate an
energy spread of at least `kT·log (M-1)`: the conformational entropy has to be paid for in energy.
This is the classical gap criterion, inherited from `IDR.FreeEnergy.folded_needs_entropic_gap`.

## 2.  A **floor**, not a ceiling, on compositional entropy
(`ordered_needs_minority`, `disordered_of_low_composition_entropy`)

The energy spread a sequence can generate is bounded by how far the sequence is from being a
homopolymer.  If a single mutation moves any conformational energy by at most `L`, and the
homopolymers have spread at most `S₀`, then a sequence whose residues are all but `n` copies of one
letter has spread at most `S₀ + 2Ln`.  Hence order *requires* a minimum number of minority
residues, and therefore a minimum compositional Shannon entropy:

`  H₁(s) < κ·log 2  with  κ = (kT·log (M-1) - S₀)/(2LN)  ⟹  the region must be disordered.`

So for a *single* sequence the information-theoretic obstruction to folding is a **lower** entropy
limit: low-complexity regions (poly-Q, poly-G, S/G/P-rich tracts) cannot fold, however favourable
the force field, because they cannot generate contrast.  `ordered_of_far_codewords` is the matching
no-go in the other direction: nothing in the model forbids a *maximally* diverse sequence from
folding, so there is no ceiling on the compositional entropy of one sequence.

## 3.  The **ceiling** is a statement about ensembles, and it is `log |F_N|`
(`entropy_ceiling`, `entropy_ceiling_attained`, `disorder_fraction_ge`)

For a *family* of sequences (a maximum-entropy design ensemble, an evolutionary family) the answer
is sharp and unconditional: with `F` the set of foldable sequences, every ensemble supported on `F`
has entropy at most `log |F|`, the uniform ensemble on `F` attains it, and an ensemble of entropy
`H` must place a fraction at least `(H - log |F| - log 2)/(N log q)` of its weight on disordered
sequences.  `log |F|` *is* the maximum entropy limit; the physics only enters through the value of
`log |F|`.

## 4.  What `log |F|` is, in physical constants
(`log_card_foldable_codeE_le`, `ceilingRate_binds`, `ceilingRate_vacuous`)

In the coding model `E s j = L·d_H(s, w j)` -- the sharpest form of "each conformation is designed
by sequences near a codeword", with `L` the per-mutation energy contrast -- the gate of part 1
confines every foldable sequence to a Hamming ball of radius `r = N(1 - σ/(βL))` around a codeword,
where `σ = (log M)/N` is the conformational entropy per residue.  Sphere packing then gives

`  (1/N)·log |F|  ≤  σ + h₂(1 - σ/(βL)) + (1 - σ/(βL))·log q  =:  ceilingRate.`

`ceilingRate_binds` exhibits a regime where this is strictly below `log q`, so the ceiling really
does bite; `ceilingRate_vacuous` shows that as soon as `σ + (1 - σ/(βL))·log q ≥ log q` -- which
holds for the mutational contrasts measured in folded domains, `βL` of order several `kT` -- the
ceiling exceeds `log q` and imposes nothing.  The honest summary is therefore: the maximum-entropy
limit exists and equals `(1/N) log |F_N|`, it is a bound on sequence *ensembles* rather than on any
one sequence, and in the parameter range of real folded domains it is *not* the operative
constraint; what forces real low-complexity regions to stay disordered is the floor of part 2 and
the gate of part 1.
-/
import Mathlib
import RequestProject.FreeEnergy
import RequestProject.SequenceEntropyCore

set_option autoImplicit false

namespace IDR

namespace SeqLimit

open Finset
open scoped Classical

/-- Sequences of length `N` over an alphabet of `q` letters. -/
abbrev Seq (N q : ℕ) := Fin N → Fin q

variable {N q M : ℕ}

/-! ## Hamming geometry of sequence space -/

/-- The Hamming distance: the number of positions at which two sequences differ. -/
noncomputable def hdist (s t : Seq N q) : ℕ := (Finset.univ.filter fun i => s i ≠ t i).card

lemma hdist_le (s t : Seq N q) : hdist s t ≤ N := by
  simpa using (Finset.card_filter_le (Finset.univ : Finset (Fin N)) fun i => s i ≠ t i)

lemma eq_of_hdist_eq_zero {s t : Seq N q} (h : hdist s t = 0) : s = t := by
  funext i
  by_contra hne
  have hmem : i ∈ (Finset.univ.filter fun k => s k ≠ t k) := by simp [hne]
  have hpos : 0 < hdist s t := Finset.card_pos.2 ⟨i, hmem⟩
  omega

/-- Repairing one mismatched position lowers the Hamming distance by exactly one. -/
lemma hdist_update_succ {s t : Seq N q} {i : Fin N} (hi : s i ≠ t i) :
    hdist (Function.update s i (t i)) t + 1 = hdist s t := by
  have hfil : (Finset.univ.filter fun k => Function.update s i (t i) k ≠ t k)
      = (Finset.univ.filter fun k => s k ≠ t k).erase i := by
    ext k
    by_cases h : k = i
    · subst h; simp
    · simp [h, Finset.mem_erase]
  have hmem : i ∈ (Finset.univ.filter fun k => s k ≠ t k) := by simp [hi]
  simp only [hdist, hfil]
  rw [Finset.card_erase_of_mem hmem]
  have : 0 < (Finset.univ.filter fun k => s k ≠ t k).card := Finset.card_pos.2 ⟨i, hmem⟩
  omega

/-- Changing one residue changes the Hamming distance to any fixed sequence by at most one. -/
lemma hdist_update_le (s t : Seq N q) (i : Fin N) (c : Fin q) :
    hdist (Function.update s i c) t ≤ hdist s t + 1 := by
  have hsub : (Finset.univ.filter fun k => Function.update s i c k ≠ t k)
      ⊆ insert i (Finset.univ.filter fun k => s k ≠ t k) := by
    intro k hk
    by_cases h : k = i
    · subst h; exact Finset.mem_insert_self _ _
    · simp only [Finset.mem_filter, Finset.mem_univ, true_and, Function.update_of_ne h] at hk
      exact Finset.mem_insert_of_mem (by simp [hk])
  calc hdist (Function.update s i c) t
      ≤ (insert i (Finset.univ.filter fun k => s k ≠ t k)).card := Finset.card_le_card hsub
    _ ≤ (Finset.univ.filter fun k => s k ≠ t k).card + 1 := Finset.card_insert_le _ _

/-! ## The model: sequences, conformations, Boltzmann order -/

/-- **Order**: a single conformation carries at least half of the equilibrium population.  Its
negation is the operational definition of an intrinsically disordered region used throughout
this project. -/
def Ordered (beta : ℝ) (E : Seq N q → Fin M → ℝ) (s : Seq N q) : Prop :=
  ∃ j, 1 / 2 ≤ FreeEnergy.boltz beta (E s) j

/-- **The thermodynamic gate.**  If the region is ordered then some conformation lies at least
`kT·log (M-1)` below the top of the sequence's own energy spectrum: the conformational entropy
must be paid for in energy. -/
theorem ordered_native_le {beta : ℝ} (hbeta : 0 < beta) (hM : 1 < M)
    {E : Seq N q → Fin M → ℝ} {s : Seq N q} {Emax : ℝ} (hmax : ∀ j, E s j ≤ Emax)
    (h : Ordered beta E s) :
    ∃ j0, E s j0 + Real.log ((M : ℝ) - 1) / beta ≤ Emax := by
  obtain ⟨j0, hj0⟩ := h
  refine ⟨j0, ?_⟩
  set D : Finset (Fin M) := Finset.univ.erase j0 with hD
  have hcard : D.card = M - 1 := by
    rw [hD, Finset.card_erase_of_mem (Finset.mem_univ _)]
    simp
  have hDne : D.Nonempty := by
    rw [← Finset.card_pos, hcard]; omega
  have hj0D : j0 ∉ D := by simp [hD]
  have hgap := FreeEnergy.folded_needs_entropic_gap hbeta (by omega : 0 < M) (E s) j0 D hDne hj0D
    Emax (fun j _ => hmax j) hj0
  rw [hcard] at hgap
  have hcast : ((M - 1 : ℕ) : ℝ) = (M : ℝ) - 1 := by
    have : (1 : ℕ) ≤ M := by omega
    simpa using (Nat.cast_sub this : ((M - 1 : ℕ) : ℝ) = (M : ℝ) - (1 : ℕ))
  rw [hcast] at hgap
  linarith

/-- **The gap criterion in spread form.**  An ordered sequence must generate an energy spread of
at least `kT·log (M-1)` across the conformational library. -/
theorem ordered_needs_spread {beta : ℝ} (hbeta : 0 < beta) (hM : 1 < M)
    {E : Seq N q → Fin M → ℝ} {s : Seq N q} {Emax Emin : ℝ} (hmax : ∀ j, E s j ≤ Emax)
    (hmin : ∀ j, Emin ≤ E s j) (h : Ordered beta E s) :
    Real.log ((M : ℝ) - 1) / beta ≤ Emax - Emin := by
  obtain ⟨j0, hj0⟩ := ordered_native_le hbeta hM hmax h
  linarith [hmin j0]

/-- **The verdict form**: too little energetic contrast, and the region *cannot* be ordered, for
any sequence and any force field with that contrast. -/
theorem disordered_of_small_spread {beta : ℝ} (hbeta : 0 < beta) (hM : 1 < M)
    {E : Seq N q → Fin M → ℝ} {s : Seq N q} {Emax Emin : ℝ} (hmax : ∀ j, E s j ≤ Emax)
    (hmin : ∀ j, Emin ≤ E s j) (hlt : Emax - Emin < Real.log ((M : ℝ) - 1) / beta) :
    ¬ Ordered beta E s := fun h =>
  absurd (ordered_needs_spread hbeta hM hmax hmin h) (not_le.2 hlt)

/-! ## Mutational sensitivity: how much contrast a sequence can generate -/

/-- **Site-Lipschitz energies**: a single substitution moves every conformational energy by at
most `L`.  For a pairwise contact energy with bounded coordination number this holds with
`L = 2·z·‖e‖∞`. -/
def SiteLip (L : ℝ) (E : Seq N q → Fin M → ℝ) : Prop :=
  ∀ (s : Seq N q) (i : Fin N) (c : Fin q) (j : Fin M), |E (Function.update s i c) j - E s j| ≤ L

/-- Site-Lipschitz energies are Lipschitz in the Hamming metric. -/
theorem siteLip_hdist {L : ℝ} {E : Seq N q → Fin M → ℝ} (hlip : SiteLip L E) :
    ∀ (n : ℕ) (s t : Seq N q), hdist s t = n → ∀ j, |E s j - E t j| ≤ L * n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro s t hn j
    rcases Nat.eq_zero_or_pos n with rfl | hpos
    · have hst : s = t := eq_of_hdist_eq_zero hn
      subst hst
      simp
    · have hdpos : 0 < hdist s t := by omega
      obtain ⟨i, hi⟩ : ∃ i, s i ≠ t i := by
        obtain ⟨i, hmem⟩ := Finset.card_pos.1 hdpos
        exact ⟨i, by simpa using hmem⟩
      set s' : Seq N q := Function.update s i (t i) with hs'
      have hstep : hdist s' t + 1 = hdist s t := hdist_update_succ hi
      have hm : hdist s' t = n - 1 := by omega
      have h1 : |E s' j - E s j| ≤ L := hlip s i (t i) j
      have h1' : |E s j - E s' j| ≤ L := by rwa [abs_sub_comm]
      have h2 : |E s' j - E t j| ≤ L * ((n - 1 : ℕ) : ℝ) := ih (n - 1) (by omega) s' t hm j
      have habs : |E s j - E t j| ≤ |E s j - E s' j| + |E s' j - E t j| := abs_sub_le _ _ _
      have hcast : ((n - 1 : ℕ) : ℝ) = (n : ℝ) - 1 := by
        have h1n : (1 : ℕ) ≤ n := hpos
        simpa using (Nat.cast_sub h1n : ((n - 1 : ℕ) : ℝ) = (n : ℝ) - (1 : ℕ))
      rw [hcast] at h2
      have hn1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hpos
      nlinarith [habs, h1', h2]

/-- The number of residues of `s` that are not the letter `c`. -/
noncomputable def minorityCount (s : Seq N q) (c : Fin q) : ℕ :=
  (Finset.univ.filter fun i => s i ≠ c).card

lemma minorityCount_eq_hdist (s : Seq N q) (c : Fin q) :
    minorityCount s c = hdist s (fun _ => c) := rfl

/-- **Order requires heterogeneity.**  With site-Lipschitz constant `L` and homopolymer spread at
most `S₀`, an ordered sequence must carry at least `(kT·log (M-1) - S₀)/(2L)` residues that differ
from any given letter. -/
theorem ordered_needs_minority {beta L S0 : ℝ} (hbeta : 0 < beta) (hM : 1 < M)
    {E : Seq N q → Fin M → ℝ} (hlip : SiteLip L E) (c : Fin q)
    (hflat : ∀ j j' : Fin M, E (fun _ => c) j - E (fun _ => c) j' ≤ S0)
    {s : Seq N q} (h : Ordered beta E s) :
    Real.log ((M : ℝ) - 1) / beta ≤ S0 + 2 * L * (minorityCount s c : ℝ) := by
  classical
  have hne : (Finset.univ : Finset (Fin M)).Nonempty := ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  obtain ⟨jmax, -, hjmax⟩ := Finset.exists_max_image Finset.univ (E s) hne
  obtain ⟨jmin, -, hjmin⟩ := Finset.exists_min_image Finset.univ (E s) hne
  have hspread := ordered_needs_spread hbeta hM (fun j => hjmax j (Finset.mem_univ j))
    (fun j => hjmin j (Finset.mem_univ j)) h
  -- compare with the homopolymer
  have hd : hdist s (fun _ => c) = minorityCount s c := rfl
  have hlipmax : |E s jmax - E (fun _ => c) jmax| ≤ L * (minorityCount s c : ℝ) := by
    have := siteLip_hdist hlip (hdist s (fun _ => c)) s (fun _ => c) rfl jmax
    rwa [hd] at this
  have hlipmin : |E s jmin - E (fun _ => c) jmin| ≤ L * (minorityCount s c : ℝ) := by
    have := siteLip_hdist hlip (hdist s (fun _ => c)) s (fun _ => c) rfl jmin
    rwa [hd] at this
  have h1 : E s jmax - E (fun _ => c) jmax ≤ L * (minorityCount s c : ℝ) :=
    (abs_le.1 hlipmax).2
  have h2 : -(L * (minorityCount s c : ℝ)) ≤ E s jmin - E (fun _ => c) jmin :=
    (abs_le.1 hlipmin).1
  have h3 : E (fun _ => c) jmax - E (fun _ => c) jmin ≤ S0 := hflat jmax jmin
  linarith

/-! ## Compositional entropy of a single sequence -/

/-- The composition of a sequence: the empirical frequency of each letter. -/
noncomputable def comp (s : Seq N q) : Fin q → ℝ :=
  fun a => ((Finset.univ.filter fun i => s i = a).card : ℝ) / N

lemma comp_nonneg (s : Seq N q) (a : Fin q) : 0 ≤ comp s a := by
  unfold comp; positivity

lemma comp_sum_one (hN : 0 < N) (s : Seq N q) : ∑ a, comp s a = 1 := by
  classical
  have hfib : (Finset.univ : Finset (Fin N)).card
      = ∑ a ∈ (Finset.univ : Finset (Fin q)), (Finset.univ.filter fun i => s i = a).card :=
    Finset.card_eq_sum_card_fiberwise (fun x _ => Finset.mem_univ (s x))
  have hN0 : (N : ℝ) ≠ 0 := by positivity
  unfold comp
  rw [← Finset.sum_div]
  have : ∑ a, ((Finset.univ.filter fun i => s i = a).card : ℝ) = (N : ℝ) := by
    have := congrArg (fun k : ℕ => (k : ℝ)) hfib
    push_cast at this
    simpa using this.symm
  rw [this]
  field_simp

/-- The weight the composition puts off a letter `c` is the minority fraction. -/
lemma sum_comp_erase (hN : 0 < N) (s : Seq N q) (c : Fin q) :
    ∑ a ∈ Finset.univ.erase c, comp s a = (minorityCount s c : ℝ) / N := by
  classical
  have hsum := comp_sum_one hN s
  have hsplit : ∑ a, comp s a = comp s c + ∑ a ∈ Finset.univ.erase c, comp s a := by
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ c)]
  have hcount : (Finset.univ.filter fun i => s i = c).card + minorityCount s c = N := by
    have hc : (Finset.univ.filter fun i => s i = c).card
        + (Finset.univ.filter fun i => ¬ (s i = c)).card
        = (Finset.univ : Finset (Fin N)).card :=
      Finset.card_filter_add_card_filter_not _
    simpa [minorityCount] using hc
  have hN0 : (N : ℝ) ≠ 0 := by positivity
  have : comp s c = 1 - (minorityCount s c : ℝ) / N := by
    unfold comp
    have : ((Finset.univ.filter fun i => s i = c).card : ℝ) = (N : ℝ) - (minorityCount s c : ℝ) := by
      have := congrArg (fun k : ℕ => (k : ℝ)) hcount
      push_cast at this
      linarith
    rw [this]
    field_simp
  linarith [hsum, hsplit, this]

/-- **A minimum entropy is required to fold.**  If the compositional Shannon entropy of the
sequence is below `κ·log 2`, where `κ` is the minority fraction demanded by the gap criterion,
then the region must be disordered -- whatever the force field, at any temperature.  This is the
low-complexity rule (poly-Q, poly-G, S/G-rich tracts) as a theorem. -/
theorem disordered_of_low_composition_entropy {beta L S0 kappa : ℝ} (hbeta : 0 < beta)
    (hM : 1 < M) (hN : 0 < N) {E : Seq N q → Fin M → ℝ} (hlip : SiteLip L E)
    (hflat : ∀ (c : Fin q) (j j' : Fin M), E (fun _ => c) j - E (fun _ => c) j' ≤ S0)
    (hk0 : 0 < kappa) (hk : kappa ≤ 1 / 2)
    (hthr : S0 + 2 * L * (kappa * N) < Real.log ((M : ℝ) - 1) / beta)
    {s : Seq N q} (hlow : SeqEnt.H (comp s) < kappa * Real.log 2) :
    ¬ Ordered beta E s := by
  classical
  intro hord
  have hMpos : 0 < M := by omega
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  -- the Lipschitz constant of a nonempty model is nonnegative
  have hL : 0 ≤ L := by
    have h0 := hlip s ⟨0, hN⟩ (s ⟨0, hN⟩) ⟨0, hMpos⟩
    rw [Function.update_eq_self] at h0
    simpa using h0
  have hnn := comp_nonneg s
  have hsum := comp_sum_one hN s
  -- the modal letter
  have hqpos : 0 < q := Fin.pos_iff_nonempty.2 ⟨s ⟨0, hN⟩⟩
  obtain ⟨c, -, hc⟩ := Finset.exists_max_image Finset.univ (comp s) ⟨⟨0, hqpos⟩,
    Finset.mem_univ _⟩
  set del : ℝ := ∑ a ∈ Finset.univ.erase c, comp s a with hdel
  have hdelval : del = (minorityCount s c : ℝ) / N := by rw [hdel, sum_comp_erase hN s c]
  have hdel0 : 0 ≤ del := by
    rw [hdelval]; positivity
  -- step 1: the modal frequency exceeds one half, so the minority weight is below one half
  have hmin_entropy : -Real.log (comp s c) ≤ SeqEnt.H (comp s) :=
    SeqEnt.H_ge_neg_log_max hnn hsum (fun a => hc a (Finset.mem_univ a))
  have hlog2 : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  have hHlt : SeqEnt.H (comp s) < Real.log 2 := by nlinarith
  have hcpos : 1 / 2 < comp s c := by
    by_contra hcon
    push_neg at hcon
    have hcnn : 0 ≤ comp s c := hnn c
    rcases eq_or_lt_of_le hcnn with h0 | hpos
    · -- all frequencies vanish, impossible since they sum to one
      have hall : ∀ a, comp s a ≤ 0 := by
        intro a
        have hle := hc a (Finset.mem_univ a)
        rw [← h0] at hle
        exact hle
      have hnp : ∑ a, comp s a ≤ 0 := Finset.sum_nonpos fun a _ => hall a
      linarith
    · have : Real.log (comp s c) ≤ Real.log (1/2) := Real.log_le_log hpos hcon
      have h12 : Real.log (1/2) = -Real.log 2 := by rw [one_div, Real.log_inv]
      rw [h12] at this
      linarith
  have hdelhalf : del < 1 / 2 := by
    have hsplit : comp s c + del = 1 := by
      rw [hdel, Finset.add_sum_erase _ _ (Finset.mem_univ c)]
      exact hsum
    linarith
  -- step 2: the minority bound turns the low entropy into few minority residues
  have hminor : del * Real.log 2 ≤ SeqEnt.H (comp s) :=
    SeqEnt.H_ge_minority hnn hsum hdel hdelhalf.le
  have hdelk : del < kappa := by
    have := lt_of_le_of_lt hminor hlow
    exact lt_of_mul_lt_mul_right (by linarith) hlog2.le
  have hcount : (minorityCount s c : ℝ) < kappa * N := by
    rw [hdelval] at hdelk
    calc (minorityCount s c : ℝ) = ((minorityCount s c : ℝ) / N) * N := by field_simp
      _ < kappa * N := by exact (mul_lt_mul_of_pos_right hdelk hN0)
  -- step 3: contradiction with the gap criterion
  have hgap := ordered_needs_minority hbeta hM hlip c (hflat c) hord
  nlinarith [hgap, hcount, hthr, hL]

/-! ## The ceiling: entropy of an ensemble of sequences -/

/-- The foldable set: the sequences that order at inverse temperature `beta`. -/
noncomputable def foldable (beta : ℝ) (E : Seq N q → Fin M → ℝ) : Finset (Seq N q) :=
  Finset.univ.filter fun s => Ordered beta E s

/-- **The entropy ceiling.**  Every ensemble of sequences that folds with certainty has Shannon
entropy at most `log |F|`. -/
theorem entropy_ceiling {beta : ℝ} {E : Seq N q → Fin M → ℝ} {p : Seq N q → ℝ}
    (hp : ∀ s, 0 ≤ p s) (hps : ∑ s, p s = 1)
    (hsupp : ∀ s, ¬ Ordered beta E s → p s = 0)
    (hne : (foldable beta E).Nonempty) :
    SeqEnt.H p ≤ Real.log (foldable beta E).card := by
  classical
  refine SeqEnt.H_le_log_card hp hps (fun s hs => hsupp s ?_) hne
  intro hord
  exact hs (by simp [foldable, hord])

/-- **And it is attained**: the uniform ensemble on the foldable set has entropy exactly
`log |F|`.  So `log |F|` is the maximum entropy compatible with order, not merely an upper
bound. -/
theorem entropy_ceiling_attained {beta : ℝ} {E : Seq N q → Fin M → ℝ}
    (hne : (foldable beta E).Nonempty) :
    SeqEnt.H (fun s => if s ∈ foldable beta E then (((foldable beta E).card : ℝ))⁻¹ else 0)
      = Real.log (foldable beta E).card :=
  SeqEnt.H_uniform hne

/-- **The disorder fraction of an over-entropic ensemble.**  An ensemble of sequences whose
entropy exceeds the ceiling must place a proportionate fraction of its weight on disordered
sequences: at least `(H - log |F| - log 2)/(N log q)`. -/
theorem disorder_fraction_ge {beta : ℝ} {E : Seq N q → Fin M → ℝ} {p : Seq N q → ℝ} {eps : ℝ}
    (hq : 1 < q) (hN : 0 < N) (hp : ∀ s, 0 ≤ p s) (hps : ∑ s, p s = 1)
    (hne : (foldable beta E).Nonempty) (hnec : (foldable beta E)ᶜ.Nonempty)
    (heps : eps = ∑ s ∈ (foldable beta E)ᶜ, p s) (h0 : 0 < eps) (h1 : eps < 1) :
    (SeqEnt.H p - Real.log (foldable beta E).card - Real.log 2) / ((N : ℝ) * Real.log q)
      ≤ eps := by
  classical
  have hqR : (1 : ℝ) < q := by exact_mod_cast hq
  have hlogq : 0 < Real.log q := Real.log_pos hqR
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hcardtot : ((foldable beta E)ᶜ).card ≤ q ^ N := by
    have h1 : ((foldable beta E)ᶜ).card ≤ Fintype.card (Seq N q) := Finset.card_le_univ _
    simpa [Fintype.card_fun] using h1
  have hC : Real.log ((foldable beta E)ᶜ).card ≤ (N : ℝ) * Real.log q := by
    have hpos : (0 : ℝ) < ((q : ℝ)) ^ N := by positivity
    have : ((((foldable beta E)ᶜ).card : ℝ)) ≤ ((q : ℝ)) ^ N := by
      exact_mod_cast hcardtot
    calc Real.log ((foldable beta E)ᶜ).card ≤ Real.log (((q : ℝ)) ^ N) :=
          Real.log_le_log (by positivity) this
      _ = (N : ℝ) * Real.log q := by rw [Real.log_pow]
  have hFcard : 1 ≤ (foldable beta E).card := Finset.card_pos.2 hne
  exact SeqEnt.escape_prob_ge hp hps hne hnec heps h0 h1 hC hFcard (by positivity)

/-! ## Evaluating `log |F|`: the coding model -/

/-- The coding energy model: conformation `j` is designed by the codeword `w j`, and the energy
cost is `L` per mismatched residue.  This is the sharpest possible form of "each conformation is
encoded by the sequences near a codeword": the native energy is minimal exactly at the codeword,
and `L` is the per-mutation energy contrast. -/
noncomputable def codeE (L : ℝ) (w : Fin M → Seq N q) : Seq N q → Fin M → ℝ :=
  fun s j => L * (hdist s (w j) : ℝ)

lemma codeE_siteLip {L : ℝ} (hL : 0 ≤ L) (w : Fin M → Seq N q) : SiteLip L (codeE L w) := by
  intro s i c j
  have h1 : hdist (Function.update s i c) (w j) ≤ hdist s (w j) + 1 :=
    hdist_update_le s (w j) i c
  have h2 : hdist s (w j) ≤ hdist (Function.update s i c) (w j) + 1 := by
    have : Function.update (Function.update s i c) i (s i) = s := by
      funext k
      by_cases h : k = i
      · subst h; simp
      · simp [Function.update_of_ne h]
    calc hdist s (w j)
        = hdist (Function.update (Function.update s i c) i (s i)) (w j) := by rw [this]
      _ ≤ hdist (Function.update s i c) (w j) + 1 := hdist_update_le _ _ i (s i)
  have h1' : ((hdist (Function.update s i c) (w j) : ℝ)) ≤ (hdist s (w j) : ℝ) + 1 := by
    exact_mod_cast h1
  have h2' : ((hdist s (w j) : ℝ)) ≤ (hdist (Function.update s i c) (w j) : ℝ) + 1 := by
    exact_mod_cast h2
  simp only [codeE]
  rw [abs_le]
  constructor <;> nlinarith [h1', h2', hL]

/-- **The gate confines foldable sequences to Hamming balls.**  In the coding model an ordered
sequence lies within `N - kT·log (M-1)/L` mutations of the codeword it folds to. -/
theorem ordered_codeE_near {beta L : ℝ} (hbeta : 0 < beta) (hL : 0 < L) (hM : 1 < M)
    {w : Fin M → Seq N q} {s : Seq N q} (h : Ordered beta (codeE L w) s) :
    ∃ j, (hdist s (w j) : ℝ) ≤ (N : ℝ) - Real.log ((M : ℝ) - 1) / (beta * L) := by
  have hmax : ∀ j, codeE L w s j ≤ L * (N : ℝ) := by
    intro j
    have : (hdist s (w j) : ℝ) ≤ (N : ℝ) := by exact_mod_cast hdist_le s (w j)
    simp only [codeE]
    nlinarith
  obtain ⟨j0, hj0⟩ := ordered_native_le hbeta hM hmax h
  refine ⟨j0, ?_⟩
  simp only [codeE] at hj0
  have hkey : L * (hdist s (w j0) : ℝ)
      ≤ L * ((N : ℝ) - Real.log ((M : ℝ) - 1) / (beta * L)) := by
    have hid : L * ((N : ℝ) - Real.log ((M : ℝ) - 1) / (beta * L))
        = L * (N : ℝ) - Real.log ((M : ℝ) - 1) / beta := by
      field_simp
    rw [hid]
    linarith
  exact le_of_mul_le_mul_left hkey hL

/-- The Hamming ball of radius `r`, as a finite set of sequences. -/
noncomputable def ball (t : Seq N q) (r : ℕ) : Finset (Seq N q) :=
  Finset.univ.filter fun s => hdist s t ≤ r

/-- **Sphere packing.**  A Hamming ball of radius `r ≤ N` contains at most `C(N,r)·q^r`
sequences. -/
theorem card_ball_le (t : Seq N q) {r : ℕ} (hr : r ≤ N) :
    (ball t r).card ≤ N.choose r * q ^ r := by
  classical
  have hsub : ball t r ⊆ (Finset.univ.powersetCard r).biUnion
      fun A => Fintype.piFinset fun i => if i ∈ A then Finset.univ else {t i} := by
    intro s hs
    simp only [ball, Finset.mem_filter, Finset.mem_univ, true_and] at hs
    set D : Finset (Fin N) := Finset.univ.filter fun i => s i ≠ t i with hDdef
    have hDcard : D.card ≤ r := hs
    obtain ⟨A, hDA, -, hAcard⟩ :=
      Finset.exists_subsuperset_card_eq (Finset.subset_univ D) hDcard (by simpa using hr)
    refine Finset.mem_biUnion.2 ⟨A, ?_, ?_⟩
    · simp [Finset.mem_powersetCard, hAcard]
    · refine Fintype.mem_piFinset.2 fun i => ?_
      by_cases h : i ∈ A
      · simp [h]
      · have : s i = t i := by
          by_contra hne
          exact h (hDA (by simp [hDdef, hne]))
        simp [h, this]
  calc (ball t r).card
      ≤ ∑ A ∈ Finset.univ.powersetCard r,
          (Fintype.piFinset fun i => if i ∈ A then (Finset.univ : Finset (Fin q)) else {t i}).card :=
        le_trans (Finset.card_le_card hsub) (Finset.card_biUnion_le)
    _ = ∑ A ∈ Finset.univ.powersetCard r, q ^ r := by
        refine Finset.sum_congr rfl fun A hA => ?_
        have hAcard : A.card = r := (Finset.mem_powersetCard.1 hA).2
        have hcards : ∀ i : Fin N,
            (if i ∈ A then (Finset.univ : Finset (Fin q)) else {t i}).card
              = if i ∈ A then q else 1 := by
          intro i; by_cases h : i ∈ A <;> simp [h]
        rw [Fintype.card_piFinset, Finset.prod_congr rfl (fun i _ => hcards i),
          Finset.prod_ite_mem, Finset.univ_inter, Finset.prod_const, hAcard]
    _ = N.choose r * q ^ r := by
        rw [Finset.sum_const, Finset.card_powersetCard]
        simp [Fintype.card_fin, mul_comm]

/-- **The foldable set is covered by `M` Hamming balls.** -/
theorem card_foldable_codeE_le {beta L : ℝ} (hbeta : 0 < beta) (hL : 0 < L) (hM : 1 < M)
    {w : Fin M → Seq N q} {r : ℕ} (hr : r ≤ N)
    (hrad : (N : ℝ) - Real.log ((M : ℝ) - 1) / (beta * L) ≤ r) :
    (foldable beta (codeE L w)).card ≤ M * (N.choose r * q ^ r) := by
  classical
  have hsub : foldable beta (codeE L w) ⊆ Finset.univ.biUnion fun j => ball (w j) r := by
    intro s hs
    simp only [foldable, Finset.mem_filter, Finset.mem_univ, true_and] at hs
    obtain ⟨j, hj⟩ := ordered_codeE_near hbeta hL hM hs
    refine Finset.mem_biUnion.2 ⟨j, Finset.mem_univ _, ?_⟩
    simp only [ball, Finset.mem_filter, Finset.mem_univ, true_and]
    have : (hdist s (w j) : ℝ) ≤ (r : ℝ) := le_trans hj hrad
    exact_mod_cast this
  calc (foldable beta (codeE L w)).card
      ≤ ∑ _j : Fin M, (N.choose r * q ^ r) := by
        refine le_trans (Finset.card_le_card hsub) (le_trans Finset.card_biUnion_le ?_)
        exact Finset.sum_le_sum fun j _ => card_ball_le (w j) hr
    _ = M * (N.choose r * q ^ r) := by simp [mul_comm]

/-- The per-residue entropy ceiling predicted by the coding model:
`σ + h₂(x) + x·log q` with `σ = (log M)/N` the conformational entropy per residue and
`x = r/N = 1 - σ/(βL)` the tolerated mutation fraction. -/
noncomputable def ceilingRate (M N q r : ℕ) : ℝ :=
  Real.log M / N + SeqEnt.h₂ ((r : ℝ) / N) + ((r : ℝ) / N) * Real.log q

/-- **The maximum sequence entropy of a foldable ensemble, in physical constants.**
`log |F| ≤ log M + N·h₂(r/N) + r·log q`, i.e. the per-residue entropy of any ensemble that folds
with certainty is at most `ceilingRate`. -/
theorem log_card_foldable_codeE_le {beta L : ℝ} (hbeta : 0 < beta) (hL : 0 < L) (hM : 1 < M)
    (hN : 0 < N) (hq : 0 < q) {w : Fin M → Seq N q} {r : ℕ} (hr0 : 0 < r) (hrN : r < N)
    (hrad : (N : ℝ) - Real.log ((M : ℝ) - 1) / (beta * L) ≤ r)
    (hne : (foldable beta (codeE L w)).Nonempty) :
    Real.log (foldable beta (codeE L w)).card ≤ (N : ℝ) * ceilingRate M N q r := by
  classical
  have hcard := card_foldable_codeE_le hbeta hL hM (le_of_lt hrN) hrad (w := w)
  have hpos : (0 : ℝ) < (M : ℝ) * ((N.choose r : ℝ) * (q : ℝ) ^ r) := by
    have h1 : (0 : ℝ) < (M : ℝ) := by positivity
    have h2 : (0 : ℝ) < (N.choose r : ℝ) := by
      have : 0 < N.choose r := Nat.choose_pos (le_of_lt hrN)
      exact_mod_cast this
    have h3 : (0 : ℝ) < (q : ℝ) ^ r := by
      have : (0 : ℝ) < (q : ℝ) := by exact_mod_cast hq
      positivity
    positivity
  have hle : ((foldable beta (codeE L w)).card : ℝ) ≤ (M : ℝ) * ((N.choose r : ℝ) * (q : ℝ) ^ r) := by
    exact_mod_cast hcard
  have hlog : Real.log (foldable beta (codeE L w)).card
      ≤ Real.log ((M : ℝ) * ((N.choose r : ℝ) * (q : ℝ) ^ r)) := by
    have hposF : (0 : ℝ) < ((foldable beta (codeE L w)).card : ℝ) := by
      have : 0 < (foldable beta (codeE L w)).card := Finset.card_pos.2 hne
      exact_mod_cast this
    exact Real.log_le_log hposF hle
  have hM0 : (0 : ℝ) < (M : ℝ) := by positivity
  have hchoose : (0 : ℝ) < (N.choose r : ℝ) := by
    have : 0 < N.choose r := Nat.choose_pos (le_of_lt hrN)
    exact_mod_cast this
  have hq0 : (0 : ℝ) < (q : ℝ) := by exact_mod_cast hq
  have hsplit : Real.log ((M : ℝ) * ((N.choose r : ℝ) * (q : ℝ) ^ r))
      = Real.log M + Real.log (N.choose r) + (r : ℝ) * Real.log q := by
    rw [Real.log_mul (ne_of_gt hM0) (by positivity), Real.log_mul (ne_of_gt hchoose) (by positivity),
      Real.log_pow]
    ring
  have hbinom : Real.log (N.choose r) ≤ (N : ℝ) * SeqEnt.h₂ ((r : ℝ) / N) :=
    SeqEnt.log_choose_le hN hr0 hrN
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hrate : (N : ℝ) * ceilingRate M N q r
      = Real.log M + (N : ℝ) * SeqEnt.h₂ ((r : ℝ) / N) + (r : ℝ) * Real.log q := by
    unfold ceilingRate
    field_simp
  rw [hrate]
  linarith [hlog, hsplit.le, hsplit.ge, hbinom]

/-! ## Does the ceiling bite?  The two regimes -/

/-- **A regime where the ceiling binds.**  With a 20-letter alphabet, two conformations per
residue (`σ = log 2`) and a mutational contrast so weak that the gate tolerates a quarter of the
residues being wrong (`r = N/4`, i.e. `βL = (4/3)·log 2 ≈ 0.9 kT`), the per-residue entropy of any
ensemble that folds with certainty is strictly below `log 20`: the sequence-entropy ceiling is a
real constraint, and a design ensemble above it must contain disordered sequences. -/
theorem ceilingRate_binds {N r : ℕ} (hN : 0 < N) (hr : 4 * r ≤ N) (hr0 : 0 < r) :
    ceilingRate (2 ^ N) N 20 r < Real.log 20 := by
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hlog2 : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  have hx : (r : ℝ) / N ≤ 1 / 4 := by
    rw [div_le_iff₀ hN0]
    have h4 : (4 : ℝ) * r ≤ N := by exact_mod_cast hr
    linarith
  have hx0 : 0 ≤ (r : ℝ) / N := by positivity
  have hx1 : (r : ℝ) / N ≤ 1 := le_trans hx (by norm_num)
  have hlogM : Real.log ((2 : ℕ) ^ N : ℕ) = (N : ℝ) * Real.log 2 := by
    push_cast
    rw [Real.log_pow]
  have hsigma : Real.log ((2 : ℕ) ^ N : ℕ) / (N : ℝ) = Real.log 2 := by
    rw [hlogM]; field_simp
  have hh2 : SeqEnt.h₂ ((r : ℝ) / N) ≤ Real.log 2 := SeqEnt.h₂_le_log_two hx0 hx1
  have hlog20 : 4 * Real.log 2 ≤ Real.log 20 := by
    have h16 : Real.log 16 ≤ Real.log 20 := Real.log_le_log (by norm_num) (by norm_num)
    have : Real.log 16 = 4 * Real.log 2 := by
      rw [show (16 : ℝ) = 2 ^ (4 : ℕ) by norm_num, Real.log_pow]
      push_cast; ring
    linarith
  have hlog20pos : (0 : ℝ) < Real.log 20 := by linarith
  have hxlog : ((r : ℝ) / N) * Real.log 20 ≤ (1 / 4) * Real.log 20 :=
    mul_le_mul_of_nonneg_right hx hlog20pos.le
  unfold ceilingRate
  rw [hsigma]
  linarith

/-- **And a regime where it does not.**  As soon as `σ + x·log q ≥ log q` -- which happens for
`x = 1 - σ/(βL)` whenever the per-mutation contrast `βL` is a few `kT`, the value measured in
folded domains -- the ceiling exceeds `log q` and constrains nothing beyond the alphabet itself.
The entropy ceiling is therefore *not* what makes real regions disordered. -/
theorem ceilingRate_vacuous {M N q r : ℕ} (hx0 : 0 ≤ (r : ℝ) / N)
    (hx1 : (r : ℝ) / N ≤ 1)
    (hweak : Real.log q ≤ Real.log M / N + ((r : ℝ) / N) * Real.log q) :
    Real.log q ≤ ceilingRate M N q r := by
  have := SeqEnt.h₂_nonneg hx0 hx1
  unfold ceilingRate
  linarith

/-! ## No ceiling on the composition of a single sequence -/

/-- **The no-go for the naive reading.**  Nothing about the composition of a sequence -- let alone
its compositional entropy -- can certify disorder from above: in the coding model *any* sequence,
including one of maximal compositional entropy, is ordered as soon as the competing conformations
are far enough away in sequence space.  The ceiling of part 3 is a statement about ensembles, and
cannot be demoted to a statement about one sequence. -/
theorem ordered_of_far_codewords {beta L d : ℝ} (hbeta : 0 < beta) (hM : 0 < M)
    {w : Fin M → Seq N q} {j0 : Fin M} (hfar : ∀ j, j ≠ j0 → d ≤ (hdist (w j0) (w j) : ℝ))
    (hd : ((M : ℝ) - 1) * Real.exp (-(beta * L * d)) ≤ 1) (hL : 0 ≤ L) :
    Ordered beta (codeE L w) (w j0) := by
  classical
  refine ⟨j0, ?_⟩
  have hzero : codeE L w (w j0) j0 = 0 := by
    simp [codeE, hdist]
  have hpart : FreeEnergy.part beta (codeE L w (w j0)) ≤ 2 := by
    have hsplit : FreeEnergy.part beta (codeE L w (w j0))
        = Real.exp (-beta * codeE L w (w j0) j0)
          + ∑ j ∈ Finset.univ.erase j0, Real.exp (-beta * codeE L w (w j0) j) := by
      rw [FreeEnergy.part, ← Finset.add_sum_erase _ _ (Finset.mem_univ j0)]
    have hterm : ∀ j ∈ Finset.univ.erase j0,
        Real.exp (-beta * codeE L w (w j0) j) ≤ Real.exp (-(beta * L * d)) := by
      intro j hj
      have hjne : j ≠ j0 := Finset.ne_of_mem_erase hj
      have : d ≤ (hdist (w j0) (w j) : ℝ) := hfar j hjne
      apply Real.exp_le_exp.2
      simp only [codeE]
      have hbl : 0 ≤ beta * L := mul_nonneg hbeta.le hL
      nlinarith [mul_le_mul_of_nonneg_left this hbl]
    have hbound : ∑ j ∈ Finset.univ.erase j0, Real.exp (-beta * codeE L w (w j0) j)
        ≤ ((M : ℝ) - 1) * Real.exp (-(beta * L * d)) := by
      have hcard : (Finset.univ.erase j0).card = M - 1 := by
        rw [Finset.card_erase_of_mem (Finset.mem_univ _)]; simp
      calc ∑ j ∈ Finset.univ.erase j0, Real.exp (-beta * codeE L w (w j0) j)
          ≤ ∑ _j ∈ Finset.univ.erase j0, Real.exp (-(beta * L * d)) := Finset.sum_le_sum hterm
        _ = ((M - 1 : ℕ) : ℝ) * Real.exp (-(beta * L * d)) := by
            rw [Finset.sum_const, hcard, nsmul_eq_mul]
        _ = ((M : ℝ) - 1) * Real.exp (-(beta * L * d)) := by
            have : (1 : ℕ) ≤ M := hM
            rw [show ((M - 1 : ℕ) : ℝ) = (M : ℝ) - 1 by
              simpa using (Nat.cast_sub this : ((M - 1 : ℕ) : ℝ) = (M : ℝ) - (1 : ℕ))]
    rw [hsplit, hzero]
    simp only [mul_zero, Real.exp_zero]
    linarith
  have hZpos : 0 < FreeEnergy.part beta (codeE L w (w j0)) := FreeEnergy.part_pos hM _ _
  rw [FreeEnergy.boltz, hzero]
  simp only [mul_zero, Real.exp_zero]
  rw [le_div_iff₀ hZpos]
  linarith

end SeqLimit

end IDR
