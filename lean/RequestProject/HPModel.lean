/-
# Part XIII.1  A microscopic sequence Hamiltonian: the HP contact model on a lattice

Every earlier part treats the target ensemble as given, or as the Gibbs measure of an abstract
energy.  This file writes down an explicit, sequence-dependent, microscopic Hamiltonian --
the standard hydrophobic/polar contact model of Lau and Dill -- on the self-avoiding chains of
Part XII, and derives from it, with no mean-field step, when a sequence can and cannot order.

* `posOf` : the position of residue `i`; on a self-avoiding chain the residues occupy distinct
  positions (`posOf_injective`).
* `contacts` : the hydrophobic pairs that are lattice neighbours without being bonded
  neighbours; `hpEnergy` gives each of them the energy `-eps`.
* `contacts_card_le` : **the energy is extensive and bounded by the coordination number**.  A
  self-avoiding chain of `n` bonds on a lattice with `q` bond vectors has at most `(n+1)·q`
  contacts, because excluded volume caps the number of neighbours of a residue.  This is the
  physical input that the mean-field treatments assume.
* `hp_ordering_threshold` / `hp_no_folding` : consequently a contact Hamiltonian can put half
  the population on one conformation only if `beta·eps·(n+1)·q ≥ log (number of conformations
  − 1)`.  Since the number of conformations is exponential in `n`, **there is a threshold
  contact energy, in units of `kT` and independent of the sequence, below which no sequence
  whatsoever folds** -- the chain is intrinsically disordered for energetic reasons, not for
  want of a better predictor.
* `polar_ens_same_saw` / `polar_chain_capacity` : a sequence with no hydrophobic residues has a
  flat landscape, so its equilibrium ensemble is the athermal self-avoiding ensemble of
  Part XII *at every temperature*, and modelling it costs one component per conformation.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Statistics
import RequestProject.Metric
import RequestProject.FreeEnergy
import RequestProject.Boltzmann
import RequestProject.LatticeWalk
import RequestProject.PartTwelve

namespace IDR.HP

open IDR.SAW IDR.FreeEnergy
open scoped Classical

section Lattice

variable {V : Type*} [AddCommGroup V] [DecidableEq V] {q n : ℕ}

/-- The position of residue `i` of the chain with bond sequence `w`: the `i`-th partial sum of
its bond vectors. -/
def posOf (dir : Fin q → V) {n : ℕ} (w : Fin n → Fin q) (i : Fin (n + 1)) : V :=
  (sites (stepsOfDir dir w)).getD (i : ℕ) 0

omit [DecidableEq V] in
lemma length_sites (l : List V) : (sites l).length = l.length + 1 :=
  List.length_scanl

omit [AddCommGroup V] [DecidableEq V] in
lemma length_stepsOfDir (dir : Fin q → V) (w : Fin n → Fin q) :
    (stepsOfDir dir w).length = n := by
  simp [stepsOfDir]

omit [DecidableEq V] in
lemma length_sites_stepsOfDir (dir : Fin q → V) (w : Fin n → Fin q) :
    (sites (stepsOfDir dir w)).length = n + 1 := by
  rw [length_sites, length_stepsOfDir]

omit [DecidableEq V] in
/-- **Excluded volume, in terms of residue positions.**  On a self-avoiding chain distinct
residues occupy distinct lattice sites. -/
theorem posOf_injective {dir : Fin q → V} {w : Fin n → Fin q}
    (h : IsSAW (stepsOfDir dir w)) : Function.Injective (posOf dir w) := by
  intro i j hij
  have hlen := length_sites_stepsOfDir dir w
  have hi : (i : ℕ) < (sites (stepsOfDir dir w)).length := by rw [hlen]; exact i.2
  have hj : (j : ℕ) < (sites (stepsOfDir dir w)).length := by rw [hlen]; exact j.2
  rw [posOf, posOf, List.getD_eq_getElem _ _ hi, List.getD_eq_getElem _ _ hj] at hij
  exact Fin.ext ((h.getElem_inj_iff).1 hij)

/-- Two lattice sites are in contact when they differ by one bond vector. -/
def Adj (dir : Fin q → V) (x y : V) : Prop := ∃ k : Fin q, y = x + dir k

instance (dir : Fin q → V) (x y : V) : Decidable (Adj dir x y) := by
  unfold Adj; infer_instance

/-- The hydrophobic contacts of the conformation `w` of the sequence `seq` (`true` =
hydrophobic): ordered pairs of residues that are hydrophobic, occupy neighbouring lattice
sites, and are not bonded neighbours along the chain. -/
def contacts (dir : Fin q → V) (seq : Fin (n + 1) → Bool) (w : Fin n → Fin q) :
    Finset (Fin (n + 1) × Fin (n + 1)) :=
  Finset.univ.filter fun p =>
    (p.1 : ℕ) + 1 < (p.2 : ℕ) ∧ seq p.1 = true ∧ seq p.2 = true ∧
      Adj dir (posOf dir w p.1) (posOf dir w p.2)

/-- **The HP Hamiltonian**: each hydrophobic contact lowers the energy by `eps`. -/
def hpEnergy (dir : Fin q → V) (eps : ℝ) (seq : Fin (n + 1) → Bool) (w : Fin n → Fin q) : ℝ :=
  -eps * (contacts dir seq w).card

/-- The number of hydrophobic residues of a sequence. -/
def hCount (seq : Fin (n + 1) → Bool) : ℕ :=
  (Finset.univ.filter fun i => seq i = true).card

lemma hCount_le (seq : Fin (n + 1) → Bool) : hCount seq ≤ n + 1 := by
  classical
  calc hCount seq ≤ (Finset.univ : Finset (Fin (n + 1))).card := Finset.card_filter_le _ _
    _ = n + 1 := by simp

/-- **Excluded volume caps the energy: the contacts are bounded by the hydrophobic content.**
A self-avoiding chain on a lattice with `q` bond vectors has at most `h·q` hydrophobic
contacts, where `h` is the number of hydrophobic residues: each of them has at most `q`
neighbouring sites, and no two residues share a site. -/
theorem contacts_card_le_hCount {dir : Fin q → V} {seq : Fin (n + 1) → Bool} {w : Fin n → Fin q}
    (h : IsSAW (stepsOfDir dir w)) : (contacts dir seq w).card ≤ hCount seq * q := by
  have hinj := posOf_injective h
  classical
  set T : Finset (Fin (n + 1) × V) :=
    (Finset.univ.filter fun i => seq i = true) ×ˢ (Finset.univ.image dir) with hT
  set F : Fin (n + 1) × Fin (n + 1) → Fin (n + 1) × V :=
    fun p => (p.1, posOf dir w p.2 - posOf dir w p.1) with hF
  have hmaps : ∀ p ∈ contacts dir seq w, F p ∈ T := by
    intro p hp
    rw [contacts, Finset.mem_filter] at hp
    obtain ⟨k, hk⟩ := hp.2.2.2.2
    refine Finset.mem_product.2 ⟨Finset.mem_filter.2 ⟨Finset.mem_univ _, hp.2.2.1⟩, ?_⟩
    refine Finset.mem_image.2 ⟨k, Finset.mem_univ _, ?_⟩
    show dir k = posOf dir w p.2 - posOf dir w p.1
    rw [hk]
    abel
  have hinjOn : ∀ p ∈ contacts dir seq w, ∀ p' ∈ contacts dir seq w, F p = F p' → p = p' := by
    intro p _ p' _ hpp
    rw [hF] at hpp
    simp only [Prod.mk.injEq] at hpp
    obtain ⟨h1, h2⟩ := hpp
    have h3 : posOf dir w p.2 = posOf dir w p'.2 := by
      rw [h1] at h2
      exact sub_left_inj.1 h2
    exact Prod.ext h1 (hinj h3)
  have hcard : (contacts dir seq w).card ≤ T.card :=
    Finset.card_le_card_of_injOn F hmaps hinjOn
  have hTcard : T.card ≤ hCount seq * q := by
    rw [hT, Finset.card_product]
    have h1 : (Finset.univ.image dir).card ≤ q := by
      calc (Finset.univ.image dir).card ≤ (Finset.univ : Finset (Fin q)).card :=
            Finset.card_image_le
        _ = q := by simp
    exact Nat.mul_le_mul_left _ h1
  exact le_trans hcard hTcard

/-- The coordination bound in its crudest form: at most `(n+1)·q` contacts, whatever the
sequence. -/
theorem contacts_card_le {dir : Fin q → V} {seq : Fin (n + 1) → Bool} {w : Fin n → Fin q}
    (h : IsSAW (stepsOfDir dir w)) : (contacts dir seq w).card ≤ (n + 1) * q :=
  le_trans (contacts_card_le_hCount h) (Nat.mul_le_mul_right _ (hCount_le seq))

/-- Contacts can only lower the energy. -/
theorem hpEnergy_nonpos {dir : Fin q → V} {eps : ℝ} (heps : 0 ≤ eps)
    (seq : Fin (n + 1) → Bool) (w : Fin n → Fin q) : hpEnergy dir eps seq w ≤ 0 := by
  rw [hpEnergy]
  have : (0 : ℝ) ≤ ((contacts dir seq w).card : ℝ) := Nat.cast_nonneg _
  nlinarith

/-- **The HP landscape has bounded depth.**  Its total energy range is at most
`eps·(n+1)·q`: linear in the length of the region, with the lattice coordination number as the
constant. -/
theorem hpEnergy_ge {dir : Fin q → V} {eps : ℝ} (heps : 0 ≤ eps)
    {seq : Fin (n + 1) → Bool} {w : Fin n → Fin q} (h : IsSAW (stepsOfDir dir w)) :
    -(eps * ((n + 1) * q)) ≤ hpEnergy dir eps seq w := by
  rw [hpEnergy]
  have hc : ((contacts dir seq w).card : ℝ) ≤ ((n + 1) * q : ℕ) := by
    exact_mod_cast contacts_card_le (seq := seq) h
  have hcast : (((n + 1) * q : ℕ) : ℝ) = ((n : ℝ) + 1) * q := by push_cast; ring
  rw [hcast] at hc
  nlinarith

/-- **The depth of the landscape is set by the hydrophobic content.**  The energy range is at
most `eps·h·q`, where `h` is the number of hydrophobic residues. -/
theorem hpEnergy_ge_hCount {dir : Fin q → V} {eps : ℝ} (heps : 0 ≤ eps)
    {seq : Fin (n + 1) → Bool} {w : Fin n → Fin q} (h : IsSAW (stepsOfDir dir w)) :
    -(eps * (hCount seq * q)) ≤ hpEnergy dir eps seq w := by
  rw [hpEnergy]
  have hc : ((contacts dir seq w).card : ℝ) ≤ ((hCount seq * q : ℕ) : ℝ) := by
    exact_mod_cast contacts_card_le_hCount (seq := seq) h
  have hcast : ((hCount seq * q : ℕ) : ℝ) = (hCount seq : ℝ) * q := by push_cast; ring
  rw [hcast] at hc
  nlinarith

end Lattice

/-! ## The equilibrium ensemble of an HP sequence -/

section Ensemble

variable {V : Type*} [AddCommGroup V] [DecidableEq V] {q n : ℕ}

/-- The energy of the `j`-th self-avoiding conformation. -/
noncomputable def hpU (dir : Fin q → V) (eps : ℝ) (seq : Fin (n + 1) → Bool) :
    Fin (cntOf dir n) → ℝ :=
  fun j => hpEnergy dir eps seq (enumOf dir n j)

/-- **The equilibrium ensemble of an HP sequence** at inverse temperature `beta`: the
Boltzmann distribution of the contact Hamiltonian over the self-avoiding conformations. -/
noncomputable def hpEns (dir : Fin q → V) (hdir : ∀ n, 0 < cntOf dir n) (eps beta : ℝ)
    (seq : Fin (n + 1) → Bool) : Ens (Fin n → Fin q) :=
  Boltz.boltzEns (hdir n) (enumOf dir n) beta (hpU dir eps seq)

/-- **An energy gap is needed to order the chain.**  If one conformation of an HP chain holds
half the equilibrium population then the accessible energy range must cover the conformational
entropy: `log (cnt n − 1) / beta ≤ eps·(n+1)·q`.  Since `cnt n` grows exponentially in `n`,
the left-hand side grows linearly with the same slope as the right-hand side, and the
comparison is a threshold on `beta·eps`. -/
theorem hp_ordering_threshold {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {eps beta : ℝ}
    (heps : 0 ≤ eps) (hbeta : 0 < beta) (seq : Fin (n + 1) → Bool)
    (hcnt : 1 < cntOf dir n) {j0 : Fin (cntOf dir n)}
    (hhalf : 1 / 2 ≤ boltz beta (hpU dir eps seq) j0) :
    Real.log (cntOf dir n - 1 : ℕ) / beta ≤ eps * ((n + 1) * q) := by
  classical
  set U := hpU dir eps seq with hU
  set D : Finset (Fin (cntOf dir n)) := Finset.univ.erase j0 with hD
  have hDne : D.Nonempty := by
    rw [hD, ← Finset.card_pos, Finset.card_erase_of_mem (Finset.mem_univ _)]
    simp only [Finset.card_univ, Fintype.card_fin]
    omega
  have hj0 : j0 ∉ D := by simp [hD]
  have hUle : ∀ j ∈ D, U j ≤ 0 := fun j _ => hpEnergy_nonpos heps seq _
  have hgap := folded_needs_entropic_gap hbeta (hdir n) U j0 D hDne hj0 0 hUle hhalf
  have hDcard : D.card = cntOf dir n - 1 := by
    rw [hD, Finset.card_erase_of_mem (Finset.mem_univ _)]
    simp
  have hlow : -(eps * ((n + 1) * q)) ≤ U j0 :=
    hpEnergy_ge heps (enumOf_isSAW dir j0)
  rw [hDcard] at hgap
  linarith [hgap]

/-- **Below a threshold contact energy no sequence folds.**  If
`beta·eps·(n+1)·q < log (cnt n − 1)`, then for *every* HP sequence and every conformation the
equilibrium population is below one half: the region is intrinsically disordered, and no
single-structure prediction can be more than half right. -/
theorem hp_no_folding {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {eps beta : ℝ}
    (heps : 0 ≤ eps) (hbeta : 0 < beta) (hcnt : 1 < cntOf dir n)
    (hthr : beta * (eps * ((n + 1) * q)) < Real.log (cntOf dir n - 1 : ℕ)) :
    ∀ (seq : Fin (n + 1) → Bool) (j : Fin (cntOf dir n)),
      boltz beta (hpU dir eps seq) j < 1 / 2 := by
  intro seq j
  by_contra hcon
  push_neg at hcon
  have h := hp_ordering_threshold hdir heps hbeta seq hcnt hcon
  rw [div_le_iff₀ hbeta] at h
  nlinarith [h]

/-- **Ordering demands hydrophobic content.**  If one conformation of an HP chain holds half
the equilibrium population, the conformational entropy must be covered by the contact energy
that the *hydrophobic residues alone* can supply: `log (cnt n − 1)/beta ≤ eps·h·q`. -/
theorem hp_ordering_threshold_hCount {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n)
    {eps beta : ℝ} (heps : 0 ≤ eps) (hbeta : 0 < beta) (seq : Fin (n + 1) → Bool)
    (hcnt : 1 < cntOf dir n) {j0 : Fin (cntOf dir n)}
    (hhalf : 1 / 2 ≤ boltz beta (hpU dir eps seq) j0) :
    Real.log (cntOf dir n - 1 : ℕ) / beta ≤ eps * (hCount seq * q) := by
  classical
  set U := hpU dir eps seq with hU
  set D : Finset (Fin (cntOf dir n)) := Finset.univ.erase j0 with hD
  have hDne : D.Nonempty := by
    rw [hD, ← Finset.card_pos, Finset.card_erase_of_mem (Finset.mem_univ _)]
    simp only [Finset.card_univ, Fintype.card_fin]
    omega
  have hj0 : j0 ∉ D := by simp [hD]
  have hUle : ∀ j ∈ D, U j ≤ 0 := fun j _ => hpEnergy_nonpos heps seq _
  have hgap := folded_needs_entropic_gap hbeta (hdir n) U j0 D hDne hj0 0 hUle hhalf
  have hDcard : D.card = cntOf dir n - 1 := by
    rw [hD, Finset.card_erase_of_mem (Finset.mem_univ _)]
    simp
  have hlow : -(eps * (hCount seq * q)) ≤ U j0 :=
    hpEnergy_ge_hCount heps (enumOf_isSAW dir j0)
  rw [hDcard] at hgap
  linarith [hgap]

/-- **A sequence of low hydrophobic content cannot order.**  If
`beta·eps·h·q < log (cnt n − 1)` then every conformation of that sequence stays below half the
equilibrium population, at that temperature. -/
theorem hp_no_folding_hCount {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {eps beta : ℝ}
    (heps : 0 ≤ eps) (hbeta : 0 < beta) {seq : Fin (n + 1) → Bool} (hcnt : 1 < cntOf dir n)
    (hthr : beta * (eps * (hCount seq * q)) < Real.log (cntOf dir n - 1 : ℕ)) :
    ∀ j : Fin (cntOf dir n), boltz beta (hpU dir eps seq) j < 1 / 2 := by
  intro j
  by_contra hcon
  push_neg at hcon
  have h := hp_ordering_threshold_hCount hdir heps hbeta seq hcnt hcon
  rw [div_le_iff₀ hbeta] at h
  nlinarith [h]

/-- The equilibrium population of the `j`-th conformation. -/
lemma hpEns_prob {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {eps beta : ℝ}
    (seq : Fin (n + 1) → Bool) (j : Fin (cntOf dir n)) :
    (hpEns dir hdir eps beta seq).prob (enumOf dir n j) = boltz beta (hpU dir eps seq) j :=
  Boltz.boltzEns_prob (hdir n) (enumOf_injective dir n) beta (hpU dir eps seq) j

/-- From a bound on the Boltzmann weights to a bound on the population of *every* point of
conformation space. -/
lemma hpEns_prob_lt {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {eps beta c : ℝ}
    (hc : 0 < c) {seq : Fin (n + 1) → Bool}
    (hlt : ∀ j, boltz beta (hpU dir eps seq) j < c) (x : Fin n → Fin q) :
    (hpEns dir hdir eps beta seq).prob x < c := by
  classical
  by_cases h : ∃ j, enumOf dir n j = x
  · obtain ⟨j, rfl⟩ := h
    rw [hpEns_prob hdir seq j]
    exact hlt j
  · push_neg at h
    rw [hpEns, Boltz.boltzEns_prob_not_mem (hdir n) h]
    exact hc

/-- **The criterion for order is sharp.**  A contact Hamiltonian with a unique lowest-energy
conformation, separated from the rest by `gap` with `beta·gap ≥ log (cnt n)`, does put half the
population on that conformation.  The threshold of `hp_ordering_threshold` is therefore the
right quantity: order occurs exactly when the energy gap is of the order of `kT` times the
conformational entropy. -/
theorem hp_folding_sufficient {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {eps beta : ℝ}
    (hbeta : 0 < beta) {seq : Fin (n + 1) → Bool} {Umin gap : ℝ} {j0 : Fin (cntOf dir n)}
    (hj0 : hpU dir eps seq j0 = Umin) (hout : ∀ j, j ≠ j0 → Umin + gap ≤ hpU dir eps seq j)
    (hgap : Real.log (cntOf dir n) ≤ beta * gap) :
    1 / 2 ≤ (hpEns dir hdir eps beta seq).prob (enumOf dir n j0) := by
  rw [hpEns_prob hdir seq j0]
  exact Boltz.boltz_unique_ground_half (hdir n) hbeta hj0 hout hgap

/-- A sequence with no hydrophobic residues has no contacts. -/
theorem contacts_polar {dir : Fin q → V} {seq : Fin (n + 1) → Bool}
    (hseq : ∀ i, seq i = false) (w : Fin n → Fin q) : contacts dir seq w = ∅ := by
  rw [contacts, Finset.filter_eq_empty_iff]
  intro p _
  simp [hseq p.1]

/-- The HP energy of a polar sequence vanishes identically. -/
theorem hpEnergy_polar {dir : Fin q → V} {eps : ℝ} {seq : Fin (n + 1) → Bool}
    (hseq : ∀ i, seq i = false) (w : Fin n → Fin q) : hpEnergy dir eps seq w = 0 := by
  rw [hpEnergy, contacts_polar hseq w]
  simp

/-- **A polar region is maximally disordered at every temperature.**  With no hydrophobic
residues the landscape is flat, so the equilibrium ensemble is exactly the athermal
self-avoiding ensemble of Part XII -- no amount of cooling narrows it. -/
theorem polar_ens_same_saw {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {eps beta : ℝ}
    {seq : Fin (n + 1) → Bool} (hseq : ∀ i, seq i = false) :
    (hpEns dir hdir eps beta seq).Same (sawEnsOf dir hdir n) := by
  have hconst : hpU dir eps seq = fun _ => (0 : ℝ) := by
    funext j
    exact hpEnergy_polar hseq _
  rw [hpEns, hconst]
  exact Boltz.boltzEns_flat_same (hdir n) (enumOf dir n) beta 0

/-- **Ground-state degeneracy of a contact Hamiltonian is a capacity requirement.**  If the
minimum of the HP energy is attained on `S` and every other conformation is at least `gap`
above it, a model within `tol` of the equilibrium ensemble needs essentially `|S|` components,
no matter how low the temperature. -/
theorem hp_degeneracy_capacity {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {eps beta : ℝ}
    (hbeta : 0 < beta) {seq : Fin (n + 1) → Bool} {Umin gap : ℝ}
    {S : Finset (Fin (cntOf dir n))} (hSmin : ∀ j ∈ S, hpU dir eps seq j = Umin)
    (hout : ∀ j ∉ S, Umin + gap ≤ hpU dir eps seq j) {k : ℕ} {M : Ens (Fin n → Fin q)}
    (hM : M.card ≤ k) {tol : ℝ} (h : ApproxSame tol M (hpEns dir hdir eps beta seq)) :
    (S.card : ℝ) - tol * ((S.card : ℝ) + cntOf dir n * Real.exp (-beta * gap)) ≤ k :=
  Boltz.ground_state_capacity (hdir n) hbeta (enumOf_injective dir n) hSmin hout hM h

/-- **Modelling a polar region costs one component per conformation, at any temperature.** -/
theorem polar_chain_capacity {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {eps beta : ℝ}
    {seq : Fin (n + 1) → Bool} (hseq : ∀ i, seq i = false)
    {M : Ens (Fin n → Fin q)} (h : M.Same (hpEns dir hdir eps beta seq)) :
    cntOf dir n ≤ M.card :=
  exact_capacity_of hdir (h.trans (polar_ens_same_saw hdir hseq))

end Ensemble

end IDR.HP
