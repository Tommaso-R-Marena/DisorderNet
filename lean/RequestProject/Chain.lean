/-
# Part IV.3  A worked disordered region: the ideal chain

Everything so far is architecture-free but abstract: capacity grows with the number of
populated conformations, with the conformational entropy, with the number of resolvable
substates.  This file computes those quantities for the standard physical caricature of an
intrinsically disordered region -- the **freely jointed (ideal) chain** of `N` bonds, each
independently pointing one of two ways -- and so turns the abstract laws into numbers in
`N`.

The conformation space is `Chain N = Fin N → Bool`, one binary torsion per bond, and the
ensemble is the uniform (athermal) one, `chainEns N`.

* `chain_endToEnd_mean = 0` and `chain_endToEnd_msd = N·b²` -- the random-walk law
  `⟨R²⟩ = N b²`.  The mean end-to-end vector vanishes while its root-mean-square is
  `b√N`: the *average structure is not a structure the chain ever visits*, and the
  discrepancy grows without bound with the length of the region.
* `chain_single_structure_floor` -- consequently every single-number (single-structure)
  prediction of the end-to-end distance has squared error at least `N·b²`.  This is the
  bias-variance floor of `RequestProject.Geometry`, made explicit and *linear in `N`*.
* `chain_entropy = N·log 2` and `chain_capacity : 2^N ≤ M.card` -- an exactly correct model
  must carry `2^N` components.  No enumeration of structures, no sampling protocol, no
  library scales: the capacity law is exponential in the length of the region.
* `chain_rate_distortion`, `chain_bit_rate` -- and this is not an artefact of demanding
  exactness: at transport distortion `D` against the (1-separated) chain conformations a
  model still needs `2^N (1 - 2D)` structures, i.e. a bit rate linear in `N`.
* `chain_folding_gap` -- what it would take to escape: to give one conformation half the
  Boltzmann population, the force field must supply an energy gap of at least
  `(1/β)·log(2^N - 1) ≈ N·kT·log 2` -- an energy *linear in the length of the region*.
  Folded domains pay it; disordered regions do not, which is precisely why they are
  disordered.
* `ideal_chain_design_laws` -- the six statements bundled.

The moral of the calculation is the design conclusion of the whole development, in a form
one can put a number to: for a disordered region the correct output object is a
*generative* distribution -- something that can encode `2^N` conformations in `O(N)`
parameters -- and never an explicit list of structures.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.ModelNature
import RequestProject.Metric
import RequestProject.Transport
import RequestProject.FreeEnergy
import RequestProject.Quantization

namespace IDR

open Finset
open scoped Classical

/-- A conformation of the freely jointed chain of `N` bonds: each bond independently points
forwards or backwards. -/
abbrev Chain (N : ℕ) := Fin N → Bool

variable {N : ℕ}

/-- The signed length contributed by bond `i` in conformation `s`, for bond length `b`. -/
noncomputable def bondVec (b : ℝ) (s : Chain N) (i : Fin N) : ℝ := if s i then b else -b

/-- The end-to-end coordinate of a chain conformation. -/
noncomputable def endToEnd (b : ℝ) (s : Chain N) : ℝ := ∑ i, bondVec b s i

/-- Reversing bond `i`. -/
def flipBond (i : Fin N) (s : Chain N) : Chain N := Function.update s i (!(s i))

lemma flipBond_involutive (i : Fin N) : Function.Involutive (flipBond (N := N) i) := by
  intro s
  funext j
  by_cases h : j = i
  · subst h; simp [flipBond]
  · simp [flipBond, Function.update_of_ne h]

/-- Reversing bond `i` is a bijection of conformation space. -/
def flipEquiv (i : Fin N) : Chain N ≃ Chain N := (flipBond_involutive i).toPerm _

lemma sum_flip (i : Fin N) (f : Chain N → ℝ) :
    ∑ s : Chain N, f (flipBond i s) = ∑ s : Chain N, f s :=
  Equiv.sum_comp (flipEquiv i) f

lemma bondVec_flip_self (b : ℝ) (i : Fin N) (s : Chain N) :
    bondVec b (flipBond i s) i = -bondVec b s i := by
  simp only [bondVec, flipBond, Function.update_self]
  cases h : s i <;> simp_all

lemma bondVec_flip_other (b : ℝ) {i j : Fin N} (h : j ≠ i) (s : Chain N) :
    bondVec b (flipBond i s) j = bondVec b s j := by
  simp [bondVec, flipBond, Function.update_of_ne h]

lemma card_chain (N : ℕ) : Fintype.card (Chain N) = 2 ^ N := by
  simp

/-- Each bond direction is unbiased: the signed bond lengths sum to zero over the whole
conformation space. -/
lemma sum_bondVec (b : ℝ) (i : Fin N) : ∑ s : Chain N, bondVec b s i = 0 := by
  have h := sum_flip i (fun s => bondVec b s i)
  have h2 : ∑ s : Chain N, bondVec b (flipBond i s) i = -∑ s : Chain N, bondVec b s i := by
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun s _ => bondVec_flip_self b i s
  rw [h2] at h
  linarith

/-- Distinct bonds are uncorrelated: the cross terms vanish. -/
lemma sum_bondVec_cross (b : ℝ) {i j : Fin N} (hij : i ≠ j) :
    ∑ s : Chain N, bondVec b s i * bondVec b s j = 0 := by
  have h := sum_flip i (fun s => bondVec b s i * bondVec b s j)
  have h2 : ∑ s : Chain N, bondVec b (flipBond i s) i * bondVec b (flipBond i s) j
      = -∑ s : Chain N, bondVec b s i * bondVec b s j := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun s _ => ?_
    rw [bondVec_flip_self b i s, bondVec_flip_other b (Ne.symm hij) s]
    ring
  rw [h2] at h
  linarith

/-- Each bond contributes `b²` to the mean square. -/
lemma sum_bondVec_diag (b : ℝ) (i : Fin N) :
    ∑ s : Chain N, bondVec b s i * bondVec b s i = 2 ^ N * b ^ 2 := by
  have hconst : ∀ s : Chain N, bondVec b s i * bondVec b s i = b ^ 2 := by
    intro s
    simp only [bondVec]
    cases h : s i <;> simp_all <;> ring
  rw [Finset.sum_congr rfl (fun s _ => hconst s), Finset.sum_const, nsmul_eq_mul,
    Finset.card_univ, card_chain]
  push_cast
  ring

/-! ## The ideal-chain ensemble -/

instance : Nonempty (Chain N) := ⟨fun _ => true⟩

/-- The athermal (uniform) ensemble of the freely jointed chain: all `2^N` bond
configurations are equally likely. -/
noncomputable def chainEns (N : ℕ) : Ens (Chain N) :=
  unif (Fintype.card_pos (α := Chain N)) (Fintype.equivFin (Chain N)).symm

lemma chainEns_expect (f : Chain N → ℝ) :
    (chainEns N).expect f = (∑ s : Chain N, f s) / 2 ^ N := by
  rw [chainEns, unif_expect]
  rw [Equiv.sum_comp (Fintype.equivFin (Chain N)).symm f, card_chain]
  push_cast
  ring

lemma chainEns_prob (s : Chain N) : (chainEns N).prob s = 1 / 2 ^ N := by
  have hinj : Function.Injective (Fintype.equivFin (Chain N)).symm :=
    (Fintype.equivFin (Chain N)).symm.injective
  have h := unif_prob_eq (Fintype.card_pos (α := Chain N)) hinj
    ((Fintype.equivFin (Chain N)) s)
  simp only [Equiv.symm_apply_apply] at h
  rw [chainEns, h, card_chain]
  push_cast
  ring

/-! ## Random-walk statistics -/

/-- **The mean end-to-end coordinate vanishes.** -/
theorem chain_endToEnd_mean (b : ℝ) : (chainEns N).expect (endToEnd b) = 0 := by
  rw [chainEns_expect]
  have : ∑ s : Chain N, endToEnd b s = 0 := by
    simp only [endToEnd]
    rw [Finset.sum_comm]
    exact Finset.sum_eq_zero fun i _ => sum_bondVec b i
  rw [this]
  simp

/-- **The random-walk law `⟨R²⟩ = N b²`.**  The chain's end-to-end coordinate has mean zero
and mean square `N b²`, so its typical magnitude is `b√N`: the ensemble is spread over a
region growing with the length of the chain, while its mean is a single point. -/
theorem chain_endToEnd_msd (b : ℝ) :
    (chainEns N).expect (fun s => (endToEnd b s) ^ 2) = N * b ^ 2 := by
  have hexpand : ∀ s : Chain N,
      (endToEnd b s) ^ 2 = ∑ i, ∑ j, bondVec b s i * bondVec b s j := by
    intro s
    simp only [endToEnd, sq]
    rw [Finset.sum_mul_sum]
  have hsum : ∑ s : Chain N, (endToEnd b s) ^ 2 = (N : ℝ) * 2 ^ N * b ^ 2 := by
    rw [Finset.sum_congr rfl (fun s _ => hexpand s)]
    rw [Finset.sum_comm]
    have hinner : ∀ i : Fin N,
        ∑ s : Chain N, ∑ j, bondVec b s i * bondVec b s j = 2 ^ N * b ^ 2 := by
      intro i
      rw [Finset.sum_comm]
      have hj : ∀ j : Fin N, ∑ s : Chain N, bondVec b s i * bondVec b s j
          = if j = i then 2 ^ N * b ^ 2 else 0 := by
        intro j
        by_cases h : j = i
        · subst h; rw [if_pos rfl]; exact sum_bondVec_diag b j
        · rw [if_neg h]; exact sum_bondVec_cross b (Ne.symm h)
      rw [Finset.sum_congr rfl (fun j _ => hj j), Finset.sum_ite_eq' Finset.univ i]
      simp
    rw [Finset.sum_congr rfl (fun i _ => hinner i), Finset.sum_const, nsmul_eq_mul]
    simp only [Finset.card_univ, Fintype.card_fin]
    ring
  rw [chainEns_expect, hsum]
  have h2 : (2 : ℝ) ^ N ≠ 0 := by positivity
  field_simp

/-- **The single-structure error floor of a disordered chain.**  Predicting any single
value `r` for the end-to-end coordinate incurs squared error exactly `N b² + r²`, hence at
least `N b²`.  The irreducible error of a single-structure model grows *linearly with the
length of the disordered region*: the longer the region, the worse a structure is as an
answer. -/
theorem chain_single_structure_floor (b r : ℝ) :
    (N : ℝ) * b ^ 2 ≤ (chainEns N).expect (fun s => (endToEnd b s - r) ^ 2) := by
  have hexp : (chainEns N).expect (fun s => (endToEnd b s - r) ^ 2)
      = (chainEns N).expect (fun s => (endToEnd b s) ^ 2)
        - 2 * r * (chainEns N).expect (endToEnd b) + r ^ 2 := by
    have h1 : ∀ s : Chain N, (endToEnd b s - r) ^ 2
        = (endToEnd b s) ^ 2 + (-(2 * r)) * endToEnd b s + r ^ 2 := by
      intro s; ring
    simp only [Ens.expect] at *
    rw [Finset.sum_congr rfl (fun j _ => by rw [h1 ((chainEns N).pt j)])]
    have : ∀ j, (chainEns N).w j * ((endToEnd b ((chainEns N).pt j)) ^ 2
        + (-(2 * r)) * endToEnd b ((chainEns N).pt j) + r ^ 2)
        = (chainEns N).w j * (endToEnd b ((chainEns N).pt j)) ^ 2
          + (-(2 * r)) * ((chainEns N).w j * endToEnd b ((chainEns N).pt j))
          + r ^ 2 * (chainEns N).w j := fun j => by ring
    rw [Finset.sum_congr rfl (fun j _ => this j), Finset.sum_add_distrib,
      Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, (chainEns N).w_sum]
    ring
  rw [hexp, chain_endToEnd_mean, chain_endToEnd_msd]
  nlinarith [sq_nonneg r]

/-! ## Capacity, entropy and bit rate of the chain -/

/-- **The conformational entropy of the chain is `N log 2`** -- one bit per bond. -/
theorem chain_entropy : entropy (chainEns N) = (N : ℝ) * Real.log 2 := by
  have h2 : (0 : ℝ) < 2 ^ N := by positivity
  have hterm : ∀ s : Chain N,
      -((chainEns N).prob s * Real.log ((chainEns N).prob s)) = (N : ℝ) * Real.log 2 / 2 ^ N := by
    intro s
    rw [chainEns_prob]
    rw [one_div, Real.log_inv, Real.log_pow]
    field_simp
  rw [entropy, Finset.sum_congr rfl (fun s _ => hterm s), Finset.sum_const, nsmul_eq_mul,
    Finset.card_univ, card_chain]
  field_simp
  push_cast
  ring

/-- **The exponential capacity law for a disordered chain.**  Any model that reproduces the
ideal-chain ensemble -- on every observable -- must carry at least `2^N` components.  A
library of structures, a set of molecular-dynamics snapshots or a bank of decoys is
therefore hopeless beyond very short regions; only a *generative* model, which represents
`2^N` conformations with `O(N)` parameters, can scale. -/
theorem chain_capacity {M : Ens (Chain N)} (h : M.Same (chainEns N)) : 2 ^ N ≤ M.card := by
  have hinj : Function.Injective (Fintype.equivFin (Chain N)).symm :=
    (Fintype.equivFin (Chain N)).symm.injective
  have hpos : ∀ l, 0 < (chainEns N).prob ((Fintype.equivFin (Chain N)).symm l) := by
    intro l
    rw [chainEns_prob]
    positivity
  have := Ens.card_le_of_same h (Fintype.equivFin (Chain N)).symm hinj hpos
  rwa [card_chain] at this

/-- **The rate--distortion law for the chain.**  Distinct chain conformations are `1`-apart
in the discrete structural metric, so even a model that is only required to be within
transport distortion `D` must still carry at least `2^N (1 - 2D)` structures. -/
theorem chain_rate_distortion {k : ℕ} {D : ℝ} {M : Ens (Chain N)} (hM : M.card ≤ k)
    (hD : transportCost (discreteDist (Chain N)) M
      (unif (Fintype.card_pos (α := Chain N)) (Fintype.equivFin (Chain N)).symm) ≤ D) :
    (2 : ℝ) ^ N * (1 - 2 * D) ≤ k := by
  have hsep : Separated (discreteDist (Chain N)) (Fintype.equivFin (Chain N)).symm 1 :=
    discreteDist_separated (Fintype.equivFin (Chain N)).symm.injective
  have h := quantization_capacity (Fintype.card_pos (α := Chain N))
    (discreteDist_structDist) (by norm_num : (0:ℝ) < 1) hsep hM (by simpa using hD)
  have hcard : ((Fintype.card (Chain N) : ℕ) : ℝ) = 2 ^ N := by
    rw [card_chain]; push_cast; ring
  rw [hcard] at h
  simpa using h

/-- **The bit rate of a chain model is linear in the length of the region.**  At distortion
`D < 1/2` a model of the ideal chain must store at least `N·log 2 + log(1 - 2D)` nats. -/
theorem chain_bit_rate {k : ℕ} {D : ℝ} {M : Ens (Chain N)} (hM : M.card ≤ k)
    (hD : transportCost (discreteDist (Chain N)) M
      (unif (Fintype.card_pos (α := Chain N)) (Fintype.equivFin (Chain N)).symm) ≤ D)
    (hDpos : 0 < 1 - 2 * D) :
    (N : ℝ) * Real.log 2 + Real.log (1 - 2 * D) ≤ Real.log k := by
  have hsep : Separated (discreteDist (Chain N)) (Fintype.equivFin (Chain N)).symm 1 :=
    discreteDist_separated (Fintype.equivFin (Chain N)).symm.injective
  have h := bit_rate_lower_bound (Fintype.card_pos (α := Chain N))
    (discreteDist_structDist) (by norm_num : (0:ℝ) < 1) hsep hM (by simpa using hD)
    (by simpa using hDpos)
  have hcard : ((Fintype.card (Chain N) : ℕ) : ℝ) = 2 ^ N := by
    rw [card_chain]; push_cast; ring
  rw [hcard, Real.log_pow] at h
  simpa using h

/-! ## What it would take to fold the chain -/

/-- **Folding must pay for the entropy.**  If one conformation of the chain is to carry at
least half of the Boltzmann population while all `2^N - 1` competitors have energy at most
`Uu`, the force field must supply an energy gap of at least `(1/β)·log(2^N - 1)`, i.e.
essentially `N·kT·log 2`: an energy **linear in the length of the region**.  Where no such
gap exists -- the defining situation of an intrinsically disordered region -- no single
conformation can dominate, and the prediction target is irreducibly an ensemble. -/
theorem chain_folding_gap (hN : 0 < N) {beta : ℝ} (hbeta : 0 < beta)
    (U : Fin (2 ^ N) → ℝ) (j0 : Fin (2 ^ N)) (Uu : ℝ)
    (hU : ∀ j, j ≠ j0 → U j ≤ Uu) (hhalf : 1 / 2 ≤ FreeEnergy.boltz beta U j0) :
    Real.log ((2 ^ N - 1 : ℕ) : ℝ) / beta ≤ Uu - U j0 := by
  have hn : 0 < 2 ^ N := Nat.two_pow_pos N
  have hcard2 : 2 ≤ 2 ^ N := by
    calc 2 = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ N := Nat.pow_le_pow_right (by norm_num) hN
  set D : Finset (Fin (2 ^ N)) := Finset.univ.erase j0 with hD
  have hDcard : D.card = 2 ^ N - 1 := by
    rw [hD, Finset.card_erase_of_mem (Finset.mem_univ j0)]
    simp
  have hDne : D.Nonempty := by
    rw [← Finset.card_pos, hDcard]
    omega
  have hj0 : j0 ∉ D := by simp [hD]
  have h := FreeEnergy.folded_needs_entropic_gap hbeta hn U j0 D hDne hj0 Uu
    (fun j hj => hU j (Finset.ne_of_mem_erase hj)) hhalf
  rwa [hDcard] at h

/-! ## The chain, all together -/

/-- **The ideal chain: the design laws with numbers in them.**  For the freely jointed
chain of `N` bonds of length `b`:

1. the mean end-to-end coordinate is `0` while its mean square is `N b²`, so the mean
   structure is unrepresentative by a margin growing with `N`;
2. every single-structure prediction has squared error at least `N b²`;
3. the conformational entropy is `N log 2`;
4. an exactly correct model needs `2^N` components;
5. even at transport distortion `D` it needs `2^N (1 - 2D)` of them;
6. and only an energy gap of order `N kT log 2` could make a single conformation dominant.

Clauses 4 and 5 are the quantitative statement of the whole development's conclusion: the
output of a model of an intrinsically disordered region must be a *generative*
distribution, since the object it has to represent has exponentially many populated
conformations, and clause 6 says that no force field of a disordered region rescues the
single-structure picture. -/
theorem ideal_chain_design_laws (N : ℕ) (hN : 0 < N) (b : ℝ) :
    ((chainEns N).expect (endToEnd b) = 0 ∧
      (chainEns N).expect (fun s => (endToEnd b s) ^ 2) = N * b ^ 2) ∧
    (∀ r : ℝ, (N : ℝ) * b ^ 2 ≤ (chainEns N).expect (fun s => (endToEnd b s - r) ^ 2)) ∧
    entropy (chainEns N) = (N : ℝ) * Real.log 2 ∧
    (∀ M : Ens (Chain N), M.Same (chainEns N) → 2 ^ N ≤ M.card) ∧
    (∀ (k : ℕ) (D : ℝ) (M : Ens (Chain N)), M.card ≤ k →
      transportCost (discreteDist (Chain N)) M
        (unif (Fintype.card_pos (α := Chain N)) (Fintype.equivFin (Chain N)).symm) ≤ D →
      (2 : ℝ) ^ N * (1 - 2 * D) ≤ k) ∧
    (∀ (beta : ℝ), 0 < beta → ∀ (U : Fin (2 ^ N) → ℝ) (j0 : Fin (2 ^ N)) (Uu : ℝ),
      (∀ j, j ≠ j0 → U j ≤ Uu) → 1 / 2 ≤ FreeEnergy.boltz beta U j0 →
      Real.log ((2 ^ N - 1 : ℕ) : ℝ) / beta ≤ Uu - U j0) :=
  ⟨⟨chain_endToEnd_mean b, chain_endToEnd_msd b⟩,
    fun r => chain_single_structure_floor b r,
    chain_entropy,
    fun _ h => chain_capacity h,
    fun _ _ _ hM hD => chain_rate_distortion hM hD,
    fun _ hbeta U j0 Uu hU hhalf => chain_folding_gap hN hbeta U j0 Uu hU hhalf⟩

end IDR
