/-
# Part XLI.1  Neutral evolution: a conserved ensemble in a diverged sequence

Disordered regions evolve fast.  Orthologues that are unambiguously functionally equivalent
routinely align at a sequence identity a globular domain would never tolerate, and the
standard resolution in the field is that what is conserved is not the sequence but a set of
*sequence-level physical descriptors* -- net charge, charge patterning, hydrophobic content,
the spacing of short motifs.  This file makes that statement, and its consequences for model
design, into theorems, using the mean-field Debye--Hückel descriptor of
`RequestProject.Electrostatics` as the concrete physical read-out.

**The symmetry.**  Every descriptor built from *pairwise* terms `q i · q j · w(|i − j|)` --
the screened electrostatic energy `screenedEnergy` at every salt concentration, and the
sequence charge decoration `scd` -- is invariant under reversal of the chain
(`screenedEnergy_rev`, `scd_rev`), because reversal preserves every sequence separation.  It is
also invariant under global charge inversion (`screenedEnergy_neg`, `scd_neg`).  Reversal is a
genuine restriction, not a triviality: a directional descriptor such as the net charge of the
N-terminal half is *not* reversal invariant (`headCharge_not_rev_invariant`), so the
invariance statements below say something specific about the level of description, and we say
explicitly in each statement which descriptor is being held fixed.  The full physics of a real
chain is of course *not* reversal invariant (the backbone is directional); what the theorems
below quantify is the behaviour of models built on descriptors that are.

**Conservation without identity.**  The block polyampholyte `+^m −^m` is carried by reversal to
`−^m +^m`, which differs from it at *every single position*: sequence identity `0`
(`block_rev_identity_zero`), the same screened energy at every salt concentration, the same
charge decoration, the same net charge (`diverged_sequences_same_physics`).  Percent identity
is therefore not merely a weak proxy for the conserved physics; the two can be at opposite
extremes simultaneously.

**The neutral set is exponentially large.**  There are `centralBinom m = C(2m, m)` distinct
`±1` sequences of length `2m` with zero net charge (`composition_class_card`,
`chargeOf_injective`), and `4^m < m · C(2m,m)` (`composition_class_exponential`): a training
set of `T` sequences covers at most a fraction `T·m/4^m` of one composition class
(`training_set_covers_vanishing_fraction`).  Generalisation across evolution cannot be
memorisation.

**Consequence for a model.**  If the target descriptor is reversal invariant then any model
that is not is provably wrong somewhere: its two errors on `x` and on the reversed `x` add up
to at least its own asymmetry (`asymmetry_forces_error`), so a model with asymmetry `a` has
error at least `a/2` on one of them (`asymmetry_error_half`).  Symmetrising the model never
increases its squared error (`symmetrised_error_le`) and strictly decreases it whenever the
model is asymmetric and the two errors differ (`symmetrised_error_lt`).  Building the neutral
group of the target into the architecture is free, and not doing it is not.
-/
import Mathlib
import RequestProject.Electrostatics

set_option autoImplicit false

namespace IDR

open Finset

namespace Evolution

open IDR.Electro

variable {N : ℕ}

/-! ## Reversal of the chain -/

/-- Reversal preserves every sequence separation. -/
lemma dist_rev (i j : Fin N) :
    Nat.dist (Fin.rev i : ℕ) (Fin.rev j : ℕ) = Nat.dist (i : ℕ) (j : ℕ) := by
  have hi := i.isLt
  have hj := j.isLt
  simp only [Fin.val_rev, Nat.dist]
  omega

/-- **Every pairwise, separation-dependent descriptor is reversal invariant.**  This is the
common content of `screenedEnergy_rev` and `scd_rev` below: the summand of such a descriptor
is symmetric in the two residues and depends on their positions only through `|i − j|`, both
of which reversal preserves. -/
theorem pairSum_rev (w : ℕ → ℝ) (q : Fin N → ℝ) :
    ∑ i, ∑ j, (if i < j then q (Fin.rev i) * q (Fin.rev j) * w (Nat.dist (i : ℕ) (j : ℕ))
      else 0)
      = ∑ i, ∑ j, (if i < j then q i * q j * w (Nat.dist (i : ℕ) (j : ℕ)) else 0) := by
  have step : ∀ G : Fin N → ℝ, ∑ i, G i = ∑ i, G (Fin.rev i) :=
    fun G => (Fintype.sum_equiv Fin.revPerm (fun i => G (Fin.rev i)) G (fun _ => rfl)).symm
  rw [step (fun i => ∑ j, (if i < j then
    q (Fin.rev i) * q (Fin.rev j) * w (Nat.dist (i : ℕ) (j : ℕ)) else 0))]
  have hinner : ∀ i : Fin N,
      ∑ j, (if Fin.rev i < j then
          q (Fin.rev (Fin.rev i)) * q (Fin.rev j) *
            w (Nat.dist ((Fin.rev i : Fin N) : ℕ) (j : ℕ)) else 0)
        = ∑ j, (if Fin.rev i < Fin.rev j then q i * q j * w (Nat.dist (i : ℕ) (j : ℕ))
            else 0) := by
    intro i
    rw [step (fun j => (if Fin.rev i < j then
      q (Fin.rev (Fin.rev i)) * q (Fin.rev j) *
        w (Nat.dist ((Fin.rev i : Fin N) : ℕ) (j : ℕ)) else 0))]
    simp only [Fin.rev_rev, dist_rev]
  simp only [hinner]
  have hlt : ∀ i j : Fin N, (Fin.rev i < Fin.rev j) ↔ j < i := fun i j => Fin.rev_lt_rev
  simp only [hlt]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  by_cases h : i < j
  · rw [if_pos h, if_pos h, Nat.dist_comm]
    ring
  · rw [if_neg h, if_neg h]

/-- **The screened electrostatic energy is the same read backwards**, at every bond length and
every salt concentration. -/
theorem screenedEnergy_rev (b kappa : ℝ) (q : Fin N → ℝ) :
    screenedEnergy b kappa (q ∘ Fin.rev) = screenedEnergy b kappa q := by
  have h := pairSum_rev
    (fun t : ℕ => Real.exp (-(kappa * (b * Real.sqrt t))) / (b * Real.sqrt t)) q
  simpa [screenedEnergy, sep, Function.comp] using h

/-- **The sequence charge decoration is the same read backwards.** -/
theorem scd_rev (q : Fin N → ℝ) : scd (q ∘ Fin.rev) = scd q := by
  have h := pairSum_rev (fun t : ℕ => Real.sqrt t) q
  simpa [scd, Function.comp, Finset.mul_sum] using congrArg (fun x => (1 / (N : ℝ)) * x) h

/-- Net charge, being a sum over residues, is reversal invariant too. -/
theorem netCharge_rev (q : Fin N → ℝ) : netCharge (q ∘ Fin.rev) = netCharge q :=
  Equiv.sum_comp Fin.revPerm q

/-- The pairwise descriptors are also invariant under global charge inversion. -/
theorem screenedEnergy_neg (b kappa : ℝ) (q : Fin N → ℝ) :
    screenedEnergy b kappa (fun i => -q i) = screenedEnergy b kappa q := by
  unfold screenedEnergy
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  by_cases h : i < j <;> simp [h]

theorem scd_neg (q : Fin N → ℝ) : scd (fun i => -q i) = scd q := by
  unfold scd
  refine congrArg _ (Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_)
  by_cases h : i < j <;> simp [h]

/-! ## Reversal invariance is a real restriction -/

/-- The net charge of the N-terminal half of the chain: a *directional* descriptor. -/
noncomputable def headCharge (q : Fin N → ℝ) : ℝ :=
  ∑ i ∈ Finset.univ.filter (fun i : Fin N => 2 * (i : ℕ) < N), q i

/-- **Not every descriptor is reversal invariant**, so the invariance theorems above are
statements about the pairwise level of description and not vacuous.  The two-residue sequence
`(+ −)` has N-terminal charge `+1`, its reversal `(− +)` has `−1`. -/
theorem headCharge_not_rev_invariant :
    headCharge (![(1 : ℝ), -1] ∘ Fin.rev) ≠ headCharge (![(1 : ℝ), -1]) := by
  have h : (Finset.univ.filter (fun i : Fin 2 => 2 * (i : ℕ) < 2)) = {(0 : Fin 2)} := by decide
  rw [headCharge, headCharge, h]
  simp [Function.comp, Fin.rev]
  norm_num

/-! ## Sequence identity -/

/-- The fraction of positions at which two sequences agree: percent identity, in `[0,1]`. -/
noncomputable def identity (x y : Fin N → ℝ) : ℝ :=
  ((Finset.univ.filter (fun i : Fin N => x i = y i)).card : ℝ) / N

lemma identity_nonneg (x y : Fin N → ℝ) : 0 ≤ identity x y := by
  unfold identity; positivity

/-- If two sequences differ everywhere their identity is zero. -/
lemma identity_eq_zero {x y : Fin N → ℝ} (h : ∀ i, x i ≠ y i) : identity x y = 0 := by
  have : (Finset.univ.filter (fun i : Fin N => x i = y i)) = ∅ :=
    Finset.filter_eq_empty_iff.mpr fun i _ => h i
  simp [identity, this]

/-! ## The block polyampholyte and its reversal -/

/-- The block polyampholyte `+^m −^m` of length `2m`: the standard maximally patterned
polyampholyte of zero net charge. -/
noncomputable def blockSeq (m : ℕ) : Fin (2 * m) → ℝ := fun i => if (i : ℕ) < m then 1 else -1

lemma blockSeq_rev (m : ℕ) (i : Fin (2 * m)) :
    blockSeq m (Fin.rev i) = - blockSeq m i := by
  have hi := i.isLt
  have hval : ((Fin.rev i : Fin (2 * m)) : ℕ) = 2 * m - (i + 1) := Fin.val_rev i
  simp only [blockSeq, hval]
  by_cases h : (i : ℕ) < m
  · rw [if_neg (by omega), if_pos h]
  · rw [if_pos (by omega), if_neg h]
    norm_num

lemma blockSeq_ne_zero (m : ℕ) (i : Fin (2 * m)) : blockSeq m i ≠ 0 := by
  by_cases h : (i : ℕ) < m <;> simp [blockSeq, h]

/-- **The reversed block polyampholyte agrees with it nowhere: identity `0`.** -/
theorem block_rev_identity_zero (m : ℕ) :
    identity (blockSeq m ∘ Fin.rev) (blockSeq m) = 0 := by
  refine identity_eq_zero fun i => ?_
  have h := blockSeq_rev m i
  simp only [Function.comp_apply, h]
  intro hcon
  have : blockSeq m i = 0 := by linarith
  exact blockSeq_ne_zero m i this

/-- **Conservation without identity.**  For every chain length, bond length and salt
concentration, the block polyampholyte and its reversal have

* sequence identity exactly `0` -- they differ at every position;
* the same screened electrostatic energy, at every salt concentration;
* the same charge decoration;
* the same (zero) net charge.

Percent identity and the conserved physical descriptor are, for this pair, at opposite
extremes at the same time. -/
theorem diverged_sequences_same_physics (m : ℕ) (b kappa : ℝ) :
    identity (blockSeq m ∘ Fin.rev) (blockSeq m) = 0 ∧
    screenedEnergy b kappa (blockSeq m ∘ Fin.rev) = screenedEnergy b kappa (blockSeq m) ∧
    scd (blockSeq m ∘ Fin.rev) = scd (blockSeq m) ∧
    netCharge (blockSeq m ∘ Fin.rev) = netCharge (blockSeq m) :=
  ⟨block_rev_identity_zero m, screenedEnergy_rev b kappa _, scd_rev _, netCharge_rev _⟩

/-! ## The neutral set is exponentially large -/

/-- The `±1` charge sequence with `+` exactly on the positions of `S`. -/
noncomputable def chargeOf {n : ℕ} (S : Finset (Fin n)) : Fin n → ℝ :=
  fun i => if i ∈ S then 1 else -1

lemma chargeOf_injective {n : ℕ} : Function.Injective (chargeOf (n := n)) := by
  intro S T h
  ext i
  by_cases hS : i ∈ S <;> by_cases hT : i ∈ T <;>
    simp only [hS, hT, iff_true, iff_false] <;>
    first
      | tauto
      | (exfalso
         have := congrFun h i
         simp [chargeOf, hS, hT] at this
         norm_num at this)

/-- The subsets of size `m` of a chain of length `2m` are exactly the zero-net-charge
composition class, and there are `C(2m, m)` of them. -/
theorem composition_class_card (m : ℕ) :
    ((Finset.univ : Finset (Fin (2 * m))).powersetCard m).card = Nat.centralBinom m := by
  rw [Finset.card_powersetCard]
  simp [Nat.centralBinom_eq_two_mul_choose]

/-- Every member of the class has zero net charge and unit charges. -/
theorem chargeOf_netCharge {m : ℕ} {S : Finset (Fin (2 * m))} (hS : S.card = m) :
    netCharge (chargeOf S) = 0 := by
  classical
  have h : ∀ i : Fin (2 * m),
      (if i ∈ S then (1 : ℝ) else -1) = 2 * (if i ∈ S then (1 : ℝ) else 0) - 1 := by
    intro i
    by_cases hi : i ∈ S
    · simp [hi]
      norm_num
    · simp [hi]
  unfold netCharge chargeOf
  rw [Finset.sum_congr rfl (fun i _ => h i)]
  simp [Finset.sum_sub_distrib, hS]
  ring

/-- **The neutral set is exponentially large.**  For `m ≥ 4` there are more than `4^m / m`
distinct `±1` sequences of length `2m` with the same (zero-net-charge, half-positive)
composition. -/
theorem composition_class_exponential (m : ℕ) (hm : 4 ≤ m) :
    4 ^ m < m * (((Finset.univ : Finset (Fin (2 * m))).powersetCard m).image
      (chargeOf (n := 2 * m))).card := by
  rw [Finset.card_image_of_injective _ chargeOf_injective, composition_class_card]
  exact Nat.four_pow_lt_mul_centralBinom m hm

/-- **A finite training set covers a vanishing fraction of one composition class.**  If a
training set is a set of `T` sequences drawn from the class, its share of the class is at most
`T·m/4^m`. -/
theorem training_set_covers_vanishing_fraction (m : ℕ) (hm : 4 ≤ m)
    (train : Finset (Fin (2 * m) → ℝ))
    (htrain : train ⊆ ((Finset.univ : Finset (Fin (2 * m))).powersetCard m).image
      (chargeOf (n := 2 * m))) :
    train.card ≤ (((Finset.univ : Finset (Fin (2 * m))).powersetCard m).image
        (chargeOf (n := 2 * m))).card ∧
    (train.card : ℝ) /
      ((((Finset.univ : Finset (Fin (2 * m))).powersetCard m).image
        (chargeOf (n := 2 * m))).card : ℝ)
      ≤ (train.card : ℝ) * m / 4 ^ m := by
  refine ⟨Finset.card_le_card htrain, ?_⟩
  set C := (((Finset.univ : Finset (Fin (2 * m))).powersetCard m).image
    (chargeOf (n := 2 * m))).card with hC
  have hexp : (4 : ℝ) ^ m < (m : ℝ) * C := by
    exact_mod_cast composition_class_exponential m hm
  have h4 : (0 : ℝ) < 4 ^ m := by positivity
  have hCpos : (0 : ℝ) < C := by
    rcases Nat.eq_zero_or_pos C with h | h
    · rw [h] at hexp; simp at hexp; linarith
    · exact_mod_cast h
  have hcard : (0 : ℝ) ≤ train.card := Nat.cast_nonneg _
  rw [div_le_div_iff₀ hCpos h4]
  nlinarith [hcard, hexp, hCpos]

/-! ## What a model must do about it -/

variable {α : Type*}

/-- **A model that lacks a symmetry of its target is wrong somewhere.**  If the target
descriptor `f` is invariant under an involution `g` of sequence space (reversal, for the
pairwise descriptors above), the two errors of a model `M` at `x` and at `g x` sum to at least
the model's own asymmetry `|M x − M (g x)|`. -/
theorem asymmetry_forces_error (f M : α → ℝ) (g : α → α) (hf : ∀ x, f (g x) = f x) (x : α) :
    |M x - M (g x)| ≤ |M x - f x| + |M (g x) - f (g x)| := by
  have h : M x - M (g x) = (M x - f x) - (M (g x) - f (g x)) := by
    rw [hf x]; ring
  calc |M x - M (g x)| = |(M x - f x) - (M (g x) - f (g x))| := by rw [h]
    _ ≤ |M x - f x| + |M (g x) - f (g x)| := abs_sub _ _

/-- Hence a model with asymmetry `a` has error at least `a/2` on one of the two sequences. -/
theorem asymmetry_error_half (f M : α → ℝ) (g : α → α) (hf : ∀ x, f (g x) = f x) (x : α) :
    |M x - M (g x)| / 2 ≤ max |M x - f x| (|M (g x) - f (g x)|) := by
  have h := asymmetry_forces_error f M g hf x
  have h1 : |M x - f x| ≤ max |M x - f x| (|M (g x) - f (g x)|) := le_max_left _ _
  have h2 : |M (g x) - f (g x)| ≤ max |M x - f x| (|M (g x) - f (g x)|) := le_max_right _ _
  linarith

/-- The symmetrised model. -/
noncomputable def symmetrise (M : α → ℝ) (g : α → α) : α → ℝ := fun x => (M x + M (g x)) / 2

/-- **Symmetrising is free.**  Against a `g`-invariant target the squared error of the
symmetrised model never exceeds the mean of the two squared errors of the original. -/
theorem symmetrised_error_le (f M : α → ℝ) (g : α → α) (hf : ∀ x, f (g x) = f x) (x : α) :
    (symmetrise M g x - f x) ^ 2
      ≤ ((M x - f x) ^ 2 + (M (g x) - f (g x)) ^ 2) / 2 := by
  have h : symmetrise M g x - f x = ((M x - f x) + (M (g x) - f (g x))) / 2 := by
    unfold symmetrise; rw [hf x]; ring
  rw [h]
  nlinarith [sq_nonneg ((M x - f x) - (M (g x) - f (g x)))]

/-- And strictly better whenever the two errors differ, which is exactly when the model is
asymmetric on the pair. -/
theorem symmetrised_error_lt (f M : α → ℝ) (g : α → α) (hf : ∀ x, f (g x) = f x) (x : α)
    (hne : M x ≠ M (g x)) :
    (symmetrise M g x - f x) ^ 2
      < ((M x - f x) ^ 2 + (M (g x) - f (g x)) ^ 2) / 2 := by
  have h : symmetrise M g x - f x = ((M x - f x) + (M (g x) - f (g x))) / 2 := by
    unfold symmetrise; rw [hf x]; ring
  have hd : (M x - f x) - (M (g x) - f (g x)) ≠ 0 := by
    rw [hf x]
    intro hc
    exact hne (by linarith [sub_eq_zero.mp (by linarith : M x - M (g x) = 0)])
  rw [h]
  nlinarith [sq_pos_of_ne_zero hd]

/-- Reversal is an involution of sequence space, so the theorems above apply to it. -/
theorem rev_involutive (q : Fin N → ℝ) : (q ∘ Fin.rev) ∘ Fin.rev = q := by
  funext i
  simp [Function.comp]

end Evolution

end IDR
