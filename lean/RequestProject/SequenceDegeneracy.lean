/-
# Part LXVII  The sequence-to-ensemble map: what a pairwise theory can never resolve

The central promise of a model of an intrinsically disordered region is that it is
*sequence-resolved*: change the pattern of charges along the chain and the predicted ensemble
changes with it.  The analytic theories that are actually used to make that prediction --
sequence charge decoration (SCD), the random-phase-approximation free energies, every
Debye-Hückel treatment in which the interaction of two charges is preaveraged over the reference
chain -- all belong to one class: the sequence enters only through *pairwise* terms depending on
the separation `d = j - i` along the chain.  This file computes exactly how much sequence
information that class can carry, and exhibits the obstruction.

* `pairSum_eq_shell` -- for any such model the sequence enters only through the charge
  autocorrelation `shell N q d = Σ_i q i · q (i+d)`: the pairwise energy is
  `Σ_d k d · shell N q d`.  This is an identity, not an approximation, and it holds for every
  kernel `k`, hence at every salt concentration, screening length and temperature.
* `confEnergy_eq_of_shell_eq`, `sequence_blind` -- consequently two charge sequences with the
  same autocorrelation have the *same conformational energy function*, and therefore the same
  partition function, the same Boltzmann ensemble, the same value of every observable, at every
  temperature: no functional whatsoever of the model distinguishes them.
* `seqA`, `seqB`, `shell_seqA_eq_seqB` -- and the autocorrelation does not determine the
  sequence.  Two explicit 12-residue charge patterns,
  `+ + + + − + − − + + − −` and `+ + − + + − + + + − − −`,
  have equal net charge, equal composition (seven positive, five negative) and equal
  autocorrelation at *every* separation, while being distinct, not each other's reverse, and not
  each other's charge inversion.
* `triple_seqA`, `triple_seqB`, `three_body_resolves` -- their nearest-neighbour three-body
  correlations are `2` and `-6`.  A three-body term therefore separates exactly the pair that
  every pairwise separation-dependent theory conflates.

The design reading, and the reason this belongs with Parts LX and LXIV: a model of a disordered
region can be *sequence-resolved* only to the extent that its sequence dependence is not
pairwise-in-separation.  SCD, RPA and preaveraged Debye-Hückel are provably degenerate on real
charge patterns; the degeneracy is not a small error, it is exact and survives every temperature
and every salt concentration; and lifting it requires terms of at least third order in the
charge sequence.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace SeqDeg

open Finset

/-! ## The charge autocorrelation and the pairwise energy -/

/-- The charge autocorrelation of a sequence at separation `d`: `Σ_i q i · q (i+d)`. -/
noncomputable def shell (N : ℕ) (q : ℕ → ℝ) (d : ℕ) : ℝ :=
  ∑ i ∈ Finset.range (N - d), q i * q (i + d)

/-- The energy of a model whose sequence dependence is pairwise and depends on the pair only
through its separation along the chain: `Σ_{i<j} k (j-i) · q i · q j`, written as a sum over a
residue and a separation. -/
noncomputable def pairSum (N : ℕ) (k : ℕ → ℝ) (q : ℕ → ℝ) : ℝ :=
  ∑ i ∈ Finset.range N, ∑ d ∈ Finset.range (N - i), if 0 < d then k d * (q i * q (i + d)) else 0

/-- The triangular exchange of a residue index against a separation. -/
lemma tri_swap (N : ℕ) (f : ℕ → ℕ → ℝ) :
    ∑ i ∈ Finset.range N, ∑ d ∈ Finset.range (N - i), f i d
      = ∑ d ∈ Finset.range N, ∑ i ∈ Finset.range (N - d), f i d := by
  refine Finset.sum_comm' ?_
  intro i d
  simp only [Finset.mem_range]
  omega

/-- **A pairwise separation-dependent model sees the sequence only through its
autocorrelation.** -/
theorem pairSum_eq_shell (N : ℕ) (k : ℕ → ℝ) (q : ℕ → ℝ) :
    pairSum N k q = ∑ d ∈ Finset.range N, if 0 < d then k d * shell N q d else 0 := by
  unfold pairSum
  rw [tri_swap N (fun i d => if 0 < d then k d * (q i * q (i + d)) else 0)]
  refine Finset.sum_congr rfl (fun d _ => ?_)
  by_cases hd : 0 < d
  · simp only [if_pos hd, shell, Finset.mul_sum]
  · simp only [if_neg hd, Finset.sum_const_zero]

/-- Two sequences with the same autocorrelation have the same pairwise energy, for **every**
kernel: at every screening length, salt concentration and temperature. -/
theorem pairSum_congr_of_shell_eq {N : ℕ} {q q' : ℕ → ℝ} (h : ∀ d, shell N q d = shell N q' d)
    (k : ℕ → ℝ) : pairSum N k q = pairSum N k q' := by
  rw [pairSum_eq_shell, pairSum_eq_shell]
  exact Finset.sum_congr rfl (fun d _ => by rw [h d])

/-! ## The ensemble is blind, not just the energy -/

variable {X : Type*}

/-- The conformational energy of a model in which the sequence enters pairwise, through an
interaction that depends on the conformation and on the separation of the two residues.  This is
the form of every preaveraged (random-phase, Debye-Hückel, SCD-type) treatment. -/
noncomputable def confEnergy (N : ℕ) (U0 : X → ℝ) (u : X → ℕ → ℝ) (q : ℕ → ℝ) (x : X) : ℝ :=
  U0 x + pairSum N (u x) q

/-- Two sequences with the same autocorrelation give the **same energy function** on
conformation space. -/
theorem confEnergy_eq_of_shell_eq (N : ℕ) (U0 : X → ℝ) (u : X → ℕ → ℝ) {q q' : ℕ → ℝ}
    (h : ∀ d, shell N q d = shell N q' d) :
    confEnergy N U0 u q = confEnergy N U0 u q' := by
  funext x
  unfold confEnergy
  rw [pairSum_congr_of_shell_eq h (u x)]

/-- Hence **no functional of the model distinguishes them**: not the partition function, not the
Boltzmann ensemble at any temperature, not any observable average, not any derived free energy
or radius of gyration. -/
theorem sequence_blind (N : ℕ) (U0 : X → ℝ) (u : X → ℕ → ℝ) {q q' : ℕ → ℝ}
    (h : ∀ d, shell N q d = shell N q' d) (F : (X → ℝ) → ℝ) :
    F (confEnergy N U0 u q) = F (confEnergy N U0 u q') := by
  rw [confEnergy_eq_of_shell_eq N U0 u h]

/-- Sequence charge decoration is one such functional, so it too is blind. -/
noncomputable def scd (N : ℕ) (q : ℕ → ℝ) : ℝ :=
  pairSum N (fun d => Real.sqrt d / N) q

theorem scd_congr_of_shell_eq {N : ℕ} {q q' : ℕ → ℝ} (h : ∀ d, shell N q d = shell N q' d) :
    scd N q = scd N q' :=
  pairSum_congr_of_shell_eq h _

/-! ## The autocorrelation does not determine the sequence -/

/-- A 12-residue charge pattern: `+ + + + − + − − + + − −`. -/
def seqA : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | 3 => 1
  | 4 => -1
  | 5 => 1
  | 6 => -1
  | 7 => -1
  | 8 => 1
  | 9 => 1
  | 10 => -1
  | 11 => -1
  | _ => 0

/-- A second 12-residue charge pattern: `+ + − + + − + + + − − −`. -/
def seqB : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | 2 => -1
  | 3 => 1
  | 4 => 1
  | 5 => -1
  | 6 => 1
  | 7 => 1
  | 8 => 1
  | 9 => -1
  | 10 => -1
  | 11 => -1
  | _ => 0

/-- The net charge of a sequence of `N` residues. -/
noncomputable def netCharge (N : ℕ) (q : ℕ → ℝ) : ℝ := ∑ i ∈ Finset.range N, q i

/-- The nearest-neighbour three-body correlation. -/
noncomputable def triple (N : ℕ) (q : ℕ → ℝ) : ℝ :=
  ∑ i ∈ Finset.range (N - 2), q i * q (i + 1) * q (i + 2)

lemma netCharge_seqA : netCharge 12 seqA = 2 := by
  unfold netCharge
  norm_num [Finset.sum_range_succ, seqA]

lemma netCharge_seqB : netCharge 12 seqB = 2 := by
  unfold netCharge
  norm_num [Finset.sum_range_succ, seqB]

/-- **Equal autocorrelation at every separation.** -/
theorem shell_seqA_eq_seqB : ∀ d, shell 12 seqA d = shell 12 seqB d := by
  intro d
  match d with
  | 0 => unfold shell; norm_num [Finset.sum_range_succ, seqA, seqB]
  | 1 => unfold shell; norm_num [Finset.sum_range_succ, seqA, seqB]
  | 2 => unfold shell; norm_num [Finset.sum_range_succ, seqA, seqB]
  | 3 => unfold shell; norm_num [Finset.sum_range_succ, seqA, seqB]
  | 4 => unfold shell; norm_num [Finset.sum_range_succ, seqA, seqB]
  | 5 => unfold shell; norm_num [Finset.sum_range_succ, seqA, seqB]
  | 6 => unfold shell; norm_num [Finset.sum_range_succ, seqA, seqB]
  | 7 => unfold shell; norm_num [Finset.sum_range_succ, seqA, seqB]
  | 8 => unfold shell; norm_num [Finset.sum_range_succ, seqA, seqB]
  | 9 => unfold shell; norm_num [Finset.sum_range_succ, seqA, seqB]
  | 10 => unfold shell; norm_num [Finset.sum_range_succ, seqA, seqB]
  | 11 => unfold shell; norm_num [Finset.sum_range_succ, seqA, seqB]
  | (m + 12) => unfold shell; norm_num

/-- The two patterns are genuinely different sequences: distinct, not each other's reverse, and
not each other's charge inversion (nor the reverse of the inversion). -/
theorem seqA_ne_seqB :
    seqA 2 ≠ seqB 2 ∧ seqA 4 ≠ seqB (11 - 4) ∧ seqA 0 ≠ -seqB 0 ∧ seqA 3 ≠ -seqB (11 - 3) := by
  refine ⟨by norm_num [seqA, seqB], by norm_num [seqA, seqB], by norm_num [seqA, seqB],
    by norm_num [seqA, seqB]⟩

/-- But their three-body correlations differ. -/
theorem triple_seqA : triple 12 seqA = 2 := by
  unfold triple
  norm_num [Finset.sum_range_succ, seqA]

theorem triple_seqB : triple 12 seqB = -6 := by
  unfold triple
  norm_num [Finset.sum_range_succ, seqB]

/-- **The obstruction, in one statement.**  Two distinct 12-residue charge patterns of equal net
charge are conflated by every model whose sequence dependence is pairwise in the separation --
identical energy function, hence identical value of every functional of the model, at every
temperature and salt concentration -- and are separated by a three-body correlation. -/
theorem three_body_resolves :
    netCharge 12 seqA = netCharge 12 seqB ∧
    (∀ d, shell 12 seqA d = shell 12 seqB d) ∧
    (∀ (X : Type) (U0 : X → ℝ) (u : X → ℕ → ℝ) (F : (X → ℝ) → ℝ),
      F (confEnergy 12 U0 u seqA) = F (confEnergy 12 U0 u seqB)) ∧
    scd 12 seqA = scd 12 seqB ∧
    triple 12 seqA ≠ triple 12 seqB := by
  refine ⟨by rw [netCharge_seqA, netCharge_seqB], shell_seqA_eq_seqB,
    fun X U0 u F => sequence_blind 12 U0 u shell_seqA_eq_seqB F,
    scd_congr_of_shell_eq shell_seqA_eq_seqB, ?_⟩
  rw [triple_seqA, triple_seqB]
  norm_num

end SeqDeg

end IDR
