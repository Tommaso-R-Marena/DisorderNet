/-
# Part LXXIII  What a pairwise model of charge patterning can see

Sequence charge patterning is the one sequence feature of a disordered region that current theory
handles quantitatively: the parameters `SCD`, `kappa`, and the various Debye--Hückel chain
energies all take the form

  `E(q) = sum_{i < j} w(j - i) * q i * q j`

for a kernel `w` depending only on the separation `j - i` along the chain.  Every such model --
whatever the kernel, screened or unscreened, Gaussian-chain or self-avoiding -- is a *pairwise
distance model* in this sense.  This file asks exactly what information about a sequence such a
model uses, and answers it completely.

* `pairEnergy_eq_sum_autocorr` -- the regrouping identity.  Every pairwise distance model is a
  linear functional of the **charge autocorrelation** `C(d) = sum_i q i * q (i+d)`:
  `E(q) = sum_{d=1}^{N-1} w d * C(d)`.  The chain of charges enters only through `N - 1` numbers.
* `pairEnergy_delta_kernel`, `autocorr_eq_iff_pairEnergy_eq` -- and no fewer numbers will do:
  choosing `w` to be a Kronecker kernel recovers each `C(d)` individually, so two sequences are
  indistinguishable by *all* pairwise distance models exactly when their autocorrelations agree.
  The autocorrelation is the complete invariant of the whole model class.
* `pairEnergy_const_kernel`, `pairEnergy_blind_of_same_composition` -- the degenerate case.  A
  constant kernel (an unscreened, distance-independent interaction) sees only the net charge and
  the mean square charge: composition, not sequence.  Sequence sensitivity is *exactly* the
  non-constancy of the kernel.
* `charge_a`, `charge_b` and the homometric theorems -- the sharp negative result.  There are two
  charge sequences of length `9`, with the *same composition* (`homometric_same_composition`: the
  two sequences give equal sums for *every* function of the individual charges), which are not
  related by chain reversal or by charge conjugation (`charge_b_ne_*`), and whose autocorrelations
  agree at every lag (`homometric_autocorr`).  Consequently **every** pairwise distance model
  whatsoever assigns them the same energy (`homometric_blind`), in particular the same `SCD`
  (`homometric_scd`) and the same Debye--Hückel energy at every screening length and bond length
  (`homometric_debye`).  Yet they are genuinely different sequences: an explicit three-body
  correlator separates them (`homometric_triple_ne`), and one of them contains a block of three
  consecutive like charges that the other does not.

The consequence for modelling intrinsically disordered regions is concrete.  A model built from
pairwise, separation-dependent charge interactions cannot be complete, however the kernel is
fitted: it is provably blind to a difference between sequences that a three-body term detects.
Completeness requires either many-body sequence terms or an explicit conformational ensemble; a
patterning parameter, of any kernel, is a projection onto `N - 1` autocorrelation coordinates.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace Pattern

open Finset

/-! ## The autocorrelation and the pairwise energy -/

/-- The charge autocorrelation of a sequence `q` of length `N` at lag `d`:
`C(d) = sum_{i} q i * q (i + d)`, the sum running over all pairs at separation `d`. -/
def autocorr (N : ℕ) (q : ℕ → ℝ) (d : ℕ) : ℝ := ∑ i ∈ range (N - d), q i * q (i + d)

/-- A pairwise, separation-dependent charge energy: `sum_{i<j} w (j-i) * q i * q j`.  Every
sequence-charge-patterning parameter in use (`SCD`, `kappa`, screened Coulomb chain energies) is
of this form for a suitable kernel `w`. -/
def pairEnergy (N : ℕ) (w q : ℕ → ℝ) : ℝ :=
  ∑ j ∈ range N, ∑ i ∈ range j, w (j - i) * (q i * q j)

/-- **Every pairwise distance model is a linear functional of the charge autocorrelation.**
Regrouping the pair sum by separation gives `E(q) = sum_{d=1}^{N-1} w d * C(d)`. -/
theorem pairEnergy_eq_sum_autocorr (N : ℕ) (w q : ℕ → ℝ) :
    pairEnergy N w q = ∑ d ∈ Ico 1 N, w d * autocorr N q d := by
  unfold pairEnergy autocorr
  rw [Finset.sum_sigma' (range N) (fun j => range j) (fun j i => w (j - i) * (q i * q j))]
  simp only [Finset.mul_sum]
  rw [Finset.sum_sigma' (Ico 1 N) (fun d => range (N - d)) (fun d i => w d * (q i * q (i + d)))]
  refine Finset.sum_nbij' (i := fun x => ⟨x.1 - x.2, x.2⟩) (j := fun y => ⟨y.2 + y.1, y.2⟩)
    ?_ ?_ ?_ ?_ ?_
  · rintro ⟨j, i⟩ h
    simp only [Finset.mem_sigma, Finset.mem_range, Finset.mem_Ico] at h ⊢
    omega
  · rintro ⟨d, i⟩ h
    simp only [Finset.mem_sigma, Finset.mem_range, Finset.mem_Ico] at h ⊢
    omega
  · rintro ⟨j, i⟩ h
    simp only [Finset.mem_sigma, Finset.mem_range] at h
    have : i + (j - i) = j := by omega
    simp [this]
  · rintro ⟨d, i⟩ h
    simp only [Finset.mem_sigma, Finset.mem_range, Finset.mem_Ico] at h
    have : i + d - i = d := by omega
    simp [this]
  · rintro ⟨j, i⟩ h
    simp only [Finset.mem_sigma, Finset.mem_range] at h
    simp only
    rw [show i + (j - i) = j from by omega]

/-- A Kronecker kernel at lag `d` reads off the autocorrelation at that lag: no coarser invariant
than the full autocorrelation suffices for the model class. -/
theorem pairEnergy_delta_kernel {N d : ℕ} (hd1 : 1 ≤ d) (hdN : d < N) (q : ℕ → ℝ) :
    pairEnergy N (fun e => if e = d then (1:ℝ) else 0) q = autocorr N q d := by
  rw [pairEnergy_eq_sum_autocorr]
  rw [Finset.sum_eq_single d]
  · simp
  · intro b _ hb
    simp [hb]
  · intro h
    exact absurd (Finset.mem_Ico.mpr ⟨hd1, hdN⟩) h

/-- **The autocorrelation is the complete invariant of pairwise distance models.**  Two charge
sequences have equal energies under every kernel exactly when their autocorrelations agree at
every lag `1 ≤ d < N`. -/
theorem autocorr_eq_iff_pairEnergy_eq (N : ℕ) (q q' : ℕ → ℝ) :
    (∀ d, 1 ≤ d → d < N → autocorr N q d = autocorr N q' d) ↔
      ∀ w : ℕ → ℝ, pairEnergy N w q = pairEnergy N w q' := by
  constructor
  · intro h w
    rw [pairEnergy_eq_sum_autocorr, pairEnergy_eq_sum_autocorr]
    refine Finset.sum_congr rfl ?_
    intro d hd
    rw [Finset.mem_Ico] at hd
    rw [h d hd.1 hd.2]
  · intro h d hd1 hdN
    have := h (fun e => if e = d then (1:ℝ) else 0)
    rwa [pairEnergy_delta_kernel hd1 hdN, pairEnergy_delta_kernel hd1 hdN] at this

/-! ## Constant kernels see composition only -/

private lemma sum_pairs_eq (N : ℕ) (q : ℕ → ℝ) :
    2 * (∑ j ∈ range N, ∑ i ∈ range j, q i * q j)
      = (∑ i ∈ range N, q i) ^ 2 - ∑ i ∈ range N, (q i) ^ 2 := by
  induction N with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_range_succ (f := fun j => ∑ i ∈ range j, q i * q j),
        Finset.sum_range_succ (f := fun i => q i), Finset.sum_range_succ (f := fun i => (q i) ^ 2)]
      have hfac : ∑ i ∈ range n, q i * q n = (∑ i ∈ range n, q i) * q n := by
        rw [← Finset.sum_mul]
      rw [hfac]
      nlinarith [ih]

/-- A distance-independent (constant) kernel produces an energy that depends only on the net
charge and the sum of squared charges -- that is, on composition alone. -/
theorem pairEnergy_const_kernel (N : ℕ) (c : ℝ) (q : ℕ → ℝ) :
    pairEnergy N (fun _ => c) q
      = c * (((∑ i ∈ range N, q i) ^ 2 - ∑ i ∈ range N, (q i) ^ 2) / 2) := by
  unfold pairEnergy
  have h : ∑ j ∈ range N, ∑ i ∈ range j, c * (q i * q j)
      = c * ∑ j ∈ range N, ∑ i ∈ range j, q i * q j := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by rw [Finset.mul_sum]
  rw [h, ← sum_pairs_eq]
  ring

/-- Two sequences of equal composition (equal net charge and equal squared-charge sum) are
indistinguishable by every constant-kernel model.  Sequence sensitivity of a pairwise model comes
entirely from the variation of its kernel with separation. -/
theorem pairEnergy_blind_of_same_composition {N : ℕ} {q q' : ℕ → ℝ}
    (h1 : ∑ i ∈ range N, q i = ∑ i ∈ range N, q' i)
    (h2 : ∑ i ∈ range N, (q i) ^ 2 = ∑ i ∈ range N, (q' i) ^ 2) (c : ℝ) :
    pairEnergy N (fun _ => c) q = pairEnergy N (fun _ => c) q' := by
  rw [pairEnergy_const_kernel, pairEnergy_const_kernel, h1, h2]

/-! ## Two named patterning models -/

/-- Sequence charge decoration, in the pairwise form: the kernel is the root-mean-square
separation `sqrt(j-i)` of a Gaussian chain, normalised by the chain length. -/
noncomputable def scd (N : ℕ) (q : ℕ → ℝ) : ℝ :=
  pairEnergy N (fun d => Real.sqrt d) q / N

/-- The Debye--Hückel energy of a charge sequence on a Gaussian chain of bond length `b` at
inverse screening length `kappa`: kernel `exp(-kappa r)/r` with `r = b sqrt(j-i)`. -/
noncomputable def debye (N : ℕ) (b kappa : ℝ) (q : ℕ → ℝ) : ℝ :=
  pairEnergy N (fun d => Real.exp (-(kappa * (b * Real.sqrt d))) / (b * Real.sqrt d)) q

/-! ## A homometric pair: two sequences no pairwise model can tell apart -/

/-- The first of two charge sequences of length `9`: `+ + - + + - - - +`. -/
def chargeA : ℕ → ℝ
  | 0 => 1 | 1 => 1 | 2 => -1 | 3 => 1 | 4 => 1
  | 5 => -1 | 6 => -1 | 7 => -1 | 8 => 1 | _ => 0

/-- The second of two charge sequences of length `9`: `- + + - + + + - -`.  It has the same
composition as `chargeA` and the same charge autocorrelation at every lag, but it is not obtained
from `chargeA` by chain reversal, by charge conjugation, or by both. -/
def chargeB : ℕ → ℝ
  | 0 => -1 | 1 => 1 | 2 => 1 | 3 => -1 | 4 => 1
  | 5 => 1 | 6 => 1 | 7 => -1 | 8 => -1 | _ => 0

/-- Reversal of a sequence of length `9`. -/
def rev9 (q : ℕ → ℝ) : ℕ → ℝ := fun i => q (8 - i)

/-- **Same composition.**  Every function of the individual charges sums to the same value over
the two sequences: they contain the same multiset of charges (five `+1` and four `-1`), so no
composition-based feature -- net charge, fraction of charged residues, mean square charge --
separates them. -/
theorem homometric_same_composition (f : ℝ → ℝ) :
    ∑ i ∈ range 9, f (chargeA i) = ∑ i ∈ range 9, f (chargeB i) := by
  simp [Finset.sum_range_succ, chargeA, chargeB]
  ring

/-- The two sequences are different sequences. -/
theorem chargeB_ne_chargeA : chargeB ≠ chargeA := by
  intro h
  have := congrArg (fun g => g 0) h
  norm_num [chargeA, chargeB] at this

/-- `chargeB` is not the reversal of `chargeA`. -/
theorem chargeB_ne_rev : chargeB ≠ rev9 chargeA := by
  intro h
  have := congrArg (fun g => g 0) h
  norm_num [chargeA, chargeB, rev9] at this

/-- `chargeB` is not the charge conjugate of `chargeA`. -/
theorem chargeB_ne_neg : chargeB ≠ fun i => -chargeA i := by
  intro h
  have := congrArg (fun g => g 1) h
  norm_num [chargeA, chargeB] at this

/-- `chargeB` is not the charge conjugate of the reversal of `chargeA` either: the pair is not
related by any symmetry of a pairwise charge model. -/
theorem chargeB_ne_neg_rev : chargeB ≠ fun i => -(rev9 chargeA i) := by
  intro h
  have := congrArg (fun g => g 3) h
  norm_num [chargeA, chargeB, rev9] at this

/-- **Equal autocorrelations at every lag.** -/
theorem homometric_autocorr (d : ℕ) : autocorr 9 chargeA d = autocorr 9 chargeB d := by
  by_cases hd : d < 9
  · interval_cases d <;>
      norm_num [autocorr, Finset.sum_range_succ, chargeA, chargeB]
  · have : 9 - d = 0 := by omega
    simp [autocorr, this]

/-- **No pairwise distance model can tell the two sequences apart.**  For every kernel `w`
whatsoever -- screened or unscreened, of any range, fitted or derived -- the two sequences have
exactly the same energy. -/
theorem homometric_blind (w : ℕ → ℝ) : pairEnergy 9 w chargeA = pairEnergy 9 w chargeB :=
  (autocorr_eq_iff_pairEnergy_eq 9 chargeA chargeB).mp
    (fun d _ _ => homometric_autocorr d) w

/-- In particular the two sequences have identical sequence charge decoration. -/
theorem homometric_scd : scd 9 chargeA = scd 9 chargeB := by
  unfold scd
  rw [homometric_blind]

/-- And identical Debye--Hückel energy at every bond length and every screening length: no
titration of salt separates them within the model class. -/
theorem homometric_debye (b kappa : ℝ) : debye 9 b kappa chargeA = debye 9 b kappa chargeB := by
  unfold debye
  rw [homometric_blind]

/-! ## But a three-body term separates them -/

/-- A three-body charge correlator: `sum_i q i * q (i+1) * q (i+3)`, the simplest statistic of a
charge sequence that is not a function of the pair separations alone. -/
def triple (N : ℕ) (q : ℕ → ℝ) : ℝ := ∑ i ∈ range (N - 3), q i * q (i + 1) * q (i + 3)

/-- **The homometric pair is separated by a three-body correlator.**  `triple` takes the value
`2` on `chargeA` and `-2` on `chargeB`.  Hence no pairwise distance model reproduces all
sequence-dependent observables: the model class of `pairEnergy` is provably incomplete. -/
theorem homometric_triple_ne : triple 9 chargeA ≠ triple 9 chargeB := by
  norm_num [triple, Finset.sum_range_succ, chargeA, chargeB]

theorem triple_chargeA : triple 9 chargeA = 2 := by
  norm_num [triple, Finset.sum_range_succ, chargeA]

theorem triple_chargeB : triple 9 chargeB = -2 := by
  norm_num [triple, Finset.sum_range_succ, chargeB]

end Pattern

end IDR
