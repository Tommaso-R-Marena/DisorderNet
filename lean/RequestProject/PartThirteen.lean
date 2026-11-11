/-
# Part XIII  From a sequence Hamiltonian: when a region can order, and when it cannot

Part XII counted the conformations of a chain that cannot overlap itself.  Part XIII puts a
*microscopic, sequence-dependent energy* on those conformations -- the hydrophobic/polar
contact Hamiltonian of `RequestProject.HPModel` -- and asks the design question again, now
with no idealised ensemble anywhere in the statement.

* `square_hp_no_folding` : on the square lattice, if the contact energy satisfies
  `8·beta·eps < log 2` then, for every chain of at least three bonds and **every** sequence,
  no conformation carries half of the equilibrium population.  Ordering a chain is not a
  matter of finding the right sequence: below a threshold contact energy, measured in units of
  `kT`, the conformational entropy of the excluded-volume chain wins outright.
* `square_low_hydrophobicity_disordered` : the same bound refined by composition -- only
  hydrophobic residues can supply contact energy, so a hydrophobic fraction `f` with
  `8·f·beta·eps < log 2` already forces disorder.
* `IDR.sequence_hamiltonian_design_laws` : the bundle -- extensivity of the contact energy,
  the bounded energy range, the entropy/energy threshold in general and on the square lattice,
  the polar chain as an exactly disordered target at every temperature, and the persistence of
  the capacity requirement as the temperature goes to zero.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Statistics
import RequestProject.Metric
import RequestProject.FreeEnergy
import RequestProject.Boltzmann
import RequestProject.SelfAvoiding
import RequestProject.PartTwelve
import RequestProject.HPModel

namespace IDR

open IDR.SAW IDR.HP IDR.FreeEnergy
open scoped Classical

namespace HP

section General

variable {V : Type*} [AddCommGroup V] [DecidableEq V] {q : ℕ}

/-- If a lattice supports at least `r ^ n` self-avoiding conformations then, after one of them
is set aside, at least `r ^ (n-1)` remain. -/
lemma pow_pred_le_cnt_pred {dir : Fin q → V} {r n : ℕ} (hr : 2 ≤ r) (hn : 1 ≤ n)
    (hpow : ∀ m, r ^ m ≤ cntOf dir m) : r ^ (n - 1) ≤ cntOf dir n - 1 := by
  have h1 : r ^ n ≤ cntOf dir n := hpow n
  have hrn : r ^ n = r * r ^ (n - 1) := by
    rw [← pow_succ']
    congr 1
    omega
  have hpos : 1 ≤ r ^ (n - 1) := Nat.one_le_pow _ _ (by omega)
  have h3 : 2 * r ^ (n - 1) ≤ r * r ^ (n - 1) := Nat.mul_le_mul_right _ hr
  omega

/-- **The conformational entropy of an excluded-volume chain is extensive.**  With at least
`r ^ n` conformations, the logarithm of the number of competitors of any one conformation is at
least `(n-1) log r`. -/
lemma log_cnt_pred_ge {dir : Fin q → V} {r n : ℕ} (hr : 2 ≤ r) (hn : 1 ≤ n)
    (hpow : ∀ m, r ^ m ≤ cntOf dir m) :
    ((n : ℝ) - 1) * Real.log r ≤ Real.log ((cntOf dir n - 1 : ℕ) : ℝ) := by
  have hple : r ^ (n - 1) ≤ cntOf dir n - 1 := pow_pred_le_cnt_pred hr hn hpow
  have hcast : ((r : ℝ)) ^ (n - 1) ≤ ((cntOf dir n - 1 : ℕ) : ℝ) := by exact_mod_cast hple
  have hrpos : (0 : ℝ) < (r : ℝ) := by
    have : (2 : ℝ) ≤ (r : ℝ) := by exact_mod_cast hr
    linarith
  have hposl : (0 : ℝ) < (r : ℝ) ^ (n - 1) := by positivity
  have hlog := Real.log_le_log hposl hcast
  rwa [Real.log_pow, Nat.cast_sub hn, Nat.cast_one] at hlog

/-- **A sequence-independent threshold for order, on any lattice.**  Let the lattice have `q`
bond vectors and at least `r ^ n` self-avoiding conformations of `n` bonds, with `r ≥ 2`.  If
the hydrophobic contact energy obeys `2·q·beta·eps < log r`, then for every chain of at least
three bonds, every hydrophobic/polar sequence and every conformation, the equilibrium
population of that conformation is below one half.  The two competing quantities are both
extensive: the contact energy can fall at most `eps·q` per residue, the conformational entropy
gains at least `log r` per residue, and the comparison of the two slopes decides. -/
theorem hp_no_folding_of_growth {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) (hq : 0 < q)
    {r n : ℕ} (hr : 2 ≤ r) (hpow : ∀ m, r ^ m ≤ cntOf dir m) (hn : 3 ≤ n)
    {eps beta : ℝ} (heps : 0 ≤ eps) (hbeta : 0 < beta)
    (hthr : 2 * (q : ℝ) * (beta * eps) < Real.log r) :
    ∀ (seq : Fin (n + 1) → Bool) (j : Fin (cntOf dir n)),
      boltz beta (hpU dir eps seq) j < 1 / 2 := by
  have hn1 : 1 ≤ n := by omega
  have hrn : 1 < r ^ n := by
    have : 2 ^ n ≤ r ^ n := Nat.pow_le_pow_left hr n
    have h2 : 2 ≤ 2 ^ n := by
      calc 2 = 2 ^ 1 := by norm_num
        _ ≤ 2 ^ n := Nat.pow_le_pow_right (by norm_num) hn1
    omega
  have hcnt : 1 < cntOf dir n := lt_of_lt_of_le hrn (hpow n)
  have hlogr : 0 < Real.log r := Real.log_pos (by exact_mod_cast hr.trans_lt' (by norm_num))
  have hqR : (0 : ℝ) < q := by exact_mod_cast hq
  have hnR : (3 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hkey : beta * (eps * (((n : ℝ) + 1) * q)) < Real.log ((cntOf dir n - 1 : ℕ) : ℝ) := by
    have hentropy : ((n : ℝ) - 1) * Real.log r ≤ Real.log ((cntOf dir n - 1 : ℕ) : ℝ) :=
      log_cnt_pred_ge hr hn1 hpow
    have hid : beta * (eps * (((n : ℝ) + 1) * q))
        = (((n : ℝ) + 1) / 2) * (2 * (q : ℝ) * (beta * eps)) := by ring
    have hposn : (0 : ℝ) < ((n : ℝ) + 1) / 2 := by linarith
    have hstep := mul_lt_mul_of_pos_left hthr hposn
    have hcmp : (((n : ℝ) + 1) / 2) * Real.log r ≤ ((n : ℝ) - 1) * Real.log r := by
      have h1 : ((n : ℝ) + 1) / 2 ≤ (n : ℝ) - 1 := by linarith
      nlinarith [hlogr]
    rw [hid]
    linarith
  exact hp_no_folding hdir heps hbeta hcnt hkey

/-- **Low hydrophobic content implies intrinsic disorder.**  Only the hydrophobic residues can
supply contact energy, so a sequence with `h` of them cannot order once
`h·q·beta·eps < (n-1)·log r`: the depth of the landscape it can build falls short of the
conformational entropy of the excluded-volume chain.  This is the microscopic form of the
empirical rule that regions of low mean hydrophobicity are disordered. -/
theorem hp_no_folding_of_hydrophobicity {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n)
    {r n : ℕ} (hr : 2 ≤ r) (hpow : ∀ m, r ^ m ≤ cntOf dir m) (hn : 1 ≤ n) {eps beta : ℝ}
    (heps : 0 ≤ eps) (hbeta : 0 < beta) {seq : Fin (n + 1) → Bool}
    (hthr : (hCount seq : ℝ) * ((q : ℝ) * (beta * eps)) < ((n : ℝ) - 1) * Real.log r) :
    ∀ j : Fin (cntOf dir n), boltz beta (hpU dir eps seq) j < 1 / 2 := by
  have hrn : 1 < r ^ n := by
    have h1 : 2 ^ n ≤ r ^ n := Nat.pow_le_pow_left hr n
    have h2 : 2 ≤ 2 ^ n := by
      calc 2 = 2 ^ 1 := by norm_num
        _ ≤ 2 ^ n := Nat.pow_le_pow_right (by norm_num) hn
    omega
  have hcnt : 1 < cntOf dir n := lt_of_lt_of_le hrn (hpow n)
  refine hp_no_folding_hCount hdir heps hbeta hcnt ?_
  calc beta * (eps * ((hCount seq : ℝ) * (q : ℝ)))
      = (hCount seq : ℝ) * ((q : ℝ) * (beta * eps)) := by ring
    _ < ((n : ℝ) - 1) * Real.log r := hthr
    _ ≤ Real.log ((cntOf dir n - 1 : ℕ) : ℝ) := log_cnt_pred_ge hr hn hpow

/-- The same bound in terms of the *fraction* `f` of hydrophobic residues, for a lattice of
coordination number `q` and growth rate `r`: `2·q·f·beta·eps < log r` forbids order. -/
theorem hp_no_folding_of_fraction {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n)
    {r n : ℕ} (hr : 2 ≤ r) (hpow : ∀ m, r ^ m ≤ cntOf dir m) (hn : 3 ≤ n) {eps beta f : ℝ}
    (heps : 0 ≤ eps) (hbeta : 0 < beta) {seq : Fin (n + 1) → Bool}
    (hf : (hCount seq : ℝ) ≤ f * ((n : ℝ) + 1))
    (hthr : 2 * (q : ℝ) * (f * (beta * eps)) < Real.log r) :
    ∀ j : Fin (cntOf dir n), boltz beta (hpU dir eps seq) j < 1 / 2 := by
  have hn1 : 1 ≤ n := by omega
  have hnR : (3 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hlogr : 0 < Real.log r := Real.log_pos (by exact_mod_cast hr.trans_lt' (by norm_num))
  have hqR : (0 : ℝ) ≤ (q : ℝ) := Nat.cast_nonneg _
  have hbe : (0 : ℝ) ≤ beta * eps := mul_nonneg (le_of_lt hbeta) heps
  refine hp_no_folding_of_hydrophobicity hdir hr hpow hn1 heps hbeta ?_
  have hstep : (hCount seq : ℝ) * ((q : ℝ) * (beta * eps))
      ≤ (f * ((n : ℝ) + 1)) * ((q : ℝ) * (beta * eps)) :=
    mul_le_mul_of_nonneg_right hf (mul_nonneg hqR hbe)
  have hid : (f * ((n : ℝ) + 1)) * ((q : ℝ) * (beta * eps))
      = (((n : ℝ) + 1) / 2) * (2 * (q : ℝ) * (f * (beta * eps))) := by ring
  have hposn : (0 : ℝ) < ((n : ℝ) + 1) / 2 := by linarith
  have hlt := mul_lt_mul_of_pos_left hthr hposn
  have hcmp : (((n : ℝ) + 1) / 2) * Real.log r ≤ ((n : ℝ) - 1) * Real.log r := by
    have h1 : ((n : ℝ) + 1) / 2 ≤ (n : ℝ) - 1 := by linarith
    nlinarith [hlogr]
  rw [hid] at hstep
  linarith

end General

/-- **The square lattice.**  `8·beta·eps < log 2` -- a contact energy below about a tenth of
`kT` -- forbids order for every sequence and every chain of at least three bonds. -/
theorem square_hp_no_folding {n : ℕ} (hn : 3 ≤ n) {eps beta : ℝ} (heps : 0 ≤ eps)
    (hbeta : 0 < beta) (hthr : 8 * (beta * eps) < Real.log 2) :
    ∀ (seq : Fin (n + 1) → Bool) (j : Fin (cnt n)),
      boltz beta (hpU dir eps seq) j < 1 / 2 := by
  refine hp_no_folding_of_growth (dir := dir) cnt_pos (by norm_num) (r := 2) (by norm_num)
    two_pow_le_cnt hn heps hbeta ?_
  have h2 : Real.log ((2 : ℕ) : ℝ) = Real.log 2 := by norm_num
  rw [h2]
  push_cast
  linarith

/-- **The cubic lattice** -- the case of a real polypeptide, six bond directions.  Here
`12·beta·eps < log 3` forbids order for every sequence and every chain of at least three
bonds. -/
theorem cubic_hp_no_folding {n : ℕ} (hn : 3 ≤ n) {eps beta : ℝ} (heps : 0 ≤ eps)
    (hbeta : 0 < beta) (hthr : 12 * (beta * eps) < Real.log 3) :
    ∀ (seq : Fin (n + 1) → Bool) (j : Fin (cnt3 n)),
      boltz beta (hpU dir3 eps seq) j < 1 / 2 := by
  refine hp_no_folding_of_growth (dir := dir3) cnt3_pos (by norm_num) (r := 3) (by norm_num)
    three_pow_le_cnt3 hn heps hbeta ?_
  have h3 : Real.log ((3 : ℕ) : ℝ) = Real.log 3 := by norm_num
  rw [h3]
  push_cast
  linarith

/-- **The square lattice, refined by hydrophobic content.**  A sequence in which a fraction at
most `f` of the residues is hydrophobic cannot order as soon as `8·f·beta·eps < log 2`. -/
theorem square_low_hydrophobicity_disordered {n : ℕ} (hn : 3 ≤ n) {eps beta f : ℝ}
    (heps : 0 ≤ eps) (hbeta : 0 < beta) {seq : Fin (n + 1) → Bool}
    (hf : (hCount seq : ℝ) ≤ f * ((n : ℝ) + 1)) (hthr : 8 * (f * (beta * eps)) < Real.log 2) :
    ∀ j : Fin (cnt n), boltz beta (hpU dir eps seq) j < 1 / 2 := by
  refine hp_no_folding_of_fraction (dir := dir) cnt_pos (r := 2) (by norm_num) two_pow_le_cnt
    hn heps hbeta hf ?_
  have h2 : Real.log ((2 : ℕ) : ℝ) = Real.log 2 := by norm_num
  rw [h2]
  push_cast
  linarith

/-- **The cubic lattice, refined by hydrophobic content**: `12·f·beta·eps < log 3` forbids
order. -/
theorem cubic_low_hydrophobicity_disordered {n : ℕ} (hn : 3 ≤ n) {eps beta f : ℝ}
    (heps : 0 ≤ eps) (hbeta : 0 < beta) {seq : Fin (n + 1) → Bool}
    (hf : (hCount seq : ℝ) ≤ f * ((n : ℝ) + 1)) (hthr : 12 * (f * (beta * eps)) < Real.log 3) :
    ∀ j : Fin (cnt3 n), boltz beta (hpU dir3 eps seq) j < 1 / 2 := by
  refine hp_no_folding_of_fraction (dir := dir3) cnt3_pos (r := 3) (by norm_num)
    three_pow_le_cnt3 hn heps hbeta hf ?_
  have h3 : Real.log ((3 : ℕ) : ℝ) = Real.log 3 := by norm_num
  rw [h3]
  push_cast
  linarith

end HP

/-- **Part XIII, the design laws of a sequence Hamiltonian.**  Everything below is derived from
an explicit microscopic energy -- one unit `eps` per hydrophobic contact between residues that
are lattice neighbours but not bonded neighbours -- on the self-avoiding conformations of
Part XII.

1. *The energy is capped by excluded volume*: a self-avoiding chain of `n` bonds on a lattice
   of coordination number `q` has at most `(n+1)·q` contacts, so the whole energy landscape
   spans at most `eps·(n+1)·q`.
2. *Order requires the energy to beat the entropy*: if any conformation holds half the
   population then `log (cnt n − 1)/beta ≤ eps·(n+1)·q`.
3. *Hence a sequence-independent threshold*: on the square lattice, `8·beta·eps < log 2`
   forbids any sequence of length `≥ 4` from populating a single conformation to one half.
3''. *Refined by hydrophobic content*: only hydrophobic residues supply contact energy, so a
   sequence in which a fraction at most `f` of the residues is hydrophobic is already
   disordered once `8·f·beta·eps < log 2` -- the microscopic form of the rule that regions of
   low mean hydrophobicity are disordered.
3'. *And the criterion is sharp*: a unique lowest-energy conformation separated by
   `beta·gap ≥ log (cnt n)` does hold half the population, so order occurs exactly when the
   energy gap is of the order of `kT` times the conformational entropy.
4. *A polar region is exactly the athermal self-avoiding ensemble*, at every temperature, and
   modelling it exactly costs `cnt n ≥ 2 ^ n` mixture components.
5. *Cooling does not help*: for any sequence, the capacity requirement at low temperature is
   the ground-state degeneracy, `|S| − tol·(|S| + cnt n · e^{−beta·gap})`. -/
theorem sequence_hamiltonian_design_laws (n : ℕ) (hn : 3 ≤ n) (eps beta : ℝ) (heps : 0 ≤ eps)
    (hbeta : 0 < beta) :
    -- 1. excluded volume caps the contact energy
    (∀ (seq : Fin (n + 1) → Bool) (w : Fin n → Fin 4), IsSAW (stepsOfDir dir w) →
        (contacts dir seq w).card ≤ (n + 1) * 4 ∧
        -(eps * (((n : ℝ) + 1) * 4)) ≤ hpEnergy dir eps seq w ∧
        hpEnergy dir eps seq w ≤ 0) ∧
    -- 2. ordering demands an energy gap of the size of the conformational entropy
    (∀ (seq : Fin (n + 1) → Bool) (j0 : Fin (cnt n)),
        1 / 2 ≤ boltz beta (hpU dir eps seq) j0 →
        Real.log ((cnt n - 1 : ℕ) : ℝ) / beta ≤ eps * (((n : ℝ) + 1) * 4)) ∧
    -- 3. below the threshold no sequence orders: every conformation stays below one half
    (8 * (beta * eps) < Real.log 2 →
        ∀ (seq : Fin (n + 1) → Bool) (x : Fin n → Fin 4),
          (hpEns dir cnt_pos eps beta seq).prob x < 1 / 2) ∧
    -- 3''. refined by hydrophobic content: a fraction `f` of hydrophobic residues with
    -- `8·f·beta·eps < log 2` already forbids order
    (∀ (f : ℝ) (seq : Fin (n + 1) → Bool), (hCount seq : ℝ) ≤ f * ((n : ℝ) + 1) →
        8 * (f * (beta * eps)) < Real.log 2 →
        ∀ x : Fin n → Fin 4, (hpEns dir cnt_pos eps beta seq).prob x < 1 / 2) ∧
    -- 3'. and the criterion is sharp: a gap of `kT · log (cnt n)` does order the chain
    (∀ (seq : Fin (n + 1) → Bool) (Umin gap : ℝ) (j0 : Fin (cnt n)),
        hpU dir eps seq j0 = Umin → (∀ j, j ≠ j0 → Umin + gap ≤ hpU dir eps seq j) →
        Real.log (cnt n) ≤ beta * gap →
        1 / 2 ≤ (hpEns dir cnt_pos eps beta seq).prob (enumOf dir n j0)) ∧
    -- 4. a polar region is the athermal self-avoiding ensemble, and costs `2 ^ n` components
    (∀ seq : Fin (n + 1) → Bool, (∀ i, seq i = false) →
        (hpEns dir cnt_pos eps beta seq).Same (sawEnsOf dir cnt_pos n) ∧
        ∀ M : Ens (Fin n → Fin 4), M.Same (hpEns dir cnt_pos eps beta seq) → 2 ^ n ≤ M.card) ∧
    -- 5. at low temperature the capacity requirement is the ground-state degeneracy
    (∀ (seq : Fin (n + 1) → Bool) (Umin gap : ℝ) (S : Finset (Fin (cnt n))),
        (∀ j ∈ S, hpU dir eps seq j = Umin) → (∀ j ∉ S, Umin + gap ≤ hpU dir eps seq j) →
        ∀ (k : ℕ) (M : Ens (Fin n → Fin 4)), M.card ≤ k → ∀ tol : ℝ,
          ApproxSame tol M (hpEns dir cnt_pos eps beta seq) →
          (S.card : ℝ) - tol * ((S.card : ℝ) + cnt n * Real.exp (-beta * gap)) ≤ k) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro seq w hw
    refine ⟨contacts_card_le hw, ?_, hpEnergy_nonpos heps seq w⟩
    have := hpEnergy_ge (dir := dir) (eps := eps) (seq := seq) (w := w) heps hw
    have hcast : ((4 : ℕ) : ℝ) = (4 : ℝ) := by norm_num
    rw [hcast] at this
    exact this
  · intro seq j0 hhalf
    have hcnt : 1 < cnt n := by
      have h1 : 2 ^ n ≤ cnt n := two_pow_le_cnt n
      have h2 : 2 ≤ 2 ^ n := by
        calc 2 = 2 ^ 1 := by norm_num
          _ ≤ 2 ^ n := Nat.pow_le_pow_right (by norm_num) (by omega)
      omega
    have := hp_ordering_threshold (dir := dir) cnt_pos heps hbeta seq hcnt hhalf
    have hcast : ((4 : ℕ) : ℝ) = (4 : ℝ) := by norm_num
    rw [hcast] at this
    exact this
  · intro hthr seq x
    exact hpEns_prob_lt cnt_pos (by norm_num) (HP.square_hp_no_folding hn heps hbeta hthr seq) x
  · intro f seq hf hthr x
    exact hpEns_prob_lt cnt_pos (by norm_num)
      (HP.square_low_hydrophobicity_disordered hn heps hbeta hf hthr) x
  · intro seq Umin gap j0 hj0 hout hgap
    exact hp_folding_sufficient cnt_pos hbeta hj0 hout hgap
  · intro seq hseq
    refine ⟨polar_ens_same_saw cnt_pos hseq, fun M hM => ?_⟩
    exact le_trans (two_pow_le_cnt n) (polar_chain_capacity cnt_pos hseq hM)
  · intro seq Umin gap S hSmin hout k M hM tol h
    exact hp_degeneracy_capacity cnt_pos hbeta hSmin hout hM h

/-- **Part XIII in three dimensions.**  The same microscopic contact Hamiltonian on the cubic
lattice, the case of a real polypeptide backbone: the contact energy of a self-avoiding chain
is capped by the coordination number, a contact energy below `log 3 / 12` in units of `kT`
leaves every conformation of every sequence below half the population, a hydrophobic fraction
`f` with `12·f·beta·eps < log 3` does the same, and a sequence with no
hydrophobic residues realises exactly the athermal self-avoiding ensemble, whose exact
modelling costs `3 ^ n` mixture components. -/
theorem cubic_sequence_hamiltonian_laws (n : ℕ) (hn : 3 ≤ n) (eps beta : ℝ) (heps : 0 ≤ eps)
    (hbeta : 0 < beta) :
    (∀ (seq : Fin (n + 1) → Bool) (w : Fin n → Fin 6), IsSAW (stepsOfDir dir3 w) →
        (contacts dir3 seq w).card ≤ (n + 1) * 6 ∧
        -(eps * (((n : ℝ) + 1) * 6)) ≤ hpEnergy dir3 eps seq w ∧
        hpEnergy dir3 eps seq w ≤ 0) ∧
    (12 * (beta * eps) < Real.log 3 →
        ∀ (seq : Fin (n + 1) → Bool) (x : Fin n → Fin 6),
          (hpEns dir3 cnt3_pos eps beta seq).prob x < 1 / 2) ∧
    (∀ (f : ℝ) (seq : Fin (n + 1) → Bool), (hCount seq : ℝ) ≤ f * ((n : ℝ) + 1) →
        12 * (f * (beta * eps)) < Real.log 3 →
        ∀ x : Fin n → Fin 6, (hpEns dir3 cnt3_pos eps beta seq).prob x < 1 / 2) ∧
    (∀ seq : Fin (n + 1) → Bool, (∀ i, seq i = false) →
        (hpEns dir3 cnt3_pos eps beta seq).Same (sawEnsOf dir3 cnt3_pos n) ∧
        ∀ M : Ens (Fin n → Fin 6), M.Same (hpEns dir3 cnt3_pos eps beta seq) →
          3 ^ n ≤ M.card) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro seq w hw
    refine ⟨contacts_card_le hw, ?_, hpEnergy_nonpos heps seq w⟩
    have := hpEnergy_ge (dir := dir3) (eps := eps) (seq := seq) (w := w) heps hw
    have hcast : ((6 : ℕ) : ℝ) = (6 : ℝ) := by norm_num
    rw [hcast] at this
    exact this
  · intro hthr seq x
    exact hpEns_prob_lt cnt3_pos (by norm_num) (HP.cubic_hp_no_folding hn heps hbeta hthr seq) x
  · intro f seq hf hthr x
    exact hpEns_prob_lt cnt3_pos (by norm_num)
      (HP.cubic_low_hydrophobicity_disordered hn heps hbeta hf hthr) x
  · intro seq hseq
    refine ⟨polar_ens_same_saw cnt3_pos hseq, fun M hM => ?_⟩
    exact le_trans (three_pow_le_cnt3 n) (polar_chain_capacity cnt3_pos hseq hM)

end IDR
