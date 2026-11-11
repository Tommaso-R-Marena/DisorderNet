import Mathlib

/-!
# Part CXLV — Disulfide topology and the loop entropy of a disordered region

Cysteines are the one chemistry that can staple a disordered region to itself.  A disulfide
between residues `i < j` closes a loop, and closing a loop costs entropy: for a random coil
the probability that the two ends meet falls off as a power of the loop length, so the free
energy of closure is

    loopFreeEnergy c0 nu L = c0 + nu * log L,

with `nu = 3/2` for an ideal chain and `nu ≈ 1.8–2.2` once excluded volume is included.  This
file proves what such a model can and cannot deliver.

* `loopFreeEnergy_strictMono` — longer loops always cost more; this is the qualitative
  content that is never in doubt.
* `single_loop_unidentifiable` — one measured loop-closure free energy is compatible with
  *every* exponent `nu`; the intercept absorbs the difference.  A model calibrated on one
  disulfide has not measured a scaling law.
* `two_loops_identify` — two loops of different length pin `c0` and `nu` uniquely, and
  `exponent_formula` gives the exponent in closed form.
* `three_loops_colinear` / `power_law_falsifiable` — three loops give a *test*: the three
  points must be colinear in `(log L, ΔG)`, and any deviation refutes the single power law.
* The topology theorems.  For four cysteines at positions `p0 < p1 < p2 < p3` there are three
  pairings.  The beads-on-a-string pairing `(01)(23)` always beats the crossed pairing
  `(02)(13)` (`sequential_lt_crossed`), but the comparison with the nested pairing `(03)(12)`
  is **not** universal: it is decided by the explicit inequality `a*c` versus `b*(a+b+c)` in
  the three gaps `a, b, c` (`sequential_lt_nested_iff`), and both outcomes actually occur
  (`nested_can_win`, `sequential_can_win`).

The last point is the design content.  A model that hard-codes a topology preference for
disulfide-bonded disordered regions — "nested is strained", "sequential is cheapest" — is
wrong for an explicitly computable set of cysteine spacings.  The preference is a function of
the spacings, and it flips.
-/

noncomputable section

namespace RequestProject.Disulfide

open Real

/-- Free energy of closing a loop of contour length `L`, in units of `k_B T`:
an offset `c0` fixed by the chemistry of the bond plus `nu * log L` of chain entropy. -/
def loopFreeEnergy (c0 nu L : ℝ) : ℝ := c0 + nu * Real.log L

/-! ### What one, two and three loops are worth -/

/-- **Longer loops cost more.** -/
theorem loopFreeEnergy_strictMono {c0 nu : ℝ} (hnu : 0 < nu) {L L' : ℝ}
    (hL : 0 < L) (hLL : L < L') :
    loopFreeEnergy c0 nu L < loopFreeEnergy c0 nu L' := by
  unfold loopFreeEnergy
  have : Real.log L < Real.log L' := Real.log_lt_log hL hLL
  nlinarith

/-- **One disulfide measures nothing about the exponent.**  Whatever value `nu'` one wants
to assume, there is an offset that reproduces the measured closure free energy exactly. -/
theorem single_loop_unidentifiable (c0 nu L : ℝ) (nu' : ℝ) :
    ∃ c0' : ℝ, loopFreeEnergy c0' nu' L = loopFreeEnergy c0 nu L :=
  ⟨c0 + nu * Real.log L - nu' * Real.log L, by unfold loopFreeEnergy; ring⟩

/-- Closed form for the exponent read off two loops. -/
theorem exponent_formula {c0 nu L L' : ℝ} (hne : Real.log L ≠ Real.log L') :
    (loopFreeEnergy c0 nu L' - loopFreeEnergy c0 nu L) / (Real.log L' - Real.log L) = nu := by
  unfold loopFreeEnergy
  have h : Real.log L' - Real.log L ≠ 0 := sub_ne_zero.2 (Ne.symm hne)
  field_simp
  ring

/-- **Two loops of different length identify the model.**  If two parameter pairs reproduce
the same two closure free energies, they are the same pair. -/
theorem two_loops_identify {c0 nu c0' nu' L L' : ℝ}
    (hne : Real.log L ≠ Real.log L')
    (h1 : loopFreeEnergy c0' nu' L = loopFreeEnergy c0 nu L)
    (h2 : loopFreeEnergy c0' nu' L' = loopFreeEnergy c0 nu L') :
    c0' = c0 ∧ nu' = nu := by
  unfold loopFreeEnergy at h1 h2
  have hd : (nu' - nu) * (Real.log L' - Real.log L) = 0 := by nlinarith
  have hlog : Real.log L' - Real.log L ≠ 0 := sub_ne_zero.2 (Ne.symm hne)
  have hnu : nu' = nu := by
    rcases mul_eq_zero.1 hd with h | h
    · linarith
    · exact absurd h hlog
  refine ⟨?_, hnu⟩
  rw [hnu] at h1
  linarith

/-- **The power law is a testable statement.**  Three loops obeying a single power law give
three colinear points in the `(log L, ΔG)` plane. -/
theorem three_loops_colinear (c0 nu L1 L2 L3 : ℝ) :
    (loopFreeEnergy c0 nu L2 - loopFreeEnergy c0 nu L1) * (Real.log L3 - Real.log L1)
      = (loopFreeEnergy c0 nu L3 - loopFreeEnergy c0 nu L1) * (Real.log L2 - Real.log L1) := by
  unfold loopFreeEnergy
  ring

/-- **Falsifiability.**  If the three measured closure free energies are *not* colinear in
`log L`, then no offset and exponent whatsoever reproduce them: the single power law is
refuted by three disulfides. -/
theorem power_law_falsifiable {L1 L2 L3 g1 g2 g3 : ℝ}
    (hdev : (g2 - g1) * (Real.log L3 - Real.log L1)
      ≠ (g3 - g1) * (Real.log L2 - Real.log L1)) :
    ¬ ∃ c0 nu : ℝ, loopFreeEnergy c0 nu L1 = g1 ∧ loopFreeEnergy c0 nu L2 = g2 ∧
        loopFreeEnergy c0 nu L3 = g3 := by
  rintro ⟨c0, nu, h1, h2, h3⟩
  apply hdev
  rw [← h1, ← h2, ← h3]
  exact three_loops_colinear c0 nu L1 L2 L3

/-! ### Topology: which pairing of four cysteines is cheapest -/

/-- Total loop free energy of a pairing whose two loops have lengths `L` and `L'`. -/
def pairingCost (c0 nu L L' : ℝ) : ℝ := loopFreeEnergy c0 nu L + loopFreeEnergy c0 nu L'

/-- Cost of the beads-on-a-string pairing `(01)(23)` of four cysteines whose consecutive
gaps are `a, b, c`. -/
def seqCost (c0 nu a _b c : ℝ) : ℝ := pairingCost c0 nu a c

/-- Cost of the crossed pairing `(02)(13)`. -/
def crossedCost (c0 nu a b c : ℝ) : ℝ := pairingCost c0 nu (a + b) (b + c)

/-- Cost of the nested pairing `(03)(12)`. -/
def nestedCost (c0 nu a b c : ℝ) : ℝ := pairingCost c0 nu (a + b + c) b

lemma pairingCost_eq (c0 nu L L' : ℝ) (hL : 0 < L) (hL' : 0 < L') :
    pairingCost c0 nu L L' = 2 * c0 + nu * Real.log (L * L') := by
  unfold pairingCost loopFreeEnergy
  rw [Real.log_mul hL.ne' hL'.ne']
  ring

/-- Comparing two pairings is comparing the products of their loop lengths. -/
lemma pairingCost_lt_iff {c0 nu : ℝ} (hnu : 0 < nu) {L L' M M' : ℝ}
    (hL : 0 < L) (hL' : 0 < L') (hM : 0 < M) (hM' : 0 < M') :
    pairingCost c0 nu L L' < pairingCost c0 nu M M' ↔ L * L' < M * M' := by
  rw [pairingCost_eq c0 nu L L' hL hL', pairingCost_eq c0 nu M M' hM hM']
  constructor
  · intro h
    have hlog : Real.log (L * L') < Real.log (M * M') := by nlinarith
    exact (Real.log_lt_log_iff (by positivity) (by positivity)).1 hlog
  · intro h
    have hlog : Real.log (L * L') < Real.log (M * M') :=
      Real.log_lt_log (by positivity) h
    nlinarith

/-- **The crossed pairing is never the cheapest.**  For any spacings, stapling
`(02)(13)` costs strictly more than stapling `(01)(23)`. -/
theorem sequential_lt_crossed {c0 nu a b c : ℝ} (hnu : 0 < nu)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    seqCost c0 nu a b c < crossedCost c0 nu a b c := by
  unfold seqCost crossedCost
  rw [pairingCost_lt_iff hnu ha hc (by linarith) (by linarith)]
  nlinarith

/-- **The nested comparison is not universal.**  Beads-on-a-string beats nested exactly when
the product of the outer gaps is smaller than `b (a+b+c)`. -/
theorem sequential_lt_nested_iff {c0 nu a b c : ℝ} (hnu : 0 < nu)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    seqCost c0 nu a b c < nestedCost c0 nu a b c ↔ a * c < (a + b + c) * b := by
  unfold seqCost nestedCost
  rw [pairingCost_lt_iff hnu ha hc (by linarith) hb]

/-- With a long middle gap the nested pairing is the more expensive one. -/
theorem sequential_can_win {c0 nu : ℝ} (hnu : 0 < nu) :
    seqCost c0 nu 1 1 1 < nestedCost c0 nu 1 1 1 := by
  rw [sequential_lt_nested_iff hnu one_pos one_pos one_pos]
  norm_num

/-- With a short middle gap and long flanks the nested pairing is the cheaper one: a model
that always prefers the sequential topology is wrong here. -/
theorem nested_can_win {c0 nu : ℝ} (hnu : 0 < nu) :
    nestedCost c0 nu 10 1 10 < seqCost c0 nu 10 1 10 := by
  unfold seqCost nestedCost
  rw [pairingCost_lt_iff hnu (by norm_num) one_pos (by norm_num) (by norm_num)]
  norm_num

/-- **The topology law.**  The three pairings of four cysteines are ordered by the products
of their loop lengths; the crossed pairing is always beaten by the sequential one; the
sequential–nested comparison is decided by an explicit inequality in the gaps, and both
orders occur. -/
theorem disulfide_topology_law {c0 nu : ℝ} (hnu : 0 < nu) :
    (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c →
        seqCost c0 nu a b c < crossedCost c0 nu a b c) ∧
    (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c →
        (seqCost c0 nu a b c < nestedCost c0 nu a b c ↔ a * c < (a + b + c) * b)) ∧
    seqCost c0 nu 1 1 1 < nestedCost c0 nu 1 1 1 ∧
    nestedCost c0 nu 10 1 10 < seqCost c0 nu 10 1 10 :=
  ⟨fun _ _ _ ha hb hc => sequential_lt_crossed hnu ha hb hc,
    fun _ _ _ ha hb hc => sequential_lt_nested_iff hnu ha hb hc,
    sequential_can_win hnu, nested_can_win hnu⟩

/-- **The loop-entropy design law.**  A single disulfide fixes no exponent; two of different
length fix the model uniquely and give the exponent in closed form; three of different length
turn the power law into a refutable claim. -/
theorem loop_entropy_design_law (c0 nu : ℝ) (hnu : 0 < nu) :
    (∀ L nu' : ℝ, ∃ c0' : ℝ, loopFreeEnergy c0' nu' L = loopFreeEnergy c0 nu L) ∧
    (∀ L L' : ℝ, Real.log L ≠ Real.log L' →
      (loopFreeEnergy c0 nu L' - loopFreeEnergy c0 nu L) / (Real.log L' - Real.log L) = nu) ∧
    (∀ c0' nu' L L' : ℝ, Real.log L ≠ Real.log L' →
      loopFreeEnergy c0' nu' L = loopFreeEnergy c0 nu L →
      loopFreeEnergy c0' nu' L' = loopFreeEnergy c0 nu L' → c0' = c0 ∧ nu' = nu) ∧
    (∀ L L' : ℝ, 0 < L → L < L' → loopFreeEnergy c0 nu L < loopFreeEnergy c0 nu L') :=
  ⟨fun L nu' => single_loop_unidentifiable c0 nu L nu',
    fun _ _ hne => exponent_formula hne,
    fun _ _ _ _ hne h1 h2 => two_loops_identify hne h1 h2,
    fun _ _ hL hLL => loopFreeEnergy_strictMono hnu hL hLL⟩

end RequestProject.Disulfide
