/-
# Part LXXX  Crosslinking mass spectrometry: what a partial yield proves

A chemical crosslinker joins two residues only in those conformations in which the two side
chains come within the reach of its spacer arm.  What the mass spectrometer then reports for a
given residue pair is *not* a distance: it is the **fraction of the ensemble** in which that pair
is within reach.  This file makes that reading exact and draws out what it does and does not
license.

The measured quantity is
`freq E d r = E.expect (indicator (d x ≤ r))`,
the yield of the crosslink whose site pair has conformation-dependent distance `d` and whose
spacer reaches `r`.

* `freq_nonneg`, `freq_le_one`, `freq_mono` -- a yield is a number in `[0,1]`, nondecreasing in
  the spacer length: it is a value of the *cumulative distribution* of that distance, not a
  distance.
* `freq_dirac` -- **a single structure predicts only `0` or `1`.**  Consequently
  `not_deterministic_of_fractional`: a single sub-stoichiometric yield, `0 < f < 1`, already
  refutes every single-structure model of the region -- no geometric analysis of the restraint is
  needed, and no second crosslink.
* `Exclusive`, `sum_freq_le_one` -- mutually exclusive crosslinks (no conformation satisfies two
  of them) have yields summing to at most one.  This is the *consistency test* on a crosslink
  data set: yields of mutually exclusive links that sum above one cannot come from any ensemble.
* `card_ge_of_exclusive` -- **the multiplicity theorem.**  `k` mutually exclusive crosslinks that
  are all observed force at least `k` distinct conformations, in the target and hence
  (`freq_eq_of_same`, `card_ge_of_exclusive_model`) in every model that reproduces the data.  The
  classical statement "the crosslinks are incompatible with one structure" is thus a quantitative
  lower bound on ensemble size, not merely a refutation.
* `exclusive_of_separated`, `triad_*` -- the geometric input, and an explicit instance: a tail tip
  visiting three anchors `0, 10, 20` with a spacer reaching `3` gives three mutually exclusive
  crosslinks of yield `1/3` each, so every model of that region needs at least three states.
* `mean_distance_from_series` -- **the positive design statement.**  Yields are values of one
  cumulative distribution, so a *series* of crosslinkers with spacer lengths `0, h, 2h, …` recovers
  the mean distance of the pair by a layer-cake sum, with error at most the spacing `h`.  One
  crosslinker measures one number; a spacer-length series measures a distribution.

Note that `freq E d r` is a linear functional of the population vector, so the counting and
identifiability laws of Parts LXIX and LXX apply verbatim: `k` crosslinks are `k` restraints and
support at most `k+1` verifiable numbers.  What this part adds is the interpretation of the
individual restraint -- a CDF value, never a distance -- and the fact that its *deviation from
`0` or `1`* is itself the evidence for disorder.
-/
import Mathlib
import RequestProject.EnsembleCore

set_option autoImplicit false

namespace IDR

namespace Crosslink

open Finset IDR

variable {X : Type*}

/-- The **crosslink yield**: the fraction of the ensemble in which the site pair whose distance
is `d` lies within the reach `r` of the spacer arm. -/
noncomputable def freq (E : Ens X) (d : X → ℝ) (r : ℝ) : ℝ :=
  E.expect (fun x => if d x ≤ r then (1 : ℝ) else 0)

lemma freq_nonneg (E : Ens X) (d : X → ℝ) (r : ℝ) : 0 ≤ freq E d r :=
  E.expect_nonneg fun x => by by_cases h : d x ≤ r <;> simp [h]

lemma freq_le_one (E : Ens X) (d : X → ℝ) (r : ℝ) : freq E d r ≤ 1 := by
  have := E.expect_mono (f := fun x => if d x ≤ r then (1 : ℝ) else 0) (g := fun _ => 1)
    (fun x => by by_cases h : d x ≤ r <;> simp [h])
  simpa [freq] using this

/-- A yield is nondecreasing in the spacer length: it is a value of the cumulative distribution
of the site–site distance. -/
lemma freq_mono (E : Ens X) (d : X → ℝ) {r r' : ℝ} (h : r ≤ r') :
    freq E d r ≤ freq E d r' :=
  E.expect_mono fun x => by
    by_cases hx : d x ≤ r
    · simp [hx, hx.trans h]
    · by_cases hx' : d x ≤ r' <;> simp [hx, hx']

/-- Yields are ensemble averages, so two observationally equal ensembles have the same yields. -/
lemma freq_eq_of_same {E F : Ens X} (h : E.Same F) (d : X → ℝ) (r : ℝ) :
    freq E d r = freq F d r := h _

lemma freq_pos_iff (E : Ens X) (d : X → ℝ) (r : ℝ) :
    0 < freq E d r ↔ ∃ j, 0 < E.w j ∧ d (E.pt j) ≤ r := by
  constructor
  · intro hpos
    by_contra hcon
    push_neg at hcon
    have hzero : freq E d r = 0 := by
      simp only [freq, Ens.expect]
      refine Finset.sum_eq_zero fun j _ => ?_
      by_cases hj : d (E.pt j) ≤ r
      · have hw : E.w j = 0 :=
          le_antisymm (by
            by_contra hlt
            exact absurd hj (not_le.2 (hcon j (lt_of_not_ge hlt)))) (E.w_nonneg j)
        simp [hj, hw]
      · simp [hj]
    exact absurd hzero (ne_of_gt hpos)
  · rintro ⟨j, hj, hd⟩
    simp only [freq, Ens.expect]
    refine Finset.sum_pos' (fun i _ => ?_) ⟨j, Finset.mem_univ j, ?_⟩
    · by_cases h : d (E.pt i) ≤ r <;> simp [h, E.w_nonneg i]
    · simp [hd, hj]

/-- **A single structure predicts a yield of `0` or `1`.** -/
@[simp] lemma freq_dirac (x : X) (d : X → ℝ) (r : ℝ) :
    freq (Ens.dirac x) d r = if d x ≤ r then 1 else 0 := by
  simp [freq]

/-- **A partial yield refutes every single-structure model.**  A crosslink seen in some but not
all of the sample is, by itself, proof that the region populates at least two conformations. -/
theorem not_deterministic_of_fractional {E : Ens X} {d : X → ℝ} {r : ℝ}
    (h0 : 0 < freq E d r) (h1 : freq E d r < 1) : ¬ E.Deterministic := by
  rintro ⟨x, hx⟩
  have : freq E d r = if d x ≤ r then (1 : ℝ) else 0 := by
    simpa using freq_eq_of_same hx d r
  by_cases hd : d x ≤ r
  · rw [this, if_pos hd] at h1; exact absurd h1 (lt_irrefl 1)
  · rw [this, if_neg hd] at h0; exact absurd h0 (lt_irrefl 0)

/-- Two crosslinks are *mutually exclusive* when no conformation can satisfy both. -/
def Exclusive {ι : Type*} (d : ι → X → ℝ) (r : ι → ℝ) : Prop :=
  ∀ (x : X) (i i' : ι), d i x ≤ r i → d i' x ≤ r i' → i = i'

/-- **The consistency test.**  The yields of mutually exclusive crosslinks sum to at most one;
a data set violating this comes from no ensemble whatsoever. -/
theorem sum_freq_le_one {ι : Type*} [DecidableEq ι] (E : Ens X) (d : ι → X → ℝ) (r : ι → ℝ)
    (hex : Exclusive d r) (s : Finset ι) :
    ∑ i ∈ s, freq E (d i) (r i) ≤ 1 := by
  have hswap : ∑ i ∈ s, freq E (d i) (r i)
      = ∑ j, E.w j * ∑ i ∈ s, (if d i (E.pt j) ≤ r i then (1 : ℝ) else 0) := by
    simp only [freq, Ens.expect, Finset.mul_sum]
    exact Finset.sum_comm
  rw [hswap]
  have hinner : ∀ j, (∑ i ∈ s, (if d i (E.pt j) ≤ r i then (1 : ℝ) else 0)) ≤ 1 := by
    intro j
    have hcard : (s.filter (fun i => d i (E.pt j) ≤ r i)).card ≤ 1 := by
      refine Finset.card_le_one.2 fun a ha b hb => ?_
      simp only [Finset.mem_filter] at ha hb
      exact hex (E.pt j) a b ha.2 hb.2
    calc (∑ i ∈ s, (if d i (E.pt j) ≤ r i then (1 : ℝ) else 0))
        = (s.filter (fun i => d i (E.pt j) ≤ r i)).card := by
          simp
      _ ≤ 1 := by exact_mod_cast hcard
  calc ∑ j, E.w j * ∑ i ∈ s, (if d i (E.pt j) ≤ r i then (1 : ℝ) else 0)
      ≤ ∑ j, E.w j * 1 :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hinner j) (E.w_nonneg j)
    _ = 1 := by simp [E.w_sum]

/-- **The multiplicity theorem.**  `k` mutually exclusive crosslinks, all observed, force at
least `k` distinct populated conformations. -/
theorem card_ge_of_exclusive {k : ℕ} (E : Ens X) (d : Fin k → X → ℝ) (r : Fin k → ℝ)
    (hex : Exclusive d r) (hobs : ∀ i, 0 < freq E (d i) (r i)) : k ≤ E.card := by
  have hpick : ∀ i : Fin k, ∃ j : Fin E.card, 0 < E.w j ∧ d i (E.pt j) ≤ r i := by
    intro i; exact (freq_pos_iff E (d i) (r i)).1 (hobs i)
  choose j hj using hpick
  have hinj : Function.Injective j := by
    intro a b hab
    have ha := (hj a).2
    have hb := (hj b).2
    rw [hab] at ha
    exact hex (E.pt (j b)) a b ha hb
  simpa using Fintype.card_le_of_injective j hinj

/-- The multiplicity bound applies to every model reproducing the data, not just to the truth. -/
theorem card_ge_of_exclusive_model {k : ℕ} {E M : Ens X} (hsame : M.Same E)
    (d : Fin k → X → ℝ) (r : Fin k → ℝ)
    (hex : Exclusive d r) (hobs : ∀ i, 0 < freq E (d i) (r i)) : k ≤ M.card :=
  card_ge_of_exclusive M d r hex fun i => by
    rw [freq_eq_of_same hsame (d i) (r i)]; exact hobs i

/-- The geometric input: crosslinks from a common site to anchors that are pairwise further
apart than twice the spacer reach are mutually exclusive. -/
theorem exclusive_of_separated {ι : Type*} (a : ι → ℝ) (r : ℝ)
    (hsep : ∀ i i', i ≠ i' → 2 * r < |a i - a i'|) :
    Exclusive (fun i (x : ℝ) => |x - a i|) (fun _ => r) := by
  intro x i i' hi hi'
  by_contra hne
  have htri : |a i - a i'| ≤ |x - a i| + |x - a i'| := by
    have h1 : |a i - a i'| ≤ |a i - x| + |x - a i'| := abs_sub_le _ _ _
    rw [abs_sub_comm (a i) x] at h1
    exact h1
  have := hsep i i' hne
  linarith

/-! ### An explicit instance: a tail tip visiting three anchors -/

/-- Three anchor positions along an axis. -/
def anchor : Fin 3 → ℝ := ![0, 10, 20]

/-- The tail tip is found at each of the three anchors a third of the time. -/
noncomputable def triad : Ens ℝ where
  card := 3
  pt := anchor
  w := fun _ => 1 / 3
  w_nonneg := by intro j; norm_num
  w_sum := by simp

@[simp] lemma triad_card : triad.card = 3 := rfl

@[simp] lemma triad_pt (j : Fin triad.card) : triad.pt j = anchor j := rfl

@[simp] lemma triad_w (j : Fin triad.card) : triad.w j = 1 / 3 := rfl

lemma triad_exclusive :
    Exclusive (fun i (x : ℝ) => |x - anchor i|) (fun _ : Fin 3 => (3 : ℝ)) := by
  refine exclusive_of_separated anchor 3 ?_
  intro i i' hne
  fin_cases i <;> fin_cases i' <;> simp_all [anchor] <;> norm_num [abs_of_nonneg, abs_of_nonpos]

lemma triad_freq (i : Fin 3) :
    freq triad (fun x => |x - anchor i|) 3 = 1 / 3 := by
  have hexp : freq triad (fun x => |x - anchor i|) 3
      = ∑ j : Fin 3, (1 / 3 : ℝ) * (if |anchor j - anchor i| ≤ 3 then 1 else 0) := rfl
  rw [hexp, Fin.sum_univ_three]
  fin_cases i <;> norm_num [anchor, Matrix.cons_val_two, Matrix.tail_cons]

/-- **Three mutually exclusive crosslinks, three states.**  Every model that reproduces the three
observed yields must carry at least three conformations. -/
theorem triad_card_ge {M : Ens ℝ} (hsame : M.Same triad) : 3 ≤ M.card :=
  card_ge_of_exclusive_model hsame _ _ triad_exclusive fun i => by
    rw [triad_freq i]; norm_num

/-! ### A spacer-length series measures a distribution -/

/-- The layer-cake estimate of the mean site–site distance from a series of crosslinkers with
spacer reaches `0, h, 2h, …, (n-1)h`. -/
noncomputable def series (E : Ens X) (d : X → ℝ) (h : ℝ) (n : ℕ) : ℝ :=
  h * ∑ i ∈ range n, (1 - freq E d (i * h))

private lemma expect_abs_le (E : Ens X) {f : X → ℝ} {c : ℝ}
    (hf : ∀ j, |f (E.pt j)| ≤ c) : |E.expect f| ≤ c := by
  calc |E.expect f| ≤ ∑ j, |E.w j * f (E.pt j)| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ j, E.w j * |f (E.pt j)| := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [abs_mul, abs_of_nonneg (E.w_nonneg j)]
    _ ≤ ∑ j, E.w j * c :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hf j) (E.w_nonneg j)
    _ = c := by rw [← Finset.sum_mul, E.w_sum, one_mul]

private lemma layer_pointwise {t h : ℝ} (hh : 0 < h) (n : ℕ) (ht0 : 0 ≤ t) (htn : t ≤ n * h) :
    |t - h * ∑ i ∈ range n, (1 - if t ≤ (i : ℝ) * h then (1 : ℝ) else 0)| ≤ h := by
  have hceil_le : ⌈t / h⌉₊ ≤ n := Nat.ceil_le.2 (by rw [div_le_iff₀ hh]; exact htn)
  have hfilter : (range n).filter (fun i : ℕ => (i : ℝ) * h < t) = range ⌈t / h⌉₊ := by
    ext i
    simp only [Finset.mem_filter, Finset.mem_range]
    constructor
    · rintro ⟨-, hlt⟩
      exact Nat.lt_ceil.2 (by rw [lt_div_iff₀ hh]; exact hlt)
    · intro hi
      refine ⟨lt_of_lt_of_le hi hceil_le, ?_⟩
      have := Nat.lt_ceil.1 hi
      rw [lt_div_iff₀ hh] at this
      exact this
  have hsum : (∑ i ∈ range n, (1 - if t ≤ (i : ℝ) * h then (1 : ℝ) else 0))
      = (⌈t / h⌉₊ : ℝ) := by
    have : ∀ i ∈ range n, (1 - if t ≤ (i : ℝ) * h then (1 : ℝ) else 0)
        = if (i : ℝ) * h < t then (1 : ℝ) else 0 := by
      intro i _
      by_cases hi : t ≤ (i : ℝ) * h
      · simp [hi, not_lt.2 hi]
      · simp [hi, lt_of_not_ge hi]
    rw [Finset.sum_congr rfl this]
    rw [Finset.sum_ite]
    simp [hfilter]
  rw [hsum]
  have hle : t ≤ h * ⌈t / h⌉₊ := by
    have := Nat.le_ceil (t / h)
    rw [div_le_iff₀ hh] at this
    linarith [this]
  have hlt : (⌈t / h⌉₊ : ℝ) < t / h + 1 := Nat.ceil_lt_add_one (by positivity)
  have : h * ⌈t / h⌉₊ < t + h := by
    have := (mul_lt_mul_of_pos_left hlt hh)
    rw [mul_add, mul_one, mul_div_cancel₀ _ (ne_of_gt hh)] at this
    linarith
  rw [abs_le]
  constructor <;> linarith

/-- **A spacer-length series measures the mean distance.**  If the site–site distance stays in
`[0, n·h]` on the ensemble, the layer-cake sum of the yields of crosslinkers with reaches
`0, h, …, (n-1)h` differs from the true mean distance by at most the spacing `h`. -/
theorem mean_distance_from_series (E : Ens X) (d : X → ℝ) {h : ℝ} (hh : 0 < h) (n : ℕ)
    (hd : ∀ j, 0 ≤ d (E.pt j) ∧ d (E.pt j) ≤ n * h) :
    |E.expect d - series E d h n| ≤ h := by
  have hser : E.expect
        (fun x => h * ∑ i ∈ range n, (1 - if d x ≤ (i : ℝ) * h then (1 : ℝ) else 0))
      = series E d h n := by
    rw [E.expect_smul, E.expect_sum]
    unfold series
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [E.expect_sub, E.expect_const]
    rfl
  rw [← hser, ← E.expect_sub]
  refine expect_abs_le E fun j => ?_
  exact layer_pointwise hh n (hd j).1 (hd j).2

end Crosslink

end IDR
