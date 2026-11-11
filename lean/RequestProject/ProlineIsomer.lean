/-
# Part CXXXIX  Slow substates: the prolines a model of a disordered region cannot average over

Disordered regions are proline-rich, and the peptidyl–prolyl bond isomerises on a timescale of
tens of seconds — six to nine orders of magnitude slower than the backbone dynamics that a
conformational ensemble is built to describe.  A region with `n` prolines therefore does not have
one ensemble: it has `2ⁿ` of them, interconverting so slowly that no simulation and no
single-molecule trajectory of realistic length visits more than one.  This part makes that
statement quantitative and draws the modelling consequences.

The isomeric state is `s : Fin n → Bool` (`true` = *cis*), populated independently with cis
fractions `pc i`.

* `sum_isoWeight_eq_one`, `isoWeight_pos` — the substate distribution is a genuine probability
  distribution, and **all `2ⁿ` substates are populated** whenever every cis fraction is strictly
  between `0` and `1` (`populated_card`).
* `isoWeight_le_pow` — no substate carries the ensemble: if every cis fraction lies in
  `[δ, 1 − δ]`, every substate has weight at most `(1 − δ)ⁿ`.
* `cover_card_lower_bound`, `cover_card_ge` — hence any collection of substates accounting for a
  fraction `m` of the population must contain at least `m·(1 − δ)^(−n)` of them.  An ensemble
  model that enumerates isomers pays an exponential price; one that ignores them is reporting a
  mixture as a structure.
* `average_not_attained` — and the mixture's average need not be attained by any member: the
  measured number can correspond to no conformation at all.
* `hidden_substate_observable` — the exact non-identifiability.  Two per-substate assignments of
  an observable that differ by an arbitrarily large amount on two substates can give *exactly*
  the same ensemble average, so an averaged measurement never determines per-isomer values.
* `no_switch_prob_ge`, `switch_needs_long_window`, `demo_frozen_on_second_timescale` — the
  kinetic half: with isomerisation rate `k`, the probability that a molecule keeps its isomer
  through a window `T` is at least `1 − kT`; to see a switch with probability one half the window
  must exceed `log 2 / k`; and at the textbook rate `10⁻² s⁻¹` a one-second experiment finds
  `99%` of the molecules frozen.
* `proline_substate_law` bundles the clauses: a realistic model must carry the isomeric state as
  an explicit discrete label with its own slow kinetics, not as one more fast degree of freedom.
-/
import Mathlib

namespace RequestProject.ProlineIsomer

open Finset

variable {n : ℕ}

/-- Isomeric state of the `n` prolines of a region: `true` at position `i` means *cis*. -/
abbrev Iso (n : ℕ) := Fin n → Bool

/-- Population of an isomeric state under independent cis fractions `pc`. -/
noncomputable def isoWeight (pc : Fin n → ℝ) (s : Iso n) : ℝ :=
  ∏ i, (if s i then pc i else 1 - pc i)

/-- The substate populations sum to one. -/
theorem sum_isoWeight_eq_one (pc : Fin n → ℝ) : ∑ s : Iso n, isoWeight pc s = 1 := by
  have h := Finset.prod_univ_sum (fun _ : Fin n => (Finset.univ : Finset Bool))
      (fun (i : Fin n) (b : Bool) => if b then pc i else 1 - pc i)
  rw [Fintype.piFinset_univ] at h
  simp only [isoWeight]
  rw [← h]
  simp

/-- Every substate is populated when no proline is locked. -/
theorem isoWeight_pos {pc : Fin n → ℝ} (h0 : ∀ i, 0 < pc i) (h1 : ∀ i, pc i < 1) (s : Iso n) :
    0 < isoWeight pc s := by
  refine Finset.prod_pos fun i _ => ?_
  by_cases hs : s i <;> simp [hs]
  · exact h0 i
  · linarith [h1 i]

lemma isoWeight_nonneg {pc : Fin n → ℝ} (h0 : ∀ i, 0 ≤ pc i) (h1 : ∀ i, pc i ≤ 1) (s : Iso n) :
    0 ≤ isoWeight pc s := by
  refine Finset.prod_nonneg fun i _ => ?_
  by_cases hs : s i <;> simp [hs]
  · exact h0 i
  · linarith [h1 i]

/-- **Exponential substate count.**  With no proline locked, the set of populated isomeric
states has exactly `2ⁿ` elements. -/
theorem populated_card {pc : Fin n → ℝ} (h0 : ∀ i, 0 < pc i) (h1 : ∀ i, pc i < 1) :
    (Finset.univ.filter (fun s : Iso n => 0 < isoWeight pc s)).card = 2 ^ n := by
  have : (Finset.univ.filter (fun s : Iso n => 0 < isoWeight pc s)) = Finset.univ :=
    Finset.filter_true_of_mem fun s _ => isoWeight_pos h0 h1 s
  rw [this, Finset.card_univ]
  simp

/-- **No substate carries the ensemble.**  If every cis fraction lies in `[δ, 1 − δ]` then every
substate has population at most `(1 − δ)ⁿ`. -/
theorem isoWeight_le_pow {pc : Fin n → ℝ} {delta : ℝ} (hd : 0 ≤ delta)
    (hlo : ∀ i, delta ≤ pc i) (hhi : ∀ i, pc i ≤ 1 - delta) (s : Iso n) :
    isoWeight pc s ≤ (1 - delta) ^ n := by
  have hfac : ∀ i : Fin n, (if s i then pc i else 1 - pc i) ≤ 1 - delta := by
    intro i
    by_cases hs : s i <;> simp [hs]
    · exact hhi i
    · linarith [hlo i]
  have hnn : ∀ i : Fin n, (0:ℝ) ≤ if s i then pc i else 1 - pc i := by
    intro i
    by_cases hs : s i <;> simp [hs]
    · linarith [hlo i, hd]
    · linarith [hhi i, hd]
  calc isoWeight pc s ≤ ∏ _i : Fin n, (1 - delta) :=
        Finset.prod_le_prod (fun i _ => hnn i) (fun i _ => hfac i)
    _ = (1 - delta) ^ n := by simp

/-- A set of substates carrying population `m` cannot be smaller than `m / M`, where `M` bounds
the population of a single substate. -/
theorem cover_card_lower_bound {pc : Fin n → ℝ} {M m : ℝ} (S : Finset (Iso n))
    (hM : ∀ s, isoWeight pc s ≤ M) (hm : m ≤ ∑ s ∈ S, isoWeight pc s) :
    m ≤ S.card * M := by
  calc m ≤ ∑ s ∈ S, isoWeight pc s := hm
    _ ≤ ∑ _s ∈ S, M := Finset.sum_le_sum fun s _ => hM s
    _ = S.card * M := by simp [mul_comm]

/-- The exponential form: covering a fraction `m` of the population needs at least
`m·(1 − δ)^(−n)` substates. -/
theorem cover_card_ge {pc : Fin n → ℝ} {delta m : ℝ} (hd : 0 < delta) (hd1 : delta < 1)
    (hlo : ∀ i, delta ≤ pc i) (hhi : ∀ i, pc i ≤ 1 - delta) (S : Finset (Iso n))
    (hm : m ≤ ∑ s ∈ S, isoWeight pc s) :
    m / (1 - delta) ^ n ≤ S.card := by
  have hpos : (0:ℝ) < (1 - delta) ^ n := pow_pos (by linarith) n
  have := cover_card_lower_bound (pc := pc) (M := (1 - delta) ^ n) S
    (fun s => isoWeight_le_pow hd.le hlo hhi s) hm
  rw [div_le_iff₀ hpos]
  linarith

/-! ## What an averaged measurement can and cannot say -/

/-- The measured ensemble average of a per-substate observable. -/
noncomputable def isoMean (pc : Fin n → ℝ) (o : Iso n → ℝ) : ℝ :=
  ∑ s, isoWeight pc s * o s

/-- **The average is not a structure.**  A two-substate mixture in equal proportions has an
average attained by neither member. -/
theorem average_not_attained :
    ∃ (pc : Fin 1 → ℝ) (o : Iso 1 → ℝ),
      (∀ i, 0 < pc i) ∧ (∀ i, pc i < 1) ∧
      isoMean pc o = 1 / 2 ∧ ∀ s, o s ≠ isoMean pc o := by
  classical
  set pc : Fin 1 → ℝ := fun _ => 1 / 2 with hpc
  set o : Iso 1 → ℝ := fun s => if s 0 then 1 else 0 with ho
  have huniv : (Finset.univ : Finset (Fin 1 → Bool))
      = {(fun _ => true), (fun _ => false)} := by decide
  have hm : isoMean pc o = 1 / 2 := by
    simp only [isoMean, isoWeight, hpc, ho]
    rw [huniv, Finset.sum_insert (by decide), Finset.sum_singleton]
    norm_num
  refine ⟨pc, o, fun _ => by norm_num [hpc], fun _ => by norm_num [hpc], hm, ?_⟩
  intro s
  rw [hm, ho]
  by_cases hs : s 0 <;> simp [hs]

/-- **Hidden substates.**  Given two distinct populated substates, an observable can be moved by
an arbitrary amount on both of them without changing the measured average at all: per-isomer
values are not identifiable from an averaged experiment. -/
theorem hidden_substate_observable {pc : Fin n → ℝ} (h0 : ∀ i, 0 < pc i) (h1 : ∀ i, pc i < 1)
    (o : Iso n → ℝ) {s₁ s₂ : Iso n} (hne : s₁ ≠ s₂) (t : ℝ) :
    ∃ o' : Iso n → ℝ, isoMean pc o' = isoMean pc o ∧
      o' s₁ - o s₁ = t / isoWeight pc s₁ ∧ o' s₂ - o s₂ = -(t / isoWeight pc s₂) := by
  classical
  have hw₁ : isoWeight pc s₁ ≠ 0 := (isoWeight_pos h0 h1 s₁).ne'
  have hw₂ : isoWeight pc s₂ ≠ 0 := (isoWeight_pos h0 h1 s₂).ne'
  refine ⟨fun s => o s + (if s = s₁ then t / isoWeight pc s₁ else 0)
      - (if s = s₂ then t / isoWeight pc s₂ else 0), ?_, ?_, ?_⟩
  · have hsplit : ∀ s : Iso n, isoWeight pc s * (o s + (if s = s₁ then t / isoWeight pc s₁ else 0)
        - (if s = s₂ then t / isoWeight pc s₂ else 0))
        = isoWeight pc s * o s
          + (if s = s₁ then isoWeight pc s * (t / isoWeight pc s₁) else 0)
          - (if s = s₂ then isoWeight pc s * (t / isoWeight pc s₂) else 0) := by
      intro s
      split_ifs <;> ring
    simp only [isoMean]
    rw [Finset.sum_congr rfl (fun s _ => hsplit s), Finset.sum_sub_distrib,
      Finset.sum_add_distrib]
    have hA : ∑ s, (if s = s₁ then isoWeight pc s * (t / isoWeight pc s₁) else 0) = t := by
      simp [Finset.sum_ite_eq', mul_div_cancel₀ _ hw₁]
    have hB : ∑ s, (if s = s₂ then isoWeight pc s * (t / isoWeight pc s₂) else 0) = t := by
      simp [Finset.sum_ite_eq', mul_div_cancel₀ _ hw₂]
    rw [hA, hB]
    ring
  · simp [hne]
  · simp [Ne.symm hne]

/-! ## The kinetic half: an isomer is frozen on the experimental timescale -/

/-- Probability that a molecule with isomerisation rate `k` keeps its isomer through a window of
length `T`. -/
noncomputable def noSwitchProb (k T : ℝ) : ℝ := Real.exp (-(k * T))

/-- The frozen fraction is at least `1 − kT`: on any window short compared with the
isomerisation time, essentially every molecule reports a single substate. -/
theorem no_switch_prob_ge (k T : ℝ) : 1 - k * T ≤ noSwitchProb k T := by
  have := Real.add_one_le_exp (-(k * T))
  simpa [noSwitchProb] using by linarith

/-- Conversely, seeing a switch with probability one half needs a window longer than
`log 2 / k`. -/
theorem switch_needs_long_window {k T : ℝ} (hk : 0 < k)
    (h : 1 / 2 ≤ 1 - noSwitchProb k T) : Real.log 2 / k ≤ T := by
  have hle : noSwitchProb k T ≤ 1 / 2 := by linarith
  have hlog : -(k * T) ≤ Real.log (1 / 2) := by
    have := Real.log_le_log (Real.exp_pos (-(k * T))) hle
    simpa [noSwitchProb, Real.log_exp] using this
  have h2 : Real.log (1 / 2) = -Real.log 2 := by
    rw [one_div, Real.log_inv]
  rw [h2] at hlog
  rw [div_le_iff₀ hk]
  linarith

/-- At the textbook peptidyl–prolyl rate of `10⁻² s⁻¹`, a one-second experiment finds at least
`99%` of the molecules in the isomer they started in. -/
theorem demo_frozen_on_second_timescale : (0.99 : ℝ) ≤ noSwitchProb (1 / 100) 1 := by
  have := no_switch_prob_ge (1 / 100 : ℝ) 1
  norm_num at this ⊢
  linarith

/-- **The proline substate law.**  A model of a proline-containing disordered region must carry
the isomeric state as an explicit slow discrete label: the substates are all populated (1) and
none dominates (2), so any faithful enumeration is exponentially large (3); an averaged
measurement determines no per-isomer value (4); and on the experimental window the label does not
move (5), so it cannot be treated as one more equilibrated degree of freedom. -/
theorem proline_substate_law {pc : Fin n → ℝ} {delta : ℝ} (hd : 0 < delta) (hd1 : delta < 1)
    (hlo : ∀ i, delta ≤ pc i) (hhi : ∀ i, pc i ≤ 1 - delta) :
    (∑ s : Iso n, isoWeight pc s = 1) ∧
    (Finset.univ.filter (fun s : Iso n => 0 < isoWeight pc s)).card = 2 ^ n ∧
    (∀ s : Iso n, isoWeight pc s ≤ (1 - delta) ^ n) ∧
    (∀ (S : Finset (Iso n)) (m : ℝ), m ≤ ∑ s ∈ S, isoWeight pc s →
      m / (1 - delta) ^ n ≤ S.card) ∧
    (∀ (o : Iso n → ℝ) (s₁ s₂ : Iso n), s₁ ≠ s₂ → ∀ t : ℝ, ∃ o' : Iso n → ℝ,
      isoMean pc o' = isoMean pc o ∧ o' s₁ - o s₁ = t / isoWeight pc s₁) ∧
    (∀ k T : ℝ, 1 - k * T ≤ noSwitchProb k T) := by
  have h0 : ∀ i, 0 < pc i := fun i => lt_of_lt_of_le hd (hlo i)
  have h1 : ∀ i, pc i < 1 := fun i => lt_of_le_of_lt (hhi i) (by linarith)
  refine ⟨sum_isoWeight_eq_one pc, populated_card h0 h1,
    fun s => isoWeight_le_pow hd.le hlo hhi s,
    fun S m hm => cover_card_ge hd hd1 hlo hhi S hm, ?_, no_switch_prob_ge⟩
  intro o s₁ s₂ hne t
  obtain ⟨o', h₁, h₂, _⟩ := hidden_substate_observable h0 h1 o hne t
  exact ⟨o', h₁, h₂⟩

end RequestProject.ProlineIsomer
