/-
# Part CXLIII  The membrane interface: binding with excluded area, and the transfer scale

Many disordered regions do their work at a lipid interface — they are amphipathic, they fold on
binding to a membrane, and their affinity is measured as a partition between water and the
bilayer surface.  Two features of that experiment have no analogue in the solution binding of
earlier parts, and both are formalised here.

**Excluded area.**  A peptide bound to a surface covers `n` lipids, so the sites are not
independent: the isotherm is the Stankowski relation `x = A·(1 − n·x)^n`, where `x` is the bound
peptide per lipid and `A` is the affinity times the free peptide concentration, not the Langmuir
relation `x = A(1 − x)`.

* `bindFun_strictMono` — the defining function is strictly increasing on the physical interval,
  so `exists_unique_coverage` gives exactly one coverage for each condition.
* `coverage_lt_saturation` and `coverage_near_saturation` — inside the physical range `[0, 1/n]`
  the coverage never reaches `1/n`, and approaches it as the free concentration grows: the surface
  saturates at one peptide per `n` lipids, not at one per lipid.
* `langmuir_underestimates_affinity` — **the design clause.**  Reading a measured coverage with
  the Langmuir isotherm, i.e. ignoring the excluded area, returns an affinity that is strictly
  *smaller* than the one the same data imply under exclusion, for every `n ≥ 2` and every nonzero
  coverage.  The bias is systematic and one-signed; it is not reduced by better statistics.

**The transfer scale.**  Interfacial binding free energies are additive over residues — the
Wimley–White construction — so a binding curve measures one number per sequence, a composition
sum.  `transfer_scale_unidentifiable` is the exact consequence: if every measured sequence happens
to carry equal counts of two residue types, then *no* amount of binding data determines their
individual transfer energies; only the sum is fixed, and the two coefficients can be moved
arbitrarily far apart.  `transfer_scale_separated` shows the same perturbation is visible on a
sequence with unequal counts, so the remedy is compositional design, not more measurement.

`interface_design_law` bundles the clauses.
-/
import Mathlib

namespace RequestProject.Interface

open Finset

/-! ## Binding with excluded area -/

/-- The Stankowski binding function: `x` is the bound peptide per lipid, `n` the number of lipids
covered by one bound peptide, `A` the affinity times the free peptide concentration.  The
physical coverage is the zero of this function on `[0, 1/n]`. -/
noncomputable def bindFun (n : ℕ) (A x : ℝ) : ℝ := x - A * (1 - n * x) ^ n

/-- On the physical interval the binding function is strictly increasing. -/
theorem bindFun_strictMono {n : ℕ} {A : ℝ} (hn : 1 ≤ n) (hA : 0 ≤ A) {x y : ℝ}
    (hxy : x < y) (hy : y ≤ 1 / n) : bindFun n A x < bindFun n A y := by
  have hnpos : (0:ℝ) < n := by exact_mod_cast Nat.lt_of_lt_of_le Nat.zero_lt_one hn
  have hy0 : 0 ≤ 1 - n * y := by
    have : (n:ℝ) * y ≤ 1 := by
      rw [le_div_iff₀ hnpos] at hy
      linarith
    linarith
  have hxy' : 1 - (n:ℝ) * y ≤ 1 - n * x := by nlinarith
  have hpow : (1 - (n:ℝ) * y) ^ n ≤ (1 - (n:ℝ) * x) ^ n :=
    pow_le_pow_left₀ hy0 hxy' n
  have hmul : A * (1 - (n:ℝ) * y) ^ n ≤ A * (1 - (n:ℝ) * x) ^ n :=
    mul_le_mul_of_nonneg_left hpow hA
  simp only [bindFun]
  linarith

private lemma bindFun_saturation {n : ℕ} {A : ℝ} (hn : 1 ≤ n) :
    0 < bindFun n A (1 / n) := by
  have hnpos : (0:ℝ) < n := by exact_mod_cast Nat.lt_of_lt_of_le Nat.zero_lt_one hn
  have hne : n ≠ 0 := by omega
  have hone : (n:ℝ) * (1 / n) = 1 := by field_simp
  have hzero : 1 - (n:ℝ) * (1 / n) = 0 := by rw [hone]; ring
  rw [bindFun, hzero, zero_pow hne, mul_zero, sub_zero]
  positivity

/-- **The coverage is well defined.**  For each condition there is exactly one physical
coverage. -/
theorem exists_unique_coverage {n : ℕ} {A : ℝ} (hn : 1 ≤ n) (hA : 0 < A) :
    ∃! x : ℝ, x ∈ Set.Icc (0:ℝ) (1 / n) ∧ bindFun n A x = 0 := by
  have hnpos : (0:ℝ) < n := by exact_mod_cast Nat.lt_of_lt_of_le Nat.zero_lt_one hn
  have hcont : ContinuousOn (fun x : ℝ => bindFun n A x) (Set.Icc 0 (1 / n)) := by
    apply Continuous.continuousOn
    unfold bindFun
    continuity
  have h0 : bindFun n A 0 < 0 := by simp [bindFun, hA]
  have h1 : 0 < bindFun n A (1 / n) := bindFun_saturation (A := A) hn
  obtain ⟨x, hxmem, hx⟩ : ∃ x ∈ Set.Ioo (0:ℝ) (1 / n), bindFun n A x = 0 := by
    have hle : (0:ℝ) ≤ 1 / n := by positivity
    have hmem : (0:ℝ) ∈ Set.Ioo (bindFun n A 0) (bindFun n A (1 / n)) := ⟨h0, h1⟩
    obtain ⟨x, hx, hxeq⟩ := intermediate_value_Ioo hle hcont hmem
    exact ⟨x, hx, hxeq⟩
  refine ⟨x, ⟨⟨hxmem.1.le, hxmem.2.le⟩, hx⟩, ?_⟩
  rintro y ⟨⟨hy0, hy1⟩, hy⟩
  by_contra hne'
  rcases lt_or_gt_of_ne hne' with h | h
  · have hlt := bindFun_strictMono hn hA.le h hxmem.2.le
    rw [hy, hx] at hlt
    exact lt_irrefl 0 hlt
  · have hlt := bindFun_strictMono hn hA.le h hy1
    rw [hy, hx] at hlt
    exact lt_irrefl 0 hlt

/-- A physical coverage stays strictly below the saturation value `1/n`: the surface never
carries more than one peptide per `n` lipids. -/
theorem coverage_lt_saturation {n : ℕ} {A x : ℝ} (hn : 1 ≤ n)
    (hxle : x ≤ 1 / n) (hx : bindFun n A x = 0) : x < 1 / n := by
  rcases eq_or_lt_of_le hxle with h | h
  · exfalso
    have h1 : 0 < bindFun n A (1 / n) := bindFun_saturation (A := A) hn
    rw [← h, hx] at h1
    exact lt_irrefl 0 h1
  · exact h

/-- **Saturation.**  For any target below `1/n` there is a condition whose physical coverage
exceeds it: the surface fills up to one peptide per `n` lipids, and to nothing more. -/
theorem coverage_near_saturation {n : ℕ} (hn : 1 ≤ n) {eps : ℝ} (heps : 0 < eps)
    (hlt : eps < 1 / n) :
    ∃ A : ℝ, 0 < A ∧ ∀ x : ℝ, x ≤ 1 / n → bindFun n A x = 0 → 1 / n - eps < x := by
  have hnpos : (0:ℝ) < n := by exact_mod_cast Nat.lt_of_lt_of_le Nat.zero_lt_one hn
  set y := 1 / (n:ℝ) - eps with hy
  have hylt : y < 1 / (n:ℝ) := by rw [hy]; linarith
  have hone : (n:ℝ) * (1 / n) = 1 := by field_simp
  have hbase : 0 < 1 - (n:ℝ) * y := by
    rw [hy]
    nlinarith
  have hpow : 0 < (1 - (n:ℝ) * y) ^ n := pow_pos hbase n
  have hy0 : 0 < y := by rw [hy]; linarith
  refine ⟨(y + 1) / (1 - (n:ℝ) * y) ^ n, div_pos (by linarith) hpow, ?_⟩
  intro x hxle hx
  set A := (y + 1) / (1 - (n:ℝ) * y) ^ n with hA
  have hApos : 0 < A := by rw [hA]; exact div_pos (by linarith) hpow
  have hFy : bindFun n A y < 0 := by
    simp only [bindFun, hA]
    rw [div_mul_eq_mul_div, mul_div_assoc, div_self hpow.ne']
    linarith
  by_contra hcon
  push_neg at hcon
  have hxy : x ≤ y := by rw [hy]; linarith
  rcases eq_or_lt_of_le hxy with h | h
  · rw [h] at hx
    rw [hx] at hFy
    exact lt_irrefl 0 hFy
  · have hmono := bindFun_strictMono hn hApos.le h hylt.le
    rw [hx] at hmono
    linarith

/-- **Ignoring the excluded area underestimates the affinity.**  A measured coverage `x`
interpreted with the Langmuir isotherm gives `x/(1 − x)`; interpreted with the exclusion isotherm
it gives `x/(1 − n x)^n`, and for `n ≥ 2` and `0 < x < 1/n` the latter is strictly larger. -/
theorem langmuir_underestimates_affinity {n : ℕ} (hn : 2 ≤ n) {x : ℝ} (hx : 0 < x)
    (hxn : x < 1 / n) : x / (1 - x) < x / (1 - (n:ℝ) * x) ^ n := by
  have hnpos : (0:ℝ) < n := by exact_mod_cast Nat.lt_of_lt_of_le Nat.zero_lt_two hn
  have hn2 : (2:ℝ) ≤ n := by exact_mod_cast hn
  have hnx : (n:ℝ) * x < 1 := by
    rw [lt_div_iff₀ hnpos] at hxn
    linarith
  have hbase0 : 0 < 1 - (n:ℝ) * x := by linarith
  have hbase1 : 1 - (n:ℝ) * x ≤ 1 := by nlinarith
  have hne : n ≠ 0 := by omega
  have hpowle : (1 - (n:ℝ) * x) ^ n ≤ 1 - (n:ℝ) * x :=
    pow_le_of_le_one hbase0.le hbase1 hne
  have hlt : (1 - (n:ℝ) * x) ^ n < 1 - x := by
    have : 1 - (n:ℝ) * x < 1 - x := by nlinarith
    linarith
  have hxle : 0 < 1 - x := by nlinarith
  have hppos : 0 < (1 - (n:ℝ) * x) ^ n := pow_pos hbase0 n
  exact div_lt_div_of_pos_left hx hppos hlt

/-! ## The transfer scale is a composition sum -/

/-- Additive interfacial transfer free energy of a sequence, given its residue composition. -/
noncomputable def transferEnergy {k : ℕ} (e : Fin k → ℝ) (comp : Fin k → ℕ) : ℝ :=
  ∑ i, (comp i : ℝ) * e i

/-- **The transfer scale is not identifiable from balanced sequences.**  If every measured
sequence carries equal counts of two residue types, then their individual transfer energies can
be moved arbitrarily far apart without changing a single predicted binding free energy. -/
theorem transfer_scale_unidentifiable {k : ℕ} {S : Type*} (e : Fin k → ℝ) {a b : Fin k}
    (hab : a ≠ b) (comp : S → Fin k → ℕ) (hbal : ∀ s, comp s a = comp s b) (t : ℝ) :
    ∃ e' : Fin k → ℝ, (∀ s, transferEnergy e' (comp s) = transferEnergy e (comp s)) ∧
      e' a - e a = t ∧ e' b - e b = -t := by
  classical
  refine ⟨fun i => e i + (if i = a then t else 0) - (if i = b then t else 0), fun s => ?_, ?_, ?_⟩
  · have hpt : ∀ i : Fin k,
        (comp s i : ℝ) * (e i + (if i = a then t else 0) - (if i = b then t else 0))
        = (comp s i : ℝ) * e i + (if i = a then (comp s i : ℝ) * t else 0)
          - (if i = b then (comp s i : ℝ) * t else 0) := by
      intro i
      split_ifs <;> ring
    simp only [transferEnergy]
    rw [Finset.sum_congr rfl (fun i _ => hpt i), Finset.sum_sub_distrib, Finset.sum_add_distrib]
    have hA : ∑ i, (if i = a then (comp s i : ℝ) * t else 0) = (comp s a : ℝ) * t := by
      simp [Finset.sum_ite_eq']
    have hB : ∑ i, (if i = b then (comp s i : ℝ) * t else 0) = (comp s b : ℝ) * t := by
      simp [Finset.sum_ite_eq']
    rw [hA, hB, hbal s]
    ring
  · simp [hab]
  · simp [Ne.symm hab]

/-- And the same perturbation *is* visible on an unbalanced sequence: the cure for the degeneracy
is a sequence whose composition breaks the balance, not more data on balanced ones. -/
theorem transfer_scale_separated {k : ℕ} (e e' : Fin k → ℝ) {a b : Fin k} (hab : a ≠ b)
    (comp : Fin k → ℕ) (ht : ∀ i, i ≠ a → i ≠ b → e' i = e i) {t : ℝ}
    (hea : e' a = e a + t) (heb : e' b = e b - t) :
    transferEnergy e' comp - transferEnergy e comp = ((comp a : ℝ) - (comp b : ℝ)) * t := by
  classical
  have hpt : ∀ i : Fin k, (comp i : ℝ) * e' i - (comp i : ℝ) * e i
      = (if i = a then (comp i : ℝ) * t else 0) - (if i = b then (comp i : ℝ) * t else 0) := by
    intro i
    by_cases hia : i = a
    · subst hia
      simp [hea, hab]
      ring
    · by_cases hib : i = b
      · subst hib
        simp [heb, hia]
        ring
      · simp [ht i hia hib, hia, hib]
  simp only [transferEnergy]
  rw [← Finset.sum_sub_distrib, Finset.sum_congr rfl (fun i _ => hpt i), Finset.sum_sub_distrib]
  have hA : ∑ i, (if i = a then (comp i : ℝ) * t else 0) = (comp a : ℝ) * t := by
    simp [Finset.sum_ite_eq']
  have hB : ∑ i, (if i = b then (comp i : ℝ) * t else 0) = (comp b : ℝ) * t := by
    simp [Finset.sum_ite_eq']
  rw [hA, hB]
  ring

/-! ## Capstone -/

/-- **The interface design law.**  A model of a disordered region at a membrane must (1) solve a
well-posed exclusion isotherm with a unique coverage, (2) saturate at one peptide per `n` lipids
rather than one per lipid, (3) acknowledge that a Langmuir reading of the same data underestimates
the affinity for every `n ≥ 2`, and (4) treat the residue transfer scale as identified only up to
the compositional degeneracies of the sequences measured. -/
theorem interface_design_law {n : ℕ} (hn : 2 ≤ n) {A : ℝ} (hA : 0 < A) :
    (∃! x : ℝ, x ∈ Set.Icc (0:ℝ) (1 / n) ∧ bindFun n A x = 0) ∧
    (∀ x : ℝ, x ≤ 1 / n → bindFun n A x = 0 → x < 1 / n) ∧
    (∀ x : ℝ, 0 < x → x < 1 / n → x / (1 - x) < x / (1 - (n:ℝ) * x) ^ n) ∧
    (∀ (k : ℕ) (e : Fin k → ℝ) (a b : Fin k), a ≠ b → ∀ (comp : ℕ → Fin k → ℕ),
      (∀ s, comp s a = comp s b) → ∀ t : ℝ, ∃ e' : Fin k → ℝ,
        (∀ s, transferEnergy e' (comp s) = transferEnergy e (comp s)) ∧ e' a - e a = t) := by
  have hn1 : 1 ≤ n := le_trans (by norm_num) hn
  refine ⟨exists_unique_coverage hn1 hA,
    fun x hxle hx => coverage_lt_saturation hn1 hxle hx,
    fun x hx hxn => langmuir_underestimates_affinity hn hx hxn, ?_⟩
  intro k e a b hab comp hbal t
  obtain ⟨e', h1, h2, _⟩ := transfer_scale_unidentifiable e hab comp hbal t
  exact ⟨e', h1, h2⟩

end RequestProject.Interface
