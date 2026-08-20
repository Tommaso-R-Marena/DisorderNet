/-
# Part CXXXVIII  The distance law is part of the model: what a complete titration identifies,
# and what it confounds

Part CXXXIV–CXXXVII read charge correlations off a salt titration and found the answer to depend
on the assumed separation kernel: for the chain kernel `d·e^{−κd}` the horizon sits at lag
`≈ log(B/eps)/κ`, for the Debye–Hückel kernel `e^{−κb√d}/(b√d)` of a Gaussian chain at
`≈ (log(B/eps)/(κb))²`.  Part CXXXVII ended with the instruction that a model must *declare* its
distance law before quoting a resolution.  This part asks the obvious next question: can the
distance law itself be calibrated from the same data?  The answer has three parts.

* **A tool.**  `expSum_amplitudes_zero` — real exponentials with distinct rates are linearly
  independent as functions of the ionic strength, on any half-line of conditions.  Proved from
  scratch, with an elementary argument (no limits): the smallest rate dominates, and
  `e^{−x} ≤ 1/x` turns the domination into a bound that forces its amplitude to vanish.

* **Identification, for a declared kernel.**  `genCurve_identifies` — for *any* separation kernel
  of the form `w(d)·e^{−κ a(d)}` with nonvanishing weights and pairwise distinct rates, a complete
  titration (all conditions on a half-line `κ ≥ κ₀`) determines the entire correlation profile.
  Instantiated at the chain kernel (`chain_profile_identifiable`) and at the Debye–Hückel kernel
  (`debye_profile_identifiable`).  Note the contrast with Part CXXXIII: *finitely* many conditions
  never suffice (`Titration.finite_titration_underdetermined`); the whole half-line always does.

* **The distance law is testable.**  `chain_debye_forces_short_range` — the two kernels are not
  interchangeable: a chain-kernel model and a Gaussian-chain Debye model can agree at every ionic
  strength only if both are blind past lag 1.  Hence `misspecified_kernel_detected`: a region with
  a non-zero lag-2 correlation, read with the chain kernel, cannot be fitted by *any* Debye
  profile — some condition separates them.  The distance law is therefore not a matter of taste;
  it is a falsifiable part of the model.

* **But the bond length is not identifiable with the profile.**  `bond_length_lag_confound` — an
  exact degeneracy: a region whose only correlation is at lag `2`, modelled with bond length `b`,
  produces *literally the same* titration curve, at every ionic strength, as a region whose only
  correlation is the same number at lag `1`, modelled with bond length `b√2`.  No resolution, no
  number of conditions, and no cleverness in the fit can separate them
  (`bond_length_not_identifiable`), because the two models are the same function.  The bond length
  must be fixed by an independent measurement; fitted jointly with the profile, it trades
  one-for-one against sequence separation.  The degeneracy is not an artefact of the toy window:
  `even_lag_bond_length_confound` shows that *every* profile supported on even lags, over a region
  of any length, is reproduced exactly at bond length `b√2` by the profile of its even lags — so
  the confound halves the apparent sequence range of the region
  (`confound_halves_apparent_range`).

* **What breaks the confound.**  `bond_length_identifiable_of_lag1` — the degeneracy lives
  entirely in profiles whose lag-1 correlation vanishes: if both candidate models carry a non-zero
  correlation at lag 1, agreement of their titration curves at every condition forces the same
  bond length, because the rate `b` of lag 1 is then strictly the slowest rate in play and nothing
  can cancel it.  Hence the design rule `bond_length_identified_given_profile`: measure the
  correlation profile with the bond-length-free separation-resolved panel of Part CXXXVI, and the
  titration then determines the bond length.

The design consequence: a calibrated model of a charged disordered region must report the distance
law it assumes, may test that law against a complete titration, and must *not* claim to have
fitted the chain's bond length from the same experiment — unless the correlation profile comes
from an independent, bond-length-free probe and does not vanish at short separation.
-/
import Mathlib
import RequestProject.DebyeResolution

set_option autoImplicit false

namespace IDR
namespace DistanceLaw

open Finset

/-! ## 1. Linear independence of real exponentials in the ionic strength -/

/-- A finite sum of exponentials `∑ A j · e^{−κ a j}` in the inverse screening length `κ`. -/
noncomputable def expSum (S : Finset ℕ) (A a : ℕ → ℝ) (kappa : ℝ) : ℝ :=
  ∑ j ∈ S, A j * Real.exp (-(kappa * a j))

/-- **The amplitude of the slowest-decaying rate vanishes.**  If a finite exponential sum vanishes
at every condition on a half-line `κ ≥ κ₀` and one of its rates is strictly smaller than all the
others, the amplitude of that rate is zero.  This is the whole content of the independence of
exponentials: at large `κ` the slowest term dominates, and `e^{−x} ≤ 1/x` turns the domination
into a bound that no non-zero amplitude survives. -/
theorem expSum_min_rate_zero {A a : ℕ → ℝ} {kappa0 : ℝ} {S : Finset ℕ} {j0 : ℕ} (hj0S : j0 ∈ S)
    (hminrate : ∀ j ∈ S, j ≠ j0 → a j0 < a j)
    (hzero : ∀ kappa, kappa0 ≤ kappa → expSum S A a kappa = 0) : A j0 = 0 := by
  -- Factor out the slowest-decaying term.
  have key : ∀ k : ℝ, kappa0 ≤ k →
      A j0 = - ∑ j ∈ S.erase j0, A j * Real.exp (-(k * (a j - a j0))) := by
    intro k hk
    have h := hzero k hk
    rw [expSum, ← Finset.add_sum_erase _ _ hj0S] at h
    have h2 : (A j0 * Real.exp (-(k * a j0))
        + ∑ j ∈ S.erase j0, A j * Real.exp (-(k * a j))) * Real.exp (k * a j0) = 0 := by
      rw [h]; ring
    rw [add_mul, Finset.sum_mul] at h2
    have e1 : ∀ j : ℕ, A j * Real.exp (-(k * a j)) * Real.exp (k * a j0)
        = A j * Real.exp (-(k * (a j - a j0))) := by
      intro j
      rw [mul_assoc, ← Real.exp_add]
      ring_nf
    simp only [e1, sub_self, mul_zero, neg_zero, Real.exp_zero, mul_one] at h2
    linarith [h2]
  rcases (S.erase j0).eq_empty_or_nonempty with hemp | hne'
  · have := key (max kappa0 0) (le_max_left _ _)
    simpa [hemp] using this
  · obtain ⟨j1, hj1, hmin⟩ := (S.erase j0).exists_min_image (fun j => a j - a j0) hne'
    set gap : ℝ := a j1 - a j0 with hgap
    have hgappos : 0 < gap :=
      sub_pos.2 (hminrate j1 (Finset.mem_of_mem_erase hj1) (Finset.ne_of_mem_erase hj1))
    set M : ℝ := ∑ j ∈ S.erase j0, |A j| with hM
    have hMnn : 0 ≤ M := Finset.sum_nonneg fun j _ => abs_nonneg _
    have bound : ∀ k : ℝ, kappa0 ≤ k → 0 ≤ k → |A j0| ≤ M * Real.exp (-(k * gap)) := by
      intro k hk hk0
      rw [key k hk, abs_neg]
      refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
      rw [hM, Finset.sum_mul]
      refine Finset.sum_le_sum ?_
      intro j hj
      rw [abs_mul, Real.abs_exp]
      have hle : -(k * (a j - a j0)) ≤ -(k * gap) := by
        have := hmin j hj
        nlinarith
      exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 hle) (abs_nonneg _)
    by_contra hne0
    have ht : 0 < |A j0| := abs_pos.2 hne0
    set k : ℝ := max kappa0 ((M + 1) / (gap * |A j0|)) with hkdef
    have hk1 : kappa0 ≤ k := le_max_left _ _
    have hk2 : (M + 1) / (gap * |A j0|) ≤ k := le_max_right _ _
    have hden : 0 < gap * |A j0| := mul_pos hgappos ht
    have hkpos : 0 < k := lt_of_lt_of_le (div_pos (by linarith) hden) hk2
    have hb := bound k hk1 hkpos.le
    have hexp : Real.exp (-(k * gap)) ≤ 1 / (k * gap) := by
      have h1 : k * gap ≤ Real.exp (k * gap) := by
        have := Real.add_one_le_exp (k * gap)
        linarith
      have h2 : 0 < k * gap := mul_pos hkpos hgappos
      rw [Real.exp_neg, inv_eq_one_div]
      exact one_div_le_one_div_of_le h2 h1
    have h3 : |A j0| ≤ M * (1 / (k * gap)) :=
      hb.trans (mul_le_mul_of_nonneg_left hexp hMnn)
    have h4 : |A j0| * (k * gap) ≤ M := by
      have hkd : 0 < k * gap := mul_pos hkpos hgappos
      calc |A j0| * (k * gap) ≤ M * (1 / (k * gap)) * (k * gap) := by nlinarith
        _ = M := by field_simp
    have h5 : (M + 1) ≤ k * (gap * |A j0|) := by
      have := (div_le_iff₀ hden).1 hk2
      linarith
    nlinarith

/-- **Exponentials with distinct rates are independent.**  If a finite exponential sum with
pairwise distinct rates vanishes at every condition on a half-line `κ ≥ κ₀`, all its amplitudes
vanish.  (Stated with the cardinality of the index set exposed, so that it can be proved by
induction; the usable form is `expSum_eq_zero`.) -/
theorem expSum_amplitudes_zero {A a : ℕ → ℝ} {kappa0 : ℝ} :
    ∀ (n : ℕ) (S : Finset ℕ), S.card = n → Set.InjOn a S →
      (∀ kappa, kappa0 ≤ kappa → expSum S A a kappa = 0) → ∀ j ∈ S, A j = 0 := by
  intro n
  induction n with
  | zero =>
      intro S hS _ _ j hj
      rw [Finset.card_eq_zero] at hS
      simp [hS] at hj
  | succ n ih =>
      intro S hcard hinj hzero
      have hSne : S.Nonempty := Finset.card_pos.1 (by omega)
      obtain ⟨j0, hj0S, hj0min⟩ := S.exists_min_image a hSne
      have hA0 : A j0 = 0 :=
        expSum_min_rate_zero hj0S
          (fun j hj hne => lt_of_le_of_ne (hj0min j hj)
            (fun heq => hne (hinj hj hj0S heq.symm))) hzero
      have hzero' : ∀ kappa : ℝ, kappa0 ≤ kappa → expSum (S.erase j0) A a kappa = 0 := by
        intro k hk
        have h := hzero k hk
        rw [expSum, ← Finset.add_sum_erase _ _ hj0S] at h
        rw [expSum]
        simp only [hA0, zero_mul, zero_add] at h
        exact h
      intro j hj
      rcases eq_or_ne j j0 with rfl | hne
      · exact hA0
      · refine ih (S.erase j0) ?_ (hinj.mono (by simp)) hzero'
          j (Finset.mem_erase.2 ⟨hne, hj⟩)
        rw [Finset.card_erase_of_mem hj0S, hcard]
        omega

/-- **Linear independence of real exponentials**, in the form used below. -/
theorem expSum_eq_zero {A a : ℕ → ℝ} {kappa0 : ℝ} {S : Finset ℕ} (hinj : Set.InjOn a S)
    (h : ∀ kappa, kappa0 ≤ kappa → expSum S A a kappa = 0) : ∀ j ∈ S, A j = 0 :=
  expSum_amplitudes_zero S.card S rfl hinj h

/-! ## 2. A complete titration identifies the profile, for any declared kernel -/

/-- The titration curve of a correlation profile `c` under a separation kernel with amplitude
`w d` and screening rate `a d` at lag `d`. -/
noncomputable def genCurve (N : ℕ) (w a c : ℕ → ℝ) (kappa : ℝ) : ℝ :=
  ∑ d ∈ Ico 1 N, w d * Real.exp (-(kappa * a d)) * c d

/-- **A complete titration identifies the whole correlation profile.**  For any kernel whose
screening rates are pairwise distinct across lags and whose amplitudes never vanish, two profiles
with the same reading at every condition `κ ≥ κ₀` agree at every lag. -/
theorem genCurve_identifies {N : ℕ} {w a : ℕ → ℝ} {kappa0 : ℝ}
    (ha : Set.InjOn a (Ico 1 N)) (hw : ∀ d ∈ Ico 1 N, w d ≠ 0) {c c' : ℕ → ℝ}
    (h : ∀ kappa, kappa0 ≤ kappa → genCurve N w a c kappa = genCurve N w a c' kappa) :
    ∀ d ∈ Ico 1 N, c d = c' d := by
  have hz : ∀ kappa, kappa0 ≤ kappa →
      expSum (Ico 1 N) (fun d => w d * (c d - c' d)) a kappa = 0 := by
    intro k hk
    have := h k hk
    rw [expSum, genCurve, genCurve] at *
    rw [← sub_eq_zero] at this
    rw [← this, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun d _ => by ring
  intro d hd
  have := expSum_eq_zero ha hz d hd
  rcases mul_eq_zero.1 this with h1 | h2
  · exact absurd h1 (hw d hd)
  · linarith [sub_eq_zero.1 h2]

/-- The chain kernel `d·e^{−κd}` of Part CXXVI, in the general form. -/
lemma titration_curve_eq_genCurve (N : ℕ) (c : ℕ → ℝ) (kappa : ℝ) :
    Titration.curve N c kappa = genCurve N (fun d => (d : ℝ)) (fun d => (d : ℝ)) c kappa := rfl

/-- The Debye–Hückel kernel `e^{−κb√d}/(b√d)` of Part CXXXVII, in the general form. -/
lemma debyeCurve_eq_genCurve (N : ℕ) (b kappa : ℝ) (c : ℕ → ℝ) :
    DebyeResolution.debyeCurve N b kappa c
      = genCurve N (fun d => 1 / (b * Real.sqrt d)) (fun d => b * Real.sqrt d) c kappa := by
  rw [DebyeResolution.debyeCurve, genCurve]
  exact Finset.sum_congr rfl fun d _ => by rw [DebyeResolution.dkern]; ring

/-- **The chain-kernel titration, read over a whole half-line of conditions, determines every
charge correlation.**  Contrast Part CXXXIII: finitely many conditions never do. -/
theorem chain_profile_identifiable {N : ℕ} {kappa0 : ℝ} {c c' : ℕ → ℝ}
    (h : ∀ kappa, kappa0 ≤ kappa → Titration.curve N c kappa = Titration.curve N c' kappa) :
    ∀ d ∈ Ico 1 N, c d = c' d := by
  refine genCurve_identifies (fun d _ e _ hde => Nat.cast_injective hde) (fun d hd => ?_)
    (fun k hk => by simpa [titration_curve_eq_genCurve] using h k hk)
  have : 1 ≤ d := (Finset.mem_Ico.1 hd).1
  positivity

/-- **The Debye–Hückel titration also determines every charge correlation**, once the bond length
is declared. -/
theorem debye_profile_identifiable {N : ℕ} {b kappa0 : ℝ} (hb : 0 < b) {c c' : ℕ → ℝ}
    (h : ∀ kappa, kappa0 ≤ kappa →
      DebyeResolution.debyeCurve N b kappa c = DebyeResolution.debyeCurve N b kappa c') :
    ∀ d ∈ Ico 1 N, c d = c' d := by
  have hsqrt : ∀ d ∈ Ico 1 N, (0 : ℝ) < Real.sqrt d := by
    intro d hd
    exact DebyeResolution.sqrt_pos_of_one_le (Finset.mem_Ico.1 hd).1
  refine genCurve_identifies (fun d hd e he hde => ?_) (fun d hd => ?_)
    (fun k hk => by simpa [debyeCurve_eq_genCurve] using h k hk)
  · simp only at hde
    have h1 : Real.sqrt d = Real.sqrt e := by
      have := mul_left_cancel₀ (ne_of_gt hb) hde
      exact this
    have h2 : (d : ℝ) = (e : ℝ) := by
      have hd0 : (0 : ℝ) ≤ d := Nat.cast_nonneg d
      have he0 : (0 : ℝ) ≤ e := Nat.cast_nonneg e
      calc (d : ℝ) = Real.sqrt d ^ 2 := (Real.sq_sqrt hd0).symm
        _ = Real.sqrt e ^ 2 := by rw [h1]
        _ = (e : ℝ) := Real.sq_sqrt he0
    exact_mod_cast h2
  · have := hsqrt d hd
    have : (0 : ℝ) < b * Real.sqrt d := mul_pos hb this
    positivity

/-! ## 3. The distance law itself is testable -/

lemma sqrt_two_lt_two : Real.sqrt 2 < 2 := by
  nlinarith [Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2), Real.sqrt_nonneg (2:ℝ)]

lemma one_lt_sqrt_two : (1 : ℝ) < Real.sqrt 2 := by
  nlinarith [Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2), Real.sqrt_nonneg (2:ℝ)]

/-- The three screening rates that occur when a chain-kernel model (rates `1, 2`) is compared with
a Gaussian-chain Debye model of unit bond length (rates `1, √2`) on a three-residue window. -/
noncomputable def rate3 : ℕ → ℝ := fun j => if j = 0 then 1 else if j = 1 then 2 else Real.sqrt 2

lemma rate3_injOn : Set.InjOn rate3 ({0, 1, 2} : Finset ℕ) := by
  have h1 := one_lt_sqrt_two
  have h2 := sqrt_two_lt_two
  intro i hi j hj hij
  simp only [Finset.coe_insert, Finset.coe_singleton, Set.mem_insert_iff,
    Set.mem_singleton_iff] at hi hj
  rcases hi with rfl | rfl | rfl <;> rcases hj with rfl | rfl | rfl <;>
    first
      | rfl
      | (exfalso; simp only [rate3] at hij; norm_num at hij; try linarith)

/-- **A chain-kernel model and a Gaussian-chain Debye model cannot be confused.**  If, on a
three-residue window, a chain-kernel curve of profile `c` agrees with a unit-bond-length Debye
curve of profile `c'` at every ionic strength `κ ≥ κ₀`, then both lag-2 correlations vanish and
the lag-1 correlations agree: the two kernels agree only where neither is being used. -/
theorem chain_debye_forces_short_range {kappa0 : ℝ} {c c' : ℕ → ℝ}
    (h : ∀ kappa, kappa0 ≤ kappa →
      Titration.curve 3 c kappa = DebyeResolution.debyeCurve 3 1 kappa c') :
    c 2 = 0 ∧ c' 2 = 0 ∧ c 1 = c' 1 := by
  set A : ℕ → ℝ := fun j =>
    if j = 0 then c 1 - c' 1 else if j = 1 then 2 * c 2 else -(c' 2 / Real.sqrt 2) with hA
  have hsq2 : (0 : ℝ) < Real.sqrt 2 := by linarith [one_lt_sqrt_two]
  have hz : ∀ kappa, kappa0 ≤ kappa → expSum ({0, 1, 2} : Finset ℕ) A rate3 kappa = 0 := by
    intro k hk
    have hcurve := h k hk
    rw [Titration.curve, DebyeResolution.debyeCurve] at hcurve
    rw [expSum]
    have hIco : Ico 1 3 = ({1, 2} : Finset ℕ) := rfl
    rw [hIco] at hcurve
    simp only [DebyeResolution.dkern] at hcurve ⊢
    norm_num [hA, rate3, Real.sqrt_one] at hcurve ⊢
    field_simp at hcurve ⊢
    nlinarith [hcurve, hsq2]
  have hzero := expSum_eq_zero rate3_injOn hz
  have h0 := hzero 0 (by decide)
  have h1 := hzero 1 (by decide)
  have h2 := hzero 2 (by decide)
  simp only [hA] at h0 h1 h2
  norm_num at h0 h1
  have h2' : c' 2 = 0 := by
    have : c' 2 / Real.sqrt 2 = 0 := by simpa using h2
    field_simp at this
    linarith
  exact ⟨h1, h2', by linarith⟩

/-- **Misspecification is detected by a complete titration.**  A three-residue window whose lag-2
charge correlation is non-zero, read with the chain kernel, is not reproduced by *any* Debye
profile: some ionic strength separates them. -/
theorem misspecified_kernel_detected {kappa0 : ℝ} {c : ℕ → ℝ} (hc : c 2 ≠ 0) (c' : ℕ → ℝ) :
    ∃ kappa, kappa0 ≤ kappa ∧
      Titration.curve 3 c kappa ≠ DebyeResolution.debyeCurve 3 1 kappa c' := by
  by_contra hcon
  push_neg at hcon
  exact hc (chain_debye_forces_short_range (fun k hk => hcon k hk)).1

/-! ## 4. The bond length is confounded with sequence separation -/

/-- A profile carrying a single correlation `t` at lag `D`. -/
noncomputable def single (D : ℕ) (t : ℝ) : ℕ → ℝ := fun d => if d = D then t else 0

/-- **An exact degeneracy between bond length and sequence separation.**  A region whose only
charge correlation sits at lag `2`, modelled with bond length `b`, has *exactly* the same
Debye–Hückel titration curve — at every ionic strength — as a region whose only correlation is the
same number at lag `1`, modelled with bond length `b√2`. -/
theorem bond_length_lag_confound (b t kappa : ℝ) :
    DebyeResolution.debyeCurve 3 b kappa (single 2 t)
      = DebyeResolution.debyeCurve 3 (b * Real.sqrt 2) kappa (single 1 t) := by
  have hIco : Ico 1 3 = ({1, 2} : Finset ℕ) := rfl
  rw [DebyeResolution.debyeCurve, DebyeResolution.debyeCurve, hIco]
  simp [DebyeResolution.dkern, single, Real.sqrt_one]

/-- **Consequently the bond length is not identifiable from the titration.**  For any non-zero
correlation `t` there are two distinct models — different bond lengths *and* different correlation
profiles — whose readings coincide at every ionic strength.  No resolution and no number of
conditions can separate them. -/
theorem bond_length_not_identifiable {b t : ℝ} (hb : 0 < b) (ht : t ≠ 0) :
    ∃ b' : ℝ, ∃ c c' : ℕ → ℝ, b ≠ b' ∧ 0 < b' ∧ c 2 ≠ c' 2 ∧
      ∀ kappa : ℝ, DebyeResolution.debyeCurve 3 b kappa c
        = DebyeResolution.debyeCurve 3 b' kappa c' := by
  have h1 := one_lt_sqrt_two
  refine ⟨b * Real.sqrt 2, single 2 t, single 1 t, ?_, by positivity, ?_,
    fun kappa => bond_length_lag_confound b t kappa⟩
  · intro hcon
    nlinarith [hcon]
  · simp [single, ht]

/-- **The confound in general.**  Every correlation profile supported on even lags, over a region
of any length, produces exactly the same Debye–Hückel titration curve at bond length `b` as the
profile of its even lags does at bond length `b√2` — at every ionic strength.  Doubling the bond
length by `√2` halves every sequence separation, and the kernel cannot tell the difference. -/
theorem even_lag_bond_length_confound {N : ℕ} (b kappa : ℝ) {c : ℕ → ℝ}
    (hodd : ∀ d, ¬ (2 ∣ d) → c d = 0) :
    DebyeResolution.debyeCurve (2 * N) b kappa c
      = DebyeResolution.debyeCurve N (b * Real.sqrt 2) kappa (fun e => c (2 * e)) := by
  classical
  rw [DebyeResolution.debyeCurve, DebyeResolution.debyeCurve]
  have hsub : (Ico 1 N).image (fun e => 2 * e) ⊆ Ico 1 (2 * N) := by
    intro d hd
    simp only [Finset.mem_image] at hd
    obtain ⟨e, he, rfl⟩ := hd
    rw [Finset.mem_Ico] at he ⊢
    omega
  have hvanish : ∀ d ∈ Ico 1 (2 * N), d ∉ (Ico 1 N).image (fun e => 2 * e) →
      DebyeResolution.dkern b kappa d * c d = 0 := by
    intro d hd hnot
    have hd2 : ¬ (2 ∣ d) := by
      rintro ⟨e, rfl⟩
      refine hnot (Finset.mem_image.2 ⟨e, ?_, rfl⟩)
      rw [Finset.mem_Ico] at hd ⊢
      omega
    rw [hodd d hd2, mul_zero]
  rw [← Finset.sum_subset hsub hvanish]
  rw [Finset.sum_image (by intro x _ y _ h; dsimp only at h; omega)]
  refine Finset.sum_congr rfl fun e _ => ?_
  have h2 : Real.sqrt ((2 * e : ℕ) : ℝ) = Real.sqrt 2 * Real.sqrt e := by
    push_cast
    rw [Real.sqrt_mul (by norm_num)]
  rw [DebyeResolution.dkern, DebyeResolution.dkern, h2]
  ring_nf

/-- **The confound halves the apparent sequence range.**  A single correlation at lag `2D`, read
with bond length `b`, is exactly a single correlation of the same size at lag `D`, read with bond
length `b√2`. -/
theorem confound_halves_apparent_range {N D : ℕ} (b t kappa : ℝ) :
    DebyeResolution.debyeCurve (2 * N) b kappa (single (2 * D) t)
      = DebyeResolution.debyeCurve N (b * Real.sqrt 2) kappa (single D t) := by
  rw [even_lag_bond_length_confound b kappa (c := single (2 * D) t) ?_]
  · congr 1
    funext e
    by_cases h : e = D
    · simp [single, h]
    · simp [single, h]
  · intro d hdiv
    have : d ≠ 2 * D := by rintro rfl; exact hdiv ⟨D, rfl⟩
    simp [single, this]

/-! ## 4½. Breaking the confound: an independently known lag-1 correlation -/

/-- The screening rates of the two competing models, indexed so that lag `d` of the `b`-model has
index `d` and lag `d` of the `b'`-model has index `N + d`. -/
noncomputable def jointRate (N : ℕ) (b b' : ℝ) : ℕ → ℝ :=
  fun j => if j < N then b * Real.sqrt j else b' * Real.sqrt ((j : ℝ) - N)

/-- The amplitudes of the difference of the two models, in the same indexing. -/
noncomputable def jointAmp (N : ℕ) (b b' : ℝ) (c c' : ℕ → ℝ) : ℕ → ℝ :=
  fun j => if j < N then c j / (b * Real.sqrt j)
    else -(c' (j - N) / (b' * Real.sqrt ((j : ℝ) - N)))

/-- The difference of two Debye–Hückel titration curves, with different bond lengths and different
correlation profiles, is an exponential sum in the ionic strength. -/
theorem debyeCurve_sub_eq_expSum (N : ℕ) (b b' : ℝ) (c c' : ℕ → ℝ) (kappa : ℝ) :
    DebyeResolution.debyeCurve N b kappa c - DebyeResolution.debyeCurve N b' kappa c'
      = expSum ((Ico 1 N) ∪ (Ico 1 N).image (fun d => N + d))
          (jointAmp N b b' c c') (jointRate N b b') kappa := by
  classical
  have hdisj : Disjoint (Ico 1 N) ((Ico 1 N).image (fun d => N + d)) := by
    rw [Finset.disjoint_left]
    intro x hx hx'
    rw [Finset.mem_Ico] at hx
    simp only [Finset.mem_image, Finset.mem_Ico] at hx'
    obtain ⟨e, he, rfl⟩ := hx'
    omega
  have e1 : ∑ d ∈ Ico 1 N, jointAmp N b b' c c' d * Real.exp (-(kappa * jointRate N b b' d))
      = DebyeResolution.debyeCurve N b kappa c := by
    rw [DebyeResolution.debyeCurve]
    refine Finset.sum_congr rfl fun d hd => ?_
    have h1 : d < N := (Finset.mem_Ico.1 hd).2
    simp only [jointAmp, jointRate, if_pos h1, DebyeResolution.dkern]
    ring
  have e2 : ∑ d ∈ Ico 1 N,
      jointAmp N b b' c c' (N + d) * Real.exp (-(kappa * jointRate N b b' (N + d)))
        = -DebyeResolution.debyeCurve N b' kappa c' := by
    rw [DebyeResolution.debyeCurve, ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun d hd => ?_
    have h1 : 1 ≤ d := (Finset.mem_Ico.1 hd).1
    have h2 : ¬ (N + d < N) := by omega
    have hc : ((N + d : ℕ) : ℝ) - (N : ℝ) = (d : ℝ) := by push_cast; ring
    simp only [jointAmp, jointRate, if_neg h2, hc, Nat.add_sub_cancel_left,
      DebyeResolution.dkern]
    ring
  rw [expSum, Finset.sum_union hdisj,
    Finset.sum_image (by intro x _ y _ h; dsimp only at h; omega), e1, e2]
  ring

/-- If the true bond length is the smaller of the two, the lag-1 correlation of the smaller-bond
model must vanish for the two titration curves to agree at every condition: its screening rate
`b` is strictly the slowest of all rates in play, so nothing can cancel it. -/
theorem lag1_zero_of_bond_length_lt {N : ℕ} (hN : 2 ≤ N) {b b' kappa0 : ℝ} (hb : 0 < b)
    (hlt : b < b') {c c' : ℕ → ℝ}
    (h : ∀ kappa, kappa0 ≤ kappa →
      DebyeResolution.debyeCurve N b kappa c = DebyeResolution.debyeCurve N b' kappa c') :
    c 1 = 0 := by
  classical
  set S : Finset ℕ := (Ico 1 N) ∪ (Ico 1 N).image (fun d => N + d) with hS
  have hzero : ∀ kappa, kappa0 ≤ kappa →
      expSum S (jointAmp N b b' c c') (jointRate N b b') kappa = 0 := by
    intro k hk
    rw [hS, ← debyeCurve_sub_eq_expSum, h k hk, sub_self]
  have h1S : (1 : ℕ) ∈ S := Finset.mem_union_left _ (Finset.mem_Ico.2 ⟨le_refl 1, by omega⟩)
  have hrate1 : jointRate N b b' 1 = b := by
    simp [jointRate, show (1:ℕ) < N by omega]
  have hmin : ∀ j ∈ S, j ≠ 1 → jointRate N b b' 1 < jointRate N b b' j := by
    intro j hj hne
    rw [hrate1, hS] at *
    rcases Finset.mem_union.1 hj with hj1 | hj2
    · rw [Finset.mem_Ico] at hj1
      have hj2 : 2 ≤ j := by omega
      have hsq : (1 : ℝ) < Real.sqrt j := by
        have hlt' : Real.sqrt 1 < Real.sqrt j := by
          refine Real.sqrt_lt_sqrt (by norm_num) ?_
          have : (2 : ℝ) ≤ (j : ℝ) := by exact_mod_cast hj2
          linarith
        simpa using hlt'
      have hval : jointRate N b b' j = b * Real.sqrt j := by simp [jointRate, hj1.2]
      rw [hval]
      nlinarith
    · simp only [Finset.mem_image, Finset.mem_Ico] at hj2
      obtain ⟨e, he, rfl⟩ := hj2
      have hlt2 : ¬ (N + e < N) := by omega
      have hcast : ((N + e : ℕ) : ℝ) - (N : ℝ) = (e : ℝ) := by push_cast; ring
      have hsq : (1 : ℝ) ≤ Real.sqrt e := by
        have hle : Real.sqrt 1 ≤ Real.sqrt e := by
          refine Real.sqrt_le_sqrt ?_
          have : (1 : ℝ) ≤ (e : ℝ) := by exact_mod_cast he.1
          linarith
        simpa using hle
      have hval : jointRate N b b' (N + e) = b' * Real.sqrt e := by simp [jointRate, hlt2]
      rw [hval]
      nlinarith
  have hamp := expSum_min_rate_zero h1S hmin hzero
  have h1 : jointAmp N b b' c c' 1 = c 1 / b := by
    simp [jointAmp, show (1:ℕ) < N by omega]
  rw [h1, div_eq_zero_iff] at hamp
  rcases hamp with h2 | h2
  · exact h2
  · exact absurd h2 (ne_of_gt hb)

/-- **The confound is broken by an independently known lag-1 correlation.**  If both models carry
a non-zero correlation at lag 1, then agreement of their titration curves at every condition
forces the *same* bond length: the degeneracy of §4 lives entirely in profiles whose short-lag
correlation vanishes. -/
theorem bond_length_identifiable_of_lag1 {N : ℕ} (hN : 2 ≤ N) {b b' kappa0 : ℝ} (hb : 0 < b)
    (hb' : 0 < b') {c c' : ℕ → ℝ} (hc : c 1 ≠ 0) (hc' : c' 1 ≠ 0)
    (h : ∀ kappa, kappa0 ≤ kappa →
      DebyeResolution.debyeCurve N b kappa c = DebyeResolution.debyeCurve N b' kappa c') :
    b = b' := by
  rcases lt_trichotomy b b' with hlt | heq | hgt
  · exact absurd (lag1_zero_of_bond_length_lt hN hb hlt h) hc
  · exact heq
  · exact absurd (lag1_zero_of_bond_length_lt hN hb' hgt (fun k hk => (h k hk).symm)) hc'

/-- **The design rule.**  If the correlation profile is measured independently — by the
separation-resolved panel of Part CXXXVI, whose inversion is `2`-Lipschitz and bond-length free —
and its lag-1 value is non-zero, then the titration determines the bond length. -/
theorem bond_length_identified_given_profile {N : ℕ} (hN : 2 ≤ N) {b b' kappa0 : ℝ} (hb : 0 < b)
    (hb' : 0 < b') {c : ℕ → ℝ} (hc : c 1 ≠ 0)
    (h : ∀ kappa, kappa0 ≤ kappa →
      DebyeResolution.debyeCurve N b kappa c = DebyeResolution.debyeCurve N b' kappa c) :
    b = b' :=
  bond_length_identifiable_of_lag1 hN hb hb' hc hc h

/-! ## 5. The law -/

/-- **The distance-law law.**  Read over a complete titration: the correlation profile of a
declared kernel is identified exactly (chain kernel and Debye–Hückel kernel alike); the choice
between the two kernels is itself testable, since agreement at every condition forces both models
to be blind past lag 1; the bond length inside the Debye kernel is *not* identifiable together
with the profile — it trades exactly against sequence separation, so two models differing in both
agree at every ionic strength; but it *is* identified once the profile is known independently and
its lag-1 value is non-zero. -/
theorem distance_law_law {N : ℕ} {b kappa0 : ℝ} (hN : 2 ≤ N) (hb : 0 < b) :
    (∀ c c' : ℕ → ℝ, (∀ kappa, kappa0 ≤ kappa →
        Titration.curve N c kappa = Titration.curve N c' kappa) → ∀ d ∈ Ico 1 N, c d = c' d) ∧
    (∀ c c' : ℕ → ℝ, (∀ kappa, kappa0 ≤ kappa →
        DebyeResolution.debyeCurve N b kappa c = DebyeResolution.debyeCurve N b kappa c') →
        ∀ d ∈ Ico 1 N, c d = c' d) ∧
    (∀ c : ℕ → ℝ, c 2 ≠ 0 → ∀ c' : ℕ → ℝ, ∃ kappa, kappa0 ≤ kappa ∧
        Titration.curve 3 c kappa ≠ DebyeResolution.debyeCurve 3 1 kappa c') ∧
    (∀ t : ℝ, t ≠ 0 → ∃ b' : ℝ, ∃ c c' : ℕ → ℝ, b ≠ b' ∧ 0 < b' ∧ c 2 ≠ c' 2 ∧
        ∀ kappa : ℝ, DebyeResolution.debyeCurve 3 b kappa c
          = DebyeResolution.debyeCurve 3 b' kappa c') ∧
    (∀ (b' : ℝ) (c : ℕ → ℝ), 0 < b' → c 1 ≠ 0 → (∀ kappa, kappa0 ≤ kappa →
        DebyeResolution.debyeCurve N b kappa c = DebyeResolution.debyeCurve N b' kappa c) →
        b = b') :=
  ⟨fun _ _ h => chain_profile_identifiable h,
   fun _ _ h => debye_profile_identifiable hb h,
   fun _ hc c' => misspecified_kernel_detected hc c',
   fun _ ht => bond_length_not_identifiable hb ht,
   fun _ _ hb' hc h => bond_length_identified_given_profile hN hb hb' hc h⟩

end DistanceLaw
end IDR
