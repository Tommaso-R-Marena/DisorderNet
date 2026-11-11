import RequestProject.FloryHuggins

/-!
# Part CXLVII — Temperature: why an enthalpic model can never produce an LCST

Part LXXV fixed the critical coupling `chiC N = (1 + √N)²/(2N)` of a disordered chain of `N`
segments.  What it left open is where `chi` itself comes from.  In a real solution `chi`
depends on temperature, and *how* it depends on temperature decides the entire topology of
the phase diagram:

* a purely enthalpic interaction gives `chi(T) = A + B/T` with `B > 0`, decreasing in `T`:
  the solution demixes on cooling and mixes on heating — an **upper** critical solution
  temperature, and that is all it can ever give;
* the hydrophobic effect is entropic, and contributes a term that *increases* with `T`.  Only
  such a term can produce a **lower** critical solution temperature, the demixing-on-heating
  that elastin-like and many condensate-forming disordered regions actually show;
* both together give the closed-loop / hourglass diagrams that are observed.

This file proves those statements against the honest demixing predicate of Part LXXV — the
existence of a genuine two-phase splitting of the Flory–Huggins density, not a proxy.

* `demixes_of_chiT_gt`, `not_demixes_of_chiT_le` — the temperature-dependent coupling is
  plugged into the exact critical criterion.
* `enthalpic_no_demixing_on_heating` — **with `C = 0` and `B ≥ 0` heating never demixes.**  If
  the solution is homogeneous at some temperature it is homogeneous at every higher one.  An
  enthalpic model has no LCST, and no amount of fitting `A` and `B` will produce one.
* `entropic_demixes_on_heating` — with `C > 0` the solution demixes at every sufficiently high
  temperature: the LCST exists.
* `hourglass_demixing` — an explicit set of parameters that demixes at `T = 1`, is homogeneous
  at `T = 10`, and demixes again at `T = 100`: both critical temperatures at once.
* `single_temperature_underdetermined` and `three_temperatures_identify` — one phase-boundary
  measurement cannot separate the three contributions; three temperatures determine them
  uniquely.
-/

noncomputable section

namespace RequestProject.SolventQuality

open Set IDR.FH IDR.Phase

/-- The temperature-dependent Flory–Huggins parameter: an entropic offset `A`, an enthalpic
contribution `B/T`, and the entropic (hydrophobic) contribution `C·T`. -/
def chiT (A B C T : ℝ) : ℝ := A + B / T + C * T

/-- The solution of chains of length `N` demixes at temperature `T`. -/
def Demixes (N A B C T : ℝ) : Prop :=
  ∃ c, PhaseSeparates (Icc (0 : ℝ) 1) (fh N (chiT A B C T)) c

/-- Above the critical coupling the temperature-dependent model demixes. -/
theorem demixes_of_chiT_gt {N A B C T : ℝ} (hN : 0 < N) (h : chiC N < chiT A B C T) :
    Demixes N A B C T :=
  fh_demixes_above_chiC hN h

/-- At or below the critical coupling it does not. -/
theorem not_demixes_of_chiT_le {N A B C T : ℝ} (hN : 0 < N) (h : chiT A B C T ≤ chiC N) :
    ¬ Demixes N A B C T := by
  rintro ⟨c, hc⟩
  exact no_demixing_below_chiC hN h c hc

/-! ### A purely enthalpic model has no lower critical solution temperature -/

/-- With no entropic term the coupling is antitone in temperature. -/
theorem chiT_antitone_of_enthalpic {A B : ℝ} (hB : 0 ≤ B) {T T' : ℝ} (hT : 0 < T)
    (hTT : T ≤ T') : chiT A B 0 T' ≤ chiT A B 0 T := by
  unfold chiT
  have h : B / T' ≤ B / T := by
    rcases eq_or_lt_of_le hB with h0 | h0
    · simp [← h0]
    · exact div_le_div_of_nonneg_left hB hT hTT
  linarith

/-- **An enthalpic model cannot demix on heating.**  If a purely enthalpic solution is
homogeneous at a temperature, it is homogeneous at every higher temperature: the model has
an upper critical solution temperature and nothing else. -/
theorem enthalpic_no_demixing_on_heating {N A B : ℝ} (hN : 0 < N) (hB : 0 ≤ B)
    {T T' : ℝ} (hT : 0 < T) (hTT : T ≤ T') (h : ¬ Demixes N A B 0 T) :
    ¬ Demixes N A B 0 T' := by
  have hle : chiT A B 0 T ≤ chiC N := by
    by_contra hcon
    exact h (demixes_of_chiT_gt hN (lt_of_not_ge hcon))
  exact not_demixes_of_chiT_le hN
    (le_trans (chiT_antitone_of_enthalpic hB hT hTT) hle)

/-- Contrapositive form: an enthalpic solution that demixes at a temperature demixes at every
lower positive temperature. -/
theorem enthalpic_demixing_downward_closed {N A B : ℝ} (hN : 0 < N) (hB : 0 ≤ B)
    {T T' : ℝ} (hT : 0 < T) (hTT : T ≤ T') (h : Demixes N A B 0 T') :
    Demixes N A B 0 T := by
  by_contra hcon
  exact enthalpic_no_demixing_on_heating hN hB hT hTT hcon h

/-! ### An entropic term produces a lower critical solution temperature -/

/-- **Demixing on heating.**  With a positive entropic term the solution demixes at every
sufficiently high temperature. -/
theorem entropic_demixes_on_heating {N A B C : ℝ} (hN : 0 < N) (hB : 0 ≤ B) (hC : 0 < C) :
    ∃ T0 : ℝ, 0 < T0 ∧ ∀ T : ℝ, T0 < T → Demixes N A B C T := by
  refine ⟨1 + |A| / C + |chiC N| / C, by positivity, fun T hT => ?_⟩
  have hT0 : 0 < T := lt_trans (by positivity) hT
  refine demixes_of_chiT_gt hN ?_
  unfold chiT
  have hBT : 0 ≤ B / T := by positivity
  have hCT : (|A| / C + |chiC N| / C) * C < T * C := by
    have : |A| / C + |chiC N| / C < T := by linarith
    exact mul_lt_mul_of_pos_right this hC
  have hexp : (|A| / C + |chiC N| / C) * C = |A| + |chiC N| := by field_simp
  rw [hexp] at hCT
  have h1 : chiC N ≤ |chiC N| := le_abs_self _
  have h2 : -|A| ≤ A := neg_abs_le A
  nlinarith

/-! ### Both temperatures at once -/

/-- **An explicit hourglass diagram.**  With `A = 0`, `B = 10`, `C = 1/10` a solution of unit
chains demixes at `T = 1`, is homogeneous at `T = 10`, and demixes again at `T = 100`: the
same model has both an upper and a lower critical solution temperature. -/
theorem hourglass_demixing :
    Demixes 1 0 10 (1/10) 1 ∧ ¬ Demixes 1 0 10 (1/10) 10 ∧ Demixes 1 0 10 (1/10) 100 := by
  have hC : chiC 1 = 2 := chiC_one
  refine ⟨demixes_of_chiT_gt one_pos ?_, not_demixes_of_chiT_le one_pos ?_,
    demixes_of_chiT_gt one_pos ?_⟩ <;> rw [hC, chiT] <;> norm_num

/-! ### Identifiability of the three contributions -/

/-- **One temperature is not enough.**  For any assumed enthalpic and entropic coefficients
there is an offset reproducing the coupling measured at a single temperature. -/
theorem single_temperature_underdetermined (A B C T B' C' : ℝ) :
    ∃ A' : ℝ, chiT A' B' C' T = chiT A B C T := by
  refine ⟨A + B / T + C * T - B' / T - C' * T, ?_⟩
  unfold chiT
  ring

/-- **Three temperatures are.**  Couplings measured at three distinct positive temperatures
determine the offset, the enthalpic coefficient and the entropic coefficient uniquely. -/
theorem three_temperatures_identify {A B C A' B' C' T1 T2 T3 : ℝ}
    (h1 : 0 < T1) (h2 : 0 < T2) (h3 : 0 < T3)
    (h12 : T1 ≠ T2) (h13 : T1 ≠ T3) (h23 : T2 ≠ T3)
    (e1 : chiT A' B' C' T1 = chiT A B C T1)
    (e2 : chiT A' B' C' T2 = chiT A B C T2)
    (e3 : chiT A' B' C' T3 = chiT A B C T3) :
    A' = A ∧ B' = B ∧ C' = C := by
  set a : ℝ := A' - A with ha
  set b : ℝ := B' - B with hb
  set c : ℝ := C' - C with hc
  -- clearing the denominators turns each equation into a quadratic in the temperature
  have q1 : c * T1 ^ 2 + a * T1 + b = 0 := by
    unfold chiT at e1
    field_simp [ha, hb, hc] at e1 ⊢
    nlinarith [e1]
  have q2 : c * T2 ^ 2 + a * T2 + b = 0 := by
    unfold chiT at e2
    field_simp [ha, hb, hc] at e2 ⊢
    nlinarith [e2]
  have q3 : c * T3 ^ 2 + a * T3 + b = 0 := by
    unfold chiT at e3
    field_simp [ha, hb, hc] at e3 ⊢
    nlinarith [e3]
  -- three distinct roots force the quadratic to vanish identically
  have d12 : (T1 - T2) * (c * (T1 + T2) + a) = 0 := by nlinarith [q1, q2]
  have d13 : (T1 - T3) * (c * (T1 + T3) + a) = 0 := by nlinarith [q1, q3]
  have s12 : c * (T1 + T2) + a = 0 := by
    rcases mul_eq_zero.1 d12 with h | h
    · exact absurd (by linarith : T1 = T2) h12
    · exact h
  have s13 : c * (T1 + T3) + a = 0 := by
    rcases mul_eq_zero.1 d13 with h | h
    · exact absurd (by linarith : T1 = T3) h13
    · exact h
  have hcz : c * (T2 - T3) = 0 := by linarith
  have hc0 : c = 0 := by
    rcases mul_eq_zero.1 hcz with h | h
    · exact h
    · exact absurd (by linarith : T2 = T3) h23
  have ha0 : a = 0 := by rw [hc0] at s12; linarith
  have hb0 : b = 0 := by rw [hc0, ha0] at q1; linarith
  exact ⟨by linarith [ha0, ha], by linarith [hb0, hb], by linarith [hc0, hc]⟩

/-- **The solvent-quality design law.**  For a disordered chain whose Flory–Huggins parameter
is `A + B/T + C·T`:

1. a purely enthalpic model (`C = 0`, `B ≥ 0`) never demixes on heating, so it cannot exhibit
   a lower critical solution temperature at all;
2. a positive entropic coefficient makes the solution demix at every sufficiently high
   temperature;
3. an explicit parameter set produces both critical temperatures at once;
4. a single phase-boundary temperature leaves the three contributions unseparated, and three
   distinct temperatures determine them uniquely. -/
theorem solvent_quality_law {N : ℝ} (hN : 0 < N) :
    (∀ A B T T' : ℝ, 0 ≤ B → 0 < T → T ≤ T' → ¬ Demixes N A B 0 T → ¬ Demixes N A B 0 T') ∧
    (∀ A B C : ℝ, 0 ≤ B → 0 < C →
      ∃ T0 : ℝ, 0 < T0 ∧ ∀ T : ℝ, T0 < T → Demixes N A B C T) ∧
    (Demixes 1 0 10 (1/10) 1 ∧ ¬ Demixes 1 0 10 (1/10) 10 ∧ Demixes 1 0 10 (1/10) 100) ∧
    (∀ A B C T B' C' : ℝ, ∃ A' : ℝ, chiT A' B' C' T = chiT A B C T) :=
  ⟨fun _ _ _ _ hB hT hTT h => enthalpic_no_demixing_on_heating hN hB hT hTT h,
    fun _ _ _ hB hC => entropic_demixes_on_heating hN hB hC,
    hourglass_demixing,
    single_temperature_underdetermined⟩

end RequestProject.SolventQuality
