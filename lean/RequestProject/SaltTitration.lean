/-
# Part CXXIX  The salt titration: what a solution series does and does not identify

Parts CXXVI–CXXVIII (`SaltCrossover.lean`, `DebyeScreening.lean`, `SaltAwareDesign.lean`) showed
that the charge-patterning contrast of a disordered region is a function of the ionic strength,
and dies once the Debye length falls below the length of the region.  That leaves the
experimental question this part answers.  A charge-patterning model is fitted to data taken at
finitely many salt concentrations.  *What is identified by such
a titration?*

The screened pairwise energy of Part CXXVI is, by the regrouping identity of Part LXXIII,

    E(κ) = ∑_{d=1}^{N−1} d · exp(−κ d) · C(d),        C(d) = ∑_i q_i q_{i+d}

— a Dirichlet series in the ionic-strength variable whose coefficients are the charge
autocorrelations.  Everything below is exact and machine-checked.

* `energy_eq_sum_autocorr` — the titration curve in that form.
* `autocorr_eq_of_energy_eq_on_infinite` — **the identifiability theorem.**  If two sequences give
  the same energy at *infinitely many* salt conditions (an interval of concentrations, say, or any
  infinite set), then their charge autocorrelations agree at every lag.  The proof turns the
  titration curve into a polynomial in `x = exp(−κ)`, which then has infinitely many roots.
* `energy_eq_of_autocorr_eq`, `titration_iff` — the converse, hence the exact statement: the salt
  titration determines the autocorrelation function and nothing beyond it.
* `titration_determines_every_kernel` — the payoff.  Because the autocorrelation is the complete
  invariant of pairwise distance models, a titration determines the region's pairwise energetics
  under *every* solution condition and *every* separation kernel, including the unscreened `κ = 0`
  limit that no experiment can reach.  Extrapolation off the measured conditions is legitimate
  inside the pairwise model class.
* `single_condition_degenerate`, `single_condition_needs_a_second` — **and one condition is never
  enough.**  At any fixed ionic strength there are two sequences of unit charges, with different
  autocorrelations, of exactly equal energy: `(+1, 0, 0)` and `(+1, −e^{−κ}, +1)`.  A model
  calibrated at a single salt concentration is not identified, and by the theorem above there
  must exist a second concentration that separates the two.
* `energy_high_salt_bound`, `high_salt_uninformative` — the realistic caveat.  The titration curve
  decays like `N³ e^{−κ}`, so an instrument of finite energy resolution `eps` sees nothing beyond
  `κ = log(2N³/eps)`: the usable range of the titration grows only *logarithmically* with the
  precision of the calorimeter.
* `finite_titration_underdetermined` — and inside that range, a titration with fewer salt points
  than lags leaves a whole subspace of correlation profiles unresolved: `N − 1` independent
  conditions are needed to pin down `N − 1` autocorrelations.

The design consequence: ionic strength is not a nuisance parameter to be held fixed, it is the
*experimental axis along which the charge-patterning part of a disordered-region model becomes
identifiable at all* — and it must be sampled at as many well-separated, low-to-moderate
concentrations as there are lags one hopes to resolve.
-/
import Mathlib
import RequestProject.ChargePatterning
import RequestProject.SaltCrossover

set_option autoImplicit false

namespace IDR
namespace Titration

open Finset Polynomial

/-! ## 1. The titration curve -/

/-- **The titration curve.**  The screened energy at inverse screening length `κ` is the
Dirichlet series `∑_d d e^{−κd} C(d)` in the charge autocorrelations. -/
theorem energy_eq_sum_autocorr (N : ℕ) (kappa : ℝ) (q : ℕ → ℝ) :
    Salt.energy N kappa q
      = ∑ d ∈ Ico 1 N, (d : ℝ) * Real.exp (-(kappa * d)) * Pattern.autocorr N q d := by
  rw [Salt.energy, Pattern.pairEnergy_eq_sum_autocorr]
  exact Finset.sum_congr rfl fun d _ => by rw [Salt.kern]

/-- The polynomial whose value at `x = exp(−κ)` is the titration curve with coefficients `c`. -/
noncomputable def titrationPoly (N : ℕ) (c : ℕ → ℝ) : ℝ[X] :=
  ∑ d ∈ Ico 1 N, Polynomial.C ((d : ℝ) * c d) * X ^ d

lemma titrationPoly_eval (N : ℕ) (c : ℕ → ℝ) (x : ℝ) :
    (titrationPoly N c).eval x = ∑ d ∈ Ico 1 N, (d : ℝ) * c d * x ^ d := by
  simp [titrationPoly, Polynomial.eval_finset_sum]

lemma titrationPoly_coeff (N : ℕ) (c : ℕ → ℝ) {d : ℕ} (hd1 : 1 ≤ d) (hdN : d < N) :
    (titrationPoly N c).coeff d = (d : ℝ) * c d := by
  rw [titrationPoly, Polynomial.finset_sum_coeff]
  rw [Finset.sum_eq_single d]
  · rw [Polynomial.coeff_C_mul, Polynomial.coeff_X_pow, if_pos rfl, mul_one]
  · intro b _ hb
    rw [Polynomial.coeff_C_mul, Polynomial.coeff_X_pow, if_neg (Ne.symm hb), mul_zero]
  · intro h
    exact absurd (Finset.mem_Ico.mpr ⟨hd1, hdN⟩) h

/-- The titration curve is the polynomial evaluated at `exp(−κ)`. -/
lemma energy_eq_poly_eval (N : ℕ) (kappa : ℝ) (q : ℕ → ℝ) :
    Salt.energy N kappa q
      = (titrationPoly N (Pattern.autocorr N q)).eval (Real.exp (-kappa)) := by
  rw [energy_eq_sum_autocorr, titrationPoly_eval]
  refine Finset.sum_congr rfl fun d _ => ?_
  have : Real.exp (-kappa) ^ d = Real.exp (-(kappa * d)) := by
    rw [← Real.exp_nat_mul]; ring_nf
  rw [this]; ring

/-! ## 2. Identifiability from an infinite set of salt conditions -/

/-- **The identifiability theorem.**  Two charge sequences whose screened energies agree at
infinitely many ionic strengths have the same charge autocorrelation at every lag. -/
theorem autocorr_eq_of_energy_eq_on_infinite {N : ℕ} {S : Set ℝ} (hS : S.Infinite)
    {q q' : ℕ → ℝ} (h : ∀ kappa ∈ S, Salt.energy N kappa q = Salt.energy N kappa q') :
    ∀ d, 1 ≤ d → d < N → Pattern.autocorr N q d = Pattern.autocorr N q' d := by
  set P : ℝ[X] :=
    titrationPoly N (fun d => Pattern.autocorr N q d - Pattern.autocorr N q' d) with hP
  have hsplit : ∀ x : ℝ,
      P.eval x = (titrationPoly N (Pattern.autocorr N q)).eval x
        - (titrationPoly N (Pattern.autocorr N q')).eval x := by
    intro x
    rw [hP, titrationPoly_eval, titrationPoly_eval, titrationPoly_eval, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun d _ => by ring
  have hroots : (Real.exp ∘ Neg.neg) '' S ⊆ {x | P.IsRoot x} := by
    rintro x ⟨kappa, hkappa, rfl⟩
    have := h kappa hkappa
    simp only [Set.mem_setOf_eq, Polynomial.IsRoot, Function.comp_apply]
    rw [hsplit, ← energy_eq_poly_eval, ← energy_eq_poly_eval, this, sub_self]
  have hinj : Set.InjOn (Real.exp ∘ Neg.neg) S := by
    intro a _ b _ hab
    have : Real.exp (-a) = Real.exp (-b) := hab
    have := Real.exp_injective this
    linarith
  have : ({x | P.IsRoot x} : Set ℝ).Infinite :=
    Set.Infinite.mono hroots (hS.image hinj)
  have hP0 : P = 0 := Polynomial.eq_zero_of_infinite_isRoot P this
  intro d hd1 hdN
  have hc := titrationPoly_coeff N (fun d => Pattern.autocorr N q d - Pattern.autocorr N q' d)
    hd1 hdN
  rw [← hP] at hc
  rw [hP0] at hc
  simp only [Polynomial.coeff_zero] at hc
  have hdpos : (0 : ℝ) < d := by exact_mod_cast hd1
  have : Pattern.autocorr N q d - Pattern.autocorr N q' d = 0 := by
    rcases mul_eq_zero.1 hc.symm with h' | h'
    · exact absurd h' (ne_of_gt hdpos)
    · exact h'
  linarith [this]

/-- The converse: equal autocorrelations give equal energies at every ionic strength. -/
theorem energy_eq_of_autocorr_eq {N : ℕ} {q q' : ℕ → ℝ}
    (h : ∀ d, 1 ≤ d → d < N → Pattern.autocorr N q d = Pattern.autocorr N q' d) :
    ∀ kappa : ℝ, Salt.energy N kappa q = Salt.energy N kappa q' := by
  intro kappa
  rw [energy_eq_sum_autocorr, energy_eq_sum_autocorr]
  refine Finset.sum_congr rfl fun d hd => ?_
  rw [Finset.mem_Ico] at hd
  rw [h d hd.1 hd.2]

/-- **What the titration identifies, exactly.**  For any infinite set of salt conditions, equality
of the measured energies there is equivalent to equality of the charge autocorrelations — no
more, and no less. -/
theorem titration_iff {N : ℕ} {S : Set ℝ} (hS : S.Infinite) (q q' : ℕ → ℝ) :
    (∀ kappa ∈ S, Salt.energy N kappa q = Salt.energy N kappa q')
      ↔ ∀ d, 1 ≤ d → d < N → Pattern.autocorr N q d = Pattern.autocorr N q' d := by
  constructor
  · exact fun h => autocorr_eq_of_energy_eq_on_infinite hS h
  · intro h kappa _
    exact energy_eq_of_autocorr_eq h kappa

/-- **Extrapolation is legitimate inside the pairwise class.**  A titration over infinitely many
salt conditions determines the energy under *every* separation kernel — in particular at zero
salt, and under any other screened, power-law or cut-off coupling. -/
theorem titration_determines_every_kernel {N : ℕ} {S : Set ℝ} (hS : S.Infinite) {q q' : ℕ → ℝ}
    (h : ∀ kappa ∈ S, Salt.energy N kappa q = Salt.energy N kappa q') :
    ∀ w : ℕ → ℝ, Pattern.pairEnergy N w q = Pattern.pairEnergy N w q' :=
  (Pattern.autocorr_eq_iff_pairEnergy_eq N q q').1
    (autocorr_eq_of_energy_eq_on_infinite hS h)

/-! ## 3. One salt condition is never enough -/

/-- A single charge at the first residue. -/
noncomputable def qOne : ℕ → ℝ := fun i => if i = 0 then 1 else 0

/-- The three-residue sequence `(+1, −e^{−κ}, +1)`, tuned to have zero screened energy at `κ`. -/
noncomputable def qTuned (kappa : ℝ) : ℕ → ℝ :=
  fun i => if i = 1 then -Real.exp (-kappa) else 1

lemma autocorr_qOne_one : Pattern.autocorr 3 qOne 1 = 0 := by
  simp [Pattern.autocorr, qOne]

lemma autocorr_qOne_two : Pattern.autocorr 3 qOne 2 = 0 := by
  simp [Pattern.autocorr, qOne]

lemma autocorr_qTuned_one (kappa : ℝ) :
    Pattern.autocorr 3 (qTuned kappa) 1 = -2 * Real.exp (-kappa) := by
  simp [Pattern.autocorr, qTuned, Finset.sum_range_succ]
  ring

lemma autocorr_qTuned_two (kappa : ℝ) : Pattern.autocorr 3 (qTuned kappa) 2 = 1 := by
  simp [Pattern.autocorr, qTuned]

lemma energy_three (kappa : ℝ) (q : ℕ → ℝ) :
    Salt.energy 3 kappa q
      = Real.exp (-(kappa * 1)) * Pattern.autocorr 3 q 1
        + 2 * Real.exp (-(kappa * 2)) * Pattern.autocorr 3 q 2 := by
  rw [energy_eq_sum_autocorr]
  norm_num [show Ico 1 3 = ({1, 2} : Finset ℕ) from rfl, Finset.sum_insert, Finset.sum_pair]

/-- **A single ionic strength does not identify the charge pattern.**  At every `κ ≥ 0` the
sequences `(+1, 0, 0)` and `(+1, −e^{−κ}, +1)` — both of unit charges — have exactly the same
screened energy, yet different charge autocorrelations. -/
theorem single_condition_degenerate (kappa : ℝ) (hk : 0 ≤ kappa) :
    (∀ i, |qOne i| ≤ 1) ∧ (∀ i, |qTuned kappa i| ≤ 1) ∧
      Salt.energy 3 kappa qOne = Salt.energy 3 kappa (qTuned kappa) ∧
      Pattern.autocorr 3 qOne 2 ≠ Pattern.autocorr 3 (qTuned kappa) 2 := by
  have hexp1 : Real.exp (-kappa) ≤ 1 := by
    rw [Real.exp_le_one_iff]; linarith
  have hexp0 : (0 : ℝ) < Real.exp (-kappa) := Real.exp_pos _
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i; unfold qOne; split <;> norm_num
  · intro i; unfold qTuned; split
    · rw [abs_neg, abs_of_pos hexp0]; exact hexp1
    · norm_num
  · rw [energy_three, energy_three, autocorr_qOne_one, autocorr_qOne_two,
      autocorr_qTuned_one, autocorr_qTuned_two]
    have h2 : Real.exp (-(kappa * 2)) = Real.exp (-(kappa * 1)) * Real.exp (-kappa) := by
      rw [← Real.exp_add]; ring_nf
    rw [h2]
    have : Real.exp (-(kappa * 1)) = Real.exp (-kappa) := by ring_nf
    rw [this]; ring
  · rw [autocorr_qOne_two, autocorr_qTuned_two]
    norm_num

/-- **Hence a second condition must exist that separates them.**  The degenerate pair of
`single_condition_degenerate` cannot be degenerate at every ionic strength. -/
theorem single_condition_needs_a_second (kappa : ℝ) :
    ∃ kappa' : ℝ, Salt.energy 3 kappa' qOne ≠ Salt.energy 3 kappa' (qTuned kappa) := by
  by_contra hcon
  push_neg at hcon
  have h := autocorr_eq_of_energy_eq_on_infinite (S := (Set.univ : Set ℝ))
    Set.infinite_univ (q := qOne) (q' := qTuned kappa) (fun k _ => hcon k) 2 (by norm_num)
    (by norm_num)
  rw [autocorr_qOne_two, autocorr_qTuned_two] at h
  norm_num at h

/-! ## 4. The usable range of the titration is logarithmic in the resolution -/

lemma abs_autocorr_le {N : ℕ} {q : ℕ → ℝ} (hq : ∀ i, |q i| ≤ 1) (d : ℕ) :
    |Pattern.autocorr N q d| ≤ N := by
  calc |Pattern.autocorr N q d| ≤ ∑ i ∈ range (N - d), |q i * q (i + d)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i ∈ range (N - d), (1 : ℝ) := by
        refine Finset.sum_le_sum fun i _ => ?_
        rw [abs_mul]
        exact mul_le_one₀ (hq i) (abs_nonneg _) (hq _)
    _ = ((N - d : ℕ) : ℝ) := by simp
    _ ≤ N := by
        have : (N - d : ℕ) ≤ N := Nat.sub_le _ _
        exact_mod_cast this

/-- **High salt is uninformative.**  The screened energy of a region of `N` unit charges is at
most `N³ e^{−κ}` at every `κ ≥ 0`: the titration curve decays exponentially in the ionic
strength. -/
theorem energy_high_salt_bound {N : ℕ} {kappa : ℝ} (hk : 0 ≤ kappa) {q : ℕ → ℝ}
    (hq : ∀ i, |q i| ≤ 1) :
    |Salt.energy N kappa q| ≤ (N : ℝ) ^ 3 * Real.exp (-kappa) := by
  rw [energy_eq_sum_autocorr]
  have hterm : ∀ d ∈ Ico 1 N,
      |(d : ℝ) * Real.exp (-(kappa * d)) * Pattern.autocorr N q d|
        ≤ (N : ℝ) ^ 2 * Real.exp (-kappa) := by
    intro d hd
    rw [Finset.mem_Ico] at hd
    have hdN : (d : ℝ) ≤ N := by exact_mod_cast hd.2.le
    have hd1 : (1 : ℝ) ≤ d := by exact_mod_cast hd.1
    have hexp : Real.exp (-(kappa * d)) ≤ Real.exp (-kappa) := by
      apply Real.exp_le_exp.2
      nlinarith
    have hexp0 : (0 : ℝ) < Real.exp (-(kappa * d)) := Real.exp_pos _
    rw [abs_mul, abs_mul, abs_of_nonneg (by positivity : (0:ℝ) ≤ (d:ℝ)),
      abs_of_pos hexp0]
    have hac := abs_autocorr_le (N := N) hq d
    have hacn : (0:ℝ) ≤ |Pattern.autocorr N q d| := abs_nonneg _
    have hN0 : (0:ℝ) ≤ N := Nat.cast_nonneg N
    calc (d : ℝ) * Real.exp (-(kappa * d)) * |Pattern.autocorr N q d|
        ≤ (N : ℝ) * Real.exp (-kappa) * (N : ℝ) := by
          apply mul_le_mul (mul_le_mul hdN hexp hexp0.le hN0) hac hacn
            (by positivity)
      _ = (N : ℝ) ^ 2 * Real.exp (-kappa) := by ring
  calc |∑ d ∈ Ico 1 N, (d : ℝ) * Real.exp (-(kappa * d)) * Pattern.autocorr N q d|
      ≤ ∑ d ∈ Ico 1 N, |(d : ℝ) * Real.exp (-(kappa * d)) * Pattern.autocorr N q d| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _d ∈ Ico 1 N, (N : ℝ) ^ 2 * Real.exp (-kappa) := Finset.sum_le_sum hterm
    _ = ((Ico 1 N).card : ℝ) * ((N : ℝ) ^ 2 * Real.exp (-kappa)) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ (N : ℝ) ^ 3 * Real.exp (-kappa) := by
        have hcard : ((Ico 1 N).card : ℝ) ≤ N := by
          rw [Nat.card_Ico]
          have : N - 1 ≤ N := Nat.sub_le _ _
          exact_mod_cast this
        have : (0:ℝ) ≤ (N : ℝ) ^ 2 * Real.exp (-kappa) := by positivity
        nlinarith [this, hcard]

/-- **The usable range of a titration is logarithmic in the instrument's resolution.**  Beyond
`κ > log (2N³/eps)` no two sequences of unit charges differ by as much as `eps` in energy, so no
measurement of resolution `eps` can learn anything there. -/
theorem high_salt_uninformative {N : ℕ} {eps kappa : ℝ} (heps : 0 < eps) (hN : 0 < N)
    (hk0 : 0 ≤ kappa) (hkap : Real.log (2 * (N : ℝ) ^ 3 / eps) < kappa) {q q' : ℕ → ℝ}
    (hq : ∀ i, |q i| ≤ 1) (hq' : ∀ i, |q' i| ≤ 1) :
    |Salt.energy N kappa q - Salt.energy N kappa q'| < eps := by
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hpos : (0 : ℝ) < 2 * (N : ℝ) ^ 3 / eps := by positivity
  have hexp : Real.exp (-kappa) < eps / (2 * (N : ℝ) ^ 3) := by
    have h1 : Real.exp (Real.log (2 * (N : ℝ) ^ 3 / eps)) < Real.exp kappa :=
      Real.exp_lt_exp.2 hkap
    rw [Real.exp_log hpos] at h1
    rw [Real.exp_neg]
    have h2 : (Real.exp kappa)⁻¹ < (2 * (N : ℝ) ^ 3 / eps)⁻¹ :=
      (inv_lt_inv₀ (Real.exp_pos _) hpos).2 h1
    have h3 : (2 * (N : ℝ) ^ 3 / eps)⁻¹ = eps / (2 * (N : ℝ) ^ 3) := by
      field_simp
    linarith [h2, h3.le, h3.ge]
  have hb := energy_high_salt_bound (N := N) (kappa := kappa) hk0 hq
  have hb' := energy_high_salt_bound (N := N) (kappa := kappa) hk0 hq'
  have : |Salt.energy N kappa q - Salt.energy N kappa q'|
      ≤ 2 * ((N : ℝ) ^ 3 * Real.exp (-kappa)) := by
    calc |Salt.energy N kappa q - Salt.energy N kappa q'|
        ≤ |Salt.energy N kappa q| + |Salt.energy N kappa q'| := abs_sub _ _
      _ ≤ (N : ℝ) ^ 3 * Real.exp (-kappa) + (N : ℝ) ^ 3 * Real.exp (-kappa) := by linarith
      _ = 2 * ((N : ℝ) ^ 3 * Real.exp (-kappa)) := by ring
  have hN3 : (0 : ℝ) < 2 * (N : ℝ) ^ 3 := by positivity
  have h4 : 2 * (N : ℝ) ^ 3 * Real.exp (-kappa) < 2 * (N : ℝ) ^ 3 * (eps / (2 * (N : ℝ) ^ 3)) :=
    mul_lt_mul_of_pos_left hexp hN3
  have h5 : 2 * (N : ℝ) ^ 3 * (eps / (2 * (N : ℝ) ^ 3)) = eps := by
    field_simp
  linarith [h4, h5.le, h5.ge]

/-! ## 5. Fewer salt conditions than lags leaves a blind subspace -/

/-- The titration curve of an arbitrary correlation profile `c`: the read-out the experiment
produces if the region's charge autocorrelations were `c`. -/
noncomputable def curve (N : ℕ) (c : ℕ → ℝ) (kappa : ℝ) : ℝ :=
  ∑ d ∈ Ico 1 N, (d : ℝ) * Real.exp (-(kappa * d)) * c d

/-- The measured energy is the titration curve of the charge autocorrelation. -/
lemma energy_eq_curve (N : ℕ) (kappa : ℝ) (q : ℕ → ℝ) :
    Salt.energy N kappa q = curve N (Pattern.autocorr N q) kappa :=
  energy_eq_sum_autocorr N kappa q

/-- The titration curve is linear in the correlation profile. -/
lemma curve_add (N : ℕ) (c c' : ℕ → ℝ) (kappa : ℝ) :
    curve N (fun d => c d + c' d) kappa = curve N c kappa + curve N c' kappa := by
  rw [curve, curve, curve, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun d _ => by ring

/-- **A titration with fewer conditions than lags is underdetermined.**  Given any `k` ionic
strengths with `k + 1 < N`, there is a correlation profile that is not identically zero on the
lags `1 ≤ d < N` and yet produces an identically zero read-out at all `k` conditions. -/
theorem finite_titration_underdetermined {N k : ℕ} (hk : k + 1 < N) (kap : Fin k → ℝ) :
    ∃ c : ℕ → ℝ, (∃ d, 1 ≤ d ∧ d < N ∧ c d ≠ 0) ∧ ∀ j, curve N c (kap j) = 0 := by
  classical
  set m := N - 1 with hm
  have hkm : k < m := by omega
  set M : Matrix (Fin k) (Fin m) ℝ :=
    Matrix.of fun j i => (((i : ℕ) + 1 : ℝ)) * Real.exp (-(kap j * ((i : ℕ) + 1))) with hM
  obtain ⟨v, hv0, hv⟩ : ∃ v : Fin m → ℝ, v ≠ 0 ∧ ∀ j, ∑ i, M j i * v i = 0 := by
    set f : (Fin m → ℝ) →ₗ[ℝ] (Fin k → ℝ) := Matrix.toLin' M with hf
    have hrank : 0 < Module.finrank ℝ (LinearMap.ker f) := by
      have h := LinearMap.finrank_range_add_finrank_ker f
      have h1 : Module.finrank ℝ (LinearMap.range f) ≤ k := by
        have := Submodule.finrank_le (LinearMap.range f)
        simpa using this
      have h2 : Module.finrank ℝ (Fin m → ℝ) = m := by simp
      omega
    have hnt : Nontrivial (LinearMap.ker f) := Module.nontrivial_of_finrank_pos hrank
    obtain ⟨w, hw⟩ := exists_ne (0 : LinearMap.ker f)
    refine ⟨(w : Fin m → ℝ), ?_, ?_⟩
    · simpa using fun h => hw (Subtype.ext h)
    · intro j
      have h0 : Matrix.toLin' M (w : Fin m → ℝ) = 0 := w.2
      exact congrFun h0 j
  set vExt : ℕ → ℝ := fun n => if h : n < m then v ⟨n, h⟩ else 0 with hvExt
  have hvExt_apply : ∀ i : Fin m, vExt (i : ℕ) = v i := by
    intro i
    rw [hvExt]
    simp [i.2]
  refine ⟨fun d => if 1 ≤ d then vExt (d - 1) else 0, ?_, ?_⟩
  · obtain ⟨i, hi⟩ : ∃ i : Fin m, v i ≠ 0 := by
      by_contra hc
      push_neg at hc
      exact hv0 (funext hc)
    refine ⟨(i : ℕ) + 1, by omega, by have := i.2; omega, ?_⟩
    simp only [Nat.add_sub_cancel, if_pos (Nat.le_add_left 1 (i : ℕ))]
    rw [hvExt_apply]
    exact hi
  · intro j
    have hsum := hv j
    rw [curve, Finset.sum_Ico_eq_sum_range]
    rw [← Fin.sum_univ_eq_sum_range (fun i => ((1 + i : ℕ) : ℝ) * Real.exp (-(kap j * (1 + i : ℕ))) *
      (if 1 ≤ 1 + i then vExt (1 + i - 1) else 0)) (N - 1)]
    rw [← hsum]
    refine Finset.sum_congr rfl fun i _ => ?_
    have h1 : (1 : ℕ) ≤ 1 + (i : ℕ) := Nat.le_add_right 1 _
    simp only [if_pos h1, Nat.add_sub_cancel_left, hvExt_apply i, hM, Matrix.of_apply]
    push_cast
    ring_nf

/-- The same statement as a pair of indistinguishable profiles: whatever the true correlations,
a `k`-point titration with `k + 1 < N` cannot rule out a different profile. -/
theorem finite_titration_two_profiles {N k : ℕ} (hk : k + 1 < N) (kap : Fin k → ℝ)
    (c₀ : ℕ → ℝ) :
    ∃ c : ℕ → ℝ, (∃ d, 1 ≤ d ∧ d < N ∧ c d ≠ c₀ d) ∧
      ∀ j, curve N c (kap j) = curve N c₀ (kap j) := by
  obtain ⟨c, ⟨d, hd1, hdN, hcd⟩, hzero⟩ := finite_titration_underdetermined hk kap
  refine ⟨fun e => c₀ e + c e, ⟨d, hd1, hdN, by simpa using hcd⟩, fun j => ?_⟩
  rw [curve_add, hzero j, add_zero]

/-! ## 6. Capstone -/

/-- **The salt-titration law for a charge-patterning model of a disordered region.**
(1) the measured energy at ionic strength `κ` is the Dirichlet series in the charge
autocorrelations; (2) a titration over infinitely many conditions identifies those
autocorrelations exactly, and with them the energy under every separation kernel — including the
unreachable zero-salt limit; (3) a single condition identifies nothing: two sequences of unit
charges with different autocorrelations share its read-out; (4) fewer conditions than lags leave
a blind direction in correlation space; and (5) beyond `κ = log(2N³/eps)` an instrument of
resolution `eps` can separate no two sequences at all. -/
theorem salt_titration_law (N : ℕ) (hN : 0 < N) :
    (∀ (kappa : ℝ) (q : ℕ → ℝ),
        Salt.energy N kappa q
          = ∑ d ∈ Ico 1 N, (d : ℝ) * Real.exp (-(kappa * d)) * Pattern.autocorr N q d) ∧
    (∀ (S : Set ℝ), S.Infinite → ∀ q q' : ℕ → ℝ,
        (∀ kappa ∈ S, Salt.energy N kappa q = Salt.energy N kappa q') →
          (∀ d, 1 ≤ d → d < N → Pattern.autocorr N q d = Pattern.autocorr N q' d) ∧
            ∀ w : ℕ → ℝ, Pattern.pairEnergy N w q = Pattern.pairEnergy N w q') ∧
    (∀ kappa : ℝ, 0 ≤ kappa →
        Salt.energy 3 kappa qOne = Salt.energy 3 kappa (qTuned kappa) ∧
          Pattern.autocorr 3 qOne 2 ≠ Pattern.autocorr 3 (qTuned kappa) 2) ∧
    (∀ (k : ℕ), k + 1 < N → ∀ kap : Fin k → ℝ,
        ∃ c : ℕ → ℝ, (∃ d, 1 ≤ d ∧ d < N ∧ c d ≠ 0) ∧ ∀ j, curve N c (kap j) = 0) ∧
    (∀ (eps kappa : ℝ), 0 < eps → 0 ≤ kappa → Real.log (2 * (N : ℝ) ^ 3 / eps) < kappa →
        ∀ q q' : ℕ → ℝ, (∀ i, |q i| ≤ 1) → (∀ i, |q' i| ≤ 1) →
          |Salt.energy N kappa q - Salt.energy N kappa q'| < eps) := by
  refine ⟨energy_eq_sum_autocorr N, ?_, ?_, ?_, ?_⟩
  · intro S hS q q' h
    exact ⟨autocorr_eq_of_energy_eq_on_infinite hS h, titration_determines_every_kernel hS h⟩
  · intro kappa hk
    obtain ⟨_, _, he, hne⟩ := single_condition_degenerate kappa hk
    exact ⟨he, hne⟩
  · intro k hk kap
    exact finite_titration_underdetermined hk kap
  · intro eps kappa heps hk0 hkap q q' hq hq'
    exact high_salt_uninformative heps hN hk0 hkap hq hq'

end Titration
end IDR
