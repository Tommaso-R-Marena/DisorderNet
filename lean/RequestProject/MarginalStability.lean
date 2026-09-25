/-
# Marginal stability: folding at 20–30% native occupancy

The model of `RequestProject.SequenceEntropyLimit` calls a region *ordered* when a single
conformation carries at least **half** of the equilibrium population.  Real folded, functional
proteins routinely violate that: a native state can hold 20–30% of the ensemble and the rest of
the population sits in alternative (often functionally relevant, "inactive") conformations, the
whole thing shifting with ligand, phosphorylation or partner binding.  A criterion phrased at a
majority threshold therefore calls such a protein disordered, which is wrong.

This file replaces the majority threshold by an *occupancy* threshold `theta` and shows that
every structural theorem of the model survives with an explicit `theta`-dependent constant.

* `IDR.Marginal.occupancy_gap` — the thermodynamic gate at occupancy `theta`: a conformation
  holding a fraction `theta` of the population against `|D|` competitors must beat them by
  `kT·log (theta·|D|/(1-theta))`.  At `theta = 1/2` this is the old `kT·log|D|`; at `theta = 1/4`
  it is `kT·log(|D|/3)` — weaker by `kT·log 3 ≈ 1.1 kT`, and *not* vacuous.
* `IDR.Marginal.Functional` — the occupancy-`theta` notion of a folded region, antitone in
  `theta` (`functional_antitone`) and equal to the old `Ordered` at `theta = 1/2`
  (`functional_half_iff_ordered`).
* `IDR.Marginal.functional_needs_spread`, `functional_needs_minority`,
  `disordered_of_low_composition_entropy'` — the gate, the heterogeneity requirement and the
  **compositional entropy floor**, all re-proved at occupancy `theta`.  The floor is the load
  bearing statement of the project and it does not depend on the majority convention.
* `IDR.Marginal.flat_four_functional_not_ordered` — the criterion really is different: on a flat
  four-state landscape every conformation holds exactly `1/4`, so the region is functional at
  occupancy `1/4` and not ordered.
* Two-state section (`occ`, `occ_odds`, `occ_shift`, `shift_to_majority`,
  `occupancy_of_free_energy`) — the active/inactive equilibrium.  Occupancy `theta` means a free
  energy `kT·log((1-theta)/theta)` (for `theta = 1/4`, `1.1 kT`); a perturbation stabilising the
  native state by `ddG` multiplies the odds by `exp(beta·ddG)` exactly, so `kT·log((1-theta)/theta)`
  of extra stabilisation — one hydrogen bond — converts a 25% ensemble into a majority one.
  This is why marginal stability is a *feature*: the ensemble is switchable at physiological
  energy scales.
* `IDR.Marginal.single_structure_error_ge` — the price of the same fact for prediction: a
  single-structure answer is wrong with probability at least `1 - theta_max`, so at 25% native
  occupancy it is wrong three times out of four.
-/
import Mathlib
import RequestProject.FreeEnergy
import RequestProject.SequenceEntropyCore
import RequestProject.SequenceEntropyLimit

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR

namespace Marginal

open Finset
open scoped Classical

/-! ## The thermodynamic gate at an arbitrary occupancy -/

variable {n : ℕ}

/-- **The gate at occupancy `theta`.**  If conformation `j0` carries at least a fraction `theta`
of the Boltzmann population, and `D` is a set of competitors all of energy at most `Uu`, then
`j0` must lie at least `kT·log (theta·|D|/(1-theta))` below `Uu`.

At `theta = 1/2` the constant is `kT·log |D|`, the classical entropic gap; at `theta = 1/4` it is
`kT·log (|D|/3)`.  Marginal stability buys the sequence exactly `kT·log ((1-theta)/theta)` of
slack — a little over one `kT` at 25% occupancy — and nothing more. -/
theorem occupancy_gap {beta theta : ℝ} (hbeta : 0 < beta) (hn : 0 < n)
    (U : Fin n → ℝ) (j0 : Fin n) (D : Finset (Fin n)) (hD : D.Nonempty) (hj0 : j0 ∉ D)
    (Uu : ℝ) (hU : ∀ j ∈ D, U j ≤ Uu) (h0 : 0 < theta) (h1 : theta < 1)
    (hocc : theta ≤ FreeEnergy.boltz beta U j0) :
    Real.log (theta * D.card / (1 - theta)) / beta ≤ Uu - U j0 := by
  have hZ : 0 < FreeEnergy.part beta U := FreeEnergy.part_pos hn beta U
  set a : ℝ := Real.exp (-beta * U j0) with ha
  have hapos : 0 < a := Real.exp_pos _
  -- occupancy hypothesis in product form
  have hthZ : theta * FreeEnergy.part beta U ≤ a := by
    have hocc' : theta ≤ a / FreeEnergy.part beta U := hocc
    exact (le_div_iff₀ hZ).1 hocc'
  -- the partition function dominates the native weight plus the competitors
  have hsplit : a + ∑ j ∈ D, Real.exp (-beta * U j) ≤ FreeEnergy.part beta U := by
    have hsum : ∑ j ∈ insert j0 D, Real.exp (-beta * U j) ≤ FreeEnergy.part beta U :=
      Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
        (fun j _ _ => le_of_lt (Real.exp_pos _))
    rwa [Finset.sum_insert hj0] at hsum
  have hDsum : (D.card : ℝ) * Real.exp (-beta * Uu) ≤ ∑ j ∈ D, Real.exp (-beta * U j) := by
    have hle : ∀ j ∈ D, Real.exp (-beta * Uu) ≤ Real.exp (-beta * U j) := by
      intro j hj
      exact Real.exp_le_exp.2 (by nlinarith [hU j hj])
    calc (D.card : ℝ) * Real.exp (-beta * Uu) = ∑ _j ∈ D, Real.exp (-beta * Uu) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ j ∈ D, Real.exp (-beta * U j) := Finset.sum_le_sum hle
  have hcardpos : (0 : ℝ) < D.card := by
    exact_mod_cast Finset.card_pos.2 hD
  -- the key inequality: theta·|D|·e^{-beta Uu} ≤ (1-theta)·a
  have hkey : theta * (D.card : ℝ) * Real.exp (-beta * Uu) ≤ (1 - theta) * a := by
    have h2 : theta * (a + (D.card : ℝ) * Real.exp (-beta * Uu)) ≤ a := by
      have : theta * (a + (D.card : ℝ) * Real.exp (-beta * Uu))
          ≤ theta * FreeEnergy.part beta U := by
        apply mul_le_mul_of_nonneg_left _ h0.le
        linarith
      linarith
    nlinarith
  have hden : 0 < 1 - theta := by linarith
  have hratio : theta * (D.card : ℝ) / (1 - theta) ≤ a * Real.exp (beta * Uu) := by
    rw [div_le_iff₀ hden]
    have hexp : Real.exp (-beta * Uu) * Real.exp (beta * Uu) = 1 := by
      rw [← Real.exp_add, show -beta * Uu + beta * Uu = 0 by ring, Real.exp_zero]
    have hmul := mul_le_mul_of_nonneg_right hkey (Real.exp_pos (beta * Uu)).le
    calc theta * (D.card : ℝ)
        = theta * (D.card : ℝ) * (Real.exp (-beta * Uu) * Real.exp (beta * Uu)) := by
          rw [hexp]; ring
      _ ≤ (1 - theta) * a * Real.exp (beta * Uu) := by rw [← mul_assoc]; exact hmul
      _ = a * Real.exp (beta * Uu) * (1 - theta) := by ring
  have hlog := Real.log_le_log (by positivity) hratio
  rw [Real.log_mul (ne_of_gt hapos) (ne_of_gt (Real.exp_pos _)), ha, Real.log_exp, Real.log_exp]
    at hlog
  rw [div_le_iff₀ hbeta]
  nlinarith [hlog]

/-! ## Functional order at occupancy `theta` -/

open SeqLimit

variable {N q M : ℕ}

/-- **Functional order at occupancy `theta`**: some conformation carries at least a fraction
`theta` of the equilibrium population.  `Functional (1/2)` is the majority criterion `Ordered`;
realistic native states are captured by `theta` around `1/5` to `3/10`. -/
def Functional (theta beta : ℝ) (E : Seq N q → Fin M → ℝ) (s : Seq N q) : Prop :=
  ∃ j, theta ≤ FreeEnergy.boltz beta (E s) j

theorem functional_half_iff_ordered {beta : ℝ} {E : Seq N q → Fin M → ℝ} {s : Seq N q} :
    Functional (1/2) beta E s ↔ Ordered beta E s := Iff.rfl

/-- Lowering the occupancy threshold can only make more sequences count as folded. -/
theorem functional_antitone {theta theta' beta : ℝ} (h : theta' ≤ theta)
    {E : Seq N q → Fin M → ℝ} {s : Seq N q} (hf : Functional theta beta E s) :
    Functional theta' beta E s := by
  obtain ⟨j, hj⟩ := hf
  exact ⟨j, le_trans h hj⟩

/-- The gate constant of the occupancy-`theta` model, in energy units. -/
noncomputable def gate (theta beta : ℝ) (M : ℕ) : ℝ :=
  Real.log (theta * ((M : ℝ) - 1) / (1 - theta)) / beta

/-- At the majority threshold the gate constant is the classical `kT·log (M-1)`. -/
theorem gate_half (beta : ℝ) (M : ℕ) :
    gate (1/2) beta M = Real.log ((M : ℝ) - 1) / beta := by
  unfold gate
  norm_num

/-- The reduced gate is not vacuous: with thirteen conformations, a native state at 25%
occupancy and `kT = 1` still has to beat its competitors by `log 4 ≈ 1.4 kT`. -/
theorem gate_quarter_thirteen : gate (1/4) 1 13 = Real.log 4 := by
  unfold gate
  norm_num

theorem gate_quarter_thirteen_pos : 0 < gate (1/4) 1 13 := by
  rw [gate_quarter_thirteen]
  exact Real.log_pos (by norm_num)

/-- **The gate, at occupancy `theta`.**  A sequence whose native state holds a fraction `theta`
of the population still has to generate an energy spread of at least `gate theta beta M`. -/
theorem functional_needs_spread {beta theta : ℝ} (hbeta : 0 < beta) (hM : 1 < M)
    (h0 : 0 < theta) (h1 : theta < 1)
    {E : Seq N q → Fin M → ℝ} {s : Seq N q} {Emax Emin : ℝ} (hmax : ∀ j, E s j ≤ Emax)
    (hmin : ∀ j, Emin ≤ E s j) (h : Functional theta beta E s) :
    gate theta beta M ≤ Emax - Emin := by
  obtain ⟨j0, hj0⟩ := h
  set D : Finset (Fin M) := Finset.univ.erase j0 with hD
  have hcard : D.card = M - 1 := by
    rw [hD, Finset.card_erase_of_mem (Finset.mem_univ _)]
    simp
  have hDne : D.Nonempty := by
    rw [← Finset.card_pos, hcard]; omega
  have hj0D : j0 ∉ D := by simp [hD]
  have hgap := occupancy_gap hbeta (by omega : 0 < M) (E s) j0 D hDne hj0D Emax
    (fun j _ => hmax j) h0 h1 hj0
  rw [hcard] at hgap
  have hcast : ((M - 1 : ℕ) : ℝ) = (M : ℝ) - 1 := by
    have : (1 : ℕ) ≤ M := by omega
    simpa using (Nat.cast_sub this : ((M - 1 : ℕ) : ℝ) = (M : ℝ) - (1 : ℕ))
  rw [hcast] at hgap
  have := hmin j0
  unfold gate
  linarith

/-- **The verdict form at occupancy `theta`**: too little contrast, and the region cannot hold
even a fraction `theta` of its population in one conformation. -/
theorem not_functional_of_small_spread {beta theta : ℝ} (hbeta : 0 < beta) (hM : 1 < M)
    (h0 : 0 < theta) (h1 : theta < 1)
    {E : Seq N q → Fin M → ℝ} {s : Seq N q} {Emax Emin : ℝ} (hmax : ∀ j, E s j ≤ Emax)
    (hmin : ∀ j, Emin ≤ E s j) (hlt : Emax - Emin < gate theta beta M) :
    ¬ Functional theta beta E s := fun h =>
  absurd (functional_needs_spread hbeta hM h0 h1 hmax hmin h) (not_le.2 hlt)

/-- **Heterogeneity requirement at occupancy `theta`.** -/
theorem functional_needs_minority {beta theta L S0 : ℝ} (hbeta : 0 < beta) (hM : 1 < M)
    (h0 : 0 < theta) (h1 : theta < 1)
    {E : Seq N q → Fin M → ℝ} (hlip : SiteLip L E) (c : Fin q)
    (hflat : ∀ j j' : Fin M, E (fun _ => c) j - E (fun _ => c) j' ≤ S0)
    {s : Seq N q} (h : Functional theta beta E s) :
    gate theta beta M ≤ S0 + 2 * L * (minorityCount s c : ℝ) := by
  have hne : (Finset.univ : Finset (Fin M)).Nonempty := ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  obtain ⟨jmax, -, hjmax⟩ := Finset.exists_max_image Finset.univ (E s) hne
  obtain ⟨jmin, -, hjmin⟩ := Finset.exists_min_image Finset.univ (E s) hne
  have hspread := functional_needs_spread hbeta hM h0 h1 (fun j => hjmax j (Finset.mem_univ j))
    (fun j => hjmin j (Finset.mem_univ j)) h
  have hd : hdist s (fun _ => c) = minorityCount s c := rfl
  have hlipmax : |E s jmax - E (fun _ => c) jmax| ≤ L * (minorityCount s c : ℝ) := by
    have := siteLip_hdist hlip (hdist s (fun _ => c)) s (fun _ => c) rfl jmax
    rwa [hd] at this
  have hlipmin : |E s jmin - E (fun _ => c) jmin| ≤ L * (minorityCount s c : ℝ) := by
    have := siteLip_hdist hlip (hdist s (fun _ => c)) s (fun _ => c) rfl jmin
    rwa [hd] at this
  have hA : E s jmax - E (fun _ => c) jmax ≤ L * (minorityCount s c : ℝ) := (abs_le.1 hlipmax).2
  have hB : -(L * (minorityCount s c : ℝ)) ≤ E s jmin - E (fun _ => c) jmin := (abs_le.1 hlipmin).1
  have hC : E (fun _ => c) jmax - E (fun _ => c) jmin ≤ S0 := hflat jmax jmin
  linarith

/-- **The compositional entropy floor at occupancy `theta`.**  A sequence whose single-residue
entropy is below `kappa·log 2` cannot hold even a fraction `theta` of its population in one
conformation, once `kappa·N` mutations of contrast cannot pay the (reduced) gate.

Marginal stability does not repeal the floor; it only lowers the gate constant by
`kT·log ((1-theta)/theta)`. -/
theorem disordered_of_low_composition_entropy' {beta theta L S0 kappa : ℝ} (hbeta : 0 < beta)
    (hM : 1 < M) (hN : 0 < N) (h0 : 0 < theta) (h1 : theta < 1)
    {E : Seq N q → Fin M → ℝ} (hlip : SiteLip L E)
    (hflat : ∀ (c : Fin q) (j j' : Fin M), E (fun _ => c) j - E (fun _ => c) j' ≤ S0)
    (hk : kappa ≤ 1 / 2)
    (hthr : S0 + 2 * L * (kappa * N) < gate theta beta M)
    {s : Seq N q} (hlow : SeqEnt.H (comp s) < kappa * Real.log 2) :
    ¬ Functional theta beta E s := by
  intro hord
  have hMpos : 0 < M := by omega
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hL : 0 ≤ L := by
    have h0' := hlip s ⟨0, hN⟩ (s ⟨0, hN⟩) ⟨0, hMpos⟩
    rw [Function.update_eq_self] at h0'
    simpa using h0'
  have hnn := comp_nonneg s
  have hsum := comp_sum_one hN s
  have hqpos : 0 < q := Fin.pos_iff_nonempty.2 ⟨s ⟨0, hN⟩⟩
  obtain ⟨c, -, hc⟩ := Finset.exists_max_image Finset.univ (comp s) ⟨⟨0, hqpos⟩,
    Finset.mem_univ _⟩
  set del : ℝ := ∑ a ∈ Finset.univ.erase c, comp s a with hdel
  have hdelval : del = (minorityCount s c : ℝ) / N := by rw [hdel, sum_comp_erase hN s c]
  have hdel0 : 0 ≤ del := by rw [hdelval]; positivity
  have hmin_entropy : -Real.log (comp s c) ≤ SeqEnt.H (comp s) :=
    SeqEnt.H_ge_neg_log_max hnn hsum (fun a => hc a (Finset.mem_univ a))
  have hlog2 : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  have hHlt : SeqEnt.H (comp s) < Real.log 2 := by nlinarith
  have hcpos : 1 / 2 < comp s c := by
    by_contra hcon
    push_neg at hcon
    have hcnn : 0 ≤ comp s c := hnn c
    rcases eq_or_lt_of_le hcnn with h0' | hpos
    · have hall : ∀ a, comp s a ≤ 0 := by
        intro a
        have hle := hc a (Finset.mem_univ a)
        rw [← h0'] at hle
        exact hle
      have hnp : ∑ a, comp s a ≤ 0 := Finset.sum_nonpos fun a _ => hall a
      linarith
    · have hlt : Real.log (comp s c) ≤ Real.log (1/2) := Real.log_le_log hpos hcon
      have h12 : Real.log (1/2) = -Real.log 2 := by rw [one_div, Real.log_inv]
      rw [h12] at hlt
      linarith
  have hdelhalf : del < 1 / 2 := by
    have hsplit : comp s c + del = 1 := by
      rw [hdel, Finset.add_sum_erase _ _ (Finset.mem_univ c)]
      exact hsum
    linarith
  have hminor : del * Real.log 2 ≤ SeqEnt.H (comp s) :=
    SeqEnt.H_ge_minority hnn hsum hdel hdelhalf.le
  have hdelk : del < kappa := by
    have := lt_of_le_of_lt hminor hlow
    exact lt_of_mul_lt_mul_right (by linarith) hlog2.le
  have hcount : (minorityCount s c : ℝ) < kappa * N := by
    rw [hdelval] at hdelk
    calc (minorityCount s c : ℝ) = ((minorityCount s c : ℝ) / N) * N := by field_simp
      _ < kappa * N := by exact (mul_lt_mul_of_pos_right hdelk hN0)
  have hgap := functional_needs_minority hbeta hM h0 h1 hlip c (hflat c) hord
  nlinarith [hgap, hcount, hthr, hL]

/-! ## The two criteria really differ -/

/-- **A flat four-state landscape.**  Every conformation holds exactly a quarter of the
population: the region is functional at occupancy `1/4` and is *not* ordered in the majority
sense.  A model that reports a Boolean "folded?" at the majority threshold therefore misses
exactly the regime — native state at 20–30% — in which many real proteins work. -/
theorem flat_four_functional_not_ordered {N q : ℕ} (beta : ℝ) (c : ℝ) (s : Seq N q) :
    Functional (1/4) beta (fun (_ : Seq N q) (_ : Fin 4) => c) s ∧
      ¬ Ordered beta (fun (_ : Seq N q) (_ : Fin 4) => c) s := by
  have hflat : FreeEnergy.boltz beta (fun _ : Fin 4 => c) = fun _ => 1 / ((4 : ℕ) : ℝ) :=
    FreeEnergy.flat_landscape_uniform (by norm_num) beta c
  constructor
  · refine ⟨⟨0, by norm_num⟩, ?_⟩
    rw [show (fun (_ : Seq N q) (_ : Fin 4) => c) s = (fun _ : Fin 4 => c) from rfl, hflat]
    norm_num
  · rintro ⟨j, hj⟩
    rw [show (fun (_ : Seq N q) (_ : Fin 4) => c) s = (fun _ : Fin 4 => c) from rfl, hflat] at hj
    norm_num at hj

/-! ## The active/inactive two-state equilibrium -/

/-- The occupancy of a state of statistical weight `a` against a rest-of-ensemble weight `R`. -/
noncomputable def occ (a R : ℝ) : ℝ := a / (a + R)

lemma occ_nonneg {a R : ℝ} (ha : 0 < a) (hR : 0 < R) : 0 ≤ occ a R := by
  unfold occ; positivity

lemma occ_lt_one {a R : ℝ} (ha : 0 < a) (hR : 0 < R) : occ a R < 1 := by
  unfold occ
  rw [div_lt_one (by linarith)]
  linarith

lemma one_sub_occ {a R : ℝ} (ha : 0 < a) (hR : 0 < R) : 1 - occ a R = R / (a + R) := by
  have h : a + R ≠ 0 := by positivity
  unfold occ
  field_simp
  ring

/-- **Occupancy and odds.**  The odds of the native state are the ratio of statistical weights. -/
theorem occ_odds {a R : ℝ} (ha : 0 < a) (hR : 0 < R) :
    occ a R / (1 - occ a R) = a / R := by
  have h : a + R ≠ 0 := by positivity
  have h2 : R ≠ 0 := ne_of_gt hR
  rw [one_sub_occ ha hR]
  unfold occ
  field_simp

/-- **Occupancy is a free energy.**  A native state at occupancy `theta` sits
`kT·log ((1-theta)/theta)` above the rest of the ensemble in free energy: `1.1 kT` at 25%,
`0.85 kT` at 30%, `0` at 50%.  Marginal stability is a small number of `kT`, not a large one. -/
theorem occupancy_of_free_energy {a R theta : ℝ} (ha : 0 < a) (hR : 0 < R)
    (h1 : theta < 1) :
    occ a R = theta ↔ a / R = theta / (1 - theta) := by
  constructor
  · intro h
    rw [← occ_odds ha hR, h]
  · intro h
    have hodds : occ a R / (1 - occ a R) = theta / (1 - theta) := by rw [occ_odds ha hR, h]
    have hone : 1 - occ a R = R / (a + R) := one_sub_occ ha hR
    have hpos : 0 < 1 - occ a R := by rw [hone]; positivity
    have h1t : 0 < 1 - theta := by linarith
    field_simp at hodds
    linarith

/-- **The population shift of a perturbation.**  Stabilising the native state by `ddG`
(a ligand, a phosphate, a partner) multiplies its odds by `exp (beta·ddG)` — exactly. -/
theorem occ_shift {a R beta ddG : ℝ} (ha : 0 < a) (hR : 0 < R) :
    occ (a * Real.exp (beta * ddG)) R / (1 - occ (a * Real.exp (beta * ddG)) R)
      = Real.exp (beta * ddG) * (occ a R / (1 - occ a R)) := by
  have hpos : 0 < a * Real.exp (beta * ddG) := by positivity
  rw [occ_odds hpos hR, occ_odds ha hR]
  field_simp

/-- Occupancy is monotone in the weight of the state. -/
theorem occ_mono {a a' R : ℝ} (ha : 0 < a) (hR : 0 < R) (h : a ≤ a') :
    occ a R ≤ occ a' R := by
  have ha' : 0 < a' := lt_of_lt_of_le ha h
  unfold occ
  rw [div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith

/-- **A conformational switch costs about one kT.**  A native state at occupancy `theta` becomes
the majority species as soon as it is stabilised by `kT·log ((1-theta)/theta)`: `1.1 kT` for a
25% ensemble, i.e. a single hydrogen bond or a modest binding event.  This is the quantitative
content of "functional proteins live in a shiftable dynamic equilibrium". -/
theorem shift_to_majority {a R beta theta ddG : ℝ} (ha : 0 < a) (hR : 0 < R) (hbeta : 0 < beta)
    (h0 : 0 < theta) (h1 : theta < 1) (hocc : theta ≤ occ a R)
    (hddG : Real.log ((1 - theta) / theta) / beta ≤ ddG) :
    1 / 2 ≤ occ (a * Real.exp (beta * ddG)) R := by
  have h1t : 0 < 1 - theta := by linarith
  -- the hypothesis on the occupancy is an inequality between weights
  have hocc' : theta * (a + R) ≤ a := by
    have : theta ≤ a / (a + R) := hocc
    exact (le_div_iff₀ (by linarith)).1 this
  have hweights : theta * R ≤ (1 - theta) * a := by nlinarith
  -- the perturbation is large enough
  have hexp : (1 - theta) / theta ≤ Real.exp (beta * ddG) := by
    have hle : Real.log ((1 - theta) / theta) ≤ beta * ddG := by
      rw [div_le_iff₀ hbeta] at hddG
      linarith
    calc (1 - theta) / theta = Real.exp (Real.log ((1 - theta) / theta)) := by
          rw [Real.exp_log (by positivity)]
      _ ≤ Real.exp (beta * ddG) := Real.exp_le_exp.2 hle
  -- combine: a·e^{beta ddG} ≥ R
  have hkey : R ≤ a * Real.exp (beta * ddG) := by
    have h2 : (1 - theta) / theta * a ≤ Real.exp (beta * ddG) * a :=
      mul_le_mul_of_nonneg_right hexp ha.le
    have h3 : R ≤ (1 - theta) / theta * a := by
      have hrw : (1 - theta) / theta * a = ((1 - theta) * a) / theta := by ring
      rw [hrw, le_div_iff₀ h0]
      linarith [hweights]
    calc R ≤ (1 - theta) / theta * a := h3
      _ ≤ Real.exp (beta * ddG) * a := h2
      _ = a * Real.exp (beta * ddG) := by ring
  have hpos : 0 < a * Real.exp (beta * ddG) := by positivity
  unfold occ
  rw [le_div_iff₀ (by linarith)]
  linarith

/-! ## What marginal stability costs a single-structure predictor -/

/-- **A single structure is mostly wrong.**  If no conformation exceeds occupancy `thetaMax`,
then any single-conformation answer has probability at least `1 - thetaMax` of being wrong: at a
native occupancy of 25%, three quarters of the ensemble is missed. -/
theorem single_structure_error_ge {beta thetaMax : ℝ} (hn : 0 < n) (U : Fin n → ℝ)
    (hmax : ∀ j, FreeEnergy.boltz beta U j ≤ thetaMax) (j0 : Fin n) :
    1 - thetaMax ≤ ∑ j ∈ Finset.univ.erase j0, FreeEnergy.boltz beta U j := by
  have hsum : ∑ j, FreeEnergy.boltz beta U j = 1 := FreeEnergy.boltz_sum_one hn beta U
  have hsplit : FreeEnergy.boltz beta U j0 + ∑ j ∈ Finset.univ.erase j0,
      FreeEnergy.boltz beta U j = 1 := by
    rw [Finset.add_sum_erase _ _ (Finset.mem_univ j0)]
    exact hsum
  have := hmax j0
  linarith

end Marginal

end IDR
