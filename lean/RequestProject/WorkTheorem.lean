/-
# Part LXIII.1  Pulling experiments out of equilibrium: work, dissipation and the estimator

Part XV.2 (`Force.lean`) treats force spectroscopy at equilibrium: the ensemble at each force is
the tilted Boltzmann one, and the force--extension curve is a functional of the law of the
extension coordinate.  A real optical-tweezer or AFM experiment on a disordered region is not
performed at equilibrium: the trap is moved at a finite speed, the region is dragged along, and
what is recorded is the *work* done in each pull.  The free energy is then extracted from the
distribution of that work.  This file formalises exactly what that extraction is worth.

Setting.  A finite set of trajectories `G`, a forward protocol with probabilities `p`, the
time-reversed protocol with probabilities `q`, an involutive time reversal `rev : G ≃ G` sending
each trajectory to its reverse, the work `W` (odd under reversal), an inverse temperature `b`,
and a free-energy difference `dF`.  *Microscopic reversibility* -- Crooks' relation -- is the
single physical hypothesis:

  `p γ = q (rev γ) · exp (b · (W γ − dF))`   (`Crooks`).

Everything below is a consequence of it.

* `jarzynski` -- **the exponential average is exact**: `⟨exp(−b·W)⟩ = exp(−b·dF)`, for every
  protocol, however fast and however dissipative.  A nonequilibrium experiment does determine an
  equilibrium free energy.
* `dissipation_eq_relEntropy` -- **and the dissipated work is a relative entropy**:
  `b·(⟨W⟩ − dF) = Σ p·log(p / q∘rev)`, exactly.  The irreversibility of the pull is the
  statistical distinguishability of the forward pull from its own time reverse.
* `second_law` -- hence `⟨W⟩ ≥ dF` (Gibbs' inequality), and `no_dissipation_iff_deterministic`
  -- the bound is attained exactly when the work is the same on every trajectory, which by
  Crooks is exactly when the forward and reversed processes coincide.  Any spread in the
  measured work is dissipation, and a mean work is an upper bound on the free energy, never an
  estimate of it.
* `crooks_histogram`, `crooks_crossing` -- the forward and reverse work histograms satisfy
  `P_F(w) = exp(b·(w − dF))·P_R(−w)`, so **they cross exactly at `dF`**: the free energy is read
  off the crossing point of two measured histograms, with no model in between.
* `rare_trajectories_dominate` -- **but the exponential average is carried by trajectories that
  are not sampled.**  For every `M > 0` there is a two-branch work distribution in which the
  rare branch has probability at most `exp(−M)` and yet contributes more than the whole of the
  rest to `⟨exp(−b·W)⟩`, and in which the free energy is `−M` or below while the estimate
  obtained from the typical branch alone is `0`.  The error of a finite-sample Jarzynski
  estimate is unbounded, and it is unbounded in a specific direction: the estimate is too high,
  drifting from `dF` towards `⟨W⟩`.
* `omitting_low_work_overestimates` -- and that direction is a theorem, not an accident of the
  example: discarding any set of trajectories whose work is below all the retained ones
  increases the estimated free energy.

For a disordered region the practical consequences are sharp, because there is no folded state
to hold the work distribution narrow: a pull on a disordered chain has a broad work
distribution, so `⟨W⟩` overestimates `dF` by an amount equal to a relative entropy, and the
exponential average that would correct it is dominated by the rare pulls in which the chain
happened to be pre-extended.  What is reliable is the crossing point of the forward and reverse
histograms; what is not is a one-directional mean.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace Work

open Finset

variable {G : Type} [Fintype G]

/-- Microscopic reversibility (Crooks' relation) for a protocol, its time reverse, the work
function and the free-energy difference. -/
def Crooks (rev : Equiv.Perm G) (p q : G → ℝ) (W : G → ℝ) (b dF : ℝ) : Prop :=
  ∀ g : G, p g = q (rev g) * Real.exp (b * (W g - dF))

/-- The mean work of the forward protocol. -/
def meanWork (p W : G → ℝ) : ℝ := ∑ g, p g * W g

/-- The forward work histogram: the probability of the work value `w`. -/
noncomputable def hist (p W : G → ℝ) (w : ℝ) : ℝ :=
  ∑ g, if W g = w then p g else 0

/-! ## The hypothesis is satisfiable -/

/-- **Crooks' relation is not vacuous.**  For every reverse protocol `q`, every work function
`W` and every temperature there is a forward protocol and a free-energy difference satisfying
it, and the free energy is then the logarithm of the exponential average -- the Jarzynski
formula read as a definition. -/
theorem crooks_satisfiable [Nonempty G] (rev : Equiv.Perm G) (q W : G → ℝ) {b : ℝ} (hb : b ≠ 0)
    (hq : ∀ g, 0 < q g) :
    ∃ (p : G → ℝ) (dF : ℝ), (∀ g, 0 < p g) ∧ (∑ g, p g = 1) ∧ Crooks rev p q W b dF := by
  set Z : ℝ := ∑ g, q (rev g) * Real.exp (b * W g) with hZ
  have hZpos : 0 < Z := by
    refine Finset.sum_pos (fun g _ => mul_pos (hq (rev g)) (Real.exp_pos _)) ?_
    exact Finset.univ_nonempty
  refine ⟨fun g => q (rev g) * Real.exp (b * (W g - (1 / b) * Real.log Z)), (1 / b) * Real.log Z,
    fun g => mul_pos (hq (rev g)) (Real.exp_pos _), ?_, fun g => rfl⟩
  have hexp : Real.exp (b * ((1 / b) * Real.log Z)) = Z := by
    have h : b * ((1 / b) * Real.log Z) = Real.log Z := by field_simp
    rw [h, Real.exp_log hZpos]
  have hterm : ∀ g : G, q (rev g) * Real.exp (b * (W g - (1 / b) * Real.log Z))
      = (1 / Z) * (q (rev g) * Real.exp (b * W g)) := by
    intro g
    rw [mul_sub, Real.exp_sub, hexp]
    field_simp
  rw [Finset.sum_congr rfl (fun g _ => hterm g), ← Finset.mul_sum, ← hZ]
  field_simp

/-! ## The Jarzynski equality -/

/-- **The exponential average of the work is exact.**  Whatever the protocol, however fast,
`⟨exp(−b·W)⟩ = exp(−b·dF)`. -/
theorem jarzynski {rev : Equiv.Perm G} {p q W : G → ℝ} {b dF : ℝ}
    (hC : Crooks rev p q W b dF) (hq : ∑ g, q g = 1) :
    ∑ g, p g * Real.exp (-b * W g) = Real.exp (-b * dF) := by
  have hterm : ∀ g : G, p g * Real.exp (-b * W g) = Real.exp (-b * dF) * q (rev g) := by
    intro g
    rw [hC g, mul_assoc, ← Real.exp_add]
    have : b * (W g - dF) + -b * W g = -b * dF := by ring
    rw [this]
    ring
  rw [Finset.sum_congr rfl (fun g _ => hterm g), ← Finset.mul_sum]
  rw [Equiv.sum_comp rev q, hq, mul_one]

/-! ## Dissipation is a relative entropy -/

/-- **The dissipated work is exactly the relative entropy between the forward process and the
time reverse of the backward process.** -/
theorem dissipation_eq_relEntropy {rev : Equiv.Perm G} {p q W : G → ℝ} {b dF : ℝ}
    (hC : Crooks rev p q W b dF) (hq : ∀ g, 0 < q g) (hsum : ∑ g, p g = 1) :
    b * (meanWork p W - dF) = ∑ g, p g * Real.log (p g / q (rev g)) := by
  have hlog : ∀ g : G, Real.log (p g / q (rev g)) = b * (W g - dF) := by
    intro g
    rw [hC g, mul_comm, mul_div_assoc, div_self (ne_of_gt (hq (rev g))), mul_one, Real.log_exp]
  rw [Finset.sum_congr rfl (fun g _ => by rw [hlog g])]
  have expand : ∀ g : G, p g * (b * (W g - dF)) = b * (p g * W g) - (b * dF) * p g := by
    intro g; ring
  rw [Finset.sum_congr rfl (fun g _ => expand g), Finset.sum_sub_distrib, ← Finset.mul_sum,
    ← Finset.mul_sum, hsum, meanWork]
  ring

/-- Gibbs' inequality for two strictly positive weight vectors summing to one. -/
theorem relEntropy_nonneg {p r : G → ℝ} (hp : ∀ g, 0 < p g) (hr : ∀ g, 0 < r g)
    (hps : ∑ g, p g = 1) (hrs : ∑ g, r g = 1) : 0 ≤ ∑ g, p g * Real.log (p g / r g) := by
  have key : ∀ g : G, p g - r g ≤ p g * Real.log (p g / r g) := by
    intro g
    have hpg := hp g
    have hrg := hr g
    have h1 : Real.log (r g / p g) ≤ r g / p g - 1 :=
      Real.log_le_sub_one_of_pos (div_pos hrg hpg)
    have h2 : Real.log (p g / r g) = - Real.log (r g / p g) := by
      rw [← Real.log_inv]
      congr 1
      field_simp
    have h3 : p g * Real.log (r g / p g) ≤ p g * (r g / p g - 1) :=
      mul_le_mul_of_nonneg_left h1 hpg.le
    have h4 : p g * (r g / p g - 1) = r g - p g := by field_simp
    rw [h2]
    nlinarith [h3, h4]
  have hsum : ∑ g, (p g - r g) ≤ ∑ g, p g * Real.log (p g / r g) :=
    Finset.sum_le_sum fun g _ => key g
  have hzero : ∑ g, (p g - r g) = 0 := by
    rw [Finset.sum_sub_distrib, hps, hrs, sub_self]
  linarith [hsum, hzero.symm.le]

/-- **The second law**: the mean work is at least the free-energy difference. -/
theorem second_law {rev : Equiv.Perm G} {p q W : G → ℝ} {b dF : ℝ} (hb : 0 < b)
    (hC : Crooks rev p q W b dF) (hp : ∀ g, 0 < p g) (hq : ∀ g, 0 < q g)
    (hps : ∑ g, p g = 1) (hqs : ∑ g, q g = 1) : dF ≤ meanWork p W := by
  have hr : ∀ g, 0 < q (rev g) := fun g => hq (rev g)
  have hrs : ∑ g, q (rev g) = 1 := by rw [Equiv.sum_comp rev q, hqs]
  have hnn : 0 ≤ ∑ g, p g * Real.log (p g / q (rev g)) := relEntropy_nonneg hp hr hps hrs
  have heq : b * (meanWork p W - dF) = ∑ g, p g * Real.log (p g / q (rev g)) :=
    dissipation_eq_relEntropy hC hq hps
  nlinarith [heq, hnn, hb]

omit [Fintype G] in
/-- **No dissipation exactly when the work is deterministic.** -/
theorem no_dissipation_iff_deterministic {rev : Equiv.Perm G} {p q W : G → ℝ} {b dF : ℝ}
    (hb : b ≠ 0) (hC : Crooks rev p q W b dF) (hq : ∀ g, 0 < q g) :
    (∀ g, p g = q (rev g)) ↔ (∀ g, W g = dF) := by
  constructor
  · intro h g
    have hqg := hq (rev g)
    have hcg := hC g
    rw [h g] at hcg
    have hexp : Real.exp (b * (W g - dF)) = 1 := by
      have h0 : q (rev g) * Real.exp (b * (W g - dF)) = q (rev g) * 1 := by
        rw [mul_one]; exact hcg.symm
      exact mul_left_cancel₀ (ne_of_gt hqg) h0
    have hz : b * (W g - dF) = 0 := (Real.exp_eq_one_iff _).mp hexp
    rcases mul_eq_zero.mp hz with h1 | h1
    · exact absurd h1 hb
    · linarith [sub_eq_zero.mp h1]
  · intro h g
    rw [hC g, h g, sub_self, mul_zero, Real.exp_zero, mul_one]

/-! ## The histograms cross at the free energy -/

/-- **Crooks' theorem in histogram form.** -/
theorem crooks_histogram {rev : Equiv.Perm G} {p q W : G → ℝ} {b dF : ℝ}
    (hC : Crooks rev p q W b dF) (hW : ∀ g, W (rev g) = - W g) (w : ℝ) :
    hist p W w = Real.exp (b * (w - dF)) * hist q W (-w) := by
  have hR : hist q W (-w) = ∑ g, if W g = w then q (rev g) else 0 := by
    have := Equiv.sum_comp rev (fun g => if W g = -w then q g else 0)
    rw [hist, ← this]
    refine Finset.sum_congr rfl fun g _ => ?_
    rw [hW g]
    by_cases hg : W g = w
    · simp [hg]
    · have : ¬ (-W g = -w) := fun hc => hg (by linarith [neg_injective hc])
      simp [hg, this]
  rw [hR, hist, Finset.mul_sum]
  refine Finset.sum_congr rfl fun g _ => ?_
  by_cases hg : W g = w
  · rw [if_pos hg, if_pos hg, hC g, hg]
    ring
  · rw [if_neg hg, if_neg hg, mul_zero]

/-- **The forward and reverse work histograms cross exactly at the free energy.** -/
theorem crooks_crossing {rev : Equiv.Perm G} {p q W : G → ℝ} {b dF : ℝ}
    (hC : Crooks rev p q W b dF) (hW : ∀ g, W (rev g) = - W g) :
    hist p W dF = hist q W (-dF) := by
  rw [crooks_histogram hC hW dF, sub_self, mul_zero, Real.exp_zero, one_mul]

/-! ## What a finite sample misses -/

/-- **The exponential average is carried by trajectories that are not sampled.**  For every `M`
there is a two-branch work distribution whose rare branch has probability at most `exp(−M)`,
contributes at least as much as the entire typical branch to the exponential average, and moves
the free energy to `−M` or below -- while the estimate from the typical branch alone is `0`. -/
theorem rare_trajectories_dominate (M : ℝ) (hM : 0 < M) :
    ∃ eps wRare : ℝ, 0 < eps ∧ eps ≤ Real.exp (-M) ∧
      (1 - eps) * Real.exp (-(0:ℝ)) ≤ eps * Real.exp (-wRare) ∧
      M ≤ Real.log (eps * Real.exp (-wRare) + (1 - eps) * Real.exp (-(0:ℝ))) := by
  refine ⟨Real.exp (-M), -(2 * M), Real.exp_pos _, le_rfl, ?_, ?_⟩
  · have h1 : Real.exp (-M) * Real.exp (-(-(2 * M))) = Real.exp M := by
      rw [← Real.exp_add]; ring_nf
    have h2 : (1 : ℝ) ≤ Real.exp M := Real.one_le_exp hM.le
    have h3 : 0 < Real.exp (-M) := Real.exp_pos _
    rw [h1]
    simp only [neg_zero, Real.exp_zero, mul_one]
    linarith
  · have h1 : Real.exp (-M) * Real.exp (-(-(2 * M))) = Real.exp M := by
      rw [← Real.exp_add]; ring_nf
    have h4 : Real.exp (-M) ≤ 1 := by
      rw [Real.exp_le_one_iff]; linarith
    rw [h1]
    simp only [neg_zero, Real.exp_zero, mul_one]
    have hpos : (0:ℝ) < Real.exp M := Real.exp_pos _
    calc M = Real.log (Real.exp M) := (Real.log_exp M).symm
      _ ≤ Real.log (Real.exp M + (1 - Real.exp (-M))) := by
          refine Real.log_le_log hpos ?_
          linarith


/-- **Discarding the low-work trajectories overestimates the free energy.**  If every discarded
trajectory has less work than every retained one, the exponential average computed from the
retained trajectories alone is at most the true one, so the free energy inferred from it is at
least the true free energy. -/
theorem omitting_low_work_overestimates [DecidableEq G] {p W : G → ℝ} (b : ℝ) (S : Finset G)
    (hp : ∀ g, 0 ≤ p g) (hps : ∑ g, p g = 1) (hS : ∑ g ∈ S, p g < 1)
    (hb : 0 < b) (hlow : ∀ g ∈ S, ∀ h ∈ Sᶜ, W g < W h) :
    (∑ g ∈ Sᶜ, p g * Real.exp (-b * W g)) / (∑ g ∈ Sᶜ, p g)
      ≤ ∑ g, p g * Real.exp (-b * W g) := by
  classical
  set f : G → ℝ := fun g => Real.exp (-b * W g) with hf
  have hsplit : ∑ g ∈ S, p g + ∑ g ∈ Sᶜ, p g = ∑ g, p g := Finset.sum_add_sum_compl S p
  have hc : 0 < ∑ g ∈ Sᶜ, p g := by
    rw [hps] at hsplit
    linarith
  have hfle : ∀ g ∈ S, ∀ h ∈ Sᶜ, f h ≤ f g := by
    intro g hg h hh
    have : -b * W g ≥ -b * W h := by
      have := hlow g hg h hh
      nlinarith
    exact Real.exp_le_exp.mpr this
  set N : ℝ := ∑ h ∈ Sᶜ, p h * f h with hN
  set c : ℝ := ∑ h ∈ Sᶜ, p h with hcdef
  have hmA : ∀ g ∈ S, N / c ≤ f g := by
    intro g hg
    have h1 : N ≤ c * f g := by
      have : ∑ h ∈ Sᶜ, p h * f h ≤ ∑ h ∈ Sᶜ, p h * f g :=
        Finset.sum_le_sum fun h hh => mul_le_mul_of_nonneg_left (hfle g hg h hh) (hp h)
      rw [← Finset.sum_mul] at this
      rw [hN, hcdef]
      linarith [this]
    rw [div_le_iff₀ hc]
    linarith [h1]
  have hSsum : (∑ g ∈ S, p g) * (N / c) ≤ ∑ g ∈ S, p g * f g := by
    have : ∑ g ∈ S, p g * (N / c) ≤ ∑ g ∈ S, p g * f g :=
      Finset.sum_le_sum fun g hg => mul_le_mul_of_nonneg_left (hmA g hg) (hp g)
    rwa [← Finset.sum_mul] at this
  have htotal : ∑ g, p g * f g = ∑ g ∈ S, p g * f g + N := by
    rw [hN]
    exact (Finset.sum_add_sum_compl S (fun g => p g * f g)).symm
  have hlam : ∑ g ∈ S, p g = 1 - c := by
    rw [hps] at hsplit
    rw [hcdef]
    linarith
  have hNc : N = (N / c) * c := by field_simp
  rw [htotal]
  rw [hlam] at hSsum
  nlinarith [hSsum, hNc, hc]

end Work

end IDR
