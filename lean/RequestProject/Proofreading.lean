import Mathlib

/-!
# Part CXLVIII — Proofreading: how a low-affinity motif can be read accurately

Recognition by intrinsically disordered regions is recognition by short linear motifs, and
short linear motifs bind weakly: the equilibrium discrimination between a cognate and a
near-cognate partner is a single ratio of dissociation constants, often no better than ten.
Signalling nevertheless discriminates far better than that.  The resolution is Hopfield's:
accuracy beyond the equilibrium ratio is bought by running the recognition through a cascade
of driven, effectively irreversible steps, each of which offers the complex another chance to
fall apart.  This file proves both halves of that statement.

The equilibrium half (`equilibrium_no_extra_discrimination`) is a no-go theorem.  If the
intermediate steps are ordinary equilibria and the intermediates themselves do not
distinguish the two partners, the overall population ratio is the *first* equilibrium
constant ratio, no matter how many steps are inserted.  Adding conformational states to an
equilibrium model of motif recognition cannot add specificity — which is the same lesson as
Part CXLIV's chaperone theorem, in a different arena.

The kinetic half is a gain theorem.  In a driven cascade of `n` stages the complex reaches
the product with probability `(kf/(kf+koff))^n`, so the discrimination between a partner with
off-rate `koffR` and one with `koffW` is `((kf+koffW)/(kf+koffR))^n`
(`discrimination_eq`): the equilibrium factor raised to the number of stages.  It is strictly
increasing in `n` (`discrimination_strictMono`) and unbounded
(`discrimination_tendsto_atTop`).

The price is exact and unavoidable.  The yield of the correct partner falls geometrically to
zero (`throughput_tendsto_zero`), and the exchange rate between the two is a constant of the
chemistry, independent of how many stages one builds: the ratio of `log` accuracy to `log`
yield loss is the same for every `n` (`accuracy_yield_exchange`).
-/

noncomputable section

namespace RequestProject.Proofreading

open Real Filter Topology

/-! ### The equilibrium no-go theorem -/

/-- **Equilibrium intermediates add no specificity.**  If two partners differ only in the
first equilibrium constant of a chain — the intermediates being conformational states that do
not themselves discriminate — the overall population ratio is that first ratio, whatever the
length of the chain. -/
theorem equilibrium_no_extra_discrimination {n : ℕ} (KR KW : Fin (n + 1) → ℝ)
    (hW : ∀ i, 0 < KW i) (h : ∀ i : Fin (n + 1), i ≠ 0 → KR i = KW i) :
    (∏ i, KR i) / (∏ i, KW i) = KR 0 / KW 0 := by
  have htail : ∀ i : Fin n, KR i.succ = KW i.succ := fun i => h i.succ (Fin.succ_ne_zero i)
  rw [Fin.prod_univ_succ, Fin.prod_univ_succ]
  rw [Finset.prod_congr rfl (fun i _ => htail i)]
  have hP : (0 : ℝ) < ∏ i : Fin n, KW i.succ := Finset.prod_pos fun i _ => hW i.succ
  rw [mul_div_mul_right _ _ hP.ne']

/-! ### The driven cascade -/

/-- Probability that a complex advances one stage rather than falling apart. -/
def passProb (kf koff : ℝ) : ℝ := kf / (kf + koff)

/-- Probability that a complex survives all `n` stages of the cascade. -/
def completion (kf koff : ℝ) (n : ℕ) : ℝ := passProb kf koff ^ n

/-- The discrimination achieved by an `n`-stage cascade between a partner with off-rate
`koffR` and one with off-rate `koffW`. -/
def discrimination (kf koffR koffW : ℝ) (n : ℕ) : ℝ :=
  completion kf koffR n / completion kf koffW n

lemma passProb_pos {kf koff : ℝ} (hf : 0 < kf) (hoff : 0 < koff) : 0 < passProb kf koff :=
  div_pos hf (by linarith)

lemma passProb_lt_one {kf koff : ℝ} (hf : 0 < kf) (hoff : 0 < koff) :
    passProb kf koff < 1 := by
  unfold passProb
  rw [div_lt_one (by linarith)]
  linarith

/-- **The discrimination of an `n`-stage cascade is the one-stage factor to the `n`-th
power.** -/
theorem discrimination_eq {kf koffR koffW : ℝ} (hf : 0 < kf) (hR : 0 < koffR)
    (hW : 0 < koffW) (n : ℕ) :
    discrimination kf koffR koffW n = ((kf + koffW) / (kf + koffR)) ^ n := by
  unfold discrimination completion passProb
  rw [← div_pow]
  congr 1
  field_simp

/-- One stage gives exactly the equilibrium discrimination factor. -/
theorem discrimination_one {kf koffR koffW : ℝ} (hf : 0 < kf) (hR : 0 < koffR)
    (hW : 0 < koffW) :
    discrimination kf koffR koffW 1 = (kf + koffW) / (kf + koffR) := by
  rw [discrimination_eq hf hR hW, pow_one]

/-- **Every additional stage multiplies the accuracy.** -/
theorem discrimination_strictMono {kf koffR koffW : ℝ} (hf : 0 < kf) (hR : 0 < koffR)
    (hW : 0 < koffW) (hRW : koffR < koffW) :
    StrictMono (discrimination kf koffR koffW) := by
  have hbase : 1 < (kf + koffW) / (kf + koffR) := by
    rw [lt_div_iff₀ (by linarith)]
    linarith
  intro a b hab
  rw [discrimination_eq hf hR hW, discrimination_eq hf hR hW]
  exact pow_lt_pow_right₀ hbase hab

/-- **The accuracy is unbounded in the number of stages.** -/
theorem discrimination_tendsto_atTop {kf koffR koffW : ℝ} (hf : 0 < kf) (hR : 0 < koffR)
    (hW : 0 < koffW) (hRW : koffR < koffW) :
    Tendsto (discrimination kf koffR koffW) atTop atTop := by
  have hbase : 1 < (kf + koffW) / (kf + koffR) := by
    rw [lt_div_iff₀ (by linarith)]
    linarith
  have h := tendsto_pow_atTop_atTop_of_one_lt hbase
  refine h.congr (fun n => ?_)
  rw [discrimination_eq hf hR hW]

/-- **The price: the yield falls geometrically to zero.** -/
theorem throughput_tendsto_zero {kf koff : ℝ} (hf : 0 < kf) (hoff : 0 < koff) :
    Tendsto (completion kf koff) atTop (𝓝 0) :=
  tendsto_pow_atTop_nhds_zero_of_lt_one (passProb_pos hf hoff).le
    (passProb_lt_one hf hoff)

/-- **The exchange rate between accuracy and yield is a constant of the chemistry.**  For
every number of stages, the logarithm of the accuracy and the logarithm of the yield loss
stand in the same fixed ratio: no cascade design escapes it. -/
theorem accuracy_yield_exchange {kf koffR koffW : ℝ} (hf : 0 < kf) (hR : 0 < koffR)
    (hW : 0 < koffW) (n : ℕ) :
    Real.log (discrimination kf koffR koffW n) * Real.log ((kf + koffR) / kf)
      = Real.log (1 / completion kf koffR n) * Real.log ((kf + koffW) / (kf + koffR)) := by
  have hcomp : completion kf koffR n = (kf / (kf + koffR)) ^ n := rfl
  rw [discrimination_eq hf hR hW, hcomp, Real.log_pow, one_div, ← inv_pow, Real.log_pow,
    inv_div]
  ring

/-- **The proofreading design law.**  Recognition of a disordered motif obeys:

1. inserting equilibrium intermediates that do not themselves discriminate leaves the
   discrimination at the single equilibrium ratio, however many are inserted;
2. an `n`-stage driven cascade raises that ratio to the `n`-th power, strictly increasing and
   unbounded in `n`;
3. the yield of the correct partner falls geometrically to zero, and
4. the logarithmic exchange rate between accuracy and yield is the same for every `n`. -/
theorem proofreading_design_law {kf koffR koffW : ℝ} (hf : 0 < kf) (hR : 0 < koffR)
    (hW : 0 < koffW) (hRW : koffR < koffW) :
    (∀ (m : ℕ) (KR KW : Fin (m + 1) → ℝ), (∀ i, 0 < KW i) →
        (∀ i : Fin (m + 1), i ≠ 0 → KR i = KW i) →
        (∏ i, KR i) / (∏ i, KW i) = KR 0 / KW 0) ∧
    (∀ n : ℕ, discrimination kf koffR koffW n = ((kf + koffW) / (kf + koffR)) ^ n) ∧
    StrictMono (discrimination kf koffR koffW) ∧
    Tendsto (discrimination kf koffR koffW) atTop atTop ∧
    Tendsto (completion kf koffR) atTop (𝓝 0) ∧
    (∀ n : ℕ, Real.log (discrimination kf koffR koffW n) * Real.log ((kf + koffR) / kf)
      = Real.log (1 / completion kf koffR n) * Real.log ((kf + koffW) / (kf + koffR))) :=
  ⟨fun _ KR KW hpos hEq => equilibrium_no_extra_discrimination KR KW hpos hEq,
    discrimination_eq hf hR hW,
    discrimination_strictMono hf hR hW hRW,
    discrimination_tendsto_atTop hf hR hW hRW,
    throughput_tendsto_zero hf hR,
    accuracy_yield_exchange hf hR hW⟩

end RequestProject.Proofreading
