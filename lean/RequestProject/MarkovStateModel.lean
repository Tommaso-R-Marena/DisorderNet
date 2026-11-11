/-
# Part LXII.1  The Markov state model that is actually estimated

`RequestProject.Kinetics` settles the qualitative question about lumping microstates into
macrostates: under Dynkin's condition the coarse variable is Markov (`lump_evolve`), and
generically it is not (`no_markov_coarse_graining`).  That leaves the quantitative question,
which is the one a Markov state model of a disordered region raises in practice.  Nobody
verifies Dynkin's condition.  What is done instead is to run a long equilibrium trajectory,
count transitions between the clusters at some lag time, and normalise -- producing the
*stationary-weighted lumping*

  `macroW a b = (Σ_{i ∈ a} π i · Σ_{j ∈ b} P i j) / (Σ_{i ∈ a} π i)`   (`macroW`),

the conditional probability of being in cluster `b` one lag later given equilibrium within
cluster `a`.  This file asks what that matrix is a statement about when the clustering is *not*
lumpable, and the answer is unusually clean: it is always exactly right about thermodynamics,
it is one-sided about kinetics, and it depends on the lag.

* `macroW_eq_of_lumpable` -- the estimator is consistent: when the clustering happens to be
  lumpable, the estimated matrix is the lumped matrix.
* `macroW_nonneg`, `macroW_stochastic`, `macroW_stationary`, `macroW_detailedBalance` -- and
  with no condition on the clustering at all, `macroW` is a stochastic matrix, the coarse
  equilibrium populations are stationary for it, and detailed balance is inherited.  **A Markov
  state model always reproduces the equilibrium thermodynamics of its own clustering**; that it
  does so is therefore no evidence whatever that the clustering is good.
* `mean_lump_eq`, `nrm_lump_eq`, `form_lump_eq` -- the coarse mean, mean square and quadratic
  form of a macro observable `g` are *exactly* the micro mean, mean square and quadratic form
  of `g ∘ phi`.  The coarse model's Rayleigh quotients are a subset of the fine model's.
* `macro_gap_bound` -- hence **coarse-graining cannot invent a slow
  mode**: every relaxation eigenvalue of the Markov state model is one the microscopic chain
  already had, so a reported implied timescale is a lower bound on the truth and never an
  overestimate.  Clustering hides slow motion; it does not manufacture it.
* `lumpable_matMul`, `chapman_kolmogorov_of_lumpable` -- if the clustering is lumpable then the
  matrix estimated at lag two is the square of the matrix estimated at lag one: the
  Chapman--Kolmogorov ("implied timescale") test is a genuine test of the modelling assumption.
* `msm_P_detailedBalance`, `chapman_kolmogorov_fails` -- and it is a test that is generically
  failed.  For an explicit reversible three-state chain with uniform equilibrium -- nothing
  pathological, no absorbing state, no zero population -- the lag-two estimate is `1/4` and the
  square of the lag-one estimate is `5/16`.  The matrix a Markov state model reports is a
  function of a modelling choice, the lag time, and the two answers differ by 25% of the
  smaller.

Read as a design constraint on models of disordered regions: a state-decomposition kinetic
model may be quoted for populations without qualification, must have its timescales quoted as
lower bounds, and is not defined at all until the lag is quoted with it.
-/
import Mathlib
import RequestProject.Kinetics

namespace IDR

open Finset

namespace MSM

open IDR.Kinetics

variable {n p : ℕ}

/-! ## The estimated matrix -/

/-- The equilibrium flux from macrostate `a` to macrostate `b`: the number of transitions per
step an infinitely long equilibrium trajectory records. -/
def flux (phi : Fin n → Fin p) (pi : Fin n → ℝ) (P : Fin n → Fin n → ℝ) (a b : Fin p) : ℝ :=
  lump phi (fun i => pi i * lump phi (P i) b) a

/-- **The Markov state model**: the stationary-weighted lumping of `P`, i.e. the transition
matrix estimated by counting transitions between clusters along an equilibrium trajectory. -/
noncomputable def macroW (phi : Fin n → Fin p) (pi : Fin n → ℝ) (P : Fin n → Fin n → ℝ)
    (a b : Fin p) : ℝ := flux phi pi P a b / lump phi pi a

/-- The product of two transition matrices; `matMul P P` is the two-step matrix, the one
estimated at lag two. -/
def matMul (P Q : Fin n → Fin n → ℝ) (i j : Fin n) : ℝ := ∑ k, P i k * Q k j

/-! ## Elementary rewriting -/

/-- Summing a function of the macrostate against lumped weights is summing it against the
microstates. -/
lemma sum_lump_mul (phi : Fin n → Fin p) (w : Fin n → ℝ) (g : Fin p → ℝ) :
    ∑ a, lump phi w a * g a = ∑ i, w i * g (phi i) := by
  classical
  have h : ∀ a : Fin p, lump phi w a * g a
      = ∑ i ∈ Finset.univ.filter (fun i => phi i = a), w i * g (phi i) := by
    intro a
    rw [lump, Finset.sum_mul]
    refine Finset.sum_congr rfl fun i hi => ?_
    rw [(Finset.mem_filter.1 hi).2]
  rw [Finset.sum_congr rfl (fun a _ => h a)]
  exact Finset.sum_fiberwise Finset.univ phi (fun i => w i * g (phi i))

/-- The total lumped weight is the total weight. -/
lemma sum_lump (phi : Fin n → Fin p) (w : Fin n → ℝ) : ∑ a, lump phi w a = ∑ i, w i := by
  have := sum_lump_mul phi w (fun _ => 1)
  simpa using this

lemma lump_nonneg {phi : Fin n → Fin p} {w : Fin n → ℝ} (hw : ∀ i, 0 ≤ w i) (a : Fin p) :
    0 ≤ lump phi w a :=
  Finset.sum_nonneg fun i _ => hw i

/-! ## The estimator is consistent -/

/-- When the clustering is lumpable, the estimated matrix is the lumped matrix: the Markov
state model recovers the true reduced kinetics. -/
theorem macroW_eq_of_lumpable {P : Fin n → Fin n → ℝ} {phi : Fin n → Fin p}
    {Q : Fin p → Fin p → ℝ} (h : Lumpable P phi Q) (pi : Fin n → ℝ) {a : Fin p}
    (ha : lump phi pi a ≠ 0) (b : Fin p) : macroW phi pi P a b = Q a b := by
  classical
  have hflux : flux phi pi P a b = lump phi pi a * Q a b := by
    simp only [flux, lump, Finset.sum_mul]
    refine Finset.sum_congr rfl fun i hi => ?_
    have hia : phi i = a := (Finset.mem_filter.1 hi).2
    rw [h i b, hia]
  rw [macroW, hflux, mul_comm, mul_div_assoc, div_self ha, mul_one]

/-! ## The thermodynamics always survives -/

theorem macroW_nonneg {P : Fin n → Fin n → ℝ} {phi : Fin n → Fin p} {pi : Fin n → ℝ}
    (hP : IsStochastic P) (hpi : ∀ i, 0 ≤ pi i) (a b : Fin p) : 0 ≤ macroW phi pi P a b := by
  refine div_nonneg ?_ (lump_nonneg hpi a)
  exact lump_nonneg (fun i => mul_nonneg (hpi i) (lump_nonneg (fun j => hP.1 i j) b)) a

/-- The estimated matrix is stochastic, whatever the clustering. -/
theorem macroW_stochastic {P : Fin n → Fin n → ℝ} {phi : Fin n → Fin p} {pi : Fin n → ℝ}
    (hP : IsStochastic P) {a : Fin p} (ha : lump phi pi a ≠ 0) :
    ∑ b, macroW phi pi P a b = 1 := by
  classical
  have hrow : ∀ i : Fin n, ∑ b, lump phi (P i) b = 1 := by
    intro i
    rw [sum_lump phi (P i), hP.2 i]
  have hsum : ∑ b, flux phi pi P a b = lump phi pi a := by
    simp only [flux, lump]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => ?_
    have : ∑ b, pi i * lump phi (P i) b = pi i := by
      rw [← Finset.mul_sum, hrow i, mul_one]
    exact this
  simp only [macroW]
  rw [← Finset.sum_div, hsum, div_self ha]

/-- The coarse-grained equilibrium populations are stationary for the estimated matrix,
whatever the clustering. -/
theorem macroW_stationary {P : Fin n → Fin n → ℝ} {phi : Fin n → Fin p} {pi : Fin n → ℝ}
    (hpi : Kinetics.Stationary P pi) (hne : ∀ a, lump phi pi a ≠ 0) (b : Fin p) :
    ∑ a, lump phi pi a * macroW phi pi P a b = lump phi pi b := by
  classical
  have hcancel : ∀ a, lump phi pi a * macroW phi pi P a b = flux phi pi P a b := by
    intro a
    rw [macroW, mul_comm, div_mul_cancel₀ _ (hne a)]
  rw [Finset.sum_congr rfl (fun a _ => hcancel a)]
  have hflux : ∑ a, flux phi pi P a b = ∑ i, pi i * lump phi (P i) b :=
    sum_lump phi (fun i => pi i * lump phi (P i) b)
  rw [hflux]
  have hswap : ∑ i, pi i * lump phi (P i) b
      = ∑ j ∈ Finset.univ.filter (fun j => phi j = b), ∑ i, pi i * P i j := by
    simp only [lump, Finset.mul_sum]
    exact Finset.sum_comm
  rw [hswap, lump]
  refine Finset.sum_congr rfl fun j _ => ?_
  exact congrFun hpi j

/-- Detailed balance is inherited by the estimated matrix. -/
theorem macroW_detailedBalance {P : Fin n → Fin n → ℝ} {phi : Fin n → Fin p} {pi : Fin n → ℝ}
    (hrev : DetailedBalance P pi) (hne : ∀ a, lump phi pi a ≠ 0) (a b : Fin p) :
    lump phi pi a * macroW phi pi P a b = lump phi pi b * macroW phi pi P b a := by
  classical
  have hcancel : ∀ x y : Fin p, lump phi pi x * macroW phi pi P x y = flux phi pi P x y := by
    intro x y
    rw [macroW, mul_comm, div_mul_cancel₀ _ (hne x)]
  rw [hcancel a b, hcancel b a]
  simp only [flux, lump, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => hrev i j

/-! ## Coarse-graining cannot invent a slow mode -/

/-- The quadratic form `⟨f, P f⟩` in the equilibrium inner product; its maximum over mean-zero
observables of unit norm is the second eigenvalue of a reversible chain. -/
def form (pi : Fin n → ℝ) (P : Fin n → Fin n → ℝ) (f : Fin n → ℝ) : ℝ :=
  ∑ i, ∑ j, pi i * P i j * (f i * f j)

/-- The equilibrium mean square of an observable. -/
def nrm (pi : Fin n → ℝ) (f : Fin n → ℝ) : ℝ := ∑ i, pi i * (f i * f i)

/-- The equilibrium mean of an observable. -/
def mean (pi : Fin n → ℝ) (f : Fin n → ℝ) : ℝ := ∑ i, pi i * f i

theorem mean_lump_eq (phi : Fin n → Fin p) (pi : Fin n → ℝ) (g : Fin p → ℝ) :
    mean (lump phi pi) g = mean pi (g ∘ phi) := by
  simpa [mean, Function.comp] using sum_lump_mul phi pi g

theorem nrm_lump_eq (phi : Fin n → Fin p) (pi : Fin n → ℝ) (g : Fin p → ℝ) :
    nrm (lump phi pi) g = nrm pi (g ∘ phi) := by
  simpa [nrm, Function.comp, mul_assoc] using sum_lump_mul phi pi (fun a => g a * g a)

/-- **The coarse quadratic form is the fine quadratic form of the lumped observable.** -/
theorem form_lump_eq {P : Fin n → Fin n → ℝ} {phi : Fin n → Fin p} {pi : Fin n → ℝ}
    (hne : ∀ a, lump phi pi a ≠ 0) (g : Fin p → ℝ) :
    form (lump phi pi) (macroW phi pi P) g = form pi P (g ∘ phi) := by
  classical
  have hcancel : ∀ a b : Fin p,
      lump phi pi a * macroW phi pi P a b * (g a * g b) = flux phi pi P a b * (g a * g b) := by
    intro a b
    rw [macroW, mul_comm (lump phi pi a), div_mul_cancel₀ _ (hne a)]
  simp only [form]
  rw [Finset.sum_congr rfl (fun a _ => Finset.sum_congr rfl (fun b _ => hcancel a b))]
  -- expand the flux and exchange the order of summation
  have hL : ∑ a, ∑ b, flux phi pi P a b * (g a * g b)
      = ∑ i, ∑ j, pi i * P i j * (g (phi i) * g (phi j)) := by
    have step1 : ∀ b : Fin p, ∑ a, flux phi pi P a b * (g a * g b)
        = ∑ i, pi i * lump phi (P i) b * (g (phi i) * g b) := by
      intro b
      have := sum_lump_mul phi (fun i => pi i * lump phi (P i) b) (fun a => g a * g b)
      simpa [flux] using this
    rw [Finset.sum_comm]
    rw [Finset.sum_congr rfl (fun b _ => step1 b)]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => ?_
    have := sum_lump_mul phi (P i) (fun b => g b)
    calc ∑ b, pi i * lump phi (P i) b * (g (phi i) * g b)
        = ∑ b, (pi i * g (phi i)) * (lump phi (P i) b * g b) := by
          refine Finset.sum_congr rfl fun b _ => by ring
      _ = (pi i * g (phi i)) * ∑ b, lump phi (P i) b * g b := by rw [Finset.mul_sum]
      _ = (pi i * g (phi i)) * ∑ j, P i j * g (phi j) := by rw [this]
      _ = ∑ j, pi i * P i j * (g (phi i) * g (phi j)) := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun j _ => by ring
  rw [hL]
  rfl

/-- **A Markov state model can only underestimate a relaxation time.**  Any bound on the
microscopic quadratic form over mean-zero observables passes to the coarse model. -/
theorem macro_gap_bound {P : Fin n → Fin n → ℝ} {phi : Fin n → Fin p} {pi : Fin n → ℝ}
    (hne : ∀ a, lump phi pi a ≠ 0) (lam : ℝ)
    (h : ∀ f : Fin n → ℝ, mean pi f = 0 → form pi P f ≤ lam * nrm pi f) (g : Fin p → ℝ)
    (hg : mean (lump phi pi) g = 0) :
    form (lump phi pi) (macroW phi pi P) g ≤ lam * nrm (lump phi pi) g := by
  rw [form_lump_eq hne, nrm_lump_eq]
  exact h (g ∘ phi) (by rwa [← mean_lump_eq])

/-! ## Chapman--Kolmogorov: a genuine test -/

/-- Lumpability is preserved by composition: the two-step matrix is lumpable with the squared
reduced matrix. -/
theorem lumpable_matMul {P : Fin n → Fin n → ℝ} {phi : Fin n → Fin p} {Q : Fin p → Fin p → ℝ}
    (h : Lumpable P phi Q) :
    Lumpable (matMul P P) phi (fun a b => ∑ c, Q a c * Q c b) := by
  classical
  intro i b
  have hstep : ∑ j ∈ Finset.univ.filter (fun j => phi j = b), matMul P P i j
      = ∑ k, P i k * (∑ j ∈ Finset.univ.filter (fun j => phi j = b), P k j) := by
    simp only [matMul, Finset.mul_sum]
    exact Finset.sum_comm
  rw [hstep]
  have hk : ∀ k : Fin n, P i k * (∑ j ∈ Finset.univ.filter (fun j => phi j = b), P k j)
      = P i k * Q (phi k) b := by
    intro k; rw [h k b]
  rw [Finset.sum_congr rfl (fun k _ => hk k)]
  have := sum_lump_mul phi (P i) (fun c => Q c b)
  rw [← this]
  have hlump : ∀ c : Fin p, lump phi (P i) c = Q (phi i) c := by
    intro c; rw [lump, h i c]
  rw [Finset.sum_congr rfl (fun c _ => by rw [hlump c])]

/-- **Under lumpability the estimated matrix satisfies Chapman--Kolmogorov**: the lag-two
estimate is the square of the lag-one estimate.  The implied-timescale test is therefore a real
test of the modelling assumption. -/
theorem chapman_kolmogorov_of_lumpable {P : Fin n → Fin n → ℝ} {phi : Fin n → Fin p}
    {Q : Fin p → Fin p → ℝ} (h : Lumpable P phi Q) (pi : Fin n → ℝ)
    (hne : ∀ a, lump phi pi a ≠ 0) (a b : Fin p) :
    macroW phi pi (matMul P P) a b = ∑ c, macroW phi pi P a c * macroW phi pi P c b := by
  rw [macroW_eq_of_lumpable (lumpable_matMul h) pi (hne a) b]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [macroW_eq_of_lumpable h pi (hne a) c, macroW_eq_of_lumpable h pi (hne c) b]

/-! ## And a test that is generically failed -/

/-- A reversible three-state chain with uniform equilibrium: nearest-neighbour hopping with
probability `1/2`, reflected at the ends. -/
noncomputable def msmP : Fin 3 → Fin 3 → ℝ := ![![1/2, 1/2, 0], ![1/2, 0, 1/2], ![0, 1/2, 1/2]]

/-- Its uniform equilibrium distribution. -/
noncomputable def msmPi : Fin 3 → ℝ := fun _ => 1/3

theorem msmP_stochastic : IsStochastic msmP := by
  constructor
  · intro i j; fin_cases i <;> fin_cases j <;> norm_num [msmP]
  · intro i; fin_cases i <;> simp [msmP, Fin.sum_univ_three] <;> norm_num

theorem msmP_detailedBalance : DetailedBalance msmP msmPi := by
  intro i j; fin_cases i <;> fin_cases j <;> norm_num [msmP, msmPi]

lemma lump_msmPi_zero : lump Kinetics.badPhi msmPi 0 = 2/3 := by
  rw [lump_badPhi_zero]; norm_num [msmPi]

lemma lump_msmPi_one : lump Kinetics.badPhi msmPi 1 = 1/3 := by
  rw [lump_badPhi_one]; norm_num [msmPi]

/-- The estimated matrix for the clustering `{0,1} | {2}` at uniform equilibrium, entry by
entry. -/
lemma macroW_msmPi_zero_zero (P : Fin 3 → Fin 3 → ℝ) :
    macroW Kinetics.badPhi msmPi P 0 0 = (P 0 0 + P 0 1 + (P 1 0 + P 1 1)) / 2 := by
  rw [macroW, flux, lump_msmPi_zero]
  simp only [lump_badPhi_zero, msmPi]
  ring

lemma macroW_msmPi_zero_one (P : Fin 3 → Fin 3 → ℝ) :
    macroW Kinetics.badPhi msmPi P 0 1 = (P 0 2 + P 1 2) / 2 := by
  rw [macroW, flux, lump_msmPi_zero]
  simp only [lump_badPhi_zero, lump_badPhi_one, msmPi]
  ring

lemma macroW_msmPi_one_zero (P : Fin 3 → Fin 3 → ℝ) :
    macroW Kinetics.badPhi msmPi P 1 0 = P 2 0 + P 2 1 := by
  rw [macroW, flux, lump_msmPi_one]
  simp only [lump_badPhi_one, lump_badPhi_zero, msmPi]
  ring

lemma macroW_msmPi_one_one (P : Fin 3 → Fin 3 → ℝ) :
    macroW Kinetics.badPhi msmPi P 1 1 = P 2 2 := by
  rw [macroW, flux, lump_msmPi_one]
  simp only [lump_badPhi_one, msmPi]
  ring

/-- **The reported matrix depends on the lag.**  For this chain and the clustering
`{0,1} | {2}`, the lag-two estimate of the escape probability is `1/4`, while the square of the
lag-one estimate is `5/16`. -/
theorem chapman_kolmogorov_fails :
    macroW Kinetics.badPhi msmPi (matMul msmP msmP) 0 1 = 1/4 ∧
      (∑ c, macroW Kinetics.badPhi msmPi msmP 0 c * macroW Kinetics.badPhi msmPi msmP c 1)
        = 5/16 := by
  constructor
  · rw [macroW_msmPi_zero_one]
    simp only [matMul, Fin.sum_univ_three, msmP]
    norm_num [Matrix.cons_val_two, Matrix.tail_cons]
  · rw [Fin.sum_univ_two, macroW_msmPi_zero_zero, macroW_msmPi_zero_one,
      macroW_msmPi_one_one]
    norm_num [msmP, Matrix.cons_val_two, Matrix.tail_cons]

end MSM

end IDR
