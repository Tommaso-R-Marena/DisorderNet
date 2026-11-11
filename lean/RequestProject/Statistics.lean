/-
# Which classes of models can, and which cannot, represent a disordered region

A predictor is never fitted to, nor evaluated on, *all* observables of a conformational
ensemble: it is fitted to a chosen family `S` of statistics (coordinates, distograms,
per-residue marginals, first and second moments, ...).  This file isolates the resulting
general obstruction.

* `AgreeOn S E F` -- the ensembles `E` and `F` are indistinguishable through `S`.
* `Separates S` -- `S` determines the ensemble completely.
* `exists_failure_of_not_separating` -- **the master negative theorem**: *any* predictor
  that depends on its target only through a non-separating family of statistics is
  necessarily wrong, in the strong sense that some ensemble it must model is not
  reproduced.  No amount of data, capacity or training changes this: the information the
  model is fitted to does not determine the answer.

The rest of the file instantiates the master theorem for the model families that are
actually used, by exhibiting, in each case, two ensembles the family cannot tell apart:

| family of statistics | what it models              | counterexample                         |
|----------------------|-----------------------------|----------------------------------------|
| `LinearObs`          | coordinate regression       | `pmOne` vs `dirac 0`                   |
| `QuadraticObs`       | mean + covariance / B-factor| `pmOne` vs `pmTwo`                     |
| `MarginalObs`        | per-residue marginals       | `corrEns` vs `prodEns`                 |
| `DistanceObs`        | distograms / contact maps   | `dirac ![0,1]` vs `dirac ![0,-1]`      |

Finally `exists_ensemble_beyond_capacity` shows that no *fixed* number of mixture
components suffices: for every `k` there is an ensemble that no `k`-component model
reproduces.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry

namespace IDR

open Finset
open scoped Classical

variable {X : Type*}

/-! ## The master theorem -/

/-- `E` and `F` are indistinguishable by the family of statistics `S`. -/
def AgreeOn (S : Set (X → ℝ)) (E F : Ens X) : Prop := ∀ f ∈ S, E.expect f = F.expect f

/-- The family of statistics `S` *separates* ensembles: matching `S` forces the ensembles
to be observationally equal. -/
def Separates (S : Set (X → ℝ)) : Prop := ∀ E F : Ens X, AgreeOn S E F → E.Same F

lemma separates_univ : Separates (Set.univ : Set (X → ℝ)) :=
  fun _ _ h f => h f (Set.mem_univ f)

lemma not_separates_iff {S : Set (X → ℝ)} :
    ¬ Separates S ↔ ∃ E F : Ens X, AgreeOn S E F ∧ ¬ E.Same F := by
  unfold Separates
  push_neg
  tauto

/-- **Master negative theorem.**  Let `A` be any predictor -- any map from target
ensembles to models -- whose output depends on the target only through the statistics `S`
(that is what "trained on `S`" means).  If `S` does not separate ensembles, then `A` fails:
there is a target ensemble that `A` does not reproduce.  The failure is information
theoretic, so it is not repaired by more data, more parameters or better optimisation. -/
theorem exists_failure_of_not_separating {S : Set (X → ℝ)} (A : Ens X → Ens X)
    (hA : ∀ E F : Ens X, AgreeOn S E F → A E = A F) (hS : ¬ Separates S) :
    ∃ E : Ens X, ¬ (A E).Same E := by
  obtain ⟨E, F, hEF, hne⟩ := not_separates_iff.1 hS
  by_contra hcon
  push_neg at hcon
  have h1 : (A E).Same E := hcon E
  have h2 : (A F).Same F := hcon F
  rw [hA E F hEF] at h1
  exact hne ((h1.symm).trans h2)

/-- Conversely, if `S` separates then matching `S` is enough. -/
theorem same_of_agreeOn {S : Set (X → ℝ)} (hS : Separates S) {E F : Ens X}
    (h : AgreeOn S E F) : E.Same F := hS E F h

/-! ## Uniform ensembles, used to build the counterexamples -/

/-- The uniform ensemble on the conformations `g 0, ..., g (m-1)`. -/
noncomputable def unif {m : ℕ} (hm : 0 < m) (g : Fin m → X) : Ens X where
  card := m
  pt := g
  w := fun _ => 1 / m
  w_nonneg := by
    intro j
    positivity
  w_sum := by
    have hm' : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp

lemma unif_expect {m : ℕ} (hm : 0 < m) (g : Fin m → X) (f : X → ℝ) :
    (unif hm g).expect f = (∑ i, f (g i)) / m := by
  simp only [unif, Ens.expect]
  rw [Finset.sum_div]
  exact Finset.sum_congr rfl fun i _ => by ring

lemma unif_prob_pos {m : ℕ} (hm : 0 < m) (g : Fin m → X) (i : Fin m) :
    0 < (unif hm g).prob (g i) := by
  refine ((unif hm g).prob_pos_iff (g i)).2 ⟨i, ?_, rfl⟩
  have : (0:ℝ) < m := by exact_mod_cast hm
  simp only [unif]
  positivity

/-! ## Families of statistics used by real predictors -/

variable {n : ℕ}

/-- Affine observables: what a coordinate-regression model is fitted to (the mean
structure). -/
def LinearObs (n : ℕ) : Set (Conf n → ℝ) :=
  {f | ∃ (a : ℝ) (b : Fin n → ℝ), ∀ c, f c = a + ∑ i, b i * c i}

/-- Observables of degree at most two: what a model predicting a mean structure together
with per-atom fluctuations (B-factors, a predicted covariance, an error bar) is fitted
to. -/
def QuadraticObs (n : ℕ) : Set (Conf n → ℝ) :=
  {f | ∃ (a : ℝ) (b : Fin n → ℝ) (Q : Fin n → Fin n → ℝ), ∀ c,
      f c = a + ∑ i, b i * c i + ∑ i, ∑ j, Q i j * (c i * c j)}

/-- Observables of a single degree of freedom: what a model predicting independent
per-residue distributions is fitted to. -/
def MarginalObs (n : ℕ) : Set (Conf n → ℝ) :=
  {f | ∃ (i : Fin n) (g : ℝ → ℝ), ∀ c, f c = g (c i)}

/-- Observables that are functions of the matrix of pairwise distances: what a distogram
or contact-map model is fitted to. -/
def DistanceObs (n : ℕ) : Set (Conf n → ℝ) :=
  {f | ∃ g : (Fin n → Fin n → ℝ) → ℝ, ∀ c, f c = g (fun i j => |c i - c j|)}

lemma expect_quadratic (E : Ens (Conf n)) (a : ℝ) (b : Fin n → ℝ) (Q : Fin n → Fin n → ℝ) :
    E.expect (fun c => a + ∑ i, b i * c i + ∑ i, ∑ j, Q i j * (c i * c j))
      = a + (∑ i, b i * E.expect (fun c => c i))
        + ∑ i, ∑ j, Q i j * E.expect (fun c => c i * c j) := by
  rw [E.expect_add (fun c => a + ∑ i, b i * c i) (fun c => ∑ i, ∑ j, Q i j * (c i * c j)),
    E.expect_add (fun _ => a) (fun c => ∑ i, b i * c i), E.expect_const,
    E.expect_sum Finset.univ (fun i c => b i * c i),
    E.expect_sum Finset.univ (fun i c => ∑ j, Q i j * (c i * c j))]
  congr 1
  · congr 1
    exact Finset.sum_congr rfl fun i _ => E.expect_smul (b i) (fun c => c i)
  · refine Finset.sum_congr rfl fun i _ => ?_
    rw [E.expect_sum Finset.univ (fun j c => Q i j * (c i * c j))]
    exact Finset.sum_congr rfl fun j _ => E.expect_smul (Q i j) (fun c => c i * c j)

/-- Matching the first two moments is exactly matching all observables of degree ≤ 2. -/
theorem agreeOn_quadraticObs_of_moments (E F : Ens (Conf n))
    (h1 : ∀ i, E.expect (fun c => c i) = F.expect (fun c => c i))
    (h2 : ∀ i j, E.expect (fun c => c i * c j) = F.expect (fun c => c i * c j)) :
    AgreeOn (QuadraticObs n) E F := by
  rintro f ⟨a, b, Q, hf⟩
  have hfe : f = fun c => a + ∑ i, b i * c i + ∑ i, ∑ j, Q i j * (c i * c j) := funext hf
  subst hfe
  rw [expect_quadratic, expect_quadratic]
  congr 1
  · congr 1
    exact Finset.sum_congr rfl fun i _ => by rw [h1 i]
  · exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by rw [h2 i j]

/-- Degree ≤ 1 observables are a subfamily of degree ≤ 2 observables. -/
lemma linearObs_subset_quadraticObs : LinearObs n ⊆ QuadraticObs n := by
  rintro f ⟨a, b, hf⟩
  exact ⟨a, b, fun _ _ => 0, fun c => by simp [hf c]⟩

/-! ## The counterexamples -/

/-- The one-dimensional two-state ensemble with values `-1` and `1`. -/
noncomputable def pmOne : Ens (Conf 1) :=
  unif (by norm_num) ![![(-1 : ℝ)], ![(1 : ℝ)]]

/-- A one-dimensional three-state ensemble with the *same* mean and variance as `pmOne`
but a different distribution: values `-2, 0, 2` with weights `1/8, 3/4, 1/8`. -/
noncomputable def pmTwo : Ens (Conf 1) where
  card := 3
  pt := ![![(-2 : ℝ)], ![(0 : ℝ)], ![(2 : ℝ)]]
  w := ![1/8, 3/4, 1/8]
  w_nonneg := by intro j; fin_cases j <;> norm_num
  w_sum := by norm_num [Fin.sum_univ_succ]

/-- A perfectly correlated two-residue ensemble: `(0,0)` and `(1,1)`. -/
noncomputable def corrEns : Ens (Conf 2) :=
  unif (by norm_num) ![![(0 : ℝ), 0], ![(1 : ℝ), 1]]

/-- The product ensemble with the same per-residue marginals as `corrEns`. -/
noncomputable def prodEns : Ens (Conf 2) :=
  unif (by norm_num) ![![(0 : ℝ), 0], ![(0 : ℝ), 1], ![(1 : ℝ), 0], ![(1 : ℝ), 1]]

lemma pmOne_expect (f : Conf 1 → ℝ) :
    pmOne.expect f = (f ![(-1 : ℝ)] + f ![(1 : ℝ)]) / 2 := by
  rw [pmOne, unif_expect]
  norm_num [Fin.sum_univ_succ]

lemma pmTwo_expect (f : Conf 1 → ℝ) :
    pmTwo.expect f = 1/8 * f ![(-2 : ℝ)] + 3/4 * f ![(0 : ℝ)] + 1/8 * f ![(2 : ℝ)] := by
  simp only [pmTwo, Ens.expect]
  norm_num [Fin.sum_univ_succ]
  ring

lemma corrEns_expect (f : Conf 2 → ℝ) :
    corrEns.expect f = (f ![(0 : ℝ), 0] + f ![(1 : ℝ), 1]) / 2 := by
  rw [corrEns, unif_expect]
  norm_num [Fin.sum_univ_succ]

lemma prodEns_expect (f : Conf 2 → ℝ) :
    prodEns.expect f =
      (f ![(0 : ℝ), 0] + f ![(0 : ℝ), 1] + f ![(1 : ℝ), 0] + f ![(1 : ℝ), 1]) / 4 := by
  rw [prodEns, unif_expect]
  norm_num [Fin.sum_univ_succ]
  ring

/-! ### Coordinate regression: matching the mean structure is not enough -/

theorem not_separates_linearObs : ¬ Separates (LinearObs 1) := by
  refine not_separates_iff.2 ⟨pmOne, Ens.dirac ![(0 : ℝ)], ?_, ?_⟩
  · rintro f ⟨a, b, hf⟩
    have hfe : f = fun c => a + ∑ i, b i * c i := funext hf
    subst hfe
    rw [pmOne_expect, Ens.expect_dirac]
    simp
    ring
  · intro h
    have := h (fun c => (c 0) ^ 2)
    rw [pmOne_expect, Ens.expect_dirac] at this
    norm_num at this

/-! ### Mean plus fluctuations: matching two moments is not enough -/

theorem not_separates_quadraticObs : ¬ Separates (QuadraticObs 1) := by
  refine not_separates_iff.2 ⟨pmOne, pmTwo, ?_, ?_⟩
  · refine agreeOn_quadraticObs_of_moments _ _ (fun i => ?_) (fun i j => ?_)
    · fin_cases i
      rw [pmOne_expect, pmTwo_expect]
      norm_num
    · fin_cases i
      fin_cases j
      rw [pmOne_expect, pmTwo_expect]
      norm_num
  · intro h
    have := h (fun c => (c 0) ^ 4)
    rw [pmOne_expect, pmTwo_expect] at this
    norm_num at this

/-! ### Per-residue marginals: matching them is not enough (correlations are lost) -/

theorem not_separates_marginalObs : ¬ Separates (MarginalObs 2) := by
  refine not_separates_iff.2 ⟨corrEns, prodEns, ?_, ?_⟩
  · rintro f ⟨i, g, hf⟩
    have hfe : f = fun c => g (c i) := funext hf
    subst hfe
    rw [corrEns_expect, prodEns_expect]
    fin_cases i <;> norm_num <;> ring
  · intro h
    have := h (fun c => c 0 * c 1)
    rw [corrEns_expect, prodEns_expect] at this
    norm_num at this

/-! ### Distograms: distances do not determine the conformation -/

theorem not_separates_distanceObs : ¬ Separates (DistanceObs 2) := by
  refine not_separates_iff.2
    ⟨Ens.dirac ![(0 : ℝ), 1], Ens.dirac ![(0 : ℝ), -1], ?_, ?_⟩
  · rintro f ⟨g, hf⟩
    rw [Ens.expect_dirac, Ens.expect_dirac, hf, hf]
    congr 1
    funext i j
    fin_cases i <;> fin_cases j <;> norm_num
  · intro h
    have := h (fun c => c 1)
    rw [Ens.expect_dirac, Ens.expect_dirac] at this
    norm_num at this

/-! ## Mean-field (independent-residue) models cannot capture correlated disorder -/

/-- A *mean-field* model: distinct degrees of freedom are uncorrelated.  This is a
necessary consequence of the per-residue independence assumed by factorised models, so
ruling out this larger class rules out all factorised models. -/
def MeanField (M : Ens (Conf n)) : Prop := ∀ i j, i ≠ j → M.cov i j = 0

lemma cov_eq_of_same {E F : Ens (Conf n)} (h : E.Same F) (i j : Fin n) :
    E.cov i j = F.cov i j := by
  have hi : E.mean i = F.mean i := h _
  have hj : E.mean j = F.mean j := h _
  rw [Ens.cov, Ens.cov, hi, hj]
  exact h _

lemma corrEns_mean (i : Fin 2) : corrEns.mean i = 1 / 2 := by
  rw [Ens.mean, corrEns_expect]
  fin_cases i <;> norm_num

lemma corrEns_cov : corrEns.cov 0 1 = 1 / 4 := by
  rw [Ens.cov, corrEns_expect, corrEns_mean, corrEns_mean]
  norm_num

/-- **Independent-residue models fail.**  No mean-field model reproduces an ensemble whose
degrees of freedom are correlated -- and correlated disorder is exactly what couples a
disordered region to its binding partner. -/
theorem no_meanField_captures_corrEns (M : Ens (Conf 2)) (hM : MeanField M) :
    ¬ M.Same corrEns := by
  intro h
  have h0 : M.cov 0 1 = corrEns.cov 0 1 := cov_eq_of_same h 0 1
  rw [corrEns_cov, hM 0 1 (by decide)] at h0
  norm_num at h0

/-! ## No fixed capacity suffices -/

/-- **Capacity must grow without bound.**  For every `k` there is a conformational
ensemble that no model with at most `k` mixture components (latent states, structural
templates, sampled decoys) reproduces. -/
theorem exists_ensemble_beyond_capacity (k : ℕ) :
    ∃ E : Ens (Conf 1), ∀ M : Ens (Conf 1), M.card ≤ k → ¬ M.Same E := by
  refine ⟨unif (Nat.succ_pos k) (fun l : Fin (k+1) => (fun _ => (l : ℝ))), ?_⟩
  intro M hM hsame
  have hinj : Function.Injective (fun l : Fin (k+1) => (fun _ => (l : ℝ)) : Fin (k+1) → Conf 1) := by
    intro a b hab
    have : ((a : ℕ) : ℝ) = ((b : ℕ) : ℝ) := congrFun hab 0
    have : (a : ℕ) = (b : ℕ) := by exact_mod_cast this
    exact Fin.ext this
  have hle : k + 1 ≤ M.card :=
    Ens.card_le_of_same hsame _ hinj (fun l => unif_prob_pos _ _ l)
  omega

end IDR
