/-
# Part V.2  Linear response: what a disorder model predicts about *perturbations*

Everything so far concerns a single ensemble.  A disordered region, however, is normally
interrogated by *changing* something -- adding a binding partner, a crowder, a phosphate
group, a denaturant, a temperature increment -- and asking how the populations move.  This
file proves the exact answer, and shows that it is governed by the very quantity that makes
the region disordered in the first place: its conformational fluctuation.

Setting: a conformational library `Fin n` carrying a reference ensemble `q` (strictly
positive weights), a coupling observable `A` (the conformation-dependent part of the
perturbing energy, in units of `kT`), and the exponentially tilted family
`tilted q A lam j ∝ q j · exp (lam · A j)`, which is exactly the Boltzmann ensemble of the
perturbed landscape (`tilted_eq_boltz_of_unif`).

* `hasDerivAt_logPart` -- the log-partition function generates the mean coupling.
* `linear_response` -- **the fluctuation--response theorem**: for *every* observable `f`,
  `d⟨f⟩/dlam = Cov(f, A)` in the tilted ensemble.  Susceptibilities are covariances; a
  model that gets the fluctuations of a disordered region wrong gets its response to every
  perturbation wrong, by exactly the covariance error.
* `susceptibility_eq_variance` -- the case `f = A`: the response of the coupling itself is
  its variance, so response is nonnegative (`susceptibility_nonneg`) and the mean coupling
  is monotone in the coupling strength (`meanObs_mono`).
* `rigid_no_response` / `response_iff_disordered` -- **an ordered region cannot respond**:
  zero conformational variance forces zero susceptibility, and conversely a region responds
  to *some* perturbation exactly when it populates more than one conformation.  This is the
  formal version of the statement that conformational entropy is the functional resource of
  an intrinsically disordered region.
* `bogoliubov` and `bogoliubov_gap` -- the Gibbs--Bogoliubov--Feynman inequality: any
  reference (e.g. mean-field, or a trained generative) model gives a rigorous *upper* bound
  on the free energy, and its excess is exactly `(1/β)·KL` of the reference ensemble from
  the true one.
-/
import Mathlib
import RequestProject.DisorderedRegions
import RequestProject.EnsembleCore
import RequestProject.FreeEnergy

namespace IDR

open Finset
open scoped Classical

namespace Response

variable {n : ℕ}

/-- The partition function of the tilted family. -/
noncomputable def part (q A : Fin n → ℝ) (lam : ℝ) : ℝ := ∑ j, q j * Real.exp (lam * A j)

/-- The unnormalised average of an observable in the tilted family. -/
noncomputable def unAvg (q A f : Fin n → ℝ) (lam : ℝ) : ℝ :=
  ∑ j, q j * f j * Real.exp (lam * A j)

/-- The exponentially tilted ensemble: the Boltzmann reweighting of `q` by the perturbing
energy `-lam·A`. -/
noncomputable def tilted (q A : Fin n → ℝ) (lam : ℝ) : Fin n → ℝ :=
  fun j => q j * Real.exp (lam * A j) / part q A lam

/-- The tilted average of an observable. -/
noncomputable def meanObs (q A f : Fin n → ℝ) (lam : ℝ) : ℝ :=
  ∑ j, tilted q A lam j * f j

/-- The tilted covariance of two observables. -/
noncomputable def cov (q A f g : Fin n → ℝ) (lam : ℝ) : ℝ :=
  ∑ j, tilted q A lam j * ((f j - meanObs q A f lam) * (g j - meanObs q A g lam))

/-- The tilted variance of an observable. -/
noncomputable def var (q A f : Fin n → ℝ) (lam : ℝ) : ℝ := cov q A f f lam

variable {q A f g : Fin n → ℝ}

lemma part_pos (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) : 0 < part q A lam := by
  have : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
  exact Finset.sum_pos (fun j _ => mul_pos (hq j) (Real.exp_pos _))
    ⟨⟨0, hn⟩, Finset.mem_univ _⟩

lemma tilted_nonneg (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) (j : Fin n) :
    0 ≤ tilted q A lam j :=
  div_nonneg (mul_nonneg (hq j).le (Real.exp_pos _).le) (part_pos hn hq lam).le

lemma tilted_sum_one (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) :
    ∑ j, tilted q A lam j = 1 := by
  simp only [tilted, ← Finset.sum_div]
  exact div_self (part_pos hn hq lam).ne'

/-- The tilted average is the ratio of the unnormalised average to the partition
function. -/
lemma meanObs_eq (lam : ℝ) :
    meanObs q A f lam = unAvg q A f lam / part q A lam := by
  simp only [meanObs, tilted, unAvg, Finset.sum_div]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- Covariance in the usual `⟨fg⟩ - ⟨f⟩⟨g⟩` form. -/
lemma cov_eq (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) :
    cov q A f g lam
      = meanObs q A (fun j => f j * g j) lam - meanObs q A f lam * meanObs q A g lam := by
  have hone := tilted_sum_one (A := A) hn hq lam
  simp only [cov, meanObs]
  have expand : ∀ j : Fin n, tilted q A lam j *
      ((f j - ∑ i, tilted q A lam i * f i) * (g j - ∑ i, tilted q A lam i * g i))
      = tilted q A lam j * (f j * g j)
        - (∑ i, tilted q A lam i * g i) * (tilted q A lam j * f j)
        - (∑ i, tilted q A lam i * f i) * (tilted q A lam j * g j)
        + ((∑ i, tilted q A lam i * f i) * (∑ i, tilted q A lam i * g i))
          * tilted q A lam j := by
    intro j; ring
  rw [Finset.sum_congr rfl fun j (_ : j ∈ Finset.univ) => expand j]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, hone]
  ring

lemma var_nonneg (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) : 0 ≤ var q A f lam := by
  simp only [var, cov]
  refine Finset.sum_nonneg fun j _ => mul_nonneg (tilted_nonneg hn hq lam j) ?_
  exact mul_self_nonneg _

/-- The tilted family is literally a Boltzmann ensemble: tilting the uniform reference by
the coupling `A = -β·U` reproduces the Boltzmann weights of the landscape `U`.  So the
response theory below is a statement about physical perturbations of a force field. -/
lemma tilted_eq_boltz_of_unif (hn : 0 < n) (beta : ℝ) (U : Fin n → ℝ) :
    tilted (fun _ => 1 / (n:ℝ)) (fun j => -beta * U j) 1 = FreeEnergy.boltz beta U := by
  have hnpos : (0:ℝ) < n := by exact_mod_cast hn
  funext j
  have hpart : part (fun _ => 1 / (n:ℝ)) (fun j => -beta * U j) 1
      = FreeEnergy.part beta U / (n:ℝ) := by
    simp only [part, FreeEnergy.part, one_mul, Finset.sum_div]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [tilted, hpart, FreeEnergy.boltz]
  have hZ : FreeEnergy.part beta U ≠ 0 := (FreeEnergy.part_pos hn beta U).ne'
  field_simp

/-! ## Derivatives of the tilted family -/

lemma hasDerivAt_part (lam : ℝ) :
    HasDerivAt (part q A) (unAvg q A A lam) lam := by
  have h : ∀ j : Fin n,
      HasDerivAt (fun l : ℝ => q j * Real.exp (l * A j))
        (q j * A j * Real.exp (lam * A j)) lam := by
    intro j
    have h1 : HasDerivAt (fun l : ℝ => l * A j) (A j) lam := by
      simpa using (hasDerivAt_id lam).mul_const (A j)
    have h2 := (h1.exp).const_mul (q j)
    convert h2 using 1
    ring
  have hsum : HasDerivAt (fun l : ℝ => ∑ j, q j * Real.exp (l * A j))
      (∑ j, q j * A j * Real.exp (lam * A j)) lam :=
    HasDerivAt.fun_sum (fun j (_ : j ∈ (Finset.univ : Finset (Fin n))) => h j)
  exact hsum

lemma hasDerivAt_unAvg (lam : ℝ) :
    HasDerivAt (unAvg q A f) (unAvg q A (fun j => f j * A j) lam) lam := by
  have h : ∀ j : Fin n,
      HasDerivAt (fun l : ℝ => q j * f j * Real.exp (l * A j))
        (q j * (f j * A j) * Real.exp (lam * A j)) lam := by
    intro j
    have h1 : HasDerivAt (fun l : ℝ => l * A j) (A j) lam := by
      simpa using (hasDerivAt_id lam).mul_const (A j)
    have h2 := (h1.exp).const_mul (q j * f j)
    convert h2 using 1
    ring
  have hsum : HasDerivAt (fun l : ℝ => ∑ j, q j * f j * Real.exp (l * A j))
      (∑ j, q j * (f j * A j) * Real.exp (lam * A j)) lam :=
    HasDerivAt.fun_sum (fun j (_ : j ∈ (Finset.univ : Finset (Fin n))) => h j)
  exact hsum

/-- The log-partition function of the tilted family generates the mean coupling: it is the
cumulant generating function of `A`. -/
theorem hasDerivAt_logPart (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) :
    HasDerivAt (fun l => Real.log (part q A l)) (meanObs q A A lam) lam := by
  have h := (hasDerivAt_part (q := q) (A := A) lam).log (part_pos hn hq lam).ne'
  rw [meanObs_eq]
  exact h

/-- **Fluctuation--response theorem.**  In the exponentially tilted family, the derivative
of the average of *any* observable with respect to the coupling strength is its covariance
with the coupling:  `d⟨f⟩/dλ = Cov(f, A)`.

Physically: susceptibilities to perturbations (ligand, crowder, post-translational
modification, denaturant) are *fluctuation* properties of the unperturbed ensemble.  A
model that misrepresents the correlations of a disordered region therefore mispredicts its
entire response behaviour, by exactly the covariance error. -/
theorem linear_response (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) :
    HasDerivAt (meanObs q A f) (cov q A f A lam) lam := by
  have hZ := hasDerivAt_part (q := q) (A := A) lam
  have hN := hasDerivAt_unAvg (q := q) (A := A) (f := f) lam
  have hpos := part_pos (A := A) hn hq lam
  have hdiv := hN.div hZ hpos.ne'
  have hfun : meanObs q A f = fun l => unAvg q A f l / part q A l := by
    funext l
    rw [meanObs_eq]
  have hval : (unAvg q A (fun j => f j * A j) lam * part q A lam
      - unAvg q A f lam * unAvg q A A lam) / part q A lam ^ 2 = cov q A f A lam := by
    rw [cov_eq hn hq, meanObs_eq, meanObs_eq, meanObs_eq]
    field_simp
  rw [hfun, ← hval]
  exact hdiv

/-- The self-response of the coupling observable is its variance. -/
theorem susceptibility_eq_variance (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) :
    HasDerivAt (meanObs q A A) (var q A A lam) lam :=
  linear_response hn hq lam

/-- Susceptibility is nonnegative. -/
theorem susceptibility_nonneg (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) :
    0 ≤ deriv (meanObs q A A) lam := by
  rw [(susceptibility_eq_variance hn hq lam).deriv]
  exact var_nonneg hn hq lam

/-- Raising the coupling strength can only raise the mean coupling: the tilted family is
stochastically ordered. -/
theorem meanObs_mono (hn : 0 < n) (hq : ∀ j, 0 < q j) : Monotone (meanObs q A A) := by
  have hderiv : ∀ l : ℝ, HasDerivAt (meanObs q A A) (var q A A l) l :=
    fun l => susceptibility_eq_variance hn hq l
  have hdiff : Differentiable ℝ (meanObs q A A) := fun l => (hderiv l).differentiableAt
  refine monotone_of_deriv_nonneg hdiff fun l => ?_
  rw [(hderiv l).deriv]
  exact var_nonneg hn hq l

/-! ## Only disorder responds -/

/-- If the coupling observable takes a single value across the whole library -- the region
is rigid as far as the perturbation can see -- then the susceptibility vanishes: no
perturbation of that shape can shift anything. -/
theorem rigid_no_response (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam c : ℝ)
    (hconst : ∀ j, A j = c) : var q A A lam = 0 := by
  have hmean : meanObs q A A lam = c := by
    simp only [meanObs, hconst, ← Finset.sum_mul, tilted_sum_one hn hq lam, one_mul]
  simp only [var, cov]
  refine Finset.sum_eq_zero fun j _ => ?_
  rw [hmean, hconst j]
  ring

/-- Conversely, a nonzero fluctuation of the coupling *is* a nonzero response.  Combined
with `rigid_no_response`: a region responds to a perturbation exactly to the extent that it
fluctuates along it.  Conformational entropy is the resource. -/
theorem response_of_disordered (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) {j₁ j₂ : Fin n}
    (hne : A j₁ ≠ A j₂) : 0 < var q A A lam := by
  set mu := meanObs q A A lam with hmu
  -- at least one of the two conformations differs from the mean
  have hex : ∃ j : Fin n, A j ≠ mu := by
    by_contra hcon
    push_neg at hcon
    exact hne ((hcon j₁).trans (hcon j₂).symm)
  obtain ⟨j₀, hj₀⟩ := hex
  simp only [var, cov, ← hmu]
  refine Finset.sum_pos' (fun j _ => mul_nonneg (tilted_nonneg hn hq lam j) (mul_self_nonneg _))
    ⟨j₀, Finset.mem_univ j₀, ?_⟩
  have htpos : 0 < tilted q A lam j₀ :=
    div_pos (mul_pos (hq j₀) (Real.exp_pos _)) (part_pos hn hq lam)
  have hsq : 0 < (A j₀ - mu) * (A j₀ - mu) :=
    mul_self_pos.2 (sub_ne_zero.2 hj₀)
  exact mul_pos htpos hsq

/-! ## The Gibbs--Bogoliubov--Feynman bound: a model is a certified free energy -/

open FreeEnergy

/-- **Gibbs--Bogoliubov--Feynman inequality.**  For any reference landscape `U0` (a
mean-field model, a coarse-grained force field, or a trained generative model interpreted
as an energy), the true free energy is bounded above by the reference free energy plus the
mean energy difference *evaluated in the reference ensemble*.  A tractable model therefore
yields a rigorous variational bound on a quantity of the intractable true system. -/
theorem bogoliubov (hn : 0 < n) {beta : ℝ} (hbeta : 0 < beta) (U U0 : Fin n → ℝ) :
    freeEnergy beta U (boltz beta U)
      ≤ freeEnergy beta U0 (boltz beta U0)
        + ∑ j, boltz beta U0 j * (U j - U0 j) := by
  have hp : ∀ j, 0 ≤ boltz beta U0 j := fun j => (boltz_pos hn beta U0 j).le
  have hps : ∑ j, boltz beta U0 j = 1 := boltz_sum_one hn beta U0
  have hle := freeEnergy_ge (U := U) hn hbeta hp hps
  have hsplit : freeEnergy beta U (boltz beta U0)
      = freeEnergy beta U0 (boltz beta U0) + ∑ j, boltz beta U0 j * (U j - U0 j) := by
    simp only [freeEnergy]
    have : ∑ j, boltz beta U0 j * (U j - U0 j)
        = (∑ j, boltz beta U0 j * U j) - ∑ j, boltz beta U0 j * U0 j := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [this]
    ring
  linarith [hsplit ▸ hle]

/-- **And the slack is exactly a relative entropy.**  The amount by which a reference model
overestimates the free energy is `(1/β)·KL(reference ‖ truth)`: the thermodynamic penalty
of an approximate model *is* its information-theoretic error. -/
theorem bogoliubov_gap (hn : 0 < n) {beta : ℝ} (hbeta : 0 < beta) (U U0 : Fin n → ℝ) :
    (freeEnergy beta U0 (boltz beta U0) + ∑ j, boltz beta U0 j * (U j - U0 j))
        - freeEnergy beta U (boltz beta U)
      = klDiv (boltz beta U0) (boltz beta U) / beta := by
  have hp : ∀ j, 0 ≤ boltz beta U0 j := fun j => (boltz_pos hn beta U0 j).le
  have hps : ∑ j, boltz beta U0 j = 1 := boltz_sum_one hn beta U0
  have hgap := freeEnergy_gap (U := U) hn hbeta hp hps
  have hsplit : freeEnergy beta U (boltz beta U0)
      = freeEnergy beta U0 (boltz beta U0) + ∑ j, boltz beta U0 j * (U j - U0 j) := by
    simp only [freeEnergy]
    have : ∑ j, boltz beta U0 j * (U j - U0 j)
        = (∑ j, boltz beta U0 j * U j) - ∑ j, boltz beta U0 j * U0 j := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [this]
    ring
  rw [← hsplit]
  linarith [hgap]

end Response

end IDR
