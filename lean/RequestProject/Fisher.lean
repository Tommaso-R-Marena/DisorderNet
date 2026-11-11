/-
# Part VI.1  How precisely can a context be read off a disordered ensemble?

Part V asks how much data are needed to *learn* an ensemble.  This file asks the dual, and
experimentally more common, question: given that the ensemble is known, how precisely can a
*perturbation* be inferred from conformational data?  A denaturant concentration, a ligand
activity, a crowding fugacity, a phosphorylation state -- each enters the statistical
mechanics as the strength `lam` of a coupling `A` in the exponentially tilted family of
`RequestProject.Response`, and reading it off a structural measurement is a parameter
estimation problem.

The answer is a precision limit that is fixed by the *fluctuations* of the region, in
exactly the sense of the fluctuation--response theorem:

* `cov_sq_le_var_mul_var` -- the Cauchy--Schwarz inequality for the tilted ensemble.
* `fisher` -- the Fisher information of the tilt parameter, which by
  `Response.susceptibility_eq_variance` is *the same object* as the susceptibility: the
  variance of the coupling observable.
* `logPart_convexOn` -- the log-partition function is convex in the coupling: thermodynamic
  stability, and the statement that the tilted family is a regular exponential family.
* `cramer_rao` -- **the Cramér--Rao bound**.  Any unbiased estimator of the perturbation
  strength built from a single conformation has variance at least `1 / fisher`.  Its proof
  is the fluctuation--response theorem plus Cauchy--Schwarz: unbiasedness forces the
  estimator's covariance with the coupling to be `1`, and covariance is bounded by the
  fluctuations that are available.
* `no_unbiased_estimator_of_rigid` -- consequently a *rigid* region carries no information
  about its own perturbation at all: with `A` constant no unbiased estimator exists, at any
  sample size.  Disorder is not noise to be removed; it is the entire measurement channel.
* `chapman_robbins` and `context_discrimination` -- the assumption-free version.  For any
  two contexts `p, q` and any conformational readout `T`, the shift of the readout is at
  most `sqrt (Var_q(T) · chi²(p‖q))`, with `chi²` the reweighting cost of
  `RequestProject.Reweighting`.  Contexts that are cheap to reweight between are contexts
  that cannot be told apart.
-/
import Mathlib
import RequestProject.Response
import RequestProject.Reweighting

namespace IDR

open Finset
open scoped Classical

namespace Fisher

variable {n : ℕ}

/-! ## A weighted Cauchy--Schwarz inequality -/

/-- Cauchy--Schwarz with nonnegative weights, on an arbitrary finite conformation space. -/
theorem weighted_cauchy_schwarz {ι : Type*} [Fintype ι] (w u v : ι → ℝ) (hw : ∀ j, 0 ≤ w j) :
    (∑ j, w j * (u j * v j)) ^ 2
      ≤ (∑ j, w j * (u j * u j)) * (∑ j, w j * (v j * v j)) := by
  have e : ∀ a b : ι → ℝ, ∑ j, Real.sqrt (w j) * a j * (Real.sqrt (w j) * b j)
      = ∑ j, w j * (a j * b j) := by
    intro a b
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [show Real.sqrt (w j) * a j * (Real.sqrt (w j) * b j)
        = (Real.sqrt (w j) * Real.sqrt (w j)) * (a j * b j) from by ring,
      Real.mul_self_sqrt (hw j)]
  have h := Finset.sum_mul_sq_le_sq_mul_sq (univ : Finset ι)
    (fun j => Real.sqrt (w j) * u j) (fun j => Real.sqrt (w j) * v j)
  simp only [pow_two] at h ⊢
  rw [e u v, e u u, e v v] at h
  exact h

variable {q A f g T : Fin n → ℝ}

open Response

/-- Cauchy--Schwarz for the tilted ensemble: a covariance is bounded by the fluctuations
that are available to produce it. -/
theorem cov_sq_le_var_mul_var (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) :
    (cov q A f g lam) ^ 2 ≤ var q A f lam * var q A g lam := by
  have := weighted_cauchy_schwarz (fun j => tilted q A lam j)
    (fun j => f j - meanObs q A f lam) (fun j => g j - meanObs q A g lam)
    (fun j => tilted_nonneg hn hq lam j)
  simpa [cov, var] using this

/-! ## Fisher information of the perturbation strength -/

/-- The Fisher information that a single conformation carries about the strength of the
perturbation `A`.  By `Response.susceptibility_eq_variance` it coincides with the
susceptibility `d⟨A⟩/dλ`: **information and response are the same quantity**. -/
noncomputable def fisher (q A : Fin n → ℝ) (lam : ℝ) : ℝ := var q A A lam

lemma fisher_nonneg (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) : 0 ≤ fisher q A lam :=
  var_nonneg hn hq lam

/-- Information equals response: the Fisher information is the derivative of the mean
coupling with respect to the coupling strength. -/
theorem fisher_eq_susceptibility (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) :
    HasDerivAt (meanObs q A A) (fisher q A lam) lam :=
  susceptibility_eq_variance hn hq lam

/-- The log-partition function is differentiable in the coupling strength. -/
lemma differentiable_logPart (hn : 0 < n) (hq : ∀ j, 0 < q j) :
    Differentiable ℝ (fun l => Real.log (part q A l)) :=
  fun l => (hasDerivAt_logPart hn hq l).differentiableAt

/-- **Convexity of the free energy in the coupling.**  The tilted family is a regular
exponential family: its log-partition function is convex, which is thermodynamic stability
(the susceptibility, being a variance, cannot be negative). -/
theorem logPart_convexOn (hn : 0 < n) (hq : ∀ j, 0 < q j) :
    ConvexOn ℝ Set.univ (fun l => Real.log (part q A l)) := by
  have hderiv : deriv (fun l => Real.log (part q A l)) = meanObs q A A := by
    funext l
    exact (hasDerivAt_logPart hn hq l).deriv
  refine Monotone.convexOn_univ_of_deriv (differentiable_logPart hn hq) ?_
  rw [hderiv]
  exact meanObs_mono hn hq

/-! ## The Cramér--Rao bound -/

/-- Unbiasedness pins the covariance of the estimator with the coupling to `1`.  This is the
fluctuation--response theorem read backwards: an estimator that tracks the perturbation must
be correlated with the observable through which the perturbation acts. -/
theorem cov_eq_one_of_unbiased (hn : 0 < n) (hq : ∀ j, 0 < q j)
    (hub : ∀ l, meanObs q A T l = l) (lam : ℝ) : cov q A T A lam = 1 := by
  have h1 : HasDerivAt (meanObs q A T) (cov q A T A lam) lam := linear_response hn hq lam
  have hfun : meanObs q A T = fun l : ℝ => l := funext hub
  have h2 : HasDerivAt (meanObs q A T) 1 lam := by
    rw [hfun]; exact hasDerivAt_id lam
  exact h1.unique h2

/-- **The Cramér--Rao bound for a disordered region.**  Any estimator of the perturbation
strength that is unbiased throughout the family has variance at least the reciprocal of the
Fisher information -- that is, at least the reciprocal of the *fluctuation* of the coupling
observable.  Precision about the context is bought with conformational disorder. -/
theorem cramer_rao (hn : 0 < n) (hq : ∀ j, 0 < q j)
    (hub : ∀ l, meanObs q A T l = l) (lam : ℝ) :
    1 ≤ var q A T lam * fisher q A lam := by
  have hcs := cov_sq_le_var_mul_var (A := A) (f := T) (g := A) hn hq lam
  rw [cov_eq_one_of_unbiased hn hq hub lam] at hcs
  simpa [fisher] using hcs

/-- The bound in its familiar form. -/
theorem cramer_rao_var_ge (hn : 0 < n) (hq : ∀ j, 0 < q j)
    (hub : ∀ l, meanObs q A T l = l) (lam : ℝ) (hI : 0 < fisher q A lam) :
    1 / fisher q A lam ≤ var q A T lam := by
  rw [div_le_iff₀ hI]
  linarith [cramer_rao hn hq hub lam]

/-- **A rigid region is uninformative.**  If the coupling observable does not vary over the
library -- the perturbation cannot see the region, or the region cannot move -- then no
unbiased estimator of the perturbation strength exists at all.  There is no estimator to
improve, no architecture to try, and no amount of data that helps: the channel from context
to conformation is closed. -/
theorem no_unbiased_estimator_of_rigid (hn : 0 < n) (hq : ∀ j, 0 < q j) (c : ℝ)
    (hconst : ∀ j, A j = c) : ¬ ∃ T : Fin n → ℝ, ∀ l, meanObs q A T l = l := by
  rintro ⟨T, hub⟩
  have h := cramer_rao (T := T) hn hq hub 0
  rw [fisher, rigid_no_response hn hq 0 c hconst] at h
  have : (1:ℝ) ≤ 0 := by simpa using h
  linarith

/-- Conversely, whenever two conformations differ along the coupling, the Fisher
information is strictly positive: a disordered region *does* report on its context, at a
rate equal to its fluctuation. -/
theorem fisher_pos_of_disordered (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) {j₁ j₂ : Fin n}
    (hne : A j₁ ≠ A j₂) : 0 < fisher q A lam :=
  response_of_disordered hn hq lam hne

/-! ## The assumption-free version: Chapman--Robbins -/

section General

variable {ι : Type*} [Fintype ι]

/-- The chi-squared divergence on an arbitrary finite conformation space; on a library
`Fin m` it is `IDR.Reweight.chiSq`. -/
noncomputable def chiSqG (p q : ι → ℝ) : ℝ := ∑ j, (p j - q j) ^ 2 / q j

lemma chiSqG_eq_chiSq {m : ℕ} (p q : Fin m → ℝ) : chiSqG p q = Reweight.chiSq p q := rfl

lemma chiSqG_nonneg {p q : ι → ℝ} (hq : ∀ j, 0 < q j) : 0 ≤ chiSqG p q :=
  Finset.sum_nonneg fun j _ => div_nonneg (sq_nonneg _) (hq j).le

/-- The variance of a readout in an ensemble. -/
noncomputable def varW (w u : ι → ℝ) : ℝ :=
  ∑ j, w j * ((u j - ∑ i, w i * u i) * (u j - ∑ i, w i * u i))

lemma varW_nonneg {w u : ι → ℝ} (hw : ∀ j, 0 ≤ w j) : 0 ≤ varW w u :=
  Finset.sum_nonneg fun j _ => mul_nonneg (hw j) (mul_self_nonneg _)

/-- **The Chapman--Robbins bound.**  For any two contexts `p` and `q` and any conformational
readout `T` whatsoever -- a FRET efficiency, a radius of gyration, a chemical shift, the
output of a trained classifier -- the squared shift of the readout between the contexts is
at most the product of its fluctuation in one context with the chi-squared divergence
between them.  No unbiasedness, differentiability or model assumption is used. -/
theorem chapman_robbins {p q T : ι → ℝ} (hq : ∀ j, 0 < q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) :
    ((∑ j, p j * T j) - ∑ j, q j * T j) ^ 2 ≤ varW q T * chiSqG p q := by
  set mu := ∑ i, q i * T i with hmu
  -- centre the readout: the difference of the means is a `q`-weighted inner product
  have hdiff : (∑ j, p j * T j) - ∑ j, q j * T j
      = ∑ j, q j * (((p j - q j) / q j) * (T j - mu)) := by
    have hterm : ∀ j : ι, q j * (((p j - q j) / q j) * (T j - mu))
        = (p j - q j) * T j - (p j - q j) * mu := by
      intro j
      have hqj : q j ≠ 0 := (hq j).ne'
      field_simp
    rw [Finset.sum_congr rfl fun j (_ : j ∈ univ) => hterm j, Finset.sum_sub_distrib,
      ← Finset.sum_mul]
    have hzero : ∑ j, (p j - q j) = 0 := by
      rw [Finset.sum_sub_distrib, hps, hqs]; ring
    rw [hzero, zero_mul, sub_zero, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  have hcs := weighted_cauchy_schwarz q (fun j => (p j - q j) / q j) (fun j => T j - mu)
    (fun j => (hq j).le)
  rw [← hdiff] at hcs
  have hchi : ∑ j, q j * (((p j - q j) / q j) * ((p j - q j) / q j)) = chiSqG p q := by
    simp only [chiSqG]
    refine Finset.sum_congr rfl fun j _ => ?_
    have hqj : q j ≠ 0 := (hq j).ne'
    field_simp
  calc ((∑ j, p j * T j) - ∑ j, q j * T j) ^ 2
      ≤ (∑ j, q j * (((p j - q j) / q j) * ((p j - q j) / q j)))
          * (∑ j, q j * ((T j - mu) * (T j - mu))) := hcs
    _ = varW q T * chiSqG p q := by rw [hchi, varW, ← hmu]; ring

/-- **The discrimination limit.**  Two contexts that are cheap to reweight between (small
chi-squared, i.e. large effective sample size in `RequestProject.Reweighting`) shift every
readout by a correspondingly small amount.  This is the precision counterpart of the
capacity laws: the experiment can separate contexts only as far as the ensembles genuinely
separate. -/
theorem context_discrimination {p q T : ι → ℝ} (hq : ∀ j, 0 < q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) :
    |(∑ j, p j * T j) - ∑ j, q j * T j| ≤ Real.sqrt (varW q T * chiSqG p q) := by
  have h := chapman_robbins (T := T) hq hps hqs
  rw [← Real.sqrt_sq_eq_abs]
  exact Real.sqrt_le_sqrt h

end General

end Fisher

end IDR
