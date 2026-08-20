/-
# A panel of structural features certifies additively

`RequestProject.TransportDuality` converts a discrepancy in *one* Lipschitz observable into
a lower bound on the structural transport error.  Applied to a panel of probes it gives one
bound per probe, and one is then forced to keep only the largest: the information in the
other probes is thrown away.

This file shows that the discrepancies can instead be **added**, provided the panel is
weighted so that its own weighted `ℓ¹` feature distance is dominated by the structural
metric.  Concretely, let `A i : X → ℝ` be structural features -- a compactness indicator, a
contact-formation indicator, a helicity fraction, a radius of gyration -- and `w i ≥ 0`
weights such that changing a conformation by `c x y` in structural distance can change the
weighted feature vector by at most `c x y` in weighted `ℓ¹`.  Then

  `∑ᵢ wᵢ · |⟨Aᵢ⟩_model − ⟨Aᵢ⟩_truth| ≤ transportCost c model truth`.

* `featureCost` -- the weighted `ℓ¹` distance between feature vectors, itself a pseudometric.
* `featureCost_le_transportCost` -- the additive certificate.
* `single_feature_certificate` -- the one-probe case, recovering the earlier certificate.
* `featureCost_le_transportCost_of_lipschitz` -- the usable form: if each feature is
  `Lᵢ`-Lipschitz and the weights satisfy `∑ᵢ wᵢ Lᵢ ≤ 1`, the additive certificate holds.

The design reading: a panel of probes should be assigned a weight budget summing (after
multiplication by each probe's Lipschitz constant) to one, and every probe then contributes
its full measured discrepancy to the certified structural error.  Probes that report on
tightly coupled features must share the budget; probes reporting on structurally
independent features each get their own.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Transport
import RequestProject.TransportGeometry
import RequestProject.TransportDuality

namespace IDR

open Finset
open scoped Classical

variable {X : Type*}

/-- The weighted `ℓ¹` distance between the feature vectors of two conformations. -/
noncomputable def featureCost {ι : Type*} [Fintype ι] (w : ι → ℝ) (A : ι → X → ℝ)
    (x y : X) : ℝ :=
  ∑ i, w i * |A i x - A i y|

lemma featureCost_nonneg {ι : Type*} [Fintype ι] {w : ι → ℝ} (hw : ∀ i, 0 ≤ w i)
    (A : ι → X → ℝ) (x y : X) : 0 ≤ featureCost w A x y :=
  Finset.sum_nonneg fun i _ => mul_nonneg (hw i) (abs_nonneg _)

lemma featureCost_self {ι : Type*} [Fintype ι] (w : ι → ℝ) (A : ι → X → ℝ) (x : X) :
    featureCost w A x x = 0 := by
  simp [featureCost]

lemma featureCost_comm {ι : Type*} [Fintype ι] (w : ι → ℝ) (A : ι → X → ℝ) (x y : X) :
    featureCost w A x y = featureCost w A y x :=
  Finset.sum_congr rfl fun i _ => by rw [abs_sub_comm]

lemma featureCost_triangle {ι : Type*} [Fintype ι] {w : ι → ℝ} (hw : ∀ i, 0 ≤ w i)
    (A : ι → X → ℝ) (x y z : X) :
    featureCost w A x z ≤ featureCost w A x y + featureCost w A y z := by
  rw [featureCost, featureCost, featureCost, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [← mul_add]
  exact mul_le_mul_of_nonneg_left (abs_sub_le _ _ _) (hw i)

/-- Writing a difference of ensemble averages against a coupling. -/
lemma expect_diff_eq_coupling_sum {E F : Ens X} {g : Fin E.card → Fin F.card → ℝ}
    (hg : IsCoupling E F g) (f : X → ℝ) :
    E.expect f - F.expect f = ∑ i, ∑ j, g i j * (f (E.pt i) - f (F.pt j)) := by
  have h1 : E.expect f = ∑ i, ∑ j, g i j * f (E.pt i) := by
    simp only [Ens.expect]
    exact Finset.sum_congr rfl fun i _ => by rw [← Finset.sum_mul, hg.row i]
  have h2 : F.expect f = ∑ i, ∑ j, g i j * f (F.pt j) := by
    simp only [Ens.expect]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun j _ => by rw [← Finset.sum_mul, hg.col j]
  rw [h1, h2, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- **The additive panel certificate.**  If the weighted `ℓ¹` distance between feature
vectors never exceeds the structural distance, then the *sum* of the weighted measured
discrepancies of the features is a lower bound on the structural transport error.  Every
probe in the panel contributes; nothing is discarded. -/
theorem featureCost_le_transportCost {ι : Type*} [Fintype ι] {w : ι → ℝ} (hw : ∀ i, 0 ≤ w i)
    {A : ι → X → ℝ} {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y)
    (hdom : ∀ x y, featureCost w A x y ≤ c x y) (E F : Ens X) :
    ∑ i, w i * |E.expect (A i) - F.expect (A i)| ≤ transportCost c E F := by
  obtain ⟨g, hg, hgc⟩ := exists_optimal_coupling hc E F
  rw [hgc, planCost]
  have hstep : ∀ i : ι, w i * |E.expect (A i) - F.expect (A i)|
      ≤ ∑ a, ∑ b, g a b * (w i * |A i (E.pt a) - A i (F.pt b)|) := by
    intro i
    rw [expect_diff_eq_coupling_sum hg (A i)]
    calc w i * |∑ a, ∑ b, g a b * (A i (E.pt a) - A i (F.pt b))|
        ≤ w i * ∑ a, ∑ b, |g a b * (A i (E.pt a) - A i (F.pt b))| := by
          refine mul_le_mul_of_nonneg_left ?_ (hw i)
          exact (Finset.abs_sum_le_sum_abs _ _).trans
            (Finset.sum_le_sum fun a _ => Finset.abs_sum_le_sum_abs _ _)
      _ = ∑ a, ∑ b, g a b * (w i * |A i (E.pt a) - A i (F.pt b)|) := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun a _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun b _ => ?_
          rw [abs_mul, abs_of_nonneg (hg.nonneg a b)]
          ring
  refine (Finset.sum_le_sum fun i _ => hstep i).trans ?_
  rw [Finset.sum_comm]
  refine Finset.sum_le_sum fun a _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_le_sum fun b _ => ?_
  rw [← Finset.mul_sum]
  exact mul_le_mul_of_nonneg_left (hdom _ _) (hg.nonneg a b)

/-- The one-probe case: a feature whose variation never exceeds the structural distance
certifies its own measured discrepancy. -/
theorem single_feature_certificate {A : X → ℝ} {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y)
    (hdom : ∀ x y, |A x - A y| ≤ c x y) (E F : Ens X) :
    |E.expect A - F.expect A| ≤ transportCost c E F := by
  have h := featureCost_le_transportCost (ι := Unit) (w := fun _ => 1)
    (fun _ => zero_le_one) (A := fun _ => A) hc (fun x y => by simpa [featureCost] using hdom x y)
    E F
  simpa using h

/-- **The usable form.**  If feature `i` is `L i`-Lipschitz with respect to the structural
metric and the weights obey the budget `∑ᵢ wᵢ · Lᵢ ≤ 1`, then the weighted measured
discrepancies add up to a certificate of structural error. -/
theorem featureCost_le_transportCost_of_lipschitz {ι : Type*} [Fintype ι] {w L : ι → ℝ}
    (hw : ∀ i, 0 ≤ w i) {A : ι → X → ℝ} {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y)
    (hlip : ∀ i x y, |A i x - A i y| ≤ L i * c x y) (hbudget : ∑ i, w i * L i ≤ 1)
    (E F : Ens X) :
    ∑ i, w i * |E.expect (A i) - F.expect (A i)| ≤ transportCost c E F := by
  refine featureCost_le_transportCost hw hc (fun x y => ?_) E F
  calc featureCost w A x y ≤ ∑ i, w i * (L i * c x y) :=
        Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hlip i x y) (hw i)
    _ = (∑ i, w i * L i) * c x y := by
        rw [Finset.sum_mul]
        exact Finset.sum_congr rfl fun i _ => by ring
    _ ≤ 1 * c x y := mul_le_mul_of_nonneg_right hbudget (hc x y)
    _ = c x y := one_mul _

/-! ## The additive certificate is sharp, and strictly beats the best single probe -/

/-- A two-coordinate caricature of a structural space: two independent descriptors, with the
structural cost of a change the sum of the two coordinate changes. -/
def planeCost (x y : ℝ × ℝ) : ℝ := |x.1 - y.1| + |x.2 - y.2|

/-- **The additive certificate is attained, and the single-probe certificate is not.**  For
two structures differing by one unit in each of two independent descriptors, the additive
panel certificate returns the exact structural distance `2`, whereas either descriptor on
its own certifies only `1`.  Combining probes is not a heuristic: it recovers information
that no single probe can. -/
theorem additive_certificate_is_sharp :
    (∀ x y : ℝ × ℝ, featureCost (fun _ : Fin 2 => (1 : ℝ)) ![Prod.fst, Prod.snd] x y
        = planeCost x y) ∧
      (∑ i : Fin 2, (1 : ℝ) * |(Ens.dirac ((0 : ℝ), (0 : ℝ))).expect (![Prod.fst, Prod.snd] i)
        - (Ens.dirac ((1 : ℝ), (1 : ℝ))).expect (![Prod.fst, Prod.snd] i)|) = 2 ∧
      transportCost planeCost (Ens.dirac ((0 : ℝ), (0 : ℝ))) (Ens.dirac ((1 : ℝ), (1 : ℝ)))
        = 2 ∧
      |(Ens.dirac ((0 : ℝ), (0 : ℝ))).expect Prod.fst
        - (Ens.dirac ((1 : ℝ), (1 : ℝ))).expect Prod.fst| = 1 := by
  refine ⟨fun x y => ?_, ?_, ?_, ?_⟩
  · simp [featureCost, planeCost, Fin.sum_univ_two]
  · norm_num [Fin.sum_univ_two]
  · rw [transportCost_dirac]
    norm_num [planeCost]
  · norm_num

end IDR
