import RequestProject.Physics.PartialCalculus

/-!
# Part CXLIII — Second-order partial calculus and the radial monomials

To solve genuine partial differential equations of continuum electrostatics and
hydrodynamics one needs to differentiate fields of the form `x_j x_a ‖x‖^k` twice and to
add the results.  This file supplies exactly that:

* `HasP2DerivAt` and `HasLaplacianAt`, predicates carrying the second partial derivative
  and the Laplacian together with the witnesses needed to add and rescale them,
* the exact first and second partial derivatives of the monomials `‖y‖^k`, `y_j ‖y‖^k`,
  `y_j y_a ‖y‖^k`, `y_j y_a y_b ‖y‖^k` at every `x ≠ 0`,
* the resulting exact Laplacians in **any** dimension `n`:
  `Δ‖x‖^k = k(n+k-2)‖x‖^{k-2}`, `Δ(x_j‖x‖^k) = k(n+k) x_j ‖x‖^{k-2}` and
  `Δ(x_j x_a ‖x‖^k) = 2δ_{ja}‖x‖^k + k(n+k+2) x_j x_a ‖x‖^{k-2}`.

These are the exact building blocks from which the Stokeslet, the pressure dipole and the
Coulomb/Born kernels are assembled.
-/

noncomputable section
namespace RequestProject.Physics
open scoped RealInnerProductSpace
open Real Filter Topology

variable {n : ℕ}

/-! ### Second-order partial derivatives -/

/-- `u` has an `i`-th second partial derivative equal to `d` at `x`: there is a field `g`
which is the `i`-th partial derivative of `u` near `x`, and whose own `i`-th partial
derivative at `x` is `d`. -/
def HasP2DerivAt (u : Sp n → ℝ) (i : Fin n) (d : ℝ) (x : Sp n) : Prop :=
  ∃ g : Sp n → ℝ, (∀ᶠ y in 𝓝 x, HasPDerivAt u i (g y) y) ∧ HasPDerivAt g i d x

namespace HasP2DerivAt

variable {u v : Sp n → ℝ} {i : Fin n} {d e : ℝ} {x : Sp n}

lemma pderiv2_eq (h : HasP2DerivAt u i d x) : pderiv2 i u x = d := by
  obtain ⟨g, hg, hgd⟩ := h
  rw [pderiv2_eq_pderiv hg, hgd.pderiv_eq]

protected lemma add (hu : HasP2DerivAt u i d x) (hv : HasP2DerivAt v i e x) :
    HasP2DerivAt (fun y => u y + v y) i (d + e) x := by
  obtain ⟨g, hg, hgd⟩ := hu
  obtain ⟨h, hh, hhd⟩ := hv
  refine ⟨fun y => g y + h y, ?_, HasPDerivAt.add hgd hhd⟩
  filter_upwards [hg, hh] with y hy hy'
  exact HasPDerivAt.add hy hy'

protected lemma const_mul (c : ℝ) (hu : HasP2DerivAt u i d x) :
    HasP2DerivAt (fun y => c * u y) i (c * d) x := by
  obtain ⟨g, hg, hgd⟩ := hu
  refine ⟨fun y => c * g y, ?_, HasPDerivAt.const_mul c hgd⟩
  filter_upwards [hg] with y hy
  exact HasPDerivAt.const_mul c hy

protected lemma sum {ι : Type*} {s : Finset ι} {U : ι → Sp n → ℝ} {D : ι → ℝ}
    (h : ∀ k ∈ s, HasP2DerivAt (U k) i (D k) x) :
    HasP2DerivAt (fun y => ∑ k ∈ s, U k y) i (∑ k ∈ s, D k) x := by
  classical
  induction s using Finset.induction with
  | empty =>
      refine ⟨fun _ => 0, ?_, by simpa using HasPDerivAt.const (n := n) (i := i) (x := x) 0⟩
      filter_upwards with y
      simpa using HasPDerivAt.const (n := n) (i := i) (x := y) 0
  | insert a s ha ih =>
      have hstep := HasP2DerivAt.add (h a (Finset.mem_insert_self a s))
        (ih fun k hk => h k (Finset.mem_insert_of_mem hk))
      simp only [Finset.sum_insert ha]
      exact hstep

end HasP2DerivAt

/-- `u` has Laplacian `d` at `x`. -/
def HasLaplacianAt (u : Sp n → ℝ) (d : ℝ) (x : Sp n) : Prop :=
  ∃ D : Fin n → ℝ, (∀ i, HasP2DerivAt u i (D i) x) ∧ ∑ i, D i = d

namespace HasLaplacianAt

variable {u v : Sp n → ℝ} {d e : ℝ} {x : Sp n}

lemma laplacian_eq (h : HasLaplacianAt u d x) : laplacian u x = d := by
  obtain ⟨D, hD, hsum⟩ := h
  rw [laplacian, ← hsum]
  exact Finset.sum_congr rfl fun i _ => (hD i).pderiv2_eq

protected lemma add (hu : HasLaplacianAt u d x) (hv : HasLaplacianAt v e x) :
    HasLaplacianAt (fun y => u y + v y) (d + e) x := by
  obtain ⟨D, hD, hs⟩ := hu
  obtain ⟨E, hE, ht⟩ := hv
  exact ⟨fun i => D i + E i, fun i => (hD i).add (hE i), by
    rw [Finset.sum_add_distrib, hs, ht]⟩

protected lemma const_mul (c : ℝ) (hu : HasLaplacianAt u d x) :
    HasLaplacianAt (fun y => c * u y) (c * d) x := by
  obtain ⟨D, hD, hs⟩ := hu
  exact ⟨fun i => c * D i, fun i => (hD i).const_mul c, by rw [← Finset.mul_sum, hs]⟩

protected lemma sum {ι : Type*} {s : Finset ι} {U : ι → Sp n → ℝ} {D : ι → ℝ}
    (h : ∀ k ∈ s, HasLaplacianAt (U k) (D k) x) :
    HasLaplacianAt (fun y => ∑ k ∈ s, U k y) (∑ k ∈ s, D k) x := by
  classical
  induction s using Finset.induction with
  | empty =>
      refine ⟨fun _ => 0, fun i => ⟨fun _ => 0, ?_, ?_⟩, by simp⟩
      · filter_upwards with y
        simpa using HasPDerivAt.const (n := n) (i := i) (x := y) 0
      · simpa using HasPDerivAt.const (n := n) (i := i) (x := x) 0
  | insert a s ha ih =>
      have hstep := HasLaplacianAt.add (h a (Finset.mem_insert_self a s))
        (ih fun k hk => h k (Finset.mem_insert_of_mem hk))
      simp only [Finset.sum_insert ha]
      exact hstep

end HasLaplacianAt

/-! ### Derivatives of the elementary radial monomials -/

section Monomials

variable {x : Sp n}

/-- `∂ᵢ ‖y‖ᵏ = k xᵢ ‖x‖^{k-2}`. -/
lemma hasPDerivAt_mono0 (k : ℤ) (hx : x ≠ 0) (i : Fin n) :
    HasPDerivAt (fun y : Sp n => ‖y‖ ^ k) i ((k : ℝ) * x i * ‖x‖ ^ (k - 2)) x := by
  have hr : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  have hf : HasDerivAt (fun t : ℝ => t ^ k) ((k : ℝ) * ‖x‖ ^ (k - 1)) ‖x‖ :=
    hasDerivAt_zpow k ‖x‖ (Or.inl hr)
  have h := HasPDerivAt.comp hf (hasPDerivAt_norm hx i)
  convert h using 1
  rw [show k - 2 = (k - 1) - 1 by ring, zpow_sub_one₀ hr]
  field_simp

/-- `∂ᵢ (y_j ‖y‖ᵏ)`. -/
lemma hasPDerivAt_mono1 (j : Fin n) (k : ℤ) (hx : x ≠ 0) (i : Fin n) :
    HasPDerivAt (fun y : Sp n => y j * ‖y‖ ^ k) i
      ((if j = i then (1 : ℝ) else 0) * ‖x‖ ^ k
        + (k : ℝ) * x j * x i * ‖x‖ ^ (k - 2)) x := by
  have h := HasPDerivAt.mul (hasPDerivAt_coord i j x) (hasPDerivAt_mono0 k hx i)
  convert h using 1
  ring

/-- `∂ᵢ (y_j y_a ‖y‖ᵏ)`. -/
lemma hasPDerivAt_mono2 (j a : Fin n) (k : ℤ) (hx : x ≠ 0) (i : Fin n) :
    HasPDerivAt (fun y : Sp n => y j * y a * ‖y‖ ^ k) i
      ((if j = i then (1 : ℝ) else 0) * x a * ‖x‖ ^ k
        + (if a = i then (1 : ℝ) else 0) * x j * ‖x‖ ^ k
        + (k : ℝ) * x j * x a * x i * ‖x‖ ^ (k - 2)) x := by
  have h := HasPDerivAt.mul (HasPDerivAt.mul (hasPDerivAt_coord i j x) (hasPDerivAt_coord i a x))
    (hasPDerivAt_mono0 k hx i)
  convert h using 1
  ring

/-- `∂ᵢ (y_j y_a y_b ‖y‖ᵏ)`. -/
lemma hasPDerivAt_mono3 (j a b : Fin n) (k : ℤ) (hx : x ≠ 0) (i : Fin n) :
    HasPDerivAt (fun y : Sp n => y j * y a * y b * ‖y‖ ^ k) i
      ((if j = i then (1 : ℝ) else 0) * x a * x b * ‖x‖ ^ k
        + (if a = i then (1 : ℝ) else 0) * x j * x b * ‖x‖ ^ k
        + (if b = i then (1 : ℝ) else 0) * x j * x a * ‖x‖ ^ k
        + (k : ℝ) * x j * x a * x b * x i * ‖x‖ ^ (k - 2)) x := by
  have h := HasPDerivAt.mul
    (HasPDerivAt.mul (HasPDerivAt.mul (hasPDerivAt_coord i j x) (hasPDerivAt_coord i a x))
      (hasPDerivAt_coord i b x)) (hasPDerivAt_mono0 k hx i)
  convert h using 1
  ring

end Monomials

/-! ### Second derivatives and Laplacians of the radial monomials -/

section Monomials2

variable {x : Sp n}

lemma eventually_ne_zero (hx : x ≠ 0) : ∀ᶠ y : Sp n in 𝓝 x, y ≠ 0 := by
  have h : ({(0 : Sp n)}ᶜ : Set (Sp n)) ∈ 𝓝 x :=
    isOpen_compl_singleton.mem_nhds (by simpa using hx)
  filter_upwards [h] with y hy
  simpa using hy

lemma sum_coord_sq (x : Sp n) : ∑ i, (x i) ^ 2 = ‖x‖ ^ 2 := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (Finset.sum_nonneg fun i _ => by positivity)]
  simp

lemma sum_delta (j : Fin n) (f : Fin n → ℝ) :
    ∑ i, (if j = i then (1 : ℝ) else 0) * f i = f j := by
  simp

lemma sum_delta2 (j a : Fin n) :
    ∑ i : Fin n, (if j = i then (1 : ℝ) else 0) * (if a = i then (1 : ℝ) else 0)
      = if j = a then (1 : ℝ) else 0 := by
  simp [Finset.sum_ite_eq]

lemma norm_zpow_shift (hx : x ≠ 0) (k : ℤ) : ‖x‖ ^ 2 * ‖x‖ ^ (k - 4) = ‖x‖ ^ (k - 2) := by
  have hr : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  rw [show (k - 2) = 2 + (k - 4) by ring, zpow_add₀ hr]
  congr 1

/-- Second partial derivative of `‖y‖ᵏ`. -/
lemma hasP2DerivAt_mono0 (k : ℤ) (hx : x ≠ 0) (i : Fin n) :
    HasP2DerivAt (fun y : Sp n => ‖y‖ ^ k) i
      ((k : ℝ) * ‖x‖ ^ (k - 2) + (k : ℝ) * ((k : ℝ) - 2) * (x i) ^ 2 * ‖x‖ ^ (k - 4)) x := by
  refine ⟨fun y => (k : ℝ) * (y i * ‖y‖ ^ (k - 2)), ?_, ?_⟩
  · filter_upwards [eventually_ne_zero hx] with y hy
    have h := hasPDerivAt_mono0 k hy i
    convert h using 1
    ring
  · have h := HasPDerivAt.const_mul (k : ℝ) (hasPDerivAt_mono1 i (k - 2) hx i)
    convert h using 1
    rw [if_pos (rfl : i = i)]
    push_cast [show k - 2 - 2 = k - 4 by ring]
    ring

/-- Second partial derivative of `y_j ‖y‖ᵏ`. -/
lemma hasP2DerivAt_mono1 (j : Fin n) (k : ℤ) (hx : x ≠ 0) (i : Fin n) :
    HasP2DerivAt (fun y : Sp n => y j * ‖y‖ ^ k) i
      (2 * (if j = i then (1 : ℝ) else 0) * (k : ℝ) * x i * ‖x‖ ^ (k - 2)
        + (k : ℝ) * x j * ‖x‖ ^ (k - 2)
        + (k : ℝ) * ((k : ℝ) - 2) * x j * (x i) ^ 2 * ‖x‖ ^ (k - 4)) x := by
  refine ⟨fun y => (if j = i then (1 : ℝ) else 0) * ‖y‖ ^ k
      + (k : ℝ) * (y j * y i * ‖y‖ ^ (k - 2)), ?_, ?_⟩
  · filter_upwards [eventually_ne_zero hx] with y hy
    have h := hasPDerivAt_mono1 j k hy i
    convert h using 1
    ring
  · have h := HasPDerivAt.add
      (HasPDerivAt.const_mul (if j = i then (1 : ℝ) else 0) (hasPDerivAt_mono0 k hx i))
      (HasPDerivAt.const_mul (k : ℝ) (hasPDerivAt_mono2 j i (k - 2) hx i))
    convert h using 1
    rw [if_pos (rfl : i = i)]
    push_cast [show k - 2 - 2 = k - 4 by ring]
    ring

/-- Second partial derivative of `y_j y_a ‖y‖ᵏ`. -/
lemma hasP2DerivAt_mono2 (j a : Fin n) (k : ℤ) (hx : x ≠ 0) (i : Fin n) :
    HasP2DerivAt (fun y : Sp n => y j * y a * ‖y‖ ^ k) i
      (2 * (if j = i then (1 : ℝ) else 0) * (if a = i then (1 : ℝ) else 0) * ‖x‖ ^ k
        + 2 * (k : ℝ) * (if j = i then (1 : ℝ) else 0) * x a * x i * ‖x‖ ^ (k - 2)
        + 2 * (k : ℝ) * (if a = i then (1 : ℝ) else 0) * x j * x i * ‖x‖ ^ (k - 2)
        + (k : ℝ) * x j * x a * ‖x‖ ^ (k - 2)
        + (k : ℝ) * ((k : ℝ) - 2) * x j * x a * (x i) ^ 2 * ‖x‖ ^ (k - 4)) x := by
  refine ⟨fun y => (if j = i then (1 : ℝ) else 0) * (y a * ‖y‖ ^ k)
      + (if a = i then (1 : ℝ) else 0) * (y j * ‖y‖ ^ k)
      + (k : ℝ) * (y j * y a * y i * ‖y‖ ^ (k - 2)), ?_, ?_⟩
  · filter_upwards [eventually_ne_zero hx] with y hy
    have h := hasPDerivAt_mono2 j a k hy i
    convert h using 1
    ring
  · have h := HasPDerivAt.add (HasPDerivAt.add
      (HasPDerivAt.const_mul (if j = i then (1 : ℝ) else 0) (hasPDerivAt_mono1 a k hx i))
      (HasPDerivAt.const_mul (if a = i then (1 : ℝ) else 0) (hasPDerivAt_mono1 j k hx i)))
      (HasPDerivAt.const_mul (k : ℝ) (hasPDerivAt_mono3 j a i (k - 2) hx i))
    convert h using 1
    rw [if_pos (rfl : i = i)]
    push_cast [show k - 2 - 2 = k - 4 by ring]
    ring

/-- **Laplacian of the radial power** `Δ ‖x‖ᵏ = k (n + k - 2) ‖x‖^{k-2}`. -/
theorem hasLaplacianAt_norm_zpow (k : ℤ) (hx : x ≠ 0) :
    HasLaplacianAt (fun y : Sp n => ‖y‖ ^ k)
      ((k : ℝ) * ((n : ℝ) + (k : ℝ) - 2) * ‖x‖ ^ (k - 2)) x := by
  refine ⟨_, fun i => hasP2DerivAt_mono0 k hx i, ?_⟩
  have hcong : ∀ i : Fin n,
      (k : ℝ) * ‖x‖ ^ (k - 2) + (k : ℝ) * ((k : ℝ) - 2) * (x i) ^ 2 * ‖x‖ ^ (k - 4)
        = (k : ℝ) * ‖x‖ ^ (k - 2) + ((k : ℝ) * ((k : ℝ) - 2) * ‖x‖ ^ (k - 4)) * (x i) ^ 2 :=
    fun i => by ring
  rw [Finset.sum_congr rfl (fun i _ => hcong i), Finset.sum_add_distrib, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, ← Finset.mul_sum, sum_coord_sq]
  have hs : ‖x‖ ^ (k - 4) * ‖x‖ ^ 2 = ‖x‖ ^ (k - 2) := by
    rw [mul_comm]; exact norm_zpow_shift hx k
  calc (n : ℝ) * ((k : ℝ) * ‖x‖ ^ (k - 2))
        + (k : ℝ) * ((k : ℝ) - 2) * ‖x‖ ^ (k - 4) * ‖x‖ ^ 2
      = (n : ℝ) * ((k : ℝ) * ‖x‖ ^ (k - 2))
        + (k : ℝ) * ((k : ℝ) - 2) * (‖x‖ ^ (k - 4) * ‖x‖ ^ 2) := by ring
    _ = (k : ℝ) * ((n : ℝ) + (k : ℝ) - 2) * ‖x‖ ^ (k - 2) := by rw [hs]; ring

/-- **Laplacian of `x_j ‖x‖ᵏ`.** -/
theorem hasLaplacianAt_mono1 (j : Fin n) (k : ℤ) (hx : x ≠ 0) :
    HasLaplacianAt (fun y : Sp n => y j * ‖y‖ ^ k)
      ((k : ℝ) * ((n : ℝ) + (k : ℝ)) * x j * ‖x‖ ^ (k - 2)) x := by
  refine ⟨_, fun i => hasP2DerivAt_mono1 j k hx i, ?_⟩
  have hs : ‖x‖ ^ (k - 4) * ‖x‖ ^ 2 = ‖x‖ ^ (k - 2) := by
    rw [mul_comm]; exact norm_zpow_shift hx k
  have hcong : ∀ i : Fin n,
      2 * (if j = i then (1 : ℝ) else 0) * (k : ℝ) * x i * ‖x‖ ^ (k - 2)
        + (k : ℝ) * x j * ‖x‖ ^ (k - 2)
        + (k : ℝ) * ((k : ℝ) - 2) * x j * (x i) ^ 2 * ‖x‖ ^ (k - 4)
      = (if j = i then (1 : ℝ) else 0) * (2 * (k : ℝ) * x i * ‖x‖ ^ (k - 2))
        + (k : ℝ) * x j * ‖x‖ ^ (k - 2)
        + ((k : ℝ) * ((k : ℝ) - 2) * x j * ‖x‖ ^ (k - 4)) * (x i) ^ 2 := fun i => by ring
  rw [Finset.sum_congr rfl (fun i _ => hcong i)]
  simp only [Finset.sum_add_distrib]
  rw [sum_delta, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    ← Finset.mul_sum, sum_coord_sq]
  calc 2 * (k : ℝ) * x j * ‖x‖ ^ (k - 2) + (n : ℝ) * ((k : ℝ) * x j * ‖x‖ ^ (k - 2))
        + (k : ℝ) * ((k : ℝ) - 2) * x j * ‖x‖ ^ (k - 4) * ‖x‖ ^ 2
      = 2 * (k : ℝ) * x j * ‖x‖ ^ (k - 2) + (n : ℝ) * ((k : ℝ) * x j * ‖x‖ ^ (k - 2))
        + (k : ℝ) * ((k : ℝ) - 2) * x j * (‖x‖ ^ (k - 4) * ‖x‖ ^ 2) := by ring
    _ = (k : ℝ) * ((n : ℝ) + (k : ℝ)) * x j * ‖x‖ ^ (k - 2) := by rw [hs]; ring

/-- **Laplacian of `x_j x_a ‖x‖ᵏ`.** -/
theorem hasLaplacianAt_mono2 (j a : Fin n) (k : ℤ) (hx : x ≠ 0) :
    HasLaplacianAt (fun y : Sp n => y j * y a * ‖y‖ ^ k)
      (2 * (if j = a then (1 : ℝ) else 0) * ‖x‖ ^ k
        + (k : ℝ) * ((n : ℝ) + (k : ℝ) + 2) * x j * x a * ‖x‖ ^ (k - 2)) x := by
  refine ⟨_, fun i => hasP2DerivAt_mono2 j a k hx i, ?_⟩
  have hs : ‖x‖ ^ (k - 4) * ‖x‖ ^ 2 = ‖x‖ ^ (k - 2) := by
    rw [mul_comm]; exact norm_zpow_shift hx k
  have hcong : ∀ i : Fin n,
      2 * (if j = i then (1 : ℝ) else 0) * (if a = i then (1 : ℝ) else 0) * ‖x‖ ^ k
        + 2 * (k : ℝ) * (if j = i then (1 : ℝ) else 0) * x a * x i * ‖x‖ ^ (k - 2)
        + 2 * (k : ℝ) * (if a = i then (1 : ℝ) else 0) * x j * x i * ‖x‖ ^ (k - 2)
        + (k : ℝ) * x j * x a * ‖x‖ ^ (k - 2)
        + (k : ℝ) * ((k : ℝ) - 2) * x j * x a * (x i) ^ 2 * ‖x‖ ^ (k - 4)
      = (2 * ‖x‖ ^ k) * ((if j = i then (1 : ℝ) else 0) * (if a = i then (1 : ℝ) else 0))
        + (if j = i then (1 : ℝ) else 0) * (2 * (k : ℝ) * x a * x i * ‖x‖ ^ (k - 2))
        + (if a = i then (1 : ℝ) else 0) * (2 * (k : ℝ) * x j * x i * ‖x‖ ^ (k - 2))
        + (k : ℝ) * x j * x a * ‖x‖ ^ (k - 2)
        + ((k : ℝ) * ((k : ℝ) - 2) * x j * x a * ‖x‖ ^ (k - 4)) * (x i) ^ 2 := fun i => by ring
  rw [Finset.sum_congr rfl (fun i _ => hcong i)]
  simp only [Finset.sum_add_distrib]
  rw [← Finset.mul_sum, sum_delta2, sum_delta, sum_delta, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, ← Finset.mul_sum, sum_coord_sq]
  calc 2 * ‖x‖ ^ k * (if j = a then (1 : ℝ) else 0)
        + 2 * (k : ℝ) * x a * x j * ‖x‖ ^ (k - 2)
        + 2 * (k : ℝ) * x j * x a * ‖x‖ ^ (k - 2)
        + (n : ℝ) * ((k : ℝ) * x j * x a * ‖x‖ ^ (k - 2))
        + (k : ℝ) * ((k : ℝ) - 2) * x j * x a * ‖x‖ ^ (k - 4) * ‖x‖ ^ 2
      = 2 * ‖x‖ ^ k * (if j = a then (1 : ℝ) else 0)
        + 2 * (k : ℝ) * x a * x j * ‖x‖ ^ (k - 2)
        + 2 * (k : ℝ) * x j * x a * ‖x‖ ^ (k - 2)
        + (n : ℝ) * ((k : ℝ) * x j * x a * ‖x‖ ^ (k - 2))
        + (k : ℝ) * ((k : ℝ) - 2) * x j * x a * (‖x‖ ^ (k - 4) * ‖x‖ ^ 2) := by ring
    _ = 2 * (if j = a then (1 : ℝ) else 0) * ‖x‖ ^ k
        + (k : ℝ) * ((n : ℝ) + (k : ℝ) + 2) * x j * x a * ‖x‖ ^ (k - 2) := by rw [hs]; ring

end Monomials2

end RequestProject.Physics
