import RequestProject.Physics.Laplacian

/-!
# Part CXXXVI — A calculus of partial derivatives on `EuclideanSpace ℝ (Fin n)`

`HasPDerivAt u i d x` says that the `i`-th partial derivative of the scalar field `u`
at `x` exists and equals `d`.  We prove the full set of algebraic rules (sum, product,
scalar multiple, quotient, chain rule, finite sums) plus the two geometric inputs that
continuum physics needs: the derivative of a coordinate function and the derivative of the
norm.  A second-order rule lets one compute `pderiv2` (and hence the Laplacian) of any
field whose first partial derivative is known on a neighbourhood.
-/

noncomputable section

namespace RequestProject.Physics

open scoped RealInnerProductSpace
open Real Filter Topology

variable {n : ℕ}

/-- `u` has `i`-th partial derivative `d` at `x`. -/
def HasPDerivAt (u : Sp n → ℝ) (i : Fin n) (d : ℝ) (x : Sp n) : Prop :=
  HasDerivAt (fun t : ℝ => u (x + t • EuclideanSpace.single i (1 : ℝ))) d 0

namespace HasPDerivAt

variable {u v : Sp n → ℝ} {i : Fin n} {d e : ℝ} {x : Sp n}

lemma pderiv_eq (h : HasPDerivAt u i d x) : pderiv i u x = d := h.deriv

protected lemma const (c : ℝ) : HasPDerivAt (fun _ : Sp n => c) i 0 x := by
  simpa [HasPDerivAt] using (hasDerivAt_const (0 : ℝ) c)

protected lemma add (hu : HasPDerivAt u i d x) (hv : HasPDerivAt v i e x) :
    HasPDerivAt (fun y => u y + v y) i (d + e) x := HasDerivAt.add hu hv

protected lemma neg (hu : HasPDerivAt u i d x) : HasPDerivAt (fun y => -u y) i (-d) x :=
  HasDerivAt.neg hu

protected lemma sub (hu : HasPDerivAt u i d x) (hv : HasPDerivAt v i e x) :
    HasPDerivAt (fun y => u y - v y) i (d - e) x := HasDerivAt.sub hu hv

protected lemma mul (hu : HasPDerivAt u i d x) (hv : HasPDerivAt v i e x) :
    HasPDerivAt (fun y => u y * v y) i (d * v x + u x * e) x := by
  have := HasDerivAt.mul hu hv
  simpa using this

protected lemma const_mul (c : ℝ) (hu : HasPDerivAt u i d x) :
    HasPDerivAt (fun y => c * u y) i (c * d) x := HasDerivAt.const_mul c hu

protected lemma mul_const (hu : HasPDerivAt u i d x) (c : ℝ) :
    HasPDerivAt (fun y => u y * c) i (d * c) x := HasDerivAt.mul_const hu c

protected lemma inv (hu : HasPDerivAt u i d x) (hx : u x ≠ 0) :
    HasPDerivAt (fun y => (u y)⁻¹) i (-d / (u x) ^ 2) x := by
  have := HasDerivAt.inv hu (by simpa using hx)
  simpa using this

protected lemma div (hu : HasPDerivAt u i d x) (hv : HasPDerivAt v i e x) (hx : v x ≠ 0) :
    HasPDerivAt (fun y => u y / v y) i ((d * v x - u x * e) / (v x) ^ 2) x := by
  have := HasDerivAt.div hu hv (by simpa using hx)
  simpa using this

/-- Chain rule with an outer function of one real variable. -/
protected lemma comp {f : ℝ → ℝ} {f' : ℝ} (hf : HasDerivAt f f' (u x))
    (hu : HasPDerivAt u i d x) : HasPDerivAt (fun y => f (u y)) i (f' * d) x := by
  have hu' : HasDerivAt (fun t : ℝ => u (x + t • EuclideanSpace.single i (1 : ℝ))) d 0 := hu
  have hf' : HasDerivAt f f' ((fun t : ℝ => u (x + t • EuclideanSpace.single i (1 : ℝ))) 0) := by
    simpa using hf
  simpa using hf'.comp 0 hu'

protected lemma pow (hu : HasPDerivAt u i d x) (m : ℕ) :
    HasPDerivAt (fun y => (u y) ^ m) i ((m : ℝ) * (u x) ^ (m - 1) * d) x := by
  have hf : HasDerivAt (fun z : ℝ => z ^ m) ((m : ℝ) * (u x) ^ (m - 1)) (u x) := by
    simpa using hasDerivAt_pow m (u x)
  simpa [mul_comm, mul_left_comm, mul_assoc] using
    (HasPDerivAt.comp (f := fun z : ℝ => z ^ m) hf hu)

protected lemma sum {ι : Type*} {s : Finset ι} {U : ι → Sp n → ℝ} {D : ι → ℝ}
    (h : ∀ k ∈ s, HasPDerivAt (U k) i (D k) x) :
    HasPDerivAt (fun y => ∑ k ∈ s, U k y) i (∑ k ∈ s, D k) x := by
  classical
  induction s using Finset.induction with
  | empty => simpa [HasPDerivAt] using (hasDerivAt_const (0 : ℝ) (0 : ℝ))
  | insert a s ha ih =>
      have hstep := HasPDerivAt.add (h a (Finset.mem_insert_self a s))
        (ih fun k hk => h k (Finset.mem_insert_of_mem hk))
      simp only [Finset.sum_insert ha]
      exact hstep

end HasPDerivAt

/-- The partial derivative of a coordinate function. -/
lemma hasPDerivAt_coord (i j : Fin n) (x : Sp n) :
    HasPDerivAt (fun y : Sp n => y j) i (if j = i then 1 else 0) x := by
  have hfun : ∀ t : ℝ, (x + t • EuclideanSpace.single i (1 : ℝ)) j
      = x j + t * (if j = i then 1 else 0) := by
    intro t
    by_cases h : j = i <;> simp [h, EuclideanSpace.single_apply]
  have : HasDerivAt (fun t : ℝ => x j + t * (if j = i then 1 else 0))
      (if j = i then 1 else 0) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (if j = i then (1 : ℝ) else 0)).const_add (x j)
  simpa [HasPDerivAt, hfun] using this

/-- The partial derivative of the norm, away from the origin. -/
lemma hasPDerivAt_norm {x : Sp n} (hx : x ≠ 0) (i : Fin n) :
    HasPDerivAt (fun y : Sp n => ‖y‖) i (x i / ‖x‖) x := by
  have hv : ‖EuclideanSpace.single i (1 : ℝ)‖ = 1 := by simp
  have h0 : x + (0 : ℝ) • EuclideanSpace.single i (1 : ℝ) ≠ 0 := by simpa using hx
  have h := hasDerivAt_norm_line (x := x) (v := EuclideanSpace.single i (1 : ℝ)) hv h0
  have hx0 : x + (0 : ℝ) • EuclideanSpace.single i (1 : ℝ) = x := by simp
  rw [hx0] at h
  have hin : ⟪x, EuclideanSpace.single i (1 : ℝ)⟫ = x i := by
    simp [EuclideanSpace.inner_single_right]
  rw [hin] at h
  simpa [HasPDerivAt] using h

/-- Second-order rule: if the `i`-th partial derivative of `u` equals `g` on a neighbourhood
of `x`, then the second partial derivative of `u` is the partial derivative of `g`. -/
lemma pderiv2_eq_pderiv {u g : Sp n → ℝ} {i : Fin n} {x : Sp n}
    (h : ∀ᶠ y in 𝓝 x, HasPDerivAt u i (g y) y) :
    pderiv2 i u x = pderiv i g x := by
  set v : Sp n := EuclideanSpace.single i (1 : ℝ) with hv
  have hcont : Continuous fun t : ℝ => x + t • v := by fun_prop
  have hmem : ∀ᶠ t in 𝓝 (0 : ℝ), HasPDerivAt u i (g (x + t • v)) (x + t • v) := by
    have htend : Filter.Tendsto (fun t : ℝ => x + t • v) (𝓝 0) (𝓝 x) := by
      simpa using hcont.tendsto (0 : ℝ)
    exact htend.eventually h
  have hEq : deriv (fun t : ℝ => u (x + t • v)) =ᶠ[𝓝 0] fun t : ℝ => g (x + t • v) := by
    filter_upwards [hmem] with t ht
    -- the derivative of the line function at `t` is the partial derivative at `x + t v`
    have hshift : (fun s : ℝ => u (x + s • v)) = fun s : ℝ => u ((x + t • v) + (s - t) • v) := by
      funext s
      congr 1
      module
    have ht' : HasDerivAt (fun s : ℝ => u ((x + t • v) + s • v)) (g (x + t • v)) 0 := ht
    have hsub : HasDerivAt (fun s : ℝ => s - t) 1 t := by
      simpa using (hasDerivAt_id t).sub_const t
    have ht'' : HasDerivAt (fun s : ℝ => u ((x + t • v) + s • v)) (g (x + t • v))
        ((fun s : ℝ => s - t) t) := by simpa using ht'
    have hshift' : HasDerivAt (fun s : ℝ => u ((x + t • v) + (s - t) • v)) (g (x + t • v)) t := by
      have hcomp := HasDerivAt.comp (h₂ := fun s : ℝ => u ((x + t • v) + s • v))
        (h := fun s : ℝ => s - t) t ht'' hsub
      simpa [Function.comp] using hcomp
    rw [hshift]
    exact hshift'.deriv
  unfold pderiv2 dirDeriv2
  rw [Filter.EventuallyEq.deriv_eq hEq]
  rfl

end RequestProject.Physics
