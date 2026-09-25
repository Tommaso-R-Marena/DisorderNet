/-
# Part CXI  Many electrons: where the two-state model comes from, and what it costs

Part CVIII (`RequestProject.BondBreaking`) writes down the two-state (diabatic) model of bond
making and breaking and says plainly what it leaves out: "a many-electron treatment of reactive
chemistry remains outside".  This file supplies one.

The system is the smallest one in which a covalent bond is a genuinely many-electron object: two
electrons in two spatial orbitals, i.e. four spin orbitals and all **six** two-electron
determinants,

    d₀ = (1↑,1↓)   d₁ = (2↑,2↓)   d₂ = (1↑,2↓)   d₃ = (1↓,2↑)   d₄ = (1↑,2↑)   d₅ = (1↓,2↓),

with hopping `t` between the two sites and on-site repulsion `U` — the Hubbard dimer, the standard
minimal model of a stretching bond.  `hub U t` is its Hamiltonian in that determinant basis, a
`6 × 6` real symmetric matrix.  Nothing here is a two-level idealisation: the ionic
configurations `d₀, d₁`, the covalent ones `d₂, d₃` and the two spin-polarised ones `d₄, d₅` are
all present and all coupled by the electron hops.

* `hub_symm`, `triplet_eigen_*` — the model is Hermitian, and the three triplet states are exact
  eigenvectors of energy `0`.
* `hub_ground_eigen` — the vector `(E, E, −2t, −2t, 0, 0)` is an exact eigenvector with eigenvalue
  `groundEnergy U t = (U − √(U² + 16t²))/2`, whose defining property is the characteristic relation
  `E² − U·E − 4t² = 0` (`groundEnergy_char`).
* `hub_qform_ge` — **and it is the ground state**: the quadratic form of `hub U t` dominates
  `groundEnergy U t` times the squared norm, at every one of the six-dimensional many-electron
  states.  This is the variational principle for this Hamiltonian, proved rather than quoted.
* `groundEnergy_eq_adiaLow` — **the two-state model of Part CVIII is exact here.**  The
  many-electron ground state energy is *identically* the lower adiabatic surface `adiaLow` of the
  two-state Hamiltonian with diabatic energies `U` and `0` and coupling `V = 2t`.  So the two-state
  model is not an ansatz bolted onto the chemistry: it is the exact projection of the many-electron
  problem onto the two states that the electrons actually mix, and `feshbach_exact` is the general
  statement — for *any* many-electron Hamiltonian in block form, the exact eigenvalue problem
  restricted to a chosen model space is the model-space Hamiltonian plus an energy-dependent
  self-energy `B (E − C)⁻¹ Bᵀ`, with no approximation whatsoever.
* `singlet_ground` — the ground state is a singlet, strictly below every triplet, whenever the
  hopping is nonzero: the bond is an electron-pairing effect, invisible to any one-electron picture.
* `superexchange` — **the strong-correlation law**: for `4|t| ≤ U` the ground state energy is
  `−4t²/U` to within `16t⁴/U³`.  The effective two-state coupling is second order in the hopping
  and inverse in the repulsion; this is the parameter a reactive force field would have to carry.
* `rhf_ge_ground`, `rhf_error_ge` — **and the single-determinant picture fails exactly where the
  chemistry is.**  Restricted Hartree–Fock (both electrons in the bonding orbital) has energy
  `U/2 − 2t`; it is variationally above the truth, and its error is at least `U/2 − 2t`, which
  tends to `U/2` as the bond stretches (`t → 0`) while the exact energy tends to `0`.  No
  reparametrisation of a one-determinant model removes a static error of half the on-site
  repulsion.

What this closes: the two-state model used for reactive chemistry is derived, not assumed, and the
price of the mean-field alternative is quantified.  What it does not claim: this is the minimal
many-electron model, not a full electronic structure theory for an arbitrary molecule.
-/
import Mathlib
import RequestProject.BondBreaking

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR.ManyElectron

open Matrix

/-! ## The general exact reduction: Feshbach/Löwdin downfolding

Any many-electron Hamiltonian, split into a model space `P` and the rest `Q`, is exactly
equivalent on the model space to `A + B (E − C)⁻¹ Bᵀ`.  The two-state model is the case
`|P| = 2`; the content of the theorem is that the reduction is *exact*, the price being that the
effective Hamiltonian depends on the eigenvalue it is solved for. -/

section Feshbach

variable {P Q : Type*} [Fintype P] [Fintype Q] [DecidableEq Q]

/-- **Exact elimination of the complementary space.**  If `(x, y)` is an eigenvector of the block
Hamiltonian with eigenvalue `E`, then the complementary component `y` is determined by `x`, and
`x` solves the model-space problem with the self-energy `B (E − C)⁻¹ Bᵀ` added. -/
theorem feshbach_exact (A : Matrix P P ℝ) (B : Matrix P Q ℝ) (C : Matrix Q Q ℝ)
    (E : ℝ) (x : P → ℝ) (y : Q → ℝ)
    [Invertible (E • (1 : Matrix Q Q ℝ) - C)]
    (heig : (fromBlocks A B Bᵀ C) *ᵥ (Sum.elim x y) = E • Sum.elim x y) :
    y = (⅟(E • (1 : Matrix Q Q ℝ) - C)) *ᵥ (Bᵀ *ᵥ x) ∧
      (A + B * (⅟(E • (1 : Matrix Q Q ℝ) - C)) * Bᵀ) *ᵥ x = E • x := by
  rw [fromBlocks_mulVec] at heig
  have hP : A *ᵥ x + B *ᵥ y = E • x := by
    funext i
    have h := congrFun heig (Sum.inl i)
    simpa using h
  have hQ : Bᵀ *ᵥ x + C *ᵥ y = E • y := by
    funext i
    have h := congrFun heig (Sum.inr i)
    simpa using h
  have hMy : (E • (1 : Matrix Q Q ℝ) - C) *ᵥ y = Bᵀ *ᵥ x := by
    have : (E • (1 : Matrix Q Q ℝ) - C) *ᵥ y = E • y - C *ᵥ y := by
      rw [sub_mulVec, Matrix.smul_mulVec, one_mulVec]
    rw [this, ← hQ]
    abel
  have hy : y = (⅟(E • (1 : Matrix Q Q ℝ) - C)) *ᵥ (Bᵀ *ᵥ x) := by
    rw [← hMy, mulVec_mulVec, invOf_mul_self, one_mulVec]
  refine ⟨hy, ?_⟩
  rw [add_mulVec, ← hP]
  congr 1
  rw [← mulVec_mulVec, ← mulVec_mulVec, ← hy]

end Feshbach

/-! ## The Hubbard dimer: two electrons, four spin orbitals, six determinants -/

/-- The Hamiltonian of the two-site, two-electron Hubbard model in the basis of the six
two-electron determinants `(1↑1↓, 2↑2↓, 1↑2↓, 1↓2↑, 1↑2↑, 1↓2↓)`: hopping `t` between the sites,
on-site repulsion `U`. -/
def hub (U t : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  !![U, 0, -t, -t, 0, 0;
     0, U, -t, -t, 0, 0;
     -t, -t, 0, 0, 0, 0;
     -t, -t, 0, 0, 0, 0;
     0, 0, 0, 0, 0, 0;
     0, 0, 0, 0, 0, 0]

/-- The Hamiltonian is symmetric. -/
theorem hub_symm (U t : ℝ) : (hub U t)ᵀ = hub U t := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [hub]

/-- The exact ground state energy of the Hubbard dimer. -/
noncomputable def groundEnergy (U t : ℝ) : ℝ := (U - Real.sqrt (U ^ 2 + 16 * t ^ 2)) / 2

lemma sqrt_sq_disc (U t : ℝ) : Real.sqrt (U ^ 2 + 16 * t ^ 2) ^ 2 = U ^ 2 + 16 * t ^ 2 :=
  Real.sq_sqrt (by positivity)

lemma sqrt_disc_nonneg (U t : ℝ) : 0 ≤ Real.sqrt (U ^ 2 + 16 * t ^ 2) := Real.sqrt_nonneg _

/-- **The characteristic relation.**  The ground state energy is the lower root of
`E² − U·E − 4t² = 0`. -/
theorem groundEnergy_char (U t : ℝ) :
    groundEnergy U t ^ 2 - U * groundEnergy U t - 4 * t ^ 2 = 0 := by
  have h := sqrt_sq_disc U t
  unfold groundEnergy
  nlinarith [h]

/-- The ground state energy is nonpositive when the repulsion is. -/
theorem groundEnergy_nonpos (U t : ℝ) : groundEnergy U t ≤ 0 := by
  have h := sqrt_sq_disc U t
  have h0 := sqrt_disc_nonneg U t
  have : U ≤ Real.sqrt (U ^ 2 + 16 * t ^ 2) := by nlinarith
  unfold groundEnergy
  linarith

/-- With a nonzero hopping the ground state energy is strictly negative: the bond binds. -/
theorem groundEnergy_neg {U t : ℝ} (ht : t ≠ 0) : groundEnergy U t < 0 := by
  have h := sqrt_sq_disc U t
  have h0 := sqrt_disc_nonneg U t
  have ht2 : 0 < t ^ 2 := by positivity
  have : U < Real.sqrt (U ^ 2 + 16 * t ^ 2) := by nlinarith
  unfold groundEnergy
  linarith

/-! ## The exact ground state -/

/-- The ground state of the Hubbard dimer in the determinant basis: a symmetric combination of
the two ionic and the two covalent configurations. -/
noncomputable def groundVec (U t : ℝ) : Fin 6 → ℝ :=
  ![groundEnergy U t, groundEnergy U t, -2 * t, -2 * t, 0, 0]

/-- **The ground state is an exact eigenvector.** -/
theorem hub_ground_eigen (U t : ℝ) :
    (hub U t) *ᵥ (groundVec U t) = (groundEnergy U t) • (groundVec U t) := by
  have hchar := groundEnergy_char U t
  funext i
  fin_cases i <;>
    simp [hub, groundVec, Matrix.mulVec, dotProduct, Fin.sum_univ_six] <;>
    nlinarith [hchar]

/-- The ground state vector is nonzero as soon as the hopping is. -/
theorem groundVec_ne_zero {U t : ℝ} (ht : t ≠ 0) : groundVec U t ≠ 0 := by
  intro h
  have := congrFun h 2
  simp [groundVec] at this
  exact ht this

/-! ## The triplet states -/

/-- The `Sz = +1` triplet determinant. -/
def tripletUp : Fin 6 → ℝ := ![0, 0, 0, 0, 1, 0]

/-- The `Sz = −1` triplet determinant. -/
def tripletDown : Fin 6 → ℝ := ![0, 0, 0, 0, 0, 1]

/-- The `Sz = 0` triplet: the antisymmetric covalent combination. -/
def tripletZero : Fin 6 → ℝ := ![0, 0, 1, -1, 0, 0]

theorem triplet_eigen_up (U t : ℝ) : (hub U t) *ᵥ tripletUp = (0 : ℝ) • tripletUp := by
  funext i; fin_cases i <;>
    simp [hub, tripletUp, Matrix.mulVec, dotProduct, Fin.sum_univ_six]

theorem triplet_eigen_down (U t : ℝ) : (hub U t) *ᵥ tripletDown = (0 : ℝ) • tripletDown := by
  funext i; fin_cases i <;>
    simp [hub, tripletDown, Matrix.mulVec, dotProduct, Fin.sum_univ_six]

theorem triplet_eigen_zero (U t : ℝ) : (hub U t) *ᵥ tripletZero = (0 : ℝ) • tripletZero := by
  funext i; fin_cases i <;>
    simp [hub, tripletZero, Matrix.mulVec, dotProduct, Fin.sum_univ_six]

/-! ## The variational principle for this Hamiltonian -/

/-- The energy expectation of a many-electron state (unnormalised). -/
noncomputable def energyForm (U t : ℝ) (v : Fin 6 → ℝ) : ℝ := v ⬝ᵥ ((hub U t) *ᵥ v)

lemma energyForm_eq (U t : ℝ) (v : Fin 6 → ℝ) :
    energyForm U t v =
      U * (v 0 ^ 2 + v 1 ^ 2) - 2 * t * (v 0 + v 1) * (v 2 + v 3) := by
  simp [energyForm, hub, Matrix.mulVec, dotProduct, Fin.sum_univ_six]
  ring

/-- **The variational principle.**  Every many-electron state has energy at least
`groundEnergy U t`; together with `hub_ground_eigen` this identifies the ground state energy
exactly. -/
theorem hub_qform_ge {U t : ℝ} (hU : 0 ≤ U) (v : Fin 6 → ℝ) :
    groundEnergy U t * (v ⬝ᵥ v) ≤ energyForm U t v := by
  set E := groundEnergy U t with hE
  have hchar := groundEnergy_char U t
  have hEle : E ≤ 0 := groundEnergy_nonpos U t
  have hUE : 0 ≤ U - E := by linarith
  rw [energyForm_eq]
  have hnorm : v ⬝ᵥ v = v 0 ^ 2 + v 1 ^ 2 + v 2 ^ 2 + v 3 ^ 2 + v 4 ^ 2 + v 5 ^ 2 := by
    simp [dotProduct, Fin.sum_univ_six]; ring
  rw [hnorm]
  set s := v 0 + v 1 with hs
  set u := v 2 + v 3 with hu
  -- the two-dimensional estimate, with the discriminant vanishing exactly
  have key : 0 ≤ (U - E) * s ^ 2 / 2 - 2 * t * s * u + (-E) * u ^ 2 / 2 := by
    rcases eq_or_lt_of_le hUE with h | h
    · -- degenerate case: `U = E` forces `E = 0`, then `t = 0`
      rw [← hE] at hchar
      have h1 : U - E = 0 := h.symm
      have hE0 : E = 0 := by linarith
      have hUz : U = 0 := by linarith
      have ht0 : t = 0 := by
        have ht2 : t ^ 2 = 0 := by
          rw [hE0, hUz] at hchar; linarith
        exact pow_eq_zero_iff (n := 2) (by norm_num) |>.1 ht2
      rw [h1, hE0, ht0]
      norm_num
    · have hsq : 0 ≤ ((U - E) * s - 2 * t * u) ^ 2 := sq_nonneg _
      have hdisc : (-E) * (U - E) = 4 * t ^ 2 := by nlinarith [hchar]
      have expand : ((U - E) * s - 2 * t * u) ^ 2
          = 2 * (U - E) * ((U - E) * s ^ 2 / 2 - 2 * t * s * u + (-E) * u ^ 2 / 2) := by
        nlinarith [hdisc]
      nlinarith [hsq, expand, h]
  nlinarith [key, sq_nonneg (v 0 - v 1), sq_nonneg (v 2 - v 3), sq_nonneg (v 4), sq_nonneg (v 5),
    hEle, hUE]

/-- **The ground state is a singlet.**  With a nonzero hopping the ground state energy is strictly
below the triplet energy `0`. -/
theorem singlet_ground {U t : ℝ} (ht : t ≠ 0) :
    groundEnergy U t < 0 ∧ energyForm U t tripletUp = 0 ∧ energyForm U t tripletZero = 0 := by
  refine ⟨groundEnergy_neg ht, ?_, ?_⟩ <;>
    simp [energyForm_eq, tripletUp, tripletZero]

/-! ## The two-state model is exact -/

/-- **The two-state model of Part CVIII is the exact many-electron answer.**  The ground state
energy of the six-determinant Hubbard dimer is identically the lower adiabatic surface of the
two-state Hamiltonian with diabatic energies `U` (ionic) and `0` (covalent) and coupling
`V = 2t`. -/
theorem groundEnergy_eq_adiaLow (U t : ℝ) :
    groundEnergy U t = IDR.BondBreaking.adiaLow U 0 (2 * t) := by
  unfold groundEnergy IDR.BondBreaking.adiaLow IDR.BondBreaking.halfGap
  have hpos : ((U - 0) / 2) ^ 2 + (2 * t) ^ 2 = (U ^ 2 + 16 * t ^ 2) / 2 ^ 2 := by ring
  rw [hpos, Real.sqrt_div' _ (by positivity), Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 2)]
  ring

/-! ## Strong correlation: the superexchange law -/

/-- **The superexchange law.**  In the strongly correlated regime `4|t| ≤ U` the exact ground
state energy is `−4t²/U` up to `16t⁴/U³`: the effective two-state coupling is second order in the
hopping and inverse in the on-site repulsion. -/
theorem superexchange {U t : ℝ} (hU : 0 < U) (ht : 4 * |t| ≤ U) :
    |groundEnergy U t - (-4 * t ^ 2 / U)| ≤ 16 * t ^ 4 / U ^ 3 := by
  have hs := sqrt_sq_disc U t
  have hs0 := sqrt_disc_nonneg U t
  set S := Real.sqrt (U ^ 2 + 16 * t ^ 2) with hSdef
  have habs : 16 * t ^ 2 ≤ U ^ 2 := by
    have h1 : |t| ≤ U / 4 := by linarith
    have h2 : |t| ^ 2 ≤ (U / 4) ^ 2 := by nlinarith [abs_nonneg t]
    rw [sq_abs] at h2
    nlinarith
  -- upper bound on the square root
  have hUp : S ≤ U + 8 * t ^ 2 / U := by
    have hnn : (0 : ℝ) ≤ U + 8 * t ^ 2 / U := by positivity
    have hexp : (U + 8 * t ^ 2 / U) ^ 2 = U ^ 2 + 16 * t ^ 2 + 64 * t ^ 4 / U ^ 2 := by
      field_simp; ring
    calc S ≤ Real.sqrt ((U + 8 * t ^ 2 / U) ^ 2) := by
          apply Real.sqrt_le_sqrt
          rw [hexp]
          have : (0 : ℝ) ≤ 64 * t ^ 4 / U ^ 2 := by positivity
          linarith
      _ = U + 8 * t ^ 2 / U := Real.sqrt_sq hnn
  -- lower bound on the square root
  have hLo : U + 8 * t ^ 2 / U - 32 * t ^ 4 / U ^ 3 ≤ S := by
    rcases le_or_gt (U + 8 * t ^ 2 / U - 32 * t ^ 4 / U ^ 3) 0 with hneg | hpos
    · linarith
    · rw [hSdef]
      refine (Real.le_sqrt hpos.le (by positivity)).2 ?_
      have hexp : (U + 8 * t ^ 2 / U - 32 * t ^ 4 / U ^ 3) ^ 2
          = U ^ 2 + 16 * t ^ 2 - 512 * t ^ 6 / U ^ 4 + 1024 * t ^ 8 / U ^ 6 := by
        field_simp; ring
      have hcmp : 1024 * t ^ 8 / U ^ 6 ≤ 512 * t ^ 6 / U ^ 4 := by
        rw [div_le_div_iff₀ (by positivity) (by positivity)]
        have h6 : (0 : ℝ) ≤ t ^ 6 * U ^ 4 := by positivity
        nlinarith [mul_nonneg h6 (by nlinarith [sq_nonneg t] : (0:ℝ) ≤ U ^ 2 - 2 * t ^ 2)]
      rw [hexp]
      linarith
  rw [abs_le]
  constructor
  · have hE : -4 * t ^ 2 / U ≤ groundEnergy U t := by
      rw [groundEnergy, ← hSdef, le_div_iff₀ (by norm_num : (0:ℝ) < 2)]
      have h8 : 8 * t ^ 2 / U * U = 8 * t ^ 2 := by field_simp
      have h4 : -4 * t ^ 2 / U * 2 = -(8 * t ^ 2 / U) := by ring
      rw [h4]
      linarith [hUp]
    have : (0 : ℝ) ≤ 16 * t ^ 4 / U ^ 3 := by positivity
    linarith
  · have hE : groundEnergy U t ≤ -4 * t ^ 2 / U + 16 * t ^ 4 / U ^ 3 := by
      rw [groundEnergy, ← hSdef, div_le_iff₀ (by norm_num : (0:ℝ) < 2)]
      have hrhs : (-4 * t ^ 2 / U + 16 * t ^ 4 / U ^ 3) * 2
          = -(8 * t ^ 2 / U) + 32 * t ^ 4 / U ^ 3 := by ring
      rw [hrhs]
      linarith [hLo]
    linarith

/-! ## The failure of the single-determinant picture -/

/-- The restricted Hartree–Fock state: both electrons in the bonding orbital.  In the determinant
basis this is the equal-weight combination of the four `Sz = 0` determinants. -/
def rhfVec : Fin 6 → ℝ := ![1, 1, 1, 1, 0, 0]

/-- The restricted Hartree–Fock energy, `U/2 − 2t`. -/
noncomputable def rhfEnergy (U t : ℝ) : ℝ := U / 2 - 2 * t

theorem rhf_rayleigh (U t : ℝ) : energyForm U t rhfVec = rhfEnergy U t * (rhfVec ⬝ᵥ rhfVec) := by
  simp [energyForm_eq, rhfEnergy, rhfVec, dotProduct, Fin.sum_univ_six]
  ring

/-- **Hartree–Fock is above the truth** (the variational principle applied to the single
determinant). -/
theorem rhf_ge_ground {U t : ℝ} (hU : 0 ≤ U) : groundEnergy U t ≤ rhfEnergy U t := by
  have h := hub_qform_ge (U := U) (t := t) hU rhfVec
  rw [rhf_rayleigh] at h
  have hn : (rhfVec ⬝ᵥ rhfVec) = 4 := by
    simp [rhfVec, dotProduct, Fin.sum_univ_six]
    norm_num
  rw [hn] at h
  linarith

/-- **Static correlation: the single-determinant error does not vanish as the bond stretches.**
The Hartree–Fock error is at least `U/2 − 2t`, so at dissociation (`t → 0`) it is half the on-site
repulsion while the exact energy tends to `0`. -/
theorem rhf_error_ge (U t : ℝ) :
    U / 2 - 2 * t ≤ rhfEnergy U t - groundEnergy U t := by
  have h := groundEnergy_nonpos U t
  unfold rhfEnergy
  linarith

/-- At dissociation the error is exactly half the on-site repulsion. -/
theorem rhf_error_dissociation {U : ℝ} (hU : 0 ≤ U) :
    rhfEnergy U 0 - groundEnergy U 0 = U / 2 := by
  have h : groundEnergy U 0 = 0 := by
    unfold groundEnergy
    rw [show U ^ 2 + 16 * (0:ℝ) ^ 2 = U ^ 2 by ring, Real.sqrt_sq hU]
    ring
  rw [h]
  unfold rhfEnergy
  ring

end IDR.ManyElectron
