/-
# Part XXIV.2  The potential terms of a molecular-mechanics force field

This file is the scalar layer of the force field: the four functional forms that a
class-I (AMBER/CHARMM-style) potential is built from, together with everything about them
that the later files need.  Nothing here is a caricature: the Lennard-Jones term is the
smooth 12-6 potential (not a hard-sphere cut-off), the Coulomb term carries an explicit
relative permittivity, the bonded terms are harmonic in the internal coordinate, and the
torsion term is the standard cosine term.

* `lj`, `lj_eq_zero_iff_sigma`, `lj_neg_iff`, `lj_pos_iff`, `lj_ge_neg_eps`,
  `lj_min_at`, `lj_eq_neg_eps_iff` -- the 12-6 potential is zero exactly at the contact
  distance `sigma`, strictly repulsive inside it, strictly attractive outside it, bounded
  below by `-eps`, and attains `-eps` exactly at `2^(1/6) sigma`.
* `lj_ge_half_core`, `lj_unbounded_above` -- the core really is a core: below `sigma/2` the
  potential exceeds `2 eps (sigma/r)^12`, so it is unbounded above as `r -> 0`.  Excluded
  volume is a *consequence* of the force field, not an extra postulate.
* `pair_bddBelow` -- **the stability theorem**: for any charge product whatsoever, the sum
  of the 12-6 term and the (attractive or repulsive) Coulomb term is bounded below on
  `r > 0`, with an explicit bound.  This is what makes the Boltzmann-Gibbs measure of the
  later files exist: a Hamiltonian with `1/r` attractions and no repulsive core has none.
* `harmonic_nonneg`, `harmonic_eq_zero_iff`, `harmonic_strictMono` -- the bonded terms.
* `torsion_mem_Icc`, `torsion_periodic`, `torsion_eq_zero_iff` -- the torsion term is
  bounded in `[0, V]` and periodic with period `2 pi / n`.
* Continuity of every term away from `r = 0`.
-/
import Mathlib
import RequestProject.Context

namespace IDR

namespace Potentials

open Real

/-! ## Two elementary facts about sixth powers -/

lemma one_lt_pow6 {x : ℝ} (hx : 0 < x) : 1 < x ^ 6 ↔ 1 < x := by
  constructor
  · intro h
    by_contra hc
    push_neg at hc
    have : x ^ 6 ≤ 1 ^ 6 := pow_le_pow_left₀ hx.le hc 6
    simp at this; linarith
  · intro h
    have : (1:ℝ) ^ 6 < x ^ 6 := pow_lt_pow_left₀ h zero_le_one (by norm_num)
    simpa using this

lemma pow6_lt_one {x : ℝ} (hx : 0 < x) : x ^ 6 < 1 ↔ x < 1 := by
  constructor
  · intro h
    by_contra hc
    push_neg at hc
    have : (1:ℝ) ^ 6 ≤ x ^ 6 := pow_le_pow_left₀ zero_le_one hc 6
    simp at this; linarith
  · intro h
    have : x ^ 6 < 1 ^ 6 := pow_lt_pow_left₀ h hx.le (by norm_num)
    simpa using this

/-! ## Lennard-Jones 12-6 -/

/-- The Lennard-Jones 12-6 potential: `4 eps ((sigma/r)^12 - (sigma/r)^6)`. -/
noncomputable def lj (eps sigma r : ℝ) : ℝ :=
  4 * eps * ((sigma / r) ^ 12 - (sigma / r) ^ 6)

/-- The factored form in the reduced variable `u = (sigma/r)^6`. -/
lemma lj_eq_factor (eps sigma r : ℝ) :
    lj eps sigma r = 4 * eps * (sigma / r) ^ 6 * ((sigma / r) ^ 6 - 1) := by
  unfold lj; ring

lemma lj_sigma (eps sigma : ℝ) (hs : sigma ≠ 0) : lj eps sigma sigma = 0 := by
  unfold lj
  rw [div_self hs]
  norm_num

/-- The 12-6 potential vanishes exactly at the contact distance. -/
theorem lj_eq_zero_iff_sigma {eps sigma r : ℝ} (he : 0 < eps) (hs : 0 < sigma) (hr : 0 < r) :
    lj eps sigma r = 0 ↔ r = sigma := by
  have hu : 0 < sigma / r := div_pos hs hr
  constructor
  · intro h
    rw [lj_eq_factor] at h
    have hpos : (0:ℝ) < 4 * eps * (sigma / r) ^ 6 := by positivity
    have h6 : (sigma / r) ^ 6 - 1 = 0 := by
      rcases mul_eq_zero.mp h with h1 | h2
      · exact absurd h1 hpos.ne'
      · exact h2
    have hone : sigma / r = 1 := by
      by_contra hne
      rcases lt_or_gt_of_ne hne with hlt | hgt
      · have := (pow6_lt_one hu).mpr hlt; linarith
      · have := (one_lt_pow6 hu).mpr hgt; linarith
    field_simp at hone
    linarith
  · rintro rfl
    exact lj_sigma eps r hr.ne'

/-- Inside the contact distance the potential is strictly positive: a soft but genuine
excluded-volume core. -/
theorem lj_pos_iff {eps sigma r : ℝ} (he : 0 < eps) (hs : 0 < sigma) (hr : 0 < r) :
    0 < lj eps sigma r ↔ r < sigma := by
  have hu : 0 < sigma / r := div_pos hs hr
  have hp : (0:ℝ) < 4 * eps * (sigma / r) ^ 6 := by positivity
  rw [lj_eq_factor, mul_pos_iff]
  constructor
  · rintro (⟨_, hB⟩ | ⟨hA, _⟩)
    · have h1 : 1 < (sigma / r) ^ 6 := by linarith
      have h2 := (one_lt_pow6 hu).mp h1
      rw [lt_div_iff₀ hr] at h2
      linarith
    · linarith
  · intro hrs
    refine Or.inl ⟨hp, ?_⟩
    have h2 : 1 < sigma / r := by rw [lt_div_iff₀ hr]; linarith
    have := (one_lt_pow6 hu).mpr h2
    linarith

/-- Outside the contact distance the potential is strictly negative: the van der Waals
attraction is present at all larger separations. -/
theorem lj_neg_iff {eps sigma r : ℝ} (he : 0 < eps) (hs : 0 < sigma) (hr : 0 < r) :
    lj eps sigma r < 0 ↔ sigma < r := by
  have hu : 0 < sigma / r := div_pos hs hr
  have hp : (0:ℝ) < 4 * eps * (sigma / r) ^ 6 := by positivity
  rw [lj_eq_factor]
  constructor
  · intro h
    have hB : (sigma / r) ^ 6 - 1 < 0 := by nlinarith
    have := (pow6_lt_one hu).mp (by linarith)
    rw [div_lt_one hr] at this
    exact this
  · intro h
    have h1 : sigma / r < 1 := (div_lt_one hr).mpr h
    have := (pow6_lt_one hu).mpr h1
    nlinarith

/-- **The well depth.**  The potential is bounded below by `-eps`. -/
theorem lj_ge_neg_eps {eps sigma r : ℝ} (he : 0 ≤ eps) : -eps ≤ lj eps sigma r := by
  unfold lj
  nlinarith [sq_nonneg ((sigma / r) ^ 6 - 1 / 2), he, sq_nonneg ((sigma / r) ^ 6)]

/-- Equality holds exactly where `(sigma/r)^6 = 1/2`, i.e. at `r = 2^(1/6) sigma`. -/
theorem lj_eq_neg_eps_iff {eps sigma r : ℝ} (he : 0 < eps) :
    lj eps sigma r = -eps ↔ (sigma / r) ^ 6 = 1 / 2 := by
  unfold lj
  constructor
  · intro h
    have hsq : (2 * (sigma / r) ^ 6 - 1) ^ 2 = 0 := by nlinarith
    have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp hsq
    linarith
  · intro h
    have h12 : (sigma / r) ^ 12 = ((sigma / r) ^ 6) ^ 2 := by ring
    rw [h12, h]; ring

/-- The minimum is attained at `r = 2^(1/6) sigma`, with value exactly `-eps`. -/
theorem lj_min_at {eps sigma : ℝ} (he : 0 < eps) (hs : 0 < sigma) :
    lj eps sigma ((2:ℝ) ^ ((1:ℝ)/6) * sigma) = -eps := by
  rw [lj_eq_neg_eps_iff he]
  have h2 : (0:ℝ) < (2:ℝ) ^ ((1:ℝ)/6) := Real.rpow_pos_of_pos (by norm_num) _
  have hquot : sigma / ((2:ℝ) ^ ((1:ℝ)/6) * sigma) = ((2:ℝ) ^ ((1:ℝ)/6))⁻¹ := by
    field_simp
  have h6 : ((2:ℝ) ^ ((1:ℝ)/6)) ^ (6:ℕ) = 2 := by
    rw [← Real.rpow_natCast ((2:ℝ) ^ ((1:ℝ)/6)) 6, ← Real.rpow_mul (by norm_num)]
    norm_num
  rw [hquot, inv_pow, h6]
  norm_num

/-- **The hard core, quantitatively.**  Below `sigma/2` the repulsion already dominates:
the potential exceeds `2 eps (sigma/r)^12`. -/
theorem lj_ge_half_core {eps sigma r : ℝ} (he : 0 < eps) (hr : 0 < r)
    (hrs : r ≤ sigma / 2) : 2 * eps * (sigma / r) ^ 12 ≤ lj eps sigma r := by
  have hu : (2:ℝ) ≤ sigma / r := by rw [le_div_iff₀ hr]; linarith
  have h6 : (64:ℝ) ≤ (sigma / r) ^ 6 := by
    calc (64:ℝ) = 2 ^ 6 := by norm_num
      _ ≤ (sigma / r) ^ 6 := pow_le_pow_left₀ (by norm_num) hu 6
  have h12 : (sigma / r) ^ 12 = ((sigma / r) ^ 6) ^ 2 := by ring
  rw [lj_eq_factor, h12]
  nlinarith [mul_nonneg (mul_nonneg he.le (by linarith : (0:ℝ) ≤ (sigma / r) ^ 6))
    (by linarith : (0:ℝ) ≤ (sigma / r) ^ 6 - 2)]

/-- The 12-6 potential is unbounded above: overlapping atoms cost unboundedly much. -/
theorem lj_unbounded_above {eps sigma : ℝ} (he : 0 < eps) (hs : 0 < sigma) (M : ℝ) :
    ∃ r : ℝ, 0 < r ∧ M < lj eps sigma r := by
  obtain ⟨K, hK⟩ := exists_nat_gt (M / (2 * eps))
  have hKnn : (0:ℝ) ≤ (K:ℝ) := Nat.cast_nonneg K
  set x : ℝ := 2 * ((K:ℝ) + 1) with hx
  have hx2 : (2:ℝ) ≤ x := by rw [hx]; linarith
  have hxpos : 0 < x := by linarith
  refine ⟨sigma / x, by positivity, ?_⟩
  have hr : 0 < sigma / x := by positivity
  have hsr : sigma / (sigma / x) = x := by field_simp
  have hrs : sigma / x ≤ sigma / 2 := by gcongr
  have hge := lj_ge_half_core he hr hrs
  rw [hsr] at hge
  have h12 : x ≤ x ^ 12 := by
    calc x = x ^ 1 := (pow_one x).symm
      _ ≤ x ^ 12 := pow_le_pow_right₀ (by linarith) (by norm_num)
  have hMx : M < 2 * eps * x := by
    have hKx : (K:ℝ) < x := by rw [hx]; linarith
    have : M / (2 * eps) < x := lt_trans hK hKx
    rw [div_lt_iff₀ (by linarith)] at this
    linarith
  nlinarith [hge, h12, he]

/-! ## Coulomb with an explicit dielectric -/

/-- Coulomb's constant `1/(4 pi eps_0)` in SI units, J·m/C². -/
noncomputable def coulombConst : ℝ :=
  1 / (4 * Real.pi * Context.vacuumPermittivity)

lemma coulombConst_pos : 0 < coulombConst := by
  unfold coulombConst
  have := Context.vacuumPermittivity_pos
  have := Real.pi_pos
  positivity

/-- The Coulomb energy of two charges (coulombs) at separation `r` (metres) in a medium of
relative permittivity `epsr`. -/
noncomputable def coulomb (epsr q1 q2 r : ℝ) : ℝ := coulombConst * q1 * q2 / (epsr * r)

/-- **The dielectric matters.**  A medium of relative permittivity `epsr` divides every
electrostatic energy by exactly `epsr`. -/
theorem coulomb_dielectric_scaling {epsr q1 q2 r : ℝ} (hd : 0 < epsr) (hr : 0 < r) :
    coulomb epsr q1 q2 r = coulomb 1 q1 q2 r / epsr := by
  unfold coulomb
  field_simp

theorem coulomb_lt_of_dielectric {epsr q1 q2 r : ℝ} (hq : 0 < q1 * q2) (hd : 1 < epsr)
    (hr : 0 < r) : coulomb epsr q1 q2 r < coulomb 1 q1 q2 r := by
  unfold coulomb
  have hnum : 0 < coulombConst * q1 * q2 := by
    have := coulombConst_pos
    nlinarith
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_pos (mul_pos hnum hr) (by linarith : (0:ℝ) < epsr - 1)]

/-! ## The stability theorem -/

/-- **A 12-6 core makes any Coulomb pair stable.**  For every charge product `c` (of either
sign) the pair energy `lj + c/r` is bounded below on `r > 0` by an explicit constant.  This
is what makes a Boltzmann-Gibbs measure exist; a model with point charges and no repulsive
core has an infinite partition function. -/
theorem pair_bddBelow {eps sigma c : ℝ} (he : 0 < eps) (hs : 0 < sigma) :
    ∃ B : ℝ, ∀ r : ℝ, 0 < r → -B ≤ lj eps sigma r + c / r := by
  set T : ℝ := max 1 (|c| / (eps * sigma ^ 12)) with hT
  refine ⟨2 * eps + |c| * T, fun r hr => ?_⟩
  set t : ℝ := 1 / r with ht
  have htpos : 0 < t := by positivity
  have hsr : sigma / r = sigma * t := by rw [ht]; ring
  have hyoung : 4 * eps * (sigma * t) ^ 6 ≤ 2 * eps * (sigma * t) ^ 12 + 2 * eps := by
    nlinarith [sq_nonneg ((sigma * t) ^ 6 - 1), he]
  have hcoul : |c| * t ≤ eps * (sigma * t) ^ 12 + |c| * T := by
    rcases le_or_gt t T with h | h
    · have hle : |c| * t ≤ |c| * T := mul_le_mul_of_nonneg_left h (abs_nonneg c)
      nlinarith [pow_nonneg (by positivity : (0:ℝ) ≤ sigma * t) 12, he]
    · have hT1 : (1:ℝ) ≤ T := le_max_left _ _
      have hTc : |c| / (eps * sigma ^ 12) ≤ T := le_max_right _ _
      have ht1 : (1:ℝ) ≤ t := le_trans hT1 h.le
      have hcle : |c| ≤ eps * sigma ^ 12 * t := by
        have hpos : 0 < eps * sigma ^ 12 := by positivity
        have hle : |c| / (eps * sigma ^ 12) ≤ t := le_trans hTc h.le
        rw [div_le_iff₀ hpos] at hle
        linarith
      have hstep : |c| * t ≤ eps * sigma ^ 12 * t * t := by nlinarith [htpos]
      have hfin : eps * sigma ^ 12 * t * t ≤ eps * (sigma * t) ^ 12 := by
        have hmono : t ^ 2 ≤ t ^ 12 := pow_le_pow_right₀ ht1 (by norm_num)
        have hrw : eps * sigma ^ 12 * t * t = eps * sigma ^ 12 * t ^ 2 := by ring
        rw [hrw, mul_pow]
        nlinarith [mul_le_mul_of_nonneg_left hmono
          (mul_nonneg he.le (pow_pos hs 12).le)]
      have hTnn : 0 ≤ |c| * T := by
        have hT0 : 0 ≤ T := le_trans zero_le_one hT1
        positivity
      linarith
  have hcr : -(|c| * t) ≤ c / r := by
    have hrw : c / r = c * t := by rw [ht]; ring
    rw [hrw]
    have habs : |c * t| = |c| * t := by
      rw [abs_mul, abs_of_pos htpos]
    have hle := neg_abs_le (c * t)
    rw [habs] at hle
    exact hle
  have hljt : lj eps sigma r = 4 * eps * ((sigma * t) ^ 12 - (sigma * t) ^ 6) := by
    unfold lj; rw [hsr]
  rw [hljt]
  nlinarith [hyoung, hcoul, hcr]

/-- **The core wins over any Coulomb attraction.**  However strong the electrostatic
attraction of a pair, at small enough separation the total pair energy exceeds any bound:
the 12-6 core is an excluded volume, not a soft penalty. -/
theorem pair_repulsive_unbounded {eps sigma c : ℝ} (he : 0 < eps) (hs : 0 < sigma) (M : ℝ) :
    ∃ d : ℝ, 0 < d ∧ ∀ r : ℝ, 0 < r → r < d → M < lj eps sigma r + c / r := by
  have hes : (0:ℝ) < eps * sigma ^ 12 := by positivity
  set S : ℝ := max (max 1 (|c| / (eps * sigma ^ 12))) (max (M / (eps * sigma ^ 12)) (2 / sigma))
    with hS
  have hS1 : (1:ℝ) ≤ S := le_trans (le_max_left _ _) (le_max_left _ _)
  have hSpos : 0 < S := lt_of_lt_of_le zero_lt_one hS1
  refine ⟨1 / S, by positivity, fun r hr hrd => ?_⟩
  set t : ℝ := 1 / r with ht
  have htS : S < t := by
    rw [ht]
    rw [lt_div_iff₀ hr]
    rw [lt_div_iff₀ hSpos] at hrd
    linarith
  have ht1 : (1:ℝ) ≤ t := le_trans hS1 htS.le
  have htpos : 0 < t := by linarith
  -- inside the core
  have hsig : 2 / sigma ≤ S := le_trans (le_max_right _ _) (le_max_right _ _)
  have hrs : r ≤ sigma / 2 := by
    have h2 : 2 / sigma < t := lt_of_le_of_lt hsig htS
    rw [ht, lt_div_iff₀ hr, div_mul_eq_mul_div, div_lt_iff₀ hs] at h2
    linarith
  have hsr : sigma / r = sigma * t := by rw [ht]; ring
  have hcore := lj_ge_half_core he hr hrs
  rw [hsr] at hcore
  -- the Coulomb term is dominated
  have hcT : |c| / (eps * sigma ^ 12) ≤ S := le_trans (le_max_right _ _) (le_max_left _ _)
  have hcle : |c| ≤ eps * sigma ^ 12 * t := by
    have hle : |c| / (eps * sigma ^ 12) ≤ t := le_trans hcT htS.le
    rw [div_le_iff₀ hes] at hle
    linarith
  have h12 : t ^ 2 ≤ t ^ 12 := pow_le_pow_right₀ ht1 (by norm_num)
  have hdom : |c| * t ≤ eps * sigma ^ 12 * t ^ 12 := by
    have hstep : |c| * t ≤ eps * sigma ^ 12 * t * t := by nlinarith [htpos]
    nlinarith [mul_le_mul_of_nonneg_left h12 hes.le]
  have hcr : -(|c| * t) ≤ c / r := by
    have hrw : c / r = c * t := by rw [ht]; ring
    rw [hrw]
    have habs : |c * t| = |c| * t := by rw [abs_mul, abs_of_pos htpos]
    have hle := neg_abs_le (c * t)
    rw [habs] at hle
    exact hle
  -- and the remaining core term beats `M`
  have hMS : M / (eps * sigma ^ 12) ≤ S := le_trans (le_max_left _ _) (le_max_right _ _)
  have hM : M < eps * sigma ^ 12 * t := by
    have hle : M / (eps * sigma ^ 12) < t := lt_of_le_of_lt hMS htS
    rw [div_lt_iff₀ hes] at hle
    linarith
  have htt : t ≤ t ^ 12 := by
    calc t = t ^ 1 := (pow_one t).symm
      _ ≤ t ^ 12 := pow_le_pow_right₀ ht1 (by norm_num)
  have hmul : (sigma * t) ^ 12 = sigma ^ 12 * t ^ 12 := by rw [mul_pow]
  rw [hmul] at hcore
  nlinarith [mul_le_mul_of_nonneg_left htt hes.le]

/-! ## Bonded terms -/

/-- A harmonic term in an internal coordinate (bond length or valence angle). -/
noncomputable def harmonic (k x0 x : ℝ) : ℝ := k / 2 * (x - x0) ^ 2

theorem harmonic_nonneg {k x0 x : ℝ} (hk : 0 ≤ k) : 0 ≤ harmonic k x0 x := by
  unfold harmonic; positivity

theorem harmonic_eq_zero_iff {k x0 x : ℝ} (hk : 0 < k) : harmonic k x0 x = 0 ↔ x = x0 := by
  unfold harmonic
  constructor
  · intro h
    have hz : (x - x0) ^ 2 = 0 := by
      by_contra hc
      have hpos : 0 < (x - x0) ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm hc)
      nlinarith
    have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp hz
    linarith
  · rintro rfl; simp

/-- Stretching a bond away from equilibrium always costs energy, strictly. -/
theorem harmonic_strictMono {k x0 x y : ℝ} (hk : 0 < k) (hx : x0 ≤ x) (hxy : x < y) :
    harmonic k x0 x < harmonic k x0 y := by
  unfold harmonic
  have h1 : (x - x0) ^ 2 < (y - x0) ^ 2 := by nlinarith
  have h2 : 0 < k / 2 := by linarith
  nlinarith [h1, h2]

/-! ## Torsions -/

/-- The standard cosine torsion term `V/2 (1 + cos (n phi - gamma))`. -/
noncomputable def torsion (V n gamma phi : ℝ) : ℝ :=
  V / 2 * (1 + Real.cos (n * phi - gamma))

theorem torsion_mem_Icc {V n gamma phi : ℝ} (hV : 0 ≤ V) :
    torsion V n gamma phi ∈ Set.Icc 0 V := by
  unfold torsion
  have h1 := Real.neg_one_le_cos (n * phi - gamma)
  have h2 := Real.cos_le_one (n * phi - gamma)
  constructor <;> nlinarith

/-- The torsion term has period `2 pi / n`. -/
theorem torsion_periodic {V n gamma phi : ℝ} (hn : n ≠ 0) :
    torsion V n gamma (phi + 2 * Real.pi / n) = torsion V n gamma phi := by
  unfold torsion
  congr 2
  have hrw : n * (phi + 2 * Real.pi / n) - gamma = (n * phi - gamma) + 2 * Real.pi := by
    field_simp; ring
  rw [hrw, Real.cos_add_two_pi]

/-- The minima of the torsion term. -/
theorem torsion_eq_zero_iff {V n gamma phi : ℝ} (hV : 0 < V) :
    torsion V n gamma phi = 0 ↔ Real.cos (n * phi - gamma) = -1 := by
  unfold torsion
  constructor
  · intro h; nlinarith
  · intro h; rw [h]; ring

/-! ## Continuity away from coincident atoms -/

theorem lj_continuousOn (eps sigma : ℝ) :
    ContinuousOn (fun r => lj eps sigma r) {r : ℝ | r ≠ 0} := by
  unfold lj
  apply ContinuousOn.mul continuousOn_const
  apply ContinuousOn.sub
  · exact ContinuousOn.pow (ContinuousOn.div continuousOn_const continuousOn_id
      (fun x hx => hx)) 12
  · exact ContinuousOn.pow (ContinuousOn.div continuousOn_const continuousOn_id
      (fun x hx => hx)) 6

theorem coulomb_continuousOn (epsr q1 q2 : ℝ) (hd : epsr ≠ 0) :
    ContinuousOn (fun r => coulomb epsr q1 q2 r) {r : ℝ | r ≠ 0} := by
  unfold coulomb
  apply ContinuousOn.div continuousOn_const
  · exact ContinuousOn.mul continuousOn_const continuousOn_id
  · intro x hx
    exact mul_ne_zero hd hx

theorem harmonic_continuous (k x0 : ℝ) : Continuous (harmonic k x0) := by
  unfold harmonic; fun_prop

theorem torsion_continuous (V n gamma : ℝ) : Continuous (torsion V n gamma) := by
  unfold torsion; fun_prop

end Potentials

end IDR
