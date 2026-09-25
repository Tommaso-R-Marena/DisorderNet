/-
# Part CXLII  The closed phase diagram: temperature and pressure together

With the pressure axis of Part CXXXVIII in place next to the temperature axis of the earlier
physics, the two can be varied together, and the second-order expansion of the stability of a
structured element inside a disordered region — a transient helix, a folded-upon-binding motif,
a residual hydrophobic cluster — becomes a quadratic form in `(t, p) = (T − T₀, P − P₀)`:

`ΔG(t,p) = f₀ + d·t + e·p + a·t² + b·t·p + c·p²`,

with `a = −ΔCp/(2T₀)`, `c = −Δβ/2`, `b = Δα` the thermal expansivity change, `d = −ΔS`,
`e = ΔV`.  A large positive heat-capacity change and a positive compressibility change — both
measured, both consequences of burying surface and packing voids — make `a` and `c` negative,
and this part shows what follows.

* `quadratic_nonpos_of_neg_disc` and `negDefinite_bound` — with `a < 0` and `4ac > b²` the
  quadratic part is negative definite with an explicit modulus `stabModulus`.
* `stability_region_bounded` — **the phase diagram is closed.**  The whole set of conditions at
  which the element is stable lies inside an explicit disc in the `(T, P)` plane.  Nothing about
  the model is extrapolable outside it: stability is not a monotone function of either axis, and
  a fit that assumes it is will be wrong on the far side of the ellipse.
* `cold_and_heat_denaturation` — **cold denaturation is a theorem, not an extra assumption.**  If
  the element is stable at the reference condition and `ΔCp > 0`, then the temperature axis
  crosses the boundary twice: there is a `t₂ > 0` (heat denaturation) *and* a `t₁ < 0` (cold
  denaturation) at which `ΔG = 0`.  Any model of a disordered region with a positive heat-capacity
  change predicts cold denaturation whether or not it was built to.
* `pressure_denaturation_exists` — the same argument on the pressure axis: a positive
  compressibility change forces a denaturing pressure.
* `elliptic_stability_law` bundles the three.
-/
import Mathlib

namespace RequestProject.StabilityDiagram

/-- Second-order expansion of a stability difference in temperature and pressure about a
reference condition. -/
noncomputable def dG (f0 d e a b c t p : ℝ) : ℝ :=
  f0 + d * t + e * p + a * t ^ 2 + b * t * p + c * p ^ 2

/-- A quadratic form with negative leading coefficient and negative discriminant is
nonpositive. -/
theorem quadratic_nonpos_of_neg_disc {a b c : ℝ} (ha : a < 0) (hdisc : b ^ 2 ≤ 4 * a * c)
    (x y : ℝ) : a * x ^ 2 + b * x * y + c * y ^ 2 ≤ 0 := by
  have hkey : 4 * a * (a * x ^ 2 + b * x * y + c * y ^ 2)
      = (2 * a * x + b * y) ^ 2 + (4 * a * c - b ^ 2) * y ^ 2 := by ring
  have hnn : 0 ≤ (2 * a * x + b * y) ^ 2 + (4 * a * c - b ^ 2) * y ^ 2 := by
    have h1 : 0 ≤ (4 * a * c - b ^ 2) * y ^ 2 :=
      mul_nonneg (by linarith) (sq_nonneg y)
    nlinarith [sq_nonneg (2 * a * x + b * y)]
  nlinarith [hkey, hnn]

/-- The modulus of negative definiteness used throughout: an explicit positive `λ` with
`a x² + b x y + c y² ≤ −λ (x² + y²)`. -/
noncomputable def stabModulus (a b c : ℝ) : ℝ :=
  min (-a / 2) ((4 * a * c - b ^ 2) / (4 * (-(a + c))))

theorem stabModulus_pos {a b c : ℝ} (ha : a < 0) (hdisc : b ^ 2 < 4 * a * c) :
    0 < stabModulus a b c := by
  have hc : c < 0 := by nlinarith [sq_nonneg b]
  have hsum : 0 < -(a + c) := by linarith
  refine lt_min (by linarith) (div_pos (by linarith) (by linarith))

/-- **Negative definiteness with an explicit modulus.** -/
theorem negDefinite_bound {a b c : ℝ} (ha : a < 0) (hdisc : b ^ 2 < 4 * a * c) (x y : ℝ) :
    a * x ^ 2 + b * x * y + c * y ^ 2 ≤ -stabModulus a b c * (x ^ 2 + y ^ 2) := by
  have hc : c < 0 := by nlinarith [sq_nonneg b]
  have hsum : 0 < -(a + c) := by linarith
  set lam := stabModulus a b c with hlam
  have hlam1 : lam ≤ -a / 2 := min_le_left _ _
  have hlam2 : lam ≤ (4 * a * c - b ^ 2) / (4 * (-(a + c))) := min_le_right _ _
  have hlampos : 0 < lam := stabModulus_pos ha hdisc
  have hA : a + lam < 0 := by linarith
  have hD : b ^ 2 ≤ 4 * (a + lam) * (c + lam) := by
    have h4 : 4 * lam * (-(a + c)) ≤ 4 * a * c - b ^ 2 := by
      rw [le_div_iff₀ (by linarith : (0:ℝ) < 4 * (-(a + c)))] at hlam2
      linarith
    nlinarith [sq_nonneg lam]
  have hq : (a + lam) * x ^ 2 + b * x * y + (c + lam) * y ^ 2 ≤ 0 :=
    quadratic_nonpos_of_neg_disc hA hD x y
  nlinarith [hq]

/-- **The phase diagram is closed.**  Every condition at which the element is stable lies inside
an explicit disc about the reference condition. -/
theorem stability_region_bounded {f0 d e a b c : ℝ} (ha : a < 0) (hdisc : b ^ 2 < 4 * a * c)
    {t p : ℝ} (hstab : 0 ≤ dG f0 d e a b c t p) :
    t ^ 2 + p ^ 2 ≤ (4 / (3 * stabModulus a b c)) *
      ((d ^ 2 + e ^ 2) / stabModulus a b c + |f0|) := by
  set lam := stabModulus a b c with hlam
  have hlampos : 0 < lam := stabModulus_pos ha hdisc
  have hquad : a * t ^ 2 + b * t * p + c * p ^ 2 ≤ -lam * (t ^ 2 + p ^ 2) :=
    negDefinite_bound ha hdisc t p
  have hd : d * t ≤ (lam / 4) * t ^ 2 + d ^ 2 / lam := by
    have key : 0 ≤ lam * ((lam / 4) * t ^ 2 + d ^ 2 / lam - d * t) := by
      have heq : lam * ((lam / 4) * t ^ 2 + d ^ 2 / lam - d * t) = (lam / 2 * t - d) ^ 2 := by
        field_simp
        ring
      rw [heq]
      exact sq_nonneg _
    nlinarith [key, hlampos]
  have he : e * p ≤ (lam / 4) * p ^ 2 + e ^ 2 / lam := by
    have key : 0 ≤ lam * ((lam / 4) * p ^ 2 + e ^ 2 / lam - e * p) := by
      have heq : lam * ((lam / 4) * p ^ 2 + e ^ 2 / lam - e * p) = (lam / 2 * p - e) ^ 2 := by
        field_simp
        ring
      rw [heq]
      exact sq_nonneg _
    nlinarith [key, hlampos]
  have hf0 : f0 ≤ |f0| := le_abs_self f0
  have hmain : (3 * lam / 4) * (t ^ 2 + p ^ 2) ≤ (d ^ 2 + e ^ 2) / lam + |f0| := by
    have hsum : 0 ≤ f0 + d * t + e * p + (a * t ^ 2 + b * t * p + c * p ^ 2) := by
      have hrw : dG f0 d e a b c t p
          = f0 + d * t + e * p + (a * t ^ 2 + b * t * p + c * p ^ 2) := by
        simp only [dG]; ring
      linarith [hrw ▸ hstab]
    have hsplit : (d ^ 2 + e ^ 2) / lam = d ^ 2 / lam + e ^ 2 / lam := by ring
    rw [hsplit]
    linarith
  have h3 : 0 < 3 * lam / 4 := by linarith
  have hfinal : t ^ 2 + p ^ 2 ≤ ((d ^ 2 + e ^ 2) / lam + |f0|) / (3 * lam / 4) := by
    rw [le_div_iff₀ h3]
    linarith
  have heq : ((d ^ 2 + e ^ 2) / lam + |f0|) / (3 * lam / 4)
      = (4 / (3 * lam)) * ((d ^ 2 + e ^ 2) / lam + |f0|) := by
    field_simp
  rw [heq] at hfinal
  exact hfinal

/-- A downward parabola that is positive at the origin has a root on each side of it. -/
private theorem parabola_two_roots {f0 d a : ℝ} (ha : a < 0) (hf0 : 0 < f0) :
    ∃ t₁ t₂ : ℝ, t₁ < 0 ∧ 0 < t₂ ∧ f0 + d * t₁ + a * t₁ ^ 2 = 0 ∧ f0 + d * t₂ + a * t₂ ^ 2 = 0 := by
  have hcont : Continuous fun t : ℝ => f0 + d * t + a * t ^ 2 := by continuity
  set K := (f0 + |d|) / (-a) with hK
  have hKnn : 0 ≤ K := div_nonneg (by positivity) (by linarith)
  set T := 1 + K with hT
  have hT1 : (1:ℝ) ≤ T := by linarith
  have hTpos : 0 < T := by linarith
  have hane : a ≠ 0 := ne_of_lt ha
  have haK : a * K = -(f0 + |d|) := by
    rw [hK]
    field_simp
  have hupper : f0 + d * T + a * T ^ 2 < 0 := by
    have hdT : d * T ≤ |d| * T := by
      have := le_abs_self d
      nlinarith
    have haT : a * T = a + a * K := by rw [hT]; ring
    have haT' : a * T = a - (f0 + |d|) := by rw [haT, haK]; ring
    nlinarith [haT', hdT, hTpos, hT1]
  have hlower : f0 + d * (-T) + a * (-T) ^ 2 < 0 := by
    have hdT : d * (-T) ≤ |d| * T := by
      have := neg_abs_le d
      nlinarith
    have haT : a * T = a - (f0 + |d|) := by
      have h1 : a * T = a + a * K := by rw [hT]; ring
      rw [h1, haK]; ring
    nlinarith [haT, hdT, hTpos, hT1]
  obtain ⟨t₂, ht₂mem, ht₂⟩ : ∃ t ∈ Set.Ioo (0:ℝ) T, f0 + d * t + a * t ^ 2 = 0 := by
    have hmem : (0:ℝ) ∈ Set.Ioo (f0 + d * T + a * T ^ 2) (f0 + d * 0 + a * 0 ^ 2) := by
      constructor
      · exact hupper
      · simpa using hf0
    have h2 := intermediate_value_Ioo' (le_of_lt hTpos) hcont.continuousOn
    obtain ⟨t, ht, hteq⟩ := h2 (by simpa using hmem)
    exact ⟨t, ht, hteq⟩
  obtain ⟨t₁, ht₁mem, ht₁⟩ : ∃ t ∈ Set.Ioo (-T) (0:ℝ), f0 + d * t + a * t ^ 2 = 0 := by
    have hcont' : ContinuousOn (fun t : ℝ => f0 + d * t + a * t ^ 2) (Set.Icc (-T) 0) :=
      hcont.continuousOn
    have hmem : (0:ℝ) ∈ Set.Ioo (f0 + d * (-T) + a * (-T) ^ 2) (f0 + d * 0 + a * 0 ^ 2) := by
      constructor
      · exact hlower
      · simpa using hf0
    have h2 := intermediate_value_Ioo (by linarith : (-T:ℝ) ≤ 0) hcont'
    obtain ⟨t, ht, hteq⟩ := h2 (by simpa using hmem)
    exact ⟨t, ht, hteq⟩
  exact ⟨t₁, t₂, ht₁mem.2, ht₂mem.1, ht₁, ht₂⟩

/-- **Cold denaturation is a theorem.**  If the element is stable at the reference condition and
the heat-capacity change is positive (`a < 0`), the temperature axis crosses the stability
boundary on both sides: there is a heat-denaturation temperature and a cold-denaturation
temperature. -/
theorem cold_and_heat_denaturation {f0 d e a b c : ℝ} (ha : a < 0) (hf0 : 0 < f0) :
    ∃ t₁ t₂ : ℝ, t₁ < 0 ∧ 0 < t₂ ∧ dG f0 d e a b c t₁ 0 = 0 ∧ dG f0 d e a b c t₂ 0 = 0 := by
  obtain ⟨t₁, t₂, h₁, h₂, e₁, e₂⟩ := parabola_two_roots (f0 := f0) (d := d) (a := a) ha hf0
  refine ⟨t₁, t₂, h₁, h₂, ?_, ?_⟩ <;> simp only [dG] <;> linarith

/-- **A positive compressibility change forces a denaturing pressure.** -/
theorem pressure_denaturation_exists {f0 d e a b c : ℝ} (hc : c < 0) (hf0 : 0 < f0) :
    ∃ p₁ p₂ : ℝ, p₁ < 0 ∧ 0 < p₂ ∧ dG f0 d e a b c 0 p₁ = 0 ∧ dG f0 d e a b c 0 p₂ = 0 := by
  obtain ⟨p₁, p₂, h₁, h₂, e₁, e₂⟩ := parabola_two_roots (f0 := f0) (d := e) (a := c) hc hf0
  refine ⟨p₁, p₂, h₁, h₂, ?_, ?_⟩ <;> simp only [dG] <;> linarith

/-- **The elliptic stability law.**  For a structured element whose heat-capacity and
compressibility changes are large enough that the quadratic part of the stability surface is
negative definite: (1) the region of conditions where it is stable is bounded, and if it is
stable at the reference condition then (2) the temperature axis and (3) the pressure axis each
cross the boundary on both sides — heat, cold and pressure denaturation are consequences of the
same closed curve. -/
theorem elliptic_stability_law {f0 d e a b c : ℝ} (ha : a < 0) (hdisc : b ^ 2 < 4 * a * c)
    (hf0 : 0 < f0) :
    (∀ t p : ℝ, 0 ≤ dG f0 d e a b c t p →
      t ^ 2 + p ^ 2 ≤ (4 / (3 * stabModulus a b c)) *
        ((d ^ 2 + e ^ 2) / stabModulus a b c + |f0|)) ∧
    (∃ t₁ t₂ : ℝ, t₁ < 0 ∧ 0 < t₂ ∧ dG f0 d e a b c t₁ 0 = 0 ∧ dG f0 d e a b c t₂ 0 = 0) ∧
    (∃ p₁ p₂ : ℝ, p₁ < 0 ∧ 0 < p₂ ∧ dG f0 d e a b c 0 p₁ = 0 ∧ dG f0 d e a b c 0 p₂ = 0) := by
  have hc : c < 0 := by nlinarith [sq_nonneg b]
  exact ⟨fun t p h => stability_region_bounded ha hdisc h,
    cold_and_heat_denaturation ha hf0,
    pressure_denaturation_exists hc hf0⟩

end RequestProject.StabilityDiagram
