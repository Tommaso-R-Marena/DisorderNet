/-
# Part CXLV  Where to put the salt conditions: the optimal spacing of a titration ladder

Part CXLII showed that an arithmetic ladder of ionic strengths `κ_j = κ₀ + j·h` inverts exactly,
with an amplification factor governed by the gap between the Vandermonde nodes `e^{−h r}`, and Part
CXLIII made that factor explicit.  The spacing `h` is a free parameter of the experiment, and the
gap depends on it in a non-monotone way: at small `h` the nodes of two distances are both near `1`
and nearly equal, at large `h` they are both near `0` and again nearly equal.  There is therefore a
best spacing, and this part computes it.

* `optimalSpacing r₁ r₂ = log(r₂/r₁)/(r₂ − r₁)` — the spacing at which the node gap of two internal
  distances `r₁ < r₂` is largest.  It is the logarithmic mean's reciprocal: for `r₂ = 2r₁` it is
  `log 2/r₁`, so the natural salt step is set by the *ratio* of the distances to be resolved as well
  as their difference.

* `node_gap_maximised` — for every spacing `h ≥ 0`, the node gap `|e^{−hr₁} − e^{−hr₂}|` is at most
  its value at `optimalSpacing r₁ r₂`.  The proof is a derivative-sign argument: the gap is
  increasing below the optimum and decreasing above it, the turning point being where
  `r₂e^{−hr₂} = r₁e^{−hr₁}`, i.e. where `log r₂ − h r₂ = log r₁ − h r₁`.

* `amplification_minimised_at_optimal_spacing` — consequently the resolution horizon of Part CXLII,
  whose amplification is inversely proportional to the node gap, is smallest at exactly this
  spacing: no titration ladder, at any spacing, separates the two distances better.  A titration
  designed to resolve two conformers should step the inverse Debye length by `log(r₂/r₁)/(r₂ − r₁)`.

* `ladder_spacing_law` collects the statement.

Design consequence, completing Parts CXLII–CXLV: the design of a salt titration for a disordered
region has three numbers in it — how many conditions (at least as many as the conformers claimed),
how they are spaced (equally, by `log(r₂/r₁)/(r₂ − r₁)` for the pair one wants to separate), and
what precision each reading carries (which, through the representer norm of Part CXLIV, fixes what
can be claimed about the ensemble at all).
-/
import Mathlib
import RequestProject.TitrationLadder

set_option autoImplicit false

namespace IDR
namespace LadderSpacing

open Real

/-- The spacing of a titration ladder that maximises the Vandermonde node gap of two internal
distances `r₁ < r₂`: `log(r₂/r₁)/(r₂ − r₁)`. -/
noncomputable def optimalSpacing (r1 r2 : ℝ) : ℝ := Real.log (r2 / r1) / (r2 - r1)

/-- The optimal spacing is a positive step in ionic strength. -/
theorem optimalSpacing_pos {r1 r2 : ℝ} (h1 : 0 < r1) (h12 : r1 < r2) :
    0 < optimalSpacing r1 r2 := by
  rw [optimalSpacing]
  refine div_pos (Real.log_pos ?_) (by linarith)
  rw [lt_div_iff₀ h1]
  linarith

/-- **The node gap is maximised at the optimal spacing.**  For two internal distances `r₁ < r₂` and
any ladder spacing `h ≥ 0`, the gap between the Vandermonde nodes `e^{−hr}` is at most its value at
`optimalSpacing r₁ r₂`. -/
theorem node_gap_maximised {r1 r2 : ℝ} (h1 : 0 < r1) (h12 : r1 < r2) :
    ∀ h : ℝ, 0 ≤ h → |Real.exp (-(h * r1)) - Real.exp (-(h * r2))|
      ≤ Real.exp (-(optimalSpacing r1 r2 * r1)) - Real.exp (-(optimalSpacing r1 r2 * r2)) := by
  set hs : ℝ := optimalSpacing r1 r2 with hhs
  have h2 : 0 < r2 := lt_trans h1 h12
  have hkey0 : hs * (r2 - r1) = Real.log r2 - Real.log r1 := by
    rw [hhs, optimalSpacing, Real.log_div (ne_of_gt h2) (ne_of_gt h1), div_mul_cancel₀]
    linarith
  set f : ℝ → ℝ := fun h => Real.exp (-(h * r1)) - Real.exp (-(h * r2)) with hf
  have hderiv : ∀ x : ℝ, HasDerivAt f
      (Real.exp (-(x * r1)) * (-r1) - Real.exp (-(x * r2)) * (-r2)) x := by
    intro x
    have d1 : HasDerivAt (fun h : ℝ => -(h * r1)) (-r1) x := by
      simpa using ((hasDerivAt_id x).mul_const r1).neg
    have d2 : HasDerivAt (fun h : ℝ => -(h * r2)) (-r2) x := by
      simpa using ((hasDerivAt_id x).mul_const r2).neg
    exact (d1.exp).sub (d2.exp)
  have hderiv_eq : ∀ x : ℝ, deriv f x
      = Real.exp (-(x * r2)) * r2 - Real.exp (-(x * r1)) * r1 := by
    intro x
    rw [(hderiv x).deriv]; ring
  have hrw1 : ∀ y : ℝ, Real.exp (-(y * r1)) * r1 = Real.exp (Real.log r1 - y * r1) := by
    intro y; rw [Real.exp_sub, Real.exp_log h1, Real.exp_neg]; ring
  have hrw2 : ∀ y : ℝ, Real.exp (-(y * r2)) * r2 = Real.exp (Real.log r2 - y * r2) := by
    intro y; rw [Real.exp_sub, Real.exp_log h2, Real.exp_neg]; ring
  have hpos_deriv : ∀ x : ℝ, x ≤ hs → 0 ≤ deriv f x := by
    intro x hx
    rw [hderiv_eq, hrw1, hrw2, sub_nonneg]
    apply Real.exp_le_exp.2
    nlinarith [mul_le_mul_of_nonneg_right hx (le_of_lt (by linarith : (0:ℝ) < r2 - r1))]
  have hneg_deriv : ∀ x : ℝ, hs ≤ x → deriv f x ≤ 0 := by
    intro x hx
    rw [hderiv_eq, hrw1, hrw2, sub_nonpos]
    apply Real.exp_le_exp.2
    nlinarith [mul_le_mul_of_nonneg_right hx (le_of_lt (by linarith : (0:ℝ) < r2 - r1))]
  have hcont : Continuous f := by rw [hf]; fun_prop
  have hmono : MonotoneOn f (Set.Iic hs) := by
    apply monotoneOn_of_deriv_nonneg (convex_Iic hs) hcont.continuousOn
    · exact fun x _ => (hderiv x).differentiableAt.differentiableWithinAt
    · intro x hx
      rw [interior_Iic] at hx
      exact hpos_deriv x (le_of_lt hx)
  have hanti : AntitoneOn f (Set.Ici hs) := by
    apply antitoneOn_of_deriv_nonpos (convex_Ici hs) hcont.continuousOn
    · exact fun x _ => (hderiv x).differentiableAt.differentiableWithinAt
    · intro x hx
      rw [interior_Ici] at hx
      exact hneg_deriv x (le_of_lt hx)
  intro h hh
  have hfnonneg : 0 ≤ f h := by
    rw [hf, sub_nonneg]
    apply Real.exp_le_exp.2
    nlinarith
  rw [abs_of_nonneg hfnonneg]
  rcases le_total h hs with hle | hge
  · exact hmono hle (le_refl hs) hle
  · exact hanti (le_refl hs) hge hge

/-- **No ladder resolves two distances better.**  The amplification factor of the two-distance
resolution horizon of Part CXLII is inversely proportional to the node gap, so by
`node_gap_maximised` it is smallest at `optimalSpacing r₁ r₂`: for every spacing `h > 0` the
horizon at the optimal spacing is at most the horizon at `h`. -/
theorem amplification_minimised_at_optimal_spacing {r1 r2 kappa0 delta : ℝ}
    (h1 : 0 < r1) (h12 : r1 < r2) (hdelta : 0 ≤ delta) :
    ∀ h : ℝ, 0 < h →
      delta * (r1 * Real.exp (kappa0 * r1))
          / (Real.exp (-(optimalSpacing r1 r2 * r1))
              - Real.exp (-(optimalSpacing r1 r2 * r2)))
        ≤ delta * (r1 * Real.exp (kappa0 * r1))
          / |Real.exp (-(h * r1)) - Real.exp (-(h * r2))| := by
  intro h hh
  have hgap_pos : 0 < |Real.exp (-(h * r1)) - Real.exp (-(h * r2))| := by
    rw [abs_pos, sub_ne_zero]
    intro hc
    have := Real.exp_injective hc
    nlinarith
  have hle := node_gap_maximised h1 h12 h (le_of_lt hh)
  have hnum : 0 ≤ delta * (r1 * Real.exp (kappa0 * r1)) := by positivity
  exact div_le_div_of_nonneg_left hnum hgap_pos hle

/-- **The spacing law.**  A titration ladder aimed at separating two internal distances `r₁ < r₂`
has a unique best step in ionic strength, `log(r₂/r₁)/(r₂ − r₁)`: it maximises the Vandermonde node
gap and therefore minimises the amplification with which reading error becomes weight error. -/
theorem ladder_spacing_law {r1 r2 : ℝ} (h1 : 0 < r1) (h12 : r1 < r2) :
    0 < optimalSpacing r1 r2 ∧
    (∀ h : ℝ, 0 ≤ h → |Real.exp (-(h * r1)) - Real.exp (-(h * r2))|
      ≤ Real.exp (-(optimalSpacing r1 r2 * r1)) - Real.exp (-(optimalSpacing r1 r2 * r2))) ∧
    (∀ (kappa0 delta : ℝ), 0 ≤ delta → ∀ h : ℝ, 0 < h →
      delta * (r1 * Real.exp (kappa0 * r1))
          / (Real.exp (-(optimalSpacing r1 r2 * r1))
              - Real.exp (-(optimalSpacing r1 r2 * r2)))
        ≤ delta * (r1 * Real.exp (kappa0 * r1))
          / |Real.exp (-(h * r1)) - Real.exp (-(h * r2))|) := by
  refine ⟨optimalSpacing_pos h1 h12, node_gap_maximised h1 h12, ?_⟩
  intro kappa0 delta hdelta h hh
  exact amplification_minimised_at_optimal_spacing h1 h12 hdelta h hh

/-! ## A worked design number -/

/-- **A concrete design number.**  To separate internal distances of `20 Å` and `25 Å` — a
plausible pair for a short disordered segment — the optimal step in inverse Debye length is
`log(1.25)/5`, which lies between `0.04 Å⁻¹` and `0.05 Å⁻¹`.  (Numerically it is `0.0446 Å⁻¹`, a
step of roughly a factor of two in salt concentration around physiological ionic strength.) -/
theorem optimalSpacing_twenty_twentyfive :
    (0.04 : ℝ) ≤ optimalSpacing 20 25 ∧ optimalSpacing 20 25 ≤ 0.05 := by
  have hval : optimalSpacing 20 25 = Real.log 1.25 / 5 := by
    rw [optimalSpacing]
    norm_num
  have hlow : (0.2 : ℝ) ≤ Real.log 1.25 := by
    have h1 : (0.8 : ℝ) ≤ Real.exp (-0.2) := by
      have := Real.add_one_le_exp (-0.2 : ℝ)
      linarith
    have h2 : Real.exp (0.2 : ℝ) ≤ 1.25 := by
      have hmul : Real.exp (0.2 : ℝ) * Real.exp (-0.2 : ℝ) = 1 := by
        rw [← Real.exp_add]; norm_num
      nlinarith [Real.exp_pos (0.2 : ℝ)]
    have h3 := Real.log_le_log (Real.exp_pos (0.2 : ℝ)) h2
    rwa [Real.log_exp] at h3
  have hhigh : Real.log 1.25 ≤ (0.25 : ℝ) := by
    have h1 : (1.25 : ℝ) ≤ Real.exp 0.25 := by
      have := Real.add_one_le_exp (0.25 : ℝ)
      linarith
    have h3 := Real.log_le_log (by norm_num : (0:ℝ) < 1.25) h1
    rwa [Real.log_exp] at h3
  rw [hval]
  constructor <;> linarith


end LadderSpacing
end IDR
