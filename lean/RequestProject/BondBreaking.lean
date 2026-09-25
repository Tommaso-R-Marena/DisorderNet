/-
# Part CVIII  Bonds that break: the two-state quantum model a force field cannot hold

Part XXIV establishes that the standard class-I molecular model is well posed, and Parts XXV, LI
and LII price four of its idealisations.  One was left standing explicitly: "what still remains
outside is quantum-mechanical bond making and breaking".  For a disordered region this is not
exotic — disulphide exchange, proton transfer along a chain of titratable side chains, and metal
coordination all make and break bonds, and all of them are modelled by chemistry that a fixed bond
topology cannot express.  This file supplies the minimal honest model, the two-state (diabatic)
Hamiltonian with electronic coupling `V`, and proves what it says.

* `adiabatic_char` — the lower adiabatic surface `adiaLow` really is an eigenvalue of the two-state
  Hamiltonian: it satisfies the characteristic equation `(E₁−λ)(E₂−λ) = V²` exactly.  Nothing below
  is a definition dressed as a result.
* `adiaLow_le_left`, `adiaLow_le_right` — **the reactive surface lies below both bonded surfaces
  everywhere.**  A force field that switches between two bonded topologies, however smoothly,
  overestimates the energy at every geometry.
* `barrier_lowering` — **and at the crossing geometry the error is exactly the coupling**: where
  the two diabatic states are degenerate the true barrier top is `|V|` below the classical one.
  Since a rate depends exponentially on the barrier, this is the term that decides whether the
  chemistry happens at all; it is not a small correction that a reparametrised bond can absorb,
  because it is largest exactly where the classical surfaces cross.
* `no_crossing` — **the two surfaces never touch when the coupling is nonzero.**  A single
  Born–Oppenheimer surface is therefore well defined along the reaction, which is what makes
  classical dynamics *on it* meaningful — and it is exactly this surface, not either bonded one,
  that a force field would have to reproduce.
* `harmonic_cannot_dissociate` — **and a harmonic bond cannot.**  With a dissociated channel at
  finite energy `D`, the reactive ground surface is bounded above by `D` at every geometry, while a
  harmonic bond grows without bound: for every stiffness and every tolerance there is a geometry
  where the classical bond is wrong by more than the tolerance.  No fitting of harmonic parameters
  repairs this, because the failure is at infinity in the bond coordinate.

What is proved is therefore the exact form of the missing physics — a lower surface, a barrier
lowered by the coupling, an avoided crossing, and an unbounded error for a fixed bond — rather than
a claim that any particular reaction occurs.  The two-state model is the smallest one in which
bond making and breaking can be written down at all; a many-electron treatment remains outside.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR.BondBreaking

/-- Half the diabatic splitting. -/
noncomputable def halfGap (E₁ E₂ : ℝ) : ℝ := (E₁ - E₂) / 2

/-- The lower adiabatic (reactive) surface of the two-state Hamiltonian
`[[E₁, V], [V, E₂]]`. -/
noncomputable def adiaLow (E₁ E₂ V : ℝ) : ℝ :=
  (E₁ + E₂) / 2 - Real.sqrt (halfGap E₁ E₂ ^ 2 + V ^ 2)

/-- The upper adiabatic surface. -/
noncomputable def adiaHigh (E₁ E₂ V : ℝ) : ℝ :=
  (E₁ + E₂) / 2 + Real.sqrt (halfGap E₁ E₂ ^ 2 + V ^ 2)

lemma sqrt_sq_eq (E₁ E₂ V : ℝ) :
    Real.sqrt (halfGap E₁ E₂ ^ 2 + V ^ 2) ^ 2 = halfGap E₁ E₂ ^ 2 + V ^ 2 :=
  Real.sq_sqrt (by positivity)

lemma abs_halfGap_le_sqrt (E₁ E₂ V : ℝ) :
    |halfGap E₁ E₂| ≤ Real.sqrt (halfGap E₁ E₂ ^ 2 + V ^ 2) := by
  rw [← Real.sqrt_sq_eq_abs]
  exact Real.sqrt_le_sqrt (by nlinarith [sq_nonneg V])

/-- **The lower adiabatic surface is an eigenvalue of the two-state Hamiltonian.** -/
theorem adiabatic_char (E₁ E₂ V : ℝ) :
    (E₁ - adiaLow E₁ E₂ V) * (E₂ - adiaLow E₁ E₂ V) = V ^ 2 := by
  have hs := sqrt_sq_eq E₁ E₂ V
  unfold adiaLow halfGap at *
  nlinarith [hs]

/-- The upper surface is the other eigenvalue. -/
theorem adiabatic_char_high (E₁ E₂ V : ℝ) :
    (E₁ - adiaHigh E₁ E₂ V) * (E₂ - adiaHigh E₁ E₂ V) = V ^ 2 := by
  have hs := sqrt_sq_eq E₁ E₂ V
  unfold adiaHigh halfGap at *
  nlinarith [hs]

/-- **The reactive surface lies below the first bonded surface, everywhere.** -/
theorem adiaLow_le_left (E₁ E₂ V : ℝ) : adiaLow E₁ E₂ V ≤ E₁ := by
  have h := abs_halfGap_le_sqrt E₁ E₂ V
  have h2 : -(halfGap E₁ E₂) ≤ |halfGap E₁ E₂| := neg_le_abs _
  unfold adiaLow halfGap at *
  linarith

/-- **The reactive surface lies below the second bonded surface, everywhere.** -/
theorem adiaLow_le_right (E₁ E₂ V : ℝ) : adiaLow E₁ E₂ V ≤ E₂ := by
  have h := abs_halfGap_le_sqrt E₁ E₂ V
  have h2 : halfGap E₁ E₂ ≤ |halfGap E₁ E₂| := le_abs_self _
  unfold adiaLow halfGap at *
  linarith

/-- **At the crossing geometry the barrier is lowered by exactly the coupling.** -/
theorem barrier_lowering (Ec V : ℝ) : adiaLow Ec Ec V = Ec - |V| := by
  unfold adiaLow halfGap
  have h : (Ec - Ec) / 2 = 0 := by ring
  rw [h]
  rw [show (0:ℝ) ^ 2 + V ^ 2 = V ^ 2 by ring, Real.sqrt_sq_eq_abs]
  ring

/-- **The two adiabatic surfaces never touch when the coupling is nonzero**: the gap is at least
`2|V|`. -/
theorem no_crossing {E₁ E₂ V : ℝ} (hV : V ≠ 0) :
    2 * |V| ≤ adiaHigh E₁ E₂ V - adiaLow E₁ E₂ V ∧ adiaLow E₁ E₂ V < adiaHigh E₁ E₂ V := by
  have hVpos : 0 < |V| := abs_pos.mpr hV
  have hge : |V| ≤ Real.sqrt (halfGap E₁ E₂ ^ 2 + V ^ 2) := by
    rw [← Real.sqrt_sq_eq_abs]
    exact Real.sqrt_le_sqrt (by nlinarith [sq_nonneg (halfGap E₁ E₂)])
  unfold adiaHigh adiaLow
  constructor
  · linarith
  · linarith

/-- A harmonic bond of stiffness `k`. -/
noncomputable def harmonicBond (k q : ℝ) : ℝ := k * q ^ 2

/-- **A harmonic bond cannot describe dissociation.**  With a dissociated channel at energy `D`,
the reactive surface never exceeds `D`, while the harmonic bond exceeds any tolerance above it. -/
theorem harmonic_cannot_dissociate {k D V B : ℝ} (hk : 0 < k) (hB : 0 ≤ B) :
    ∃ q : ℝ, adiaLow (harmonicBond k q) D V + B < harmonicBond k q := by
  refine ⟨Real.sqrt ((|D| + B + 1) / k), ?_⟩
  have hq : harmonicBond k (Real.sqrt ((|D| + B + 1) / k)) = |D| + B + 1 := by
    unfold harmonicBond
    rw [Real.sq_sqrt (by positivity)]
    field_simp
  have hle : adiaLow (harmonicBond k (Real.sqrt ((|D| + B + 1) / k))) D V ≤ D :=
    adiaLow_le_right _ _ _
  have hD : D ≤ |D| := le_abs_self D
  rw [hq] at hle ⊢
  linarith

end IDR.BondBreaking
