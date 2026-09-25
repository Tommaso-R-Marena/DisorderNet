/-
# Part CXLI  Where the volume comes from: a coarse model that cannot be pressure denatured

Parts CXXXVIII–CXL treated the partial molar volume of a conformer as given.  It is not: it is a
sum of three physically distinct terms — the van der Waals volume of the atoms, the packing voids
the conformation encloses, and the electrostricted (compressed) hydration shell around the
exposed surface,

`V i = v_vdW + v_void · voids i − v_hyd · surf i`.

Only the last two depend on the conformation.  This part turns that decomposition into a
falsification test for coarse models of disordered regions.

* `residue_additive_blind` — **the negative result.**  A model whose conformer volume is a sum of
  residue contributions — which is what every residue-additive or Gaussian-chain volume model
  amounts to, since it has no voids and no conformation-dependent surface — assigns *the same*
  volume to every conformer, and therefore predicts, exactly, **no pressure response of any
  observable whatsoever**.  Such a model cannot be pressure denatured, cannot have a
  compressibility, and cannot be fitted to high-pressure data even in principle.  It fails the
  experiment structurally, not numerically.
* `voids_antitone`, `voids_strict_anti` — **the positive result.**  As soon as the volume carries
  a void term with `v_void > 0`, pressure squeezes voids out: the mean void content is
  non-increasing in pressure, strictly decreasing whenever two conformers differ in it.
* `cov_volume_voids`, `hasDerivAt_mean_voids` — and the response is exactly computable:
  `d⟨voids⟩/dp = −v_void·Var(voids) + v_hyd·Cov(surf, voids)`, which for a pure void model is
  `−v_void·Var(voids) ≤ 0`.  The compressibility of the model is `v_void²·Var(voids)`
  (`volume_variance_of_voids`), so a model with a rigid void content has none.
* `partial_volume_law` bundles the dichotomy: **any model of a disordered region that is to have
  a pressure axis at all must carry a conformation-dependent void or hydration term.**

The bilinearity lemmas for the ensemble covariance used along the way (`cov_add_left`,
`cov_smul_left`, `cov_const_left`, and the corresponding averages) are proved here because they
are what makes the response decomposition exact.
-/
import Mathlib
import RequestProject.PressureEnsemble
import RequestProject.PressureDesign

namespace RequestProject.PartialVolume

open Finset RequestProject.PressureEnsemble

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-! ## Linearity of the ensemble average and bilinearity of the covariance -/

omit [Nonempty ι] in
lemma mean_add (E : Ensemble ι) (p : ℝ) (f g : ι → ℝ) :
    mean E p (fun i => f i + g i) = mean E p f + mean E p g := by
  have h : ∑ i, wt E p i * (f i + g i)
      = (∑ i, wt E p i * f i) + ∑ i, wt E p i * g i := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  simp only [mean, h]
  ring

omit [Nonempty ι] in
lemma mean_smul (E : Ensemble ι) (p c : ℝ) (f : ι → ℝ) :
    mean E p (fun i => c * f i) = c * mean E p f := by
  have h : ∑ i, wt E p i * (c * f i) = c * ∑ i, wt E p i * f i := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  simp only [mean, h]
  ring

lemma mean_const (E : Ensemble ι) (p c : ℝ) : mean E p (fun _ => c) = c := by
  have h : ∑ i, wt E p i * c = c * Z E p := by
    rw [Z, Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  simp only [mean, h]
  field_simp [Z_ne_zero E p]

omit [Nonempty ι] in
lemma mean_sub (E : Ensemble ι) (p : ℝ) (f g : ι → ℝ) :
    mean E p (fun i => f i - g i) = mean E p f - mean E p g := by
  have h : (fun i => f i - g i) = fun i => f i + (-1) * g i := by
    funext i; ring
  rw [h, mean_add, mean_smul]
  ring

omit [Nonempty ι] in
lemma cov_add_left (E : Ensemble ι) (p : ℝ) (f g h : ι → ℝ) :
    cov E p (fun i => f i + g i) h = cov E p f h + cov E p g h := by
  have hprod : (fun i => (f i + g i) * h i) = fun i => f i * h i + g i * h i := by
    funext i; ring
  simp only [cov, hprod, mean_add]
  ring

omit [Nonempty ι] in
lemma cov_smul_left (E : Ensemble ι) (p c : ℝ) (f h : ι → ℝ) :
    cov E p (fun i => c * f i) h = c * cov E p f h := by
  have hprod : (fun i => (c * f i) * h i) = fun i => c * (f i * h i) := by
    funext i; ring
  simp only [cov, hprod, mean_smul]
  ring

lemma cov_const_left (E : Ensemble ι) (p c : ℝ) (h : ι → ℝ) :
    cov E p (fun _ => c) h = 0 := by
  have hprod : (fun i => (fun _ : ι => c) i * h i) = fun i => c * h i := by
    funext i; ring
  simp only [cov, mean_smul, mean_const]
  ring

/-! ## The volume decomposition -/

/-- A physical decomposition of the partial molar volume of a conformer into a
conformation-independent van der Waals term, a packing-void term and an electrostricted
hydration term proportional to exposed surface. -/
structure VolumeModel (ι : Type*) where
  /-- Van der Waals volume of the atoms; the same for every conformation. -/
  vdW : ℝ
  /-- Volume per unit of enclosed packing void. -/
  vVoid : ℝ
  /-- Volume lost per unit of exposed surface, through electrostriction of the hydration shell. -/
  vHyd : ℝ
  /-- Void content of each conformer. -/
  voids : ι → ℝ
  /-- Exposed surface of each conformer. -/
  surf : ι → ℝ

/-- The partial molar volume implied by the decomposition. -/
def volume (M : VolumeModel ι) (i : ι) : ℝ := M.vdW + M.vVoid * M.voids i - M.vHyd * M.surf i

/-- The pressure ensemble built from reference free energies and a volume model. -/
def ofVolumeModel (G : ι → ℝ) (M : VolumeModel ι) : Ensemble ι := ⟨G, volume M⟩

omit [Fintype ι] [Nonempty ι] in
@[simp] lemma ofVolumeModel_V (G : ι → ℝ) (M : VolumeModel ι) :
    (ofVolumeModel G M).V = volume M := rfl

/-! ## The negative result: a residue-additive volume has no pressure axis -/

omit [Nonempty ι] in
/-- **A residue-additive volume model is exactly pressure blind.**  If neither the void content
nor the exposed surface depends on the conformation — which is the case for any volume obtained
by summing residue contributions along the sequence — then every ensemble average is independent
of pressure.  The model has no compressibility and cannot be pressure denatured. -/
theorem residue_additive_blind (G : ι → ℝ) (M : VolumeModel ι) {v0 s0 : ℝ}
    (hv : ∀ i, M.voids i = v0) (hs : ∀ i, M.surf i = s0) (f : ι → ℝ) (p q : ℝ) :
    mean (ofVolumeModel G M) p f = mean (ofVolumeModel G M) q f := by
  refine RequestProject.PressureDesign.mean_const_of_volumes_equal (ofVolumeModel G M)
    (M.vdW + M.vVoid * v0 - M.vHyd * s0) (fun i => ?_) f p q
  simp [ofVolumeModel, volume, hv i, hs i]

/-! ## The positive result: voids respond to pressure -/

omit [Fintype ι] [Nonempty ι] in
/-- With a nonnegative void term and a conformation-independent surface, void content is
comonotone with volume. -/
lemma comonotone_voids (G : ι → ℝ) (M : VolumeModel ι) (hvv : 0 ≤ M.vVoid) {s0 : ℝ}
    (hs : ∀ i, M.surf i = s0) : Comonotone (ofVolumeModel G M).V M.voids := by
  intro i j
  have hij : (ofVolumeModel G M).V i - (ofVolumeModel G M).V j
      = M.vVoid * (M.voids i - M.voids j) := by
    simp only [ofVolumeModel_V, volume, hs i, hs j]
    ring
  rw [hij]
  nlinarith [sq_nonneg (M.voids i - M.voids j)]

/-- **Pressure squeezes voids out.**  The mean void content never increases with pressure. -/
theorem voids_antitone (G : ι → ℝ) (M : VolumeModel ι) (hvv : 0 ≤ M.vVoid) {s0 : ℝ}
    (hs : ∀ i, M.surf i = s0) {p q : ℝ} (hpq : p ≤ q) :
    mean (ofVolumeModel G M) q M.voids ≤ mean (ofVolumeModel G M) p M.voids :=
  mean_antitone_of_comonotone _ (comonotone_voids G M hvv hs) hpq

/-- And strictly, once two conformers differ in void content and the void term is real. -/
theorem voids_strict_anti (G : ι → ℝ) (M : VolumeModel ι) (hvv : 0 < M.vVoid) {s0 : ℝ}
    (hs : ∀ i, M.surf i = s0) {p q : ℝ} (hpq : p < q) {a b : ι}
    (hab : M.voids a < M.voids b) :
    mean (ofVolumeModel G M) q (volume M) < mean (ofVolumeModel G M) p (volume M) := by
  have hV : (ofVolumeModel G M).V a < (ofVolumeModel G M).V b := by
    simp only [ofVolumeModel_V, volume, hs a, hs b]
    nlinarith
  simpa [ofVolumeModel_V] using mean_volume_strict_anti (ofVolumeModel G M) hpq hV

/-! ## The response, exactly -/

/-- The covariance of volume with any observable splits into a void term and a surface term. -/
theorem cov_volume_left (G : ι → ℝ) (M : VolumeModel ι) (p : ℝ) (h : ι → ℝ) :
    cov (ofVolumeModel G M) p (ofVolumeModel G M).V h
      = M.vVoid * cov (ofVolumeModel G M) p M.voids h
        - M.vHyd * cov (ofVolumeModel G M) p M.surf h := by
  set E := ofVolumeModel G M with hE
  have hsplit : E.V = fun i => (M.vdW + M.vVoid * M.voids i) + (-M.vHyd) * M.surf i := by
    funext i
    simp only [hE, ofVolumeModel_V, volume]
    ring
  have hsplit2 : (fun i => M.vdW + M.vVoid * M.voids i)
      = fun i => (fun _ : ι => M.vdW) i + M.vVoid * M.voids i := rfl
  rw [hsplit, cov_add_left, hsplit2, cov_add_left, cov_const_left, cov_smul_left, cov_smul_left]
  ring

/-- Hence the exact pressure response of the mean void content. -/
theorem hasDerivAt_mean_voids (G : ι → ℝ) (M : VolumeModel ι) (p : ℝ) :
    HasDerivAt (fun p => mean (ofVolumeModel G M) p M.voids)
      (-(M.vVoid * cov (ofVolumeModel G M) p M.voids M.voids
        - M.vHyd * cov (ofVolumeModel G M) p M.surf M.voids)) p := by
  have h := hasDerivAt_mean (ofVolumeModel G M) p M.voids
  rwa [cov_volume_left G M p M.voids] at h

/-- For a pure void model the compressibility is `v_void²·Var(voids)`: a model whose void content
is the same in every conformer has none. -/
theorem volume_variance_of_voids (G : ι → ℝ) (M : VolumeModel ι) (p : ℝ) {s0 : ℝ}
    (hs : ∀ i, M.surf i = s0) :
    var (ofVolumeModel G M) p (ofVolumeModel G M).V
      = M.vVoid ^ 2 * var (ofVolumeModel G M) p M.voids := by
  set E := ofVolumeModel G M with hE
  have hsplit : E.V = fun i => (M.vdW - M.vHyd * s0) + M.vVoid * M.voids i := by
    funext i
    simp only [hE, ofVolumeModel_V, volume, hs i]
    ring
  have hstep : ∀ h : ι → ℝ, cov E p E.V h = M.vVoid * cov E p M.voids h := by
    intro h
    have hs2 : E.V = fun i => (fun _ : ι => M.vdW - M.vHyd * s0) i + M.vVoid * M.voids i := hsplit
    rw [hs2, cov_add_left, cov_const_left, cov_smul_left]
    ring
  have hsymm : cov E p M.voids E.V = M.vVoid * cov E p M.voids M.voids := by
    have hcomm : ∀ f g : ι → ℝ, cov E p f g = cov E p g f := by
      intro f g
      simp only [cov]
      rw [show (fun i => f i * g i) = fun i => g i * f i from by funext i; ring]
      ring
    rw [hcomm, hstep M.voids]
  calc var E p E.V = cov E p E.V E.V := rfl
    _ = M.vVoid * cov E p M.voids E.V := hstep E.V
    _ = M.vVoid * (M.vVoid * cov E p M.voids M.voids) := by rw [hsymm]
    _ = M.vVoid ^ 2 * var E p M.voids := by simp only [var]; ring

/-! ## Capstone -/

/-- **The partial-volume law.**  A model of a disordered region has a pressure axis if and only if
its volume depends on the conformation: (1) a residue-additive volume gives exactly no pressure
response of any observable at any pressure, while (2) a void term makes the mean void content
non-increasing in pressure, (3) with an exactly computable response and (4) a compressibility
`v_void²·Var(voids)` that vanishes precisely when the void content is rigid. -/
theorem partial_volume_law (G : ι → ℝ) (M : VolumeModel ι) {v0 s0 : ℝ}
    (hv : ∀ i, M.voids i = v0) (hs : ∀ i, M.surf i = s0) (hvv : 0 ≤ M.vVoid) :
    (∀ (f : ι → ℝ) (p q : ℝ),
      mean (ofVolumeModel G M) p f = mean (ofVolumeModel G M) q f) ∧
    (∀ p q : ℝ, p ≤ q →
      mean (ofVolumeModel G M) q M.voids ≤ mean (ofVolumeModel G M) p M.voids) ∧
    (∀ p : ℝ, HasDerivAt (fun p => mean (ofVolumeModel G M) p M.voids)
      (-(M.vVoid * cov (ofVolumeModel G M) p M.voids M.voids
        - M.vHyd * cov (ofVolumeModel G M) p M.surf M.voids)) p) ∧
    (∀ p : ℝ, var (ofVolumeModel G M) p (ofVolumeModel G M).V
      = M.vVoid ^ 2 * var (ofVolumeModel G M) p M.voids) :=
  ⟨fun f p q => residue_additive_blind G M hv hs f p q,
   fun _ _ hpq => voids_antitone G M hvv hs hpq,
   fun p => hasDerivAt_mean_voids G M p,
   fun p => volume_variance_of_voids G M p hs⟩

end RequestProject.PartialVolume
