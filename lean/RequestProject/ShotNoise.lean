/-
# Part CLI  From photon counts to conformational entropy: the shot-noise floor of a FRET histogram

Parts CXLVI–CL certify populations and entropy from a measured *mean* and a measured *variance* of
a conformational observable.  In a single-molecule FRET experiment neither is observed directly.
What is observed is a histogram of burst efficiencies `k/n`, where a burst of `n` photons from a
molecule momentarily at transfer efficiency `E` yields `k` acceptor photons — a binomial count.  The
histogram is therefore *always* broader than the ensemble, even for a perfectly homogeneous chain,
and the classical practice is to subtract the shot-noise width before speaking about heterogeneity.
This part proves that practice correct, and quantifies exactly what remains.

The binomial machinery is developed from scratch, by a transfer recursion rather than by generating
functions:

* `binPmf_succ_succ` — Pascal's recurrence for the binomial weights `C(n,k)p^k(1−p)^{n−k}`.
* `moment_succ` — the transfer lemma: `∑ₖ f(k)·b_{n+1}(k) = ∑ₖ (p·f(k+1) + (1−p)·f(k))·bₙ(k)`.
* `binPmf_sum`, `binPmf_mean`, `binPmf_sq` — normalisation, `⟨k⟩ = np`, and
  `⟨k²⟩ = np + n(n−1)p²`, each a two-line induction on the transfer lemma.
* `binPmf_dev_sq` — hence the burst variance about an arbitrary centre:
  `⟨(k/n − m)²⟩ = p(1−p)/n + (p − m)²`.

The ensemble statements then follow.

* `burst_mean_eq` — **the histogram is unbiased in the mean**: the mean burst efficiency is exactly
  the ensemble mean efficiency, whatever the photon budget.

* `burst_var_decomposition` — **and biased in the width, by exactly the shot noise**:

  `Var(histogram) = Var(ensemble) + (1/n)·⟨E(1 − E)⟩`.

  The second term is the shot-noise contribution; it is positive whenever any conformer has
  `0 < E < 1`, and it is the entire discrepancy.

* `ensemble_var_lower` — **the deconvolution certificate**: since `E(1 − E) ≤ 1/4`,

  `Var(ensemble) ≥ Var(histogram) − 1/(4n)`,

  a lower bound on genuine conformational heterogeneity computed from the histogram and the photon
  budget alone.

* `homogeneous_histogram_has_width` — **and it cannot be improved to an equality**: a completely
  homogeneous ensemble at efficiency `p` produces a histogram of variance `p(1−p)/n`, so any
  histogram no broader than shot noise is consistent with a single conformation.  A claim of
  heterogeneity must exceed the shot-noise width or it is empty.

* `entropy_floor_from_burst_histogram` — the chain closes: a histogram broader than shot noise
  certifies conformational entropy at least

  `(Var(histogram) − 1/(4n))² / 2`  nats,

  by Part CL applied to the efficiency, which lives in `[0,1]` by construction.  From photon counts
  to a thermodynamic bound on the entropic cost of ordering, with no polymer model anywhere.

Design consequence: the photon budget `n` is a design parameter of the same standing as the number
of salt conditions in Parts CXLII–CXLV.  It fixes the smallest heterogeneity that can be claimed —
`1/(4n)` in efficiency variance — and hence, through the floor above, the smallest conformational
entropy an experiment can certify.
-/
import Mathlib
import RequestProject.PopulationCertificate
import RequestProject.EntropyFloor

set_option autoImplicit false

namespace IDR
namespace ShotNoise

open Finset IDR.PopulationCertificate IDR.EntropyFloor

/-- The binomial weight: the probability of `k` acceptor photons in a burst of `n` from a molecule
at transfer efficiency `p`. -/
noncomputable def binPmf (p : ℝ) (n k : ℕ) : ℝ := (n.choose k : ℝ) * p ^ k * (1 - p) ^ (n - k)

theorem binPmf_zero_succ (p : ℝ) (n : ℕ) : binPmf p (n + 1) 0 = (1 - p) * binPmf p n 0 := by
  simp [binPmf, pow_succ]
  ring

/-- Pascal's recurrence for the binomial weights. -/
theorem binPmf_succ_succ (p : ℝ) (n k : ℕ) :
    binPmf p (n + 1) (k + 1) = p * binPmf p n k + (1 - p) * binPmf p n (k + 1) := by
  rcases lt_or_ge k n with h | h
  · have h1 : n - k = (n - (k + 1)) + 1 := by omega
    simp only [binPmf, Nat.succ_sub_succ, Nat.choose_succ_succ, Nat.cast_add]
    rw [h1]
    ring
  · have hc : n.choose (k + 1) = 0 := Nat.choose_eq_zero_of_lt (by omega)
    have hnk : n - k = 0 := by omega
    have hnk1 : n - (k + 1) = 0 := by omega
    simp only [binPmf, Nat.succ_sub_succ, Nat.choose_succ_succ, Nat.cast_add, hc, hnk, hnk1]
    push_cast
    ring

theorem binPmf_out (p : ℝ) (n k : ℕ) (h : n < k) : binPmf p n k = 0 := by
  simp [binPmf, Nat.choose_eq_zero_of_lt h]

/-- **The transfer lemma.**  Any moment of a burst of `n+1` photons reduces to a moment of a burst
of `n`. -/
theorem moment_succ (p : ℝ) (n : ℕ) (f : ℕ → ℝ) :
    ∑ k ∈ range (n + 2), f k * binPmf p (n + 1) k
      = ∑ k ∈ range (n + 1), (p * f (k + 1) + (1 - p) * f k) * binPmf p n k := by
  rw [Finset.sum_range_succ' (fun k => f k * binPmf p (n + 1) k) (n + 1)]
  have h1 : ∑ k ∈ range (n + 1), f (k + 1) * binPmf p (n + 1) (k + 1)
      = ∑ k ∈ range (n + 1), (p * f (k + 1)) * binPmf p n k
        + ∑ k ∈ range (n + 1), ((1 - p) * f (k + 1)) * binPmf p n (k + 1) := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [binPmf_succ_succ]
    ring
  rw [h1, binPmf_zero_succ]
  have h2 : ∑ k ∈ range (n + 1), ((1 - p) * f (k + 1)) * binPmf p n (k + 1)
        + f 0 * ((1 - p) * binPmf p n 0)
      = ∑ k ∈ range (n + 2), ((1 - p) * f k) * binPmf p n k := by
    rw [Finset.sum_range_succ' (fun k => ((1 - p) * f k) * binPmf p n k) (n + 1)]
    ring
  have h3 : ∑ k ∈ range (n + 2), ((1 - p) * f k) * binPmf p n k
      = ∑ k ∈ range (n + 1), ((1 - p) * f k) * binPmf p n k := by
    rw [Finset.sum_range_succ, binPmf_out p n (n + 1) (by omega)]
    ring
  rw [add_assoc, h2, h3, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  ring

theorem binPmf_sum (p : ℝ) (n : ℕ) : ∑ k ∈ range (n + 1), binPmf p n k = 1 := by
  induction n with
  | zero => simp [binPmf]
  | succ m ih =>
    have h := moment_succ p m (fun _ => 1)
    simp only [one_mul] at h
    rw [h]
    have h2 : ∀ k ∈ range (m + 1), (p * 1 + (1 - p) * 1) * binPmf p m k = binPmf p m k :=
      fun k _ => by ring
    rw [Finset.sum_congr rfl h2, ih]

theorem binPmf_mean (p : ℝ) (n : ℕ) : ∑ k ∈ range (n + 1), (k : ℝ) * binPmf p n k = n * p := by
  induction n with
  | zero => simp [binPmf]
  | succ m ih =>
    rw [moment_succ p m (fun k => (k : ℝ))]
    have hexp : ∀ k ∈ range (m + 1),
        (p * ((k + 1 : ℕ) : ℝ) + (1 - p) * (k : ℝ)) * binPmf p m k
          = (k : ℝ) * binPmf p m k + p * binPmf p m k := by
      intro k _
      push_cast
      ring
    rw [Finset.sum_congr rfl hexp, Finset.sum_add_distrib, ih, ← Finset.mul_sum, binPmf_sum]
    push_cast
    ring

theorem binPmf_sq (p : ℝ) (n : ℕ) :
    ∑ k ∈ range (n + 1), (k : ℝ) ^ 2 * binPmf p n k = n * p + (n : ℝ) * (n - 1) * p ^ 2 := by
  induction n with
  | zero => simp [binPmf]
  | succ m ih =>
    rw [moment_succ p m (fun k => (k : ℝ) ^ 2)]
    have hexp : ∀ k ∈ range (m + 1),
        (p * ((k + 1 : ℕ) : ℝ) ^ 2 + (1 - p) * (k : ℝ) ^ 2) * binPmf p m k
          = (k : ℝ) ^ 2 * binPmf p m k + 2 * p * ((k : ℝ) * binPmf p m k)
            + p * binPmf p m k := by
      intro k _
      push_cast
      ring
    rw [Finset.sum_congr rfl hexp, Finset.sum_add_distrib, Finset.sum_add_distrib, ih,
      ← Finset.mul_sum, ← Finset.mul_sum, binPmf_mean, binPmf_sum]
    push_cast
    ring

/-- **The burst spread about an arbitrary centre.**  A burst of `n` photons from a molecule at
efficiency `p` has `⟨(k/n − m)²⟩ = p(1−p)/n + (p − m)²`: the shot-noise term plus the displacement
of the conformer from the centre. -/
theorem binPmf_dev_sq (p m : ℝ) {n : ℕ} (hn : 0 < n) :
    ∑ k ∈ range (n + 1), ((k : ℝ) / n - m) ^ 2 * binPmf p n k
      = p * (1 - p) / n + (p - m) ^ 2 := by
  have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  have hexp : ∀ k ∈ range (n + 1), ((k : ℝ) / n - m) ^ 2 * binPmf p n k
      = (1 / (n : ℝ) ^ 2) * ((k : ℝ) ^ 2 * binPmf p n k)
        - (2 * m / (n : ℝ)) * ((k : ℝ) * binPmf p n k) + m ^ 2 * binPmf p n k := by
    intro k _
    field_simp
    ring
  rw [Finset.sum_congr rfl hexp, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, binPmf_sq, binPmf_mean, binPmf_sum]
  field_simp
  ring

/-! ### The histogram of an ensemble -/

variable {N : ℕ}

/-- The mean burst efficiency recorded from an ensemble with photon budget `n`. -/
noncomputable def burstMean (w E : Fin N → ℝ) (n : ℕ) : ℝ :=
  ∑ j, w j * ∑ k ∈ range (n + 1), ((k : ℝ) / n) * binPmf (E j) n k

/-- The variance of the recorded histogram of burst efficiencies, about the centre `m`. -/
noncomputable def burstVar (w E : Fin N → ℝ) (n : ℕ) (m : ℝ) : ℝ :=
  ∑ j, w j * ∑ k ∈ range (n + 1), ((k : ℝ) / n - m) ^ 2 * binPmf (E j) n k

/-- **The histogram is unbiased in the mean.**  The mean burst efficiency equals the ensemble mean
efficiency, for any photon budget. -/
theorem burst_mean_eq {w E : Fin N → ℝ} {n : ℕ} (hn : 0 < n) :
    burstMean w E n = wmean w E := by
  have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  rw [burstMean, wmean]
  refine Finset.sum_congr rfl fun j _ => ?_
  have hinner : ∑ k ∈ range (n + 1), ((k : ℝ) / n) * binPmf (E j) n k = E j := by
    have hexp : ∀ k ∈ range (n + 1), ((k : ℝ) / n) * binPmf (E j) n k
        = (1 / (n : ℝ)) * ((k : ℝ) * binPmf (E j) n k) := fun k _ => by ring
    rw [Finset.sum_congr rfl hexp, ← Finset.mul_sum, binPmf_mean]
    field_simp
  rw [hinner]

/-- **The shot-noise decomposition.**  The variance of the recorded histogram is the variance of
the ensemble plus the mean shot-noise term `⟨E(1−E)⟩/n`, exactly. -/
theorem burst_var_decomposition {w E : Fin N → ℝ} {n : ℕ} (hn : 0 < n) :
    burstVar w E n (wmean w E) = wvar w E + (∑ j, w j * (E j * (1 - E j))) / n := by
  rw [burstVar, wvar]
  have hinner : ∀ j ∈ (Finset.univ : Finset (Fin N)),
      w j * ∑ k ∈ range (n + 1), ((k : ℝ) / n - wmean w E) ^ 2 * binPmf (E j) n k
        = w j * (E j - wmean w E) ^ 2 + (w j * (E j * (1 - E j))) / n := by
    intro j _
    rw [binPmf_dev_sq (E j) (wmean w E) hn]
    ring
  rw [Finset.sum_congr rfl hinner, Finset.sum_add_distrib, ← Finset.sum_div]

/-- The shot-noise term never exceeds `1/(4n)`. -/
theorem shot_noise_le {w E : Fin N → ℝ} {n : ℕ} (hn : 0 < n) (hw : ∀ j, 0 ≤ w j)
    (hsum : ∑ j, w j = 1) :
    (∑ j, w j * (E j * (1 - E j))) / n ≤ 1 / (4 * n) := by
  have hn0 : (0:ℝ) < n := Nat.cast_pos.mpr hn
  have hb : ∑ j, w j * (E j * (1 - E j)) ≤ 1 / 4 := by
    calc ∑ j, w j * (E j * (1 - E j)) ≤ ∑ j, w j * (1 / 4) := by
          refine Finset.sum_le_sum fun j _ => ?_
          have : E j * (1 - E j) ≤ 1 / 4 := by nlinarith [sq_nonneg (E j - 1 / 2)]
          exact mul_le_mul_of_nonneg_left this (hw j)
      _ = 1 / 4 := by rw [← Finset.sum_mul, hsum, one_mul]
  rw [div_le_div_iff₀ hn0 (by positivity)]
  nlinarith [hb, hn0]

/-- **The deconvolution certificate.**  Genuine conformational heterogeneity is at least the
histogram width minus the shot-noise ceiling `1/(4n)`. -/
theorem ensemble_var_lower {w E : Fin N → ℝ} {n : ℕ} (hn : 0 < n) (hw : ∀ j, 0 ≤ w j)
    (hsum : ∑ j, w j = 1) :
    burstVar w E n (wmean w E) - 1 / (4 * n) ≤ wvar w E := by
  have hdec := burst_var_decomposition (w := w) (E := E) hn
  have hnoise := shot_noise_le (w := w) (E := E) hn hw hsum
  linarith

/-- **The certificate cannot be strengthened to an equality.**  A completely homogeneous ensemble
at efficiency `p` still produces a histogram of variance `p(1−p)/n`, so a histogram no broader than
shot noise is consistent with a single conformation. -/
theorem homogeneous_histogram_has_width {p : ℝ} {n : ℕ} (hn : 0 < n) :
    ∃ w E : Fin 1 → ℝ, (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ (∀ j, E j = p) ∧
      wvar w E = 0 ∧ burstVar w E n (wmean w E) = p * (1 - p) / n := by
  refine ⟨![1], ![p], ?_, ?_, ?_, ?_, ?_⟩
  · intro j
    fin_cases j
    norm_num
  · simp
  · intro j
    fin_cases j
    rfl
  · have hm : wmean ![(1:ℝ)] ![p] = p := by
      rw [wmean]
      simp
    rw [wvar, hm]
    simp
  · have hm : wmean ![(1:ℝ)] ![p] = p := by
      rw [wmean]
      simp
    rw [burstVar, hm]
    simp only [Finset.univ_unique, Finset.sum_singleton]
    rw [show (![(1:ℝ)] : Fin 1 → ℝ) default = 1 from rfl, show (![p] : Fin 1 → ℝ) default = p from rfl,
      binPmf_dev_sq p p hn]
    ring

/-- **From photon counts to conformational entropy.**  A histogram broader than the shot-noise
ceiling certifies conformational entropy at least `(Var(histogram) − 1/(4n))²/2` nats — the chain
photon counts → histogram width → ensemble heterogeneity → entropy, with no model of the chain. -/
theorem entropy_floor_from_burst_histogram {w E : Fin N → ℝ} {n : ℕ} (hn : 0 < n)
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) (hE0 : ∀ j, 0 ≤ E j) (hE1 : ∀ j, E j ≤ 1)
    (hbroad : 1 / (4 * n) ≤ burstVar w E n (wmean w E)) :
    (burstVar w E n (wmean w E) - 1 / (4 * n)) ^ 2 / 2 ≤ ent w := by
  have hfloor := entropy_floor_from_variance (x := E) (B := 1) one_pos hw hsum hE0 hE1
  have hvar := ensemble_var_lower (w := w) (E := E) hn hw hsum
  have hd0 : 0 ≤ burstVar w E n (wmean w E) - 1 / (4 * n) := by linarith
  have hsq : (burstVar w E n (wmean w E) - 1 / (4 * n)) ^ 2 ≤ wvar w E ^ 2 := by
    have := wvar_nonneg (x := E) hw
    nlinarith
  have hone : wvar w E ^ 2 / (2 * (1:ℝ) ^ 4) = wvar w E ^ 2 / 2 := by norm_num
  rw [hone] at hfloor
  linarith

end ShotNoise
end IDR
