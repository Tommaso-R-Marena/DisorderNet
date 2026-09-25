/-
# Part CII  What a temperature ladder costs: round trips

Replica exchange appears in this development as a sampling device whose ladder is assumed to work:
the assumptions list recorded that "round-trip statistics, and the scheduling controls that govern
them, are not modelled".  They are the quantity that actually decides whether a simulation of a
disordered region has sampled anything.  A replica that never travels from the hot end of the
ladder to the cold end and back has not carried the decorrelation that the hot end provides down
to the temperature the ensemble is reported at; the cold replicas are then no better than
independent short runs, however many exchange attempts were accepted.

This file proves two lower bounds on the cost, one deterministic and one diffusive.

* `travel_le`, `bottom_to_top`, `round_trip_time` — **the ballistic bound.**  A replica's ladder
  position changes by at most one rung per sweep, because that is what a nearest-neighbour
  exchange move does.  So a bottom-to-top traversal of a `K`-rung ladder needs at least `K` sweeps
  and a round trip at least `2K`, no matter how the swaps are scheduled or how high the acceptance
  rate is.  `round_trips_time` accumulates this: `n` round trips need `2Kn` sweeps.
* `sqDisp_step`, `sqDisp_growth` — **the diffusive bound**, which is the one that bites.  If the
  ladder walk has no net drift (the acceptance rates are balanced, as they are by construction when
  the ladder is tuned to a uniform acceptance rate), the mean squared ladder displacement grows by
  at most `1` per sweep.  Travel is a random walk, not a march.
* `reach_prob_le` — hence **the probability that a replica started at the bottom has got `R` rungs
  away after `T` sweeps is at most `T / R²`**, and `sweeps_needed_for_traversal`: getting there
  with probability at least `1/2` takes at least `R² / 2` sweeps.

The two together are the honest statement of what a ladder buys.  Adding rungs raises the
acceptance rate — the reason ladders are made dense — but the sweeps needed for a round trip grow
*quadratically* in the number of rungs, so the exchange machinery has an optimum and past it a
denser ladder makes the sampling worse.  None of this is visible to a convergence diagnostic run on
the cold replica alone, which is why round-trip counting is the diagnostic that has to be reported.

The diffusive results are stated for a time-inhomogeneous ladder walk (`M t` may differ at every
sweep), so an adaptive schedule that retunes the temperatures as it goes is covered too: no
schedule that keeps the walk driftless escapes the quadratic cost.
-/
import Mathlib
import RequestProject.Ageing

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset

namespace IDR.ReplicaTrip

open IDR.Ageing (vecMul IsStoch)

/-! ## The ballistic bound -/

section Ballistic

variable {pos : ℕ → ℤ}

/-- A replica whose ladder position moves by at most one rung per sweep travels at most `b − a`
rungs between sweep `a` and sweep `b`. -/
theorem travel_le (hstep : ∀ t, |pos (t + 1) - pos t| ≤ 1) {a b : ℕ} (hab : a ≤ b) :
    |pos b - pos a| ≤ (b : ℤ) - (a : ℤ) := by
  induction b, hab using Nat.le_induction with
  | base => simp
  | succ n hn ih =>
      have h1 := hstep n
      rw [abs_le] at h1 ih ⊢
      push_cast
      constructor <;> linarith [h1.1, h1.2, ih.1, ih.2]

/-- **A bottom-to-top traversal of a `K`-rung ladder needs at least `K` sweeps.** -/
theorem bottom_to_top (hstep : ∀ t, |pos (t + 1) - pos t| ≤ 1) {K : ℤ} {a b : ℕ} (hab : a ≤ b)
    (h0 : pos a = 0) (hK : pos b = K) : K ≤ (b : ℤ) - (a : ℤ) := by
  have h2 := travel_le hstep hab
  rw [h0, hK] at h2
  exact le_trans (by simpa using le_abs_self K) h2

/-- **A round trip needs at least `2K` sweeps**: bottom at sweep `a`, top at some sweep `m` in
between, bottom again at sweep `b`. -/
theorem round_trip_time (hstep : ∀ t, |pos (t + 1) - pos t| ≤ 1) {K : ℤ} {a m b : ℕ}
    (ham : a ≤ m) (hmb : m ≤ b) (h0 : pos a = 0) (hK : pos m = K) (h0' : pos b = 0) :
    2 * K ≤ (b : ℤ) - (a : ℤ) := by
  have h1 : K ≤ (m : ℤ) - (a : ℤ) := bottom_to_top hstep ham h0 hK
  have h2 : |pos b - pos m| ≤ (b : ℤ) - (m : ℤ) := travel_le hstep hmb
  rw [hK, h0'] at h2
  have h3 : K ≤ (b : ℤ) - (m : ℤ) := le_trans (by simpa using le_abs_self K) h2
  linarith

/-- `n` round trips need `2Kn` sweeps. -/
theorem round_trips_time (hstep : ∀ t, |pos (t + 1) - pos t| ≤ 1) {K : ℤ} {u : ℕ → ℕ}
    (hbot : ∀ i, pos (u i) = 0)
    (htop : ∀ i, ∃ s, u i ≤ s ∧ s ≤ u (i + 1) ∧ pos s = K) :
    ∀ n : ℕ, 2 * K * n ≤ (u n : ℤ) - (u 0 : ℤ) := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
      obtain ⟨s, hs1, hs2, hs3⟩ := htop n
      have := round_trip_time hstep hs1 hs2 (hbot n) hs3 (hbot (n + 1))
      push_cast
      push_cast at ih
      linarith

end Ballistic

/-! ## The diffusive bound -/

variable {S : Type*} [Fintype S]

/-- Mean squared ladder displacement of a population vector `p`, for a rung labelling `h`. -/
def sqDisp (h : S → ℝ) (p : S → ℝ) : ℝ := ∑ x, p x * (h x) ^ 2

lemma vecMul_nonneg {M : S → S → ℝ} (hM : IsStoch M) {p : S → ℝ} (hp : ∀ x, 0 ≤ p x) :
    ∀ z, 0 ≤ vecMul p M z := by
  intro z
  exact Finset.sum_nonneg fun x _ => mul_nonneg (hp x) (hM.1 x z)

lemma vecMul_sum {M : S → S → ℝ} (hM : IsStoch M) {p : S → ℝ} (hp1 : ∑ x, p x = 1) :
    ∑ z, vecMul p M z = 1 := by
  unfold vecMul
  rw [Finset.sum_comm]
  calc ∑ x, ∑ z, p x * M x z = ∑ x, p x := by
        refine Finset.sum_congr rfl fun x _ => ?_
        rw [← Finset.mul_sum, hM.2 x, mul_one]
    _ = 1 := hp1

/-- **One sweep of a driftless nearest-neighbour ladder walk increases the mean squared
displacement by at most one.** -/
theorem sqDisp_step {M : S → S → ℝ} {h p : S → ℝ} (hM : IsStoch M) (hp : ∀ x, 0 ≤ p x)
    (hp1 : ∑ x, p x = 1)
    (hjump : ∀ x z, M x z * (h z - h x) ^ 2 ≤ M x z)
    (hdrift : ∀ x, ∑ z, M x z * (h z - h x) = 0) :
    sqDisp h (vecMul p M) ≤ sqDisp h p + 1 := by
  have key : ∀ x, ∑ z, M x z * (h z) ^ 2 ≤ (h x) ^ 2 + 1 := by
    intro x
    have expand : ∀ z, M x z * (h z) ^ 2
        = M x z * (h z - h x) ^ 2 + 2 * h x * (M x z * (h z - h x)) + (h x) ^ 2 * M x z := by
      intro z; ring
    calc ∑ z, M x z * (h z) ^ 2
        = (∑ z, M x z * (h z - h x) ^ 2) + 2 * h x * (∑ z, M x z * (h z - h x))
            + (h x) ^ 2 * ∑ z, M x z := by
          simp only [expand]
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
      _ ≤ (h x) ^ 2 + 1 := by
          have h4 : ∑ z, M x z * (h z - h x) ^ 2 ≤ 1 := by
            calc ∑ z, M x z * (h z - h x) ^ 2 ≤ ∑ z, M x z :=
                  Finset.sum_le_sum fun z _ => hjump x z
              _ = 1 := hM.2 x
          rw [hdrift x, hM.2 x]
          linarith
  calc sqDisp h (vecMul p M) = ∑ x, p x * ∑ z, M x z * (h z) ^ 2 := by
        unfold sqDisp vecMul
        simp only [Finset.sum_mul]
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun x _ => ?_
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun z _ => by ring
    _ ≤ ∑ x, p x * ((h x) ^ 2 + 1) := by
        refine Finset.sum_le_sum fun x _ => ?_
        exact mul_le_mul_of_nonneg_left (key x) (hp x)
    _ = sqDisp h p + 1 := by
        unfold sqDisp
        simp only [mul_add, mul_one]
        rw [Finset.sum_add_distrib, hp1]

/-- The ladder walk stays a probability distribution. -/
lemma isDist_iter {p : ℕ → S → ℝ} {M : ℕ → S → S → ℝ} (hM : ∀ t, IsStoch (M t))
    (hstep : ∀ t, p (t + 1) = vecMul (p t) (M t)) (hp0 : ∀ x, 0 ≤ p 0 x) (hp01 : ∑ x, p 0 x = 1) :
    ∀ T, (∀ x, 0 ≤ p T x) ∧ ∑ x, p T x = 1 := by
  intro T
  induction T with
  | zero => exact ⟨hp0, hp01⟩
  | succ n ih =>
      rw [hstep n]
      exact ⟨vecMul_nonneg (hM n) ih.1, vecMul_sum (hM n) ih.2⟩

/-- **Mean squared ladder displacement grows at most linearly in the number of sweeps.** -/
theorem sqDisp_growth {p : ℕ → S → ℝ} {M : ℕ → S → S → ℝ} {h : S → ℝ}
    (hM : ∀ t, IsStoch (M t)) (hstep : ∀ t, p (t + 1) = vecMul (p t) (M t))
    (hp0 : ∀ x, 0 ≤ p 0 x) (hp01 : ∑ x, p 0 x = 1)
    (hjump : ∀ t x z, M t x z * (h z - h x) ^ 2 ≤ M t x z)
    (hdrift : ∀ t x, ∑ z, M t x z * (h z - h x) = 0) :
    ∀ T : ℕ, sqDisp h (p T) ≤ sqDisp h (p 0) + T := by
  intro T
  induction T with
  | zero => simp
  | succ n ih =>
      have hd := isDist_iter hM hstep hp0 hp01 n
      have := sqDisp_step (hM n) hd.1 hd.2 (hjump n) (hdrift n)
      rw [hstep n]
      push_cast
      linarith

/-- **The probability of having travelled `R` rungs in `T` sweeps is at most `T / R²`.**
The replica starts at the bottom rung (`sqDisp h (p 0) = 0`, i.e. all initial weight on rungs
labelled `0`). -/
theorem reach_prob_le {p : ℕ → S → ℝ} {M : ℕ → S → S → ℝ} {h : S → ℝ} {R : ℝ}
    (hM : ∀ t, IsStoch (M t)) (hstep : ∀ t, p (t + 1) = vecMul (p t) (M t))
    (hp0 : ∀ x, 0 ≤ p 0 x) (hp01 : ∑ x, p 0 x = 1) (hstart : sqDisp h (p 0) = 0)
    (hjump : ∀ t x z, M t x z * (h z - h x) ^ 2 ≤ M t x z)
    (hdrift : ∀ t x, ∑ z, M t x z * (h z - h x) = 0)
    (hR : 0 < R) (T : ℕ) :
    ∑ x ∈ Finset.univ.filter (fun x => R ≤ |h x|), p T x ≤ (T : ℝ) / R ^ 2 := by
  have hpT := isDist_iter hM hstep hp0 hp01 T
  have hgrow := sqDisp_growth hM hstep hp0 hp01 hjump hdrift T
  rw [hstart, zero_add] at hgrow
  have hsub : R ^ 2 * ∑ x ∈ Finset.univ.filter (fun x => R ≤ |h x|), p T x ≤ sqDisp h (p T) := by
    rw [Finset.mul_sum]
    calc ∑ x ∈ Finset.univ.filter (fun x => R ≤ |h x|), R ^ 2 * p T x
        ≤ ∑ x ∈ Finset.univ.filter (fun x => R ≤ |h x|), p T x * (h x) ^ 2 := by
          refine Finset.sum_le_sum fun x hx => ?_
          have hx' : R ≤ |h x| := (Finset.mem_filter.mp hx).2
          have : R ^ 2 ≤ (h x) ^ 2 := by
            have : R ^ 2 ≤ |h x| ^ 2 := by nlinarith [hR.le]
            simpa [sq_abs] using this
          nlinarith [hpT.1 x]
      _ ≤ ∑ x, p T x * (h x) ^ 2 := by
          refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) ?_
          intro x _ _
          exact mul_nonneg (hpT.1 x) (sq_nonneg _)
      _ = sqDisp h (p T) := rfl
  rw [le_div_iff₀ (by positivity)]
  nlinarith [hsub, hgrow]

/-- **Reaching `R` rungs with probability at least one half takes at least `R²/2` sweeps.** -/
theorem sweeps_needed_for_traversal {p : ℕ → S → ℝ} {M : ℕ → S → S → ℝ} {h : S → ℝ} {R : ℝ}
    (hM : ∀ t, IsStoch (M t)) (hstep : ∀ t, p (t + 1) = vecMul (p t) (M t))
    (hp0 : ∀ x, 0 ≤ p 0 x) (hp01 : ∑ x, p 0 x = 1) (hstart : sqDisp h (p 0) = 0)
    (hjump : ∀ t x z, M t x z * (h z - h x) ^ 2 ≤ M t x z)
    (hdrift : ∀ t x, ∑ z, M t x z * (h z - h x) = 0)
    (hR : 0 < R) {T : ℕ}
    (hreach : (1 : ℝ) / 2 ≤ ∑ x ∈ Finset.univ.filter (fun x => R ≤ |h x|), p T x) :
    R ^ 2 / 2 ≤ (T : ℝ) := by
  have h1 := reach_prob_le hM hstep hp0 hp01 hstart hjump hdrift hR T
  have h2 : (1 : ℝ) / 2 ≤ (T : ℝ) / R ^ 2 := le_trans hreach h1
  rw [le_div_iff₀ (by positivity)] at h2
  linarith

end IDR.ReplicaTrip
