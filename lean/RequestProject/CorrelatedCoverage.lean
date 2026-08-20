/-
# Part XCV  How much of the ensemble a *trajectory* has seen

Part LXXXVII (`Coverage.lean`) computes the population an ensemble sample has never visited, and
does it for `N` **independent** draws.  A molecular-dynamics trajectory, a Monte-Carlo chain and a
replica-exchange run are not independent draws, and the assumptions list recorded the gap: "a
molecular-dynamics trajectory is correlated, so its effective sample size is smaller and the
missing mass correspondingly larger; the bounds proved here are therefore optimistic for
trajectory data".  This file proves the trajectory version.

The sampler is an arbitrary finite-state Markov chain `M` — a Metropolis walk, a discretised
dynamics, any stochastic matrix.  Correlation is unrestricted; what is assumed instead is the
standard mixing input, a **Doeblin minorisation**: after `k` steps, from *any* starting
conformation, the chain reaches conformation `y` with probability at least `eps · w y`
(`Minorises`).  `eps = 1, k = 1` is exactly the independent sampler.

The machinery is deliberately elementary: no path measure is constructed.  `avoidOp M y` is the
sub-stochastic operator that deletes conformation `y`, and `iter (avoidOp M y) N 1` *is* the
probability that the trajectory misses `y` in its first `N` steps (`avoid_eq_one_of_frozen`
exhibits the extreme case).

* `avoid_le_one_sub` — one mixing block costs a factor: from every start, the chain misses `y`
  over `k` steps with probability at most `1 − eps·w y`.
* `avoid_block_le` — hence over `j` blocks, at most `(1 − eps·w y)^j`.
* `unseen_le` — **the theorem**: the expected unseen population of a trajectory of length `N` is
  at most `Σ_y w y · (1 − eps·w y)^(N/k)`.  The independent bound of Part LXXXVII, with `N`
  replaced by the *effective* sample size `N/k` and each population by `eps·w y`.
* `unseen_le_iid` — and it is a genuine generalisation: for the independent sampler
  (`M x z = w z`, `eps = 1`, `k = 1`) the bound is exactly the missing mass
  `Σ_y w y (1 − w y)^N` of Part LXXXVII.
* `avoid_eq_one_of_frozen` — the necessity of a mixing hypothesis: a chain that never moves
  misses every other conformation forever, at every trajectory length.  Sample size alone
  certifies nothing about coverage; only sample size divided by the mixing time does.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset

namespace IDR.CorrCoverage

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- One step of a (sub)stochastic operator acting on functions of the conformation. -/
def app (M : S → S → ℝ) (f : S → ℝ) : S → ℝ := fun x => ∑ z, M x z * f z

/-- `n` steps. -/
def iter (M : S → S → ℝ) : ℕ → (S → ℝ) → (S → ℝ)
  | 0, f => f
  | (n+1), f => app M (iter M n f)

/-- A stochastic matrix: the sampler. -/
def IsStoch (M : S → S → ℝ) : Prop := (∀ x z, 0 ≤ M x z) ∧ (∀ x, ∑ z, M x z = 1)

/-- The constant function `1`. -/
def one' : S → ℝ := fun _ => 1

/-- The indicator of a single conformation. -/
def ind (y : S) : S → ℝ := fun z => if z = y then 1 else 0

/-- The operator with conformation `y` deleted: `iter (avoidOp M y) N one'` is the probability
that the trajectory avoids `y` at each of its first `N` steps. -/
def avoidOp (M : S → S → ℝ) (y : S) : S → S → ℝ := fun x z => if z = y then 0 else M x z

omit [DecidableEq S] in
lemma app_mono {M : S → S → ℝ} (hM : ∀ x z, 0 ≤ M x z) {f g : S → ℝ} (h : ∀ z, f z ≤ g z) :
    ∀ x, app M f x ≤ app M g x := by
  intro x
  apply Finset.sum_le_sum
  intro z _
  exact mul_le_mul_of_nonneg_left (h z) (hM x z)

omit [DecidableEq S] in
lemma iter_mono {M : S → S → ℝ} (hM : ∀ x z, 0 ≤ M x z) :
    ∀ (n : ℕ) {f g : S → ℝ}, (∀ z, f z ≤ g z) → ∀ x, iter M n f x ≤ iter M n g x := by
  intro n
  induction n with
  | zero => intro f g h x; exact h x
  | succ n ih => intro f g h x; exact app_mono hM (fun z => ih (f := f) (g := g) h z) x

omit [DecidableEq S] in
lemma iter_nonneg {M : S → S → ℝ} (hM : ∀ x z, 0 ≤ M x z) :
    ∀ (n : ℕ) {f : S → ℝ}, (∀ z, 0 ≤ f z) → ∀ x, 0 ≤ iter M n f x := by
  intro n
  induction n with
  | zero => intro f h x; exact h x
  | succ n ih =>
      intro f h x
      exact Finset.sum_nonneg fun z _ => mul_nonneg (hM x z) (ih h z)

omit [DecidableEq S] in
lemma iter_smul (M : S → S → ℝ) (c : ℝ) :
    ∀ (n : ℕ) (f : S → ℝ) (x : S), iter M n (fun z => c * f z) x = c * iter M n f x := by
  intro n
  induction n with
  | zero => intro f x; rfl
  | succ n ih =>
      intro f x
      show ∑ z, M x z * iter M n (fun z => c * f z) z = c * ∑ z, M x z * iter M n f z
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl ?_
      intro z _
      rw [ih f z]
      ring

omit [DecidableEq S] in
lemma iter_sub (M : S → S → ℝ) :
    ∀ (n : ℕ) (f g : S → ℝ) (x : S),
      iter M n (fun z => f z - g z) x = iter M n f x - iter M n g x := by
  intro n
  induction n with
  | zero => intro f g x; rfl
  | succ n ih =>
      intro f g x
      show ∑ z, M x z * iter M n (fun z => f z - g z) z
          = (∑ z, M x z * iter M n f z) - ∑ z, M x z * iter M n g z
      rw [← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl ?_
      intro z _
      rw [ih f g z]
      ring

omit [DecidableEq S] in
lemma iter_one' {M : S → S → ℝ} (hM : IsStoch M) :
    ∀ (n : ℕ) (x : S), iter M n (one' : S → ℝ) x = 1 := by
  intro n
  induction n with
  | zero => intro x; rfl
  | succ n ih =>
      intro x
      show ∑ z, M x z * iter M n (one' : S → ℝ) z = 1
      simp only [ih]
      simpa using hM.2 x

omit [DecidableEq S] in
lemma iter_add (M : S → S → ℝ) :
    ∀ (a b : ℕ) (f : S → ℝ) (x : S), iter M (a + b) f x = iter M a (iter M b f) x := by
  intro a
  induction a with
  | zero => intro b f x; simp [iter]
  | succ a ih =>
      intro b f x
      have hre : a + 1 + b = (a + b) + 1 := by ring
      rw [hre]
      show app M (iter M (a + b) f) x = app M (iter M a (iter M b f)) x
      unfold app
      exact Finset.sum_congr rfl fun z _ => by rw [ih b f z]

lemma avoidOp_nonneg {M : S → S → ℝ} (hM : IsStoch M) (y : S) :
    ∀ x z, 0 ≤ avoidOp M y x z := by
  intro x z
  unfold avoidOp
  split_ifs
  · exact le_rfl
  · exact hM.1 x z

lemma avoidOp_le {M : S → S → ℝ} (hM : IsStoch M) (y : S) : ∀ x z, avoidOp M y x z ≤ M x z := by
  intro x z
  unfold avoidOp
  split_ifs
  · exact hM.1 x z
  · exact le_rfl

/-- Missing `y` at every step of `n+1` steps is at most missing it at the last step. -/
lemma avoid_le_indicator {M : S → S → ℝ} (hM : IsStoch M) (y : S) :
    ∀ (n : ℕ) (x : S),
      iter (avoidOp M y) (n+1) (one' : S → ℝ) x
        ≤ iter M (n+1) (fun z => one' z - ind y z) x := by
  intro n
  induction n with
  | zero =>
      intro x
      show ∑ z, avoidOp M y x z * one' z ≤ ∑ z, M x z * (one' z - ind y z)
      apply Finset.sum_le_sum
      intro z _
      by_cases h : z = y
      · simp [avoidOp, ind, one', h]
      · simp [avoidOp, ind, one', h]
  | succ n ih =>
      intro x
      show ∑ z, avoidOp M y x z * iter (avoidOp M y) (n+1) (one' : S → ℝ) z
          ≤ ∑ z, M x z * iter M (n+1) (fun z => one' z - ind y z) z
      have hnn : ∀ z, 0 ≤ iter M (n+1) (fun z => one' z - ind y z) z := by
        intro z
        apply iter_nonneg hM.1
        intro u
        by_cases h : u = y <;> simp [one', ind, h]
      apply Finset.sum_le_sum
      intro z _
      have h1 : avoidOp M y x z * iter (avoidOp M y) (n+1) (one' : S → ℝ) z
          ≤ avoidOp M y x z * iter M (n+1) (fun z => one' z - ind y z) z :=
        mul_le_mul_of_nonneg_left (ih z) (avoidOp_nonneg hM y x z)
      have h2 : avoidOp M y x z * iter M (n+1) (fun z => one' z - ind y z) z
          ≤ M x z * iter M (n+1) (fun z => one' z - ind y z) z :=
        mul_le_mul_of_nonneg_right (avoidOp_le hM y x z) (hnn z)
      linarith

/-- **The Doeblin minorisation.**  After `k` steps, from every start, the sampler reaches `y` with
probability at least `eps · w y`.  For the independent sampler this holds with `eps = 1, k = 1`. -/
def Minorises (M : S → S → ℝ) (k : ℕ) (eps : ℝ) (w : S → ℝ) : Prop :=
  ∀ y x, eps * w y ≤ iter M k (ind y) x

/-- One mixing block costs a factor `1 − eps·w y`. -/
theorem avoid_le_one_sub {M : S → S → ℝ} (hM : IsStoch M) {k : ℕ} (hk : 0 < k) {eps : ℝ}
    {w : S → ℝ} (hmin : Minorises M k eps w) (y : S) (x : S) :
    iter (avoidOp M y) k (one' : S → ℝ) x ≤ 1 - eps * w y := by
  obtain ⟨n, rfl⟩ : ∃ n, k = n + 1 := ⟨k - 1, by omega⟩
  have h1 := avoid_le_indicator hM y n x
  have h2 : iter M (n+1) (fun z => one' z - ind y z) x
      = iter M (n+1) (one' : S → ℝ) x - iter M (n+1) (ind y) x :=
    iter_sub M (n+1) one' (ind y) x
  have h3 : iter M (n+1) (one' : S → ℝ) x = 1 := iter_one' hM (n+1) x
  have h4 : eps * w y ≤ iter M (n+1) (ind y) x := hmin y x
  linarith [h1, h2 ▸ h1]

/-- Over `j` mixing blocks the avoidance probability is at most `(1 − eps·w y)^j`. -/
theorem avoid_block_le {M : S → S → ℝ} (hM : IsStoch M) {k : ℕ} (hk : 0 < k) {eps : ℝ}
    {w : S → ℝ} (hmin : Minorises M k eps w) (hnn : ∀ y, 0 ≤ 1 - eps * w y) (y : S) :
    ∀ (j : ℕ) (x : S), iter (avoidOp M y) (j * k) (one' : S → ℝ) x ≤ (1 - eps * w y) ^ j := by
  intro j
  induction j with
  | zero => intro x; simp [iter, one']
  | succ j ih =>
      intro x
      have hstep : (j + 1) * k = k + j * k := by ring
      rw [hstep, iter_add (avoidOp M y) k (j * k) one' x]
      have hb : ∀ z, iter (avoidOp M y) (j * k) (one' : S → ℝ) z
          ≤ (1 - eps * w y) ^ j * one' z := by
        intro z; simpa [one'] using ih z
      have hmono := iter_mono (avoidOp_nonneg hM y) k (f := iter (avoidOp M y) (j*k) one')
        (g := fun z => (1 - eps * w y) ^ j * one' z) hb x
      have hsm := iter_smul (avoidOp M y) ((1 - eps * w y) ^ j) k one' x
      have hone := avoid_le_one_sub hM hk hmin y x
      have hpow : (0:ℝ) ≤ (1 - eps * w y) ^ j := pow_nonneg (hnn y) j
      calc iter (avoidOp M y) k (iter (avoidOp M y) (j * k) one') x
          ≤ iter (avoidOp M y) k (fun z => (1 - eps * w y) ^ j * one' z) x := hmono
        _ = (1 - eps * w y) ^ j * iter (avoidOp M y) k one' x := hsm
        _ ≤ (1 - eps * w y) ^ j * (1 - eps * w y) := by
              exact mul_le_mul_of_nonneg_left hone hpow
        _ = (1 - eps * w y) ^ (j + 1) := by ring

/-- The expected unseen population of a trajectory of `N` steps started from `w`. -/
noncomputable def unseen (M : S → S → ℝ) (w : S → ℝ) (N : ℕ) : ℝ :=
  ∑ y, w y * ∑ x, w x * iter (avoidOp M y) N (one' : S → ℝ) x

/-- **The coverage bound for a correlated sampler.**  With a Doeblin minorisation at lag `k` and
constant `eps`, the expected unseen population after `N` steps is at most
`Σ_y w y (1 − eps·w y)^(N/k)`: the independent bound of Part LXXXVII at the *effective* sample
size `N/k`. -/
theorem unseen_le {M : S → S → ℝ} (hM : IsStoch M) {k : ℕ} (hk : 0 < k) {eps : ℝ} {w : S → ℝ}
    (hw : ∀ y, 0 ≤ w y) (hwsum : ∑ y, w y = 1) (hmin : Minorises M k eps w)
    (hnn : ∀ y, 0 ≤ 1 - eps * w y) (N : ℕ) :
    unseen M w N ≤ ∑ y, w y * (1 - eps * w y) ^ (N / k) := by
  have hstep : ∀ y x, iter (avoidOp M y) N (one' : S → ℝ) x ≤ (1 - eps * w y) ^ (N / k) := by
    intro y x
    have hdec : N = (N / k) * k + N % k := by
      rw [Nat.mul_comm]
      exact (Nat.div_add_mod N k).symm
    have hrest : ∀ z, iter (avoidOp M y) (N % k) (one' : S → ℝ) z ≤ one' z := by
      intro z
      have : iter (avoidOp M y) (N % k) (one' : S → ℝ) z ≤ iter M (N % k) (one' : S → ℝ) z := by
        clear hdec
        induction (N % k) generalizing z with
        | zero => exact le_rfl
        | succ n ih =>
            show ∑ u, avoidOp M y z u * iter (avoidOp M y) n (one' : S → ℝ) u
                ≤ ∑ u, M z u * iter M n (one' : S → ℝ) u
            apply Finset.sum_le_sum
            intro u _
            have h1 : avoidOp M y z u * iter (avoidOp M y) n (one' : S → ℝ) u
                ≤ avoidOp M y z u * iter M n (one' : S → ℝ) u :=
              mul_le_mul_of_nonneg_left (ih u) (avoidOp_nonneg hM y z u)
            have h2 : avoidOp M y z u * iter M n (one' : S → ℝ) u
                ≤ M z u * iter M n (one' : S → ℝ) u := by
              refine mul_le_mul_of_nonneg_right (avoidOp_le hM y z u) ?_
              exact iter_nonneg hM.1 n (fun _ => by norm_num [one']) u
            linarith
      simpa [one', iter_one' hM (N % k) z] using this
    calc iter (avoidOp M y) N (one' : S → ℝ) x
        = iter (avoidOp M y) ((N / k) * k + N % k) (one' : S → ℝ) x := by rw [← hdec]
      _ = iter (avoidOp M y) ((N / k) * k) (iter (avoidOp M y) (N % k) one') x :=
          iter_add (avoidOp M y) _ _ _ _
      _ ≤ iter (avoidOp M y) ((N / k) * k) (one' : S → ℝ) x :=
          iter_mono (avoidOp_nonneg hM y) _ hrest x
      _ ≤ (1 - eps * w y) ^ (N / k) := avoid_block_le hM hk hmin hnn y (N / k) x
  unfold unseen
  apply Finset.sum_le_sum
  intro y _
  have hinner : ∑ x, w x * iter (avoidOp M y) N (one' : S → ℝ) x ≤ (1 - eps * w y) ^ (N / k) := by
    calc ∑ x, w x * iter (avoidOp M y) N (one' : S → ℝ) x
        ≤ ∑ x, w x * (1 - eps * w y) ^ (N / k) :=
          Finset.sum_le_sum fun x _ => mul_le_mul_of_nonneg_left (hstep y x) (hw x)
      _ = (1 - eps * w y) ^ (N / k) := by rw [← Finset.sum_mul, hwsum, one_mul]
  exact mul_le_mul_of_nonneg_left hinner (hw y)

/-- The independent sampler satisfies the minorisation with `eps = 1` at lag `1`, so the bound
above reduces to exactly the missing mass `Σ_y w y (1 − w y)^N` of Part LXXXVII: the trajectory
theorem contains the independent one. -/
theorem unseen_le_iid {w : S → ℝ} (hw : ∀ y, 0 ≤ w y) (hwsum : ∑ y, w y = 1)
    (hle : ∀ y, w y ≤ 1) (N : ℕ) :
    unseen (fun _ z => w z) w N ≤ ∑ y, w y * (1 - w y) ^ N := by
  have hM : IsStoch (fun _ z => w z) := ⟨fun _ z => hw z, fun _ => hwsum⟩
  have hmin : Minorises (fun _ z => w z) 1 1 w := by
    intro y x
    show 1 * w y ≤ ∑ z, w z * ind y z
    rw [Finset.sum_eq_single y]
    · simp [ind]
    · intro b _ hb; simp [ind, hb]
    · intro h; exact absurd (Finset.mem_univ y) h
  have := unseen_le hM (by norm_num) hw hwsum hmin (fun y => by linarith [hle y]) N
  simpa using this

/-- **Why a mixing hypothesis is unavoidable.**  A sampler that never moves misses every other
conformation at every trajectory length: without a bound on the correlation, sample size says
nothing at all about coverage. -/
theorem avoid_eq_one_of_frozen (y : S) :
    ∀ (N : ℕ) (x : S), x ≠ y →
      iter (avoidOp (fun a b : S => if b = a then (1:ℝ) else 0) y) N (one' : S → ℝ) x = 1 := by
  intro N
  induction N with
  | zero => intro x _; rfl
  | succ n ih =>
      intro x hx
      show ∑ z, avoidOp (fun a b : S => if b = a then (1:ℝ) else 0) y x z
          * iter (avoidOp (fun a b : S => if b = a then (1:ℝ) else 0) y) n (one' : S → ℝ) z = 1
      rw [Finset.sum_eq_single x]
      · simp [avoidOp, hx, ih x hx]
      · intro b _ hb
        simp [avoidOp, hb]
      · intro h
        exact absurd (Finset.mem_univ x) h

end IDR.CorrCoverage
