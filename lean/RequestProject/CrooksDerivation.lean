/-
# Part XCVI  Microscopic reversibility, derived from the dynamics

Part LXIII (`WorkTheorem.lean`) prices a pulling experiment on a disordered chain — Jarzynski,
the dissipation as a relative entropy, the histogram crossing, the estimator's dependence on rare
trajectories — but every one of those statements is a consequence of Crooks' relation, which that
part **assumes**.  The assumptions list said so: "Part LXIII takes Crooks' relation as a hypothesis
on a finite set of trajectories rather than deriving it from an underlying dynamics".

This file derives it, for the standard discrete-time driven dynamics.

The protocol.  Energies `H 0, H 1, …, H n` on a finite conformation space; between step `t` and
step `t+1` the energy is switched at fixed conformation — that is the *work* — and the chain then
relaxes with a kernel `K (t+1)` which satisfies **local detailed balance** with respect to the new
energy `H (t+1)` (`DetailedBalance`).  This is exactly what a Metropolis or Glauber move at
temperature `1/b` does, and it is the only physical input.

* `first_law` — the identity that organises the accounting: over the whole protocol,
  `H n (x n) − H 0 (x 0) = work + heat`.
* `path_weight_ratio` — the engine, by induction over the protocol: for *every* trajectory,
  `(∏ forward kernels)·e^{−b H₀(x₀)} = (∏ reverse kernels)·e^{−b Hₙ(xₙ)}·e^{b·W}`.  No
  normalisation, no partition function, just detailed balance applied `n` times.
* `revLaw_revPath` — the reversed trajectory under the reversed protocol has exactly the reverse
  product weight, by reflecting the index.
* `crooks_derived` — **the theorem**: with `dF = −(1/b)·log(Zₙ/Z₀)` the free-energy difference of
  the two equilibrium ensembles,

      fwd(x) = rev(reverse x) · exp( b · (W(x) − dF) )

  for every trajectory `x`.  Crooks' relation is a property of the dynamics, not an assumption
  about the experiment, and the `dF` appearing in it is the thermodynamic free-energy difference
  and nothing else.  Every consequence proved in Part LXIII therefore applies to this dynamics.
* `crooks_derived_ratio` — the same statement as the ratio of path probabilities, which is the
  form the histogram method uses.

What is still assumed, and cannot be removed, is local detailed balance itself: a dynamics that
violates it (a non-thermal noise source, a chemically driven step) has no Crooks relation, and
Parts XVIII–XIX are where such driven steady states are treated.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset

namespace IDR.CrooksDeriv

variable {S : Type} [Fintype S]

/-- Unnormalised Boltzmann weight at inverse temperature `b`. -/
noncomputable def bw (b : ℝ) (H : S → ℝ) (x : S) : ℝ := Real.exp (-b * H x)

/-- Partition function. -/
noncomputable def Zpart (b : ℝ) (H : S → ℝ) : ℝ := ∑ x, bw b H x

/-- Equilibrium (Boltzmann) population. -/
noncomputable def boltz (b : ℝ) (H : S → ℝ) (x : S) : ℝ := bw b H x / Zpart b H

omit [Fintype S] in
lemma bw_pos (b : ℝ) (H : S → ℝ) (x : S) : 0 < bw b H x := Real.exp_pos _

lemma Zpart_pos [Nonempty S] (b : ℝ) (H : S → ℝ) : 0 < Zpart b H :=
  Finset.sum_pos (fun x _ => bw_pos b H x) Finset.univ_nonempty

/-- **Local detailed balance** for one relaxation kernel with respect to one energy. -/
def DetailedBalance (b : ℝ) (H : S → ℝ) (K : S → S → ℝ) : Prop :=
  ∀ x y, bw b H x * K x y = bw b H y * K y x

/-- The work done on the chain: the energy changes made at fixed conformation. -/
def work (H : ℕ → S → ℝ) (n : ℕ) (x : ℕ → S) : ℝ :=
  ∑ t ∈ range n, (H (t+1) (x t) - H t (x t))

/-- The heat absorbed: the energy changes made at fixed energy function. -/
def heat (H : ℕ → S → ℝ) (n : ℕ) (x : ℕ → S) : ℝ :=
  ∑ t ∈ range n, (H (t+1) (x (t+1)) - H (t+1) (x t))

omit [Fintype S] in
/-- **The first law**: over the protocol, the energy change is the work plus the heat. -/
theorem first_law (H : ℕ → S → ℝ) (n : ℕ) (x : ℕ → S) :
    H n (x n) - H 0 (x 0) = work H n x + heat H n x := by
  unfold work heat
  rw [← Finset.sum_add_distrib]
  have : ∀ t ∈ range n, (H (t+1) (x t) - H t (x t)) + (H (t+1) (x (t+1)) - H (t+1) (x t))
      = H (t+1) (x (t+1)) - H t (x t) := by intro t _; ring
  rw [Finset.sum_congr rfl this]
  exact (Finset.sum_range_sub (fun t => H t (x t)) n).symm

/-- The product of forward transition probabilities along a trajectory. -/
def fwdProd (K : ℕ → S → S → ℝ) (n : ℕ) (x : ℕ → S) : ℝ :=
  ∏ t ∈ range n, K (t+1) (x t) (x (t+1))

/-- The product of the same kernels taken backwards. -/
def revProd (K : ℕ → S → S → ℝ) (n : ℕ) (x : ℕ → S) : ℝ :=
  ∏ t ∈ range n, K (t+1) (x (t+1)) (x t)

omit [Fintype S] in
/-- **The path weight ratio.**  Detailed balance applied along the whole protocol. -/
theorem path_weight_ratio {b : ℝ} {H : ℕ → S → ℝ} {K : ℕ → S → S → ℝ}
    (hDB : ∀ t, DetailedBalance b (H t) (K t)) (x : ℕ → S) :
    ∀ n : ℕ, fwdProd K n x * bw b (H 0) (x 0)
      = revProd K n x * bw b (H n) (x n) * Real.exp (b * work H n x) := by
  intro n
  induction n with
  | zero => simp [fwdProd, revProd, work]
  | succ n ih =>
      have hdb := hDB (n+1) (x n) (x (n+1))
      have hsplit : bw b (H n) (x n) * K (n+1) (x n) (x (n+1))
          = Real.exp (b * (H (n+1) (x n) - H n (x n))) *
            (bw b (H (n+1)) (x (n+1)) * K (n+1) (x (n+1)) (x n)) := by
        rw [← hdb]
        have hcomb : Real.exp (b * (H (n+1) (x n) - H n (x n))) * bw b (H (n+1)) (x n)
            = bw b (H n) (x n) := by
          unfold bw
          rw [← Real.exp_add]
          congr 1
          ring
        rw [← hcomb]
        ring
      calc fwdProd K (n+1) x * bw b (H 0) (x 0)
          = (fwdProd K n x * bw b (H 0) (x 0)) * K (n+1) (x n) (x (n+1)) := by
            unfold fwdProd
            rw [Finset.prod_range_succ]
            ring
        _ = (revProd K n x * bw b (H n) (x n) * Real.exp (b * work H n x))
              * K (n+1) (x n) (x (n+1)) := by rw [ih]
        _ = revProd K n x * Real.exp (b * work H n x)
              * (bw b (H n) (x n) * K (n+1) (x n) (x (n+1))) := by ring
        _ = revProd K n x * Real.exp (b * work H n x)
              * (Real.exp (b * (H (n+1) (x n) - H n (x n))) *
                 (bw b (H (n+1)) (x (n+1)) * K (n+1) (x (n+1)) (x n))) := by rw [hsplit]
        _ = revProd K (n+1) x * bw b (H (n+1)) (x (n+1)) * Real.exp (b * work H (n+1) x) := by
            unfold revProd work
            rw [Finset.prod_range_succ, Finset.sum_range_succ]
            rw [mul_add, Real.exp_add]
            ring

/-- The law of a trajectory under the forward protocol, started from equilibrium at `H 0`. -/
noncomputable def fwdLaw (b : ℝ) (H : ℕ → S → ℝ) (K : ℕ → S → S → ℝ) (n : ℕ) (x : ℕ → S) : ℝ :=
  boltz b (H 0) (x 0) * fwdProd K n x

/-- The law of a trajectory under the *reversed* protocol, started from equilibrium at `H n`:
the energies and the kernels are run backwards. -/
noncomputable def revLaw (b : ℝ) (H : ℕ → S → ℝ) (K : ℕ → S → S → ℝ) (n : ℕ) (y : ℕ → S) : ℝ :=
  boltz b (H n) (y 0) * ∏ t ∈ range n, K (n - t) (y t) (y (t+1))

/-- Time reversal of a trajectory of `n` steps. -/
def revPath (n : ℕ) (x : ℕ → S) : ℕ → S := fun t => x (n - t)

/-- The reversed protocol assigns the reversed trajectory exactly the backwards product weight. -/
theorem revLaw_revPath (b : ℝ) (H : ℕ → S → ℝ) (K : ℕ → S → S → ℝ) (n : ℕ) (x : ℕ → S) :
    revLaw b H K n (revPath n x) = boltz b (H n) (x n) * revProd K n x := by
  unfold revLaw revPath revProd
  simp only [Nat.sub_zero]
  congr 1
  rw [← Finset.prod_range_reflect (fun s => K (s+1) (x (s+1)) (x s)) n]
  refine Finset.prod_congr rfl ?_
  intro t ht
  have htn : t < n := Finset.mem_range.mp ht
  have h1 : n - 1 - t + 1 = n - t := by omega
  have h2 : n - (t + 1) = n - 1 - t := by omega
  rw [h1, h2]

/-- The free-energy difference of the two equilibrium ensembles. -/
noncomputable def dF (b : ℝ) (H : ℕ → S → ℝ) (n : ℕ) : ℝ :=
  -(1 / b) * Real.log (Zpart b (H n) / Zpart b (H 0))

/-- **Crooks' relation, derived.**  For a chain driven by any protocol whose relaxation steps
obey local detailed balance, the forward law of every trajectory and the reverse law of its time
reverse stand in the ratio `exp(b(W − ΔF))`, with `ΔF` the thermodynamic free-energy difference. -/
theorem crooks_derived [Nonempty S] {b : ℝ} (hb : b ≠ 0) {H : ℕ → S → ℝ} {K : ℕ → S → S → ℝ}
    (hDB : ∀ t, DetailedBalance b (H t) (K t)) (n : ℕ) (x : ℕ → S) :
    fwdLaw b H K n x
      = revLaw b H K n (revPath n x) * Real.exp (b * (work H n x - dF b H n)) := by
  have hZ0 : 0 < Zpart b (H 0) := Zpart_pos b (H 0)
  have hZn : 0 < Zpart b (H n) := Zpart_pos b (H n)
  have hexpdF : Real.exp (b * dF b H n) = Zpart b (H 0) / Zpart b (H n) := by
    unfold dF
    have h : b * (-(1 / b) * Real.log (Zpart b (H n) / Zpart b (H 0)))
        = Real.log (Zpart b (H 0) / Zpart b (H n)) := by
      rw [Real.log_div (ne_of_gt hZn) (ne_of_gt hZ0), Real.log_div (ne_of_gt hZ0) (ne_of_gt hZn)]
      field_simp
      ring
    rw [h, Real.exp_log (by positivity)]
  have hkey := path_weight_ratio hDB x n
  rw [revLaw_revPath]
  unfold fwdLaw boltz
  rw [mul_sub, Real.exp_sub, hexpdF]
  have hz0 : Zpart b (H 0) ≠ 0 := ne_of_gt hZ0
  have hzn : Zpart b (H n) ≠ 0 := ne_of_gt hZn
  field_simp
  nlinarith [hkey]

/-- The same statement in the ratio form used by the histogram method. -/
theorem crooks_derived_ratio [Nonempty S] {b : ℝ} (hb : b ≠ 0) {H : ℕ → S → ℝ}
    {K : ℕ → S → S → ℝ} (hDB : ∀ t, DetailedBalance b (H t) (K t)) (n : ℕ) (x : ℕ → S)
    (hpos : revLaw b H K n (revPath n x) ≠ 0) :
    fwdLaw b H K n x / revLaw b H K n (revPath n x)
      = Real.exp (b * (work H n x - dF b H n)) := by
  rw [crooks_derived hb hDB n x]
  field_simp

end IDR.CrooksDeriv
