/-
# The *decision* problem, and the size of the reduction

`AUCHardness.lean` builds, from an arbitrary weight matrix `W : Fin K → Fin K → ℕ`, a residue
table whose reachable pooled Mann–Whitney optimum over per-protein biases is

  `max_b U_pooled(s + b) = U_within(s) + C + lopOpt W`     (`bias_optimum_eq_lop`)

with `C = |basePairs|` a constant read off the table.  That is the *optimisation* form of the
reduction.  This file supplies the three things a hardness claim actually quotes.

* `bias_threshold_iff_lop` — **the decision form.**  For every threshold `m : ℕ`,

    `(∃ b, U_pooled(s + b) ≥ U_within(s) + C + m)  ↔  m ≤ lopOpt W`.

  Deciding whether a per-protein recalibration can push the pooled statistic past a stated target
  is therefore *exactly* the decision version of weighted linear ordering, which is NP-complete
  (Karp; Garey–Johnson GT44).

* `ceiling_deficit_threshold_iff_lop`, `ceiling_deficit_eq` — **the "ceiling minus `d` pairs"
  form**, which is the way the question is posed in practice: the ceiling
  `U_within + |betweenPairs|` is the value at which every cross-protein comparison comes out
  right, and the question is how many pairs below it the best bias lands.  The deficit of the
  constructed table is `|betweenPairs| − C − lopOpt W`, so deciding "is the ceiling attainable up
  to `d` pairs?" is again linear ordering.

* `exists_good_multiplicity_card_le`, `lop_reduces_to_bias_optimum_poly` — **the reduction is
  polynomially sized.**  The background multiplicity `M` needed to pin the optimum inside the box
  can be chosen so that the whole table has at most `26·(K+1)²·(T+1)` residues, where
  `T = slotCount K W` is the total weight of the instance.  Since weighted linear ordering is
  NP-hard already for 0/1 weights, where `T ≤ K²`, the table has `O(K⁴)` residues: the map from a
  linear ordering instance to a score table is a genuine polynomial-time many-one reduction, not
  merely an identity between two optima.

What is machine-checked here (as in `AUCHardness.lean`) is the reduction; the NP-hardness of
weighted linear ordering itself is the classical input, and is quoted, not proved.
-/
import RequestProject.AUCHardness

set_option autoImplicit false

namespace IDR.GroupedAUC

open Finset

namespace Hardness

section Decision

variable {K M : ℕ} {W : Fin K → Fin K → ℕ}

/-- **The decision version of the recalibration problem is the decision version of linear
ordering.**

For the score table built from `W`, a per-protein bias pushing the pooled Mann–Whitney statistic
to at least `U_within + C + m` exists precisely when the linear ordering instance `W` has value at
least `m`. -/
theorem bias_threshold_iff_lop (hW : ∀ k, W k k = 0)
    (hM : 2 * (Fintype.card (Itm K M W) * slotCount K W) ≤ M * M) (m : ℕ) :
    (∃ b : Fin K → ℝ,
        U (withinPairs (lab K M W) (grp K M W)) (scr K M W)
            + (basePairs K M W).card + (m : ℝ)
          ≤ U (allPairs (lab K M W)) (shift (grp K M W) b (scr K M W)))
      ↔ m ≤ lopOpt W := by
  obtain ⟨hup, b₀, hb₀⟩ := bias_optimum_eq_lop hW hM
  constructor
  · rintro ⟨b, hb⟩
    have h := hup b
    have : (m : ℝ) ≤ (lopOpt W : ℝ) := by linarith
    exact_mod_cast this
  · intro hm
    refine ⟨b₀, ?_⟩
    have : (m : ℝ) ≤ (lopOpt W : ℝ) := by exact_mod_cast hm
    rw [hb₀]
    linarith

/-- The exact deficit of the constructed table from its ceiling: the best bias falls short of
`U_within + |betweenPairs|` by exactly `|betweenPairs| − C − lopOpt W` comparison pairs. -/
theorem ceiling_deficit_eq (hW : ∀ k, W k k = 0)
    (hM : 2 * (Fintype.card (Itm K M W) * slotCount K W) ≤ M * M) :
    ∃ b : Fin K → ℝ,
      U (allPairs (lab K M W)) (shift (grp K M W) b (scr K M W))
        = (U (withinPairs (lab K M W) (grp K M W)) (scr K M W)
            + (betweenPairs (lab K M W) (grp K M W)).card)
          - (((betweenPairs (lab K M W) (grp K M W)).card - (basePairs K M W).card
              - (lopOpt W : ℝ))) := by
  obtain ⟨-, b₀, hb₀⟩ := bias_optimum_eq_lop hW hM
  exact ⟨b₀, by rw [hb₀]; ring⟩

/-- **"Is the ceiling attainable up to `d` pairs?" is linear ordering.**

The best bias comes within `d` comparison pairs of the ceiling `U_within + |betweenPairs|`
exactly when the linear ordering optimum of `W` is at least `|betweenPairs| − C − d`. -/
theorem ceiling_deficit_threshold_iff_lop (hW : ∀ k, W k k = 0)
    (hM : 2 * (Fintype.card (Itm K M W) * slotCount K W) ≤ M * M) (d : ℝ) :
    (∃ b : Fin K → ℝ,
        (U (withinPairs (lab K M W) (grp K M W)) (scr K M W)
            + (betweenPairs (lab K M W) (grp K M W)).card) - d
          ≤ U (allPairs (lab K M W)) (shift (grp K M W) b (scr K M W)))
      ↔ ((betweenPairs (lab K M W) (grp K M W)).card - (basePairs K M W).card - d
            ≤ (lopOpt W : ℝ)) := by
  obtain ⟨hup, b₀, hb₀⟩ := bias_optimum_eq_lop hW hM
  constructor
  · rintro ⟨b, hb⟩
    have h := hup b
    linarith
  · intro hd
    exact ⟨b₀, by rw [hb₀]; linarith⟩

end Decision

/-! ## The reduction is polynomially sized -/

section Size

/-- A background multiplicity that both pins the optimum inside the box and keeps the table
polynomially large: at most `26·(K+1)²·(T+1)` residues, with `T = slotCount K W` the total weight
of the linear ordering instance. -/
theorem exists_good_multiplicity_card_le (K : ℕ) (W : Fin K → Fin K → ℕ) :
    ∃ M : ℕ, 2 * (Fintype.card (Itm K M W) * slotCount K W) ≤ M * M ∧
      Fintype.card (Itm K M W) ≤ 26 * (K + 1) ^ 2 * (slotCount K W + 1) := by
  classical
  set T := slotCount K W with hT
  refine ⟨4 * K * T + 4 * T + 4, ?_, ?_⟩
  · rw [card_Itm]
    set M := 4 * K * T + 4 * T + 4 with hM
    have h : 2 * ((2 * (K * M) + 2 * T) * T) = 4 * K * M * T + 4 * T * T := by ring
    rw [h, hM]
    nlinarith [Nat.zero_le K, Nat.zero_le T, Nat.zero_le (K * T), Nat.zero_le (T * T),
      Nat.zero_le (K * T * T)]
  · rw [card_Itm]
    nlinarith [Nat.zero_le K, Nat.zero_le T, Nat.zero_le (K * T), Nat.zero_le (T * T),
      Nat.zero_le (K * T * T), Nat.zero_le (K * K * T)]

/-- **Every weighted linear ordering instance is a per-protein recalibration problem, through a
polynomially sized table.**

Given any weight matrix `W` on `K` proteins with vanishing diagonal there is a residue table with
at most `26·(K+1)²·(T+1)` residues (`T` the total weight of `W`) whose reachable pooled optimum
over per-protein biases is `U_within + C + lopOpt W`, and for which reaching a stated target is
equivalent to the linear ordering instance meeting the corresponding threshold. -/
theorem lop_reduces_to_bias_optimum_poly (K : ℕ) (W : Fin K → Fin K → ℕ) (hW : ∀ k, W k k = 0) :
    ∃ (I : Type) (_ : Fintype I) (_ : DecidableEq I) (lab : I → Bool) (grp : I → Fin K)
      (s : I → ℝ) (C : ℝ),
      Fintype.card I ≤ 26 * (K + 1) ^ 2 * (slotCount K W + 1) ∧
      (∀ b : Fin K → ℝ, U (allPairs lab) (shift grp b s)
          ≤ U (withinPairs lab grp) s + C + (lopOpt W : ℝ)) ∧
      (∃ b : Fin K → ℝ, U (allPairs lab) (shift grp b s)
          = U (withinPairs lab grp) s + C + (lopOpt W : ℝ)) ∧
      (∀ m : ℕ, (∃ b : Fin K → ℝ,
          U (withinPairs lab grp) s + C + (m : ℝ) ≤ U (allPairs lab) (shift grp b s))
        ↔ m ≤ lopOpt W) := by
  classical
  obtain ⟨M, hM, hcard⟩ := exists_good_multiplicity_card_le K W
  obtain ⟨hup, hex⟩ := bias_optimum_eq_lop (M := M) hW hM
  exact ⟨Itm K M W, inferInstance, inferInstance, lab K M W, grp K M W, scr K M W,
    ((basePairs K M W).card : ℝ), hcard, hup, hex, fun m => bias_threshold_iff_lop hW hM m⟩

end Size

end Hardness

end IDR.GroupedAUC
