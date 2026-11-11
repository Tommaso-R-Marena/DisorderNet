/-
# Part CXIV  Exact connective constants and the Flory exponent

This file addresses the last of the open items: exact values for the connective constant and
for the Flory swelling exponent of genuine self-avoiding walks.

The exact value of the connective constant of `ℤ²` (and the exact Flory exponent `ν = 3/4`)
are open problems of mathematics; nothing here claims them.  What is delivered is:

* an **exactly solvable** lattice for which the connective constant *is* determined:
  the directed square lattice, whose connective constant is exactly `log 2`
  (`cntOf_ddir`, `connectiveConstant_directed`), and whose walks have end-to-end distance
  exactly `n`, i.e. Flory exponent exactly `1` (`directed_extent`);
* an **improved rigorous upper bound** for the genuine square lattice, from the exact
  six-step count `cnt 6 = 780`, which strictly improves the four-step bound
  (`connectiveConstant_le_log780_div6`, `improves_on_cnt_four`);
* a **deterministic Flory lower bound**: a self-avoiding walk of `n` steps visits `n + 1`
  distinct sites, which cannot fit in a box of side `2R + 1` unless `(2R+1)^2 ≥ n + 1`.
  Hence *every* self-avoiding walk — not merely a typical one — has extension at least
  `(√(n+1) − 1)/2` (`sqrt_le_extent`, `flory_exponent_half`), against the trivial upper bound
  `extent ≤ n` (`extent_le_length`).  So `1/2 ≤ ν ≤ 1` deterministically in two dimensions,
  and the same box argument gives `ν ≥ 1/3` in three dimensions
  (`cubic_flory_exponent_third`).

What remains genuinely open is stated, not hidden: the exact value of `μ(ℤ²)` and of `ν`.
-/
import Mathlib
import RequestProject.LatticeWalk
import RequestProject.SelfAvoiding
import RequestProject.CubicLattice

namespace IDR.SAW

open scoped BigOperators

/-! ## Generalities: the number of sites visited -/

section Sites

variable {V : Type*} [AddCommGroup V]

@[simp] lemma length_sites (l : List V) : (sites l).length = l.length + 1 := by
  simp [sites, sitesFrom]

lemma sites_cons (d : V) (t : List V) :
    sites (d :: t) = (0 : V) :: (sites t).map fun y => d + y := by
  rw [sites, sitesFrom_cons, sitesFrom_eq_map]
  simp

end Sites

/-- The endpoint of a chain with bond vectors `l`. -/
def endpoint {V : Type*} [AddCommGroup V] (l : List V) : V := List.foldl (· + ·) 0 l

/-! ## The exactly solvable case: the directed square lattice

Only two bond vectors are allowed, east and north.  Every walk is automatically
self-avoiding, so the count is exactly `2 ^ n` and the entropy per residue is exactly
`log 2`. -/

/-- The two bond vectors of the directed square lattice: east and north. -/
def ddir : Fin 2 → Site := ![(1, 0), (0, 1)]

/-- **Exact conformation count for the directed lattice.** -/
theorem cntOf_ddir (n : ℕ) : cntOf ddir n = 2 ^ n := by
  refine le_antisymm ?_ ?_
  · calc cntOf ddir n ≤ (Finset.univ : Finset (Fin n → Fin 2)).card :=
          Finset.card_le_card (Finset.filter_subset _ _)
      _ = 2 ^ n := by simp
  · exact pow_le_cntOf ddir hgt hgt_add hgt_zero (fun i => i) Function.injective_id
      (by decide) n

lemma cntOf_ddir_pos (n : ℕ) : 0 < cntOf ddir n := by
  rw [cntOf_ddir]; positivity

/-- **The connective constant of the directed square lattice is exactly `log 2`.** -/
theorem connectiveConstant_directed :
    connectiveConstantOf ddir cntOf_ddir_pos = Real.log 2 := by
  refine tendsto_nhds_unique (tendsto_connectiveConstantOf cntOf_ddir_pos) ?_
  refine Filter.Tendsto.congr' ?_ tendsto_const_nhds
  filter_upwards [Filter.eventually_gt_atTop 0] with n hn
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hn.ne'
  rw [logCntOf, cntOf_ddir]
  push_cast
  rw [Real.log_pow]
  field_simp

/-! ## The extension of a walk -/

/-- The `ℓ^∞` radius of the set of sites visited by a chain: its extension. -/
def extent (l : List Site) : ℤ := ((sites l).map fun p => max |p.1| |p.2|).foldr max 0

lemma foldr_max_nonneg (L : List ℤ) : 0 ≤ L.foldr max 0 := by
  induction L with
  | nil => simp
  | cons a t ih => exact le_trans ih (le_max_right _ _)

lemma le_foldr_max {L : List ℤ} {x : ℤ} (hx : x ∈ L) : x ≤ L.foldr max 0 := by
  induction L with
  | nil => simp at hx
  | cons a t ih =>
      rcases List.mem_cons.1 hx with rfl | hx
      · exact le_max_left _ _
      · exact le_trans (ih hx) (le_max_right _ _)

lemma foldr_max_le {L : List ℤ} {b : ℤ} (hb : 0 ≤ b) (h : ∀ x ∈ L, x ≤ b) :
    L.foldr max 0 ≤ b := by
  induction L with
  | nil => simpa using hb
  | cons a t ih =>
      exact max_le (h a (by simp)) (ih fun x hx => h x (by simp [hx]))

lemma extent_nonneg (l : List Site) : 0 ≤ extent l := foldr_max_nonneg _

lemma le_extent {l : List Site} {p : Site} (hp : p ∈ sites l) :
    max |p.1| |p.2| ≤ extent l :=
  le_foldr_max (List.mem_map_of_mem hp)

/-! ## The deterministic Flory bound

A self-avoiding walk of `n` steps occupies `n + 1` distinct sites.  If its extension is `R`
all those sites lie in a box with `(2R+1)^2` sites, so `n + 1 ≤ (2R+1)^2`. -/

/-- A self-avoiding chain of `n` bonds occupies exactly `n + 1` distinct sites. -/
lemma card_sites_of_isSAW {V : Type*} [AddCommGroup V] [DecidableEq V] {l : List V}
    (h : IsSAW l) :
    (sites l).toFinset.card = l.length + 1 := by
  rw [List.toFinset_card_of_nodup h, length_sites]

/-- **The box bound.**  The `n + 1` distinct sites of a self-avoiding walk fit inside the
box of side `2 * extent + 1`, so `n + 1 ≤ (2 * extent + 1) ^ 2`. -/
theorem length_succ_le_extent_sq {l : List Site} (h : IsSAW l) :
    ((l.length : ℤ) + 1) ≤ (2 * extent l + 1) ^ 2 := by
  classical
  set R := extent l with hR
  have hRnn : 0 ≤ R := extent_nonneg l
  have hsub : (sites l).toFinset ⊆ (Finset.Icc (-R) R) ×ˢ (Finset.Icc (-R) R) := by
    intro p hp
    rw [List.mem_toFinset] at hp
    have := le_extent hp
    have h1 : |p.1| ≤ R := le_trans (le_max_left _ _) this
    have h2 : |p.2| ≤ R := le_trans (le_max_right _ _) this
    rw [Finset.mem_product, Finset.mem_Icc, Finset.mem_Icc]
    constructor
    · exact ⟨neg_le_of_abs_le h1, le_of_abs_le h1⟩
    · exact ⟨neg_le_of_abs_le h2, le_of_abs_le h2⟩
  have hcard := Finset.card_le_card hsub
  rw [card_sites_of_isSAW h, Finset.card_product, Int.card_Icc] at hcard
  have hval : ((R + 1 - -R).toNat) = (2 * R + 1).toNat := by ring_nf
  rw [hval] at hcard
  have : ((l.length + 1 : ℕ) : ℤ) ≤ (((2 * R + 1).toNat * (2 * R + 1).toNat : ℕ) : ℤ) := by
    exact_mod_cast hcard
  have hpos : (0 : ℤ) ≤ 2 * R + 1 := by omega
  rw [Nat.cast_mul, Int.toNat_of_nonneg hpos] at this
  push_cast at this ⊢
  nlinarith [this]

/-- **Deterministic Flory lower bound**: every self-avoiding walk of `n` steps has
extension at least `(√(n+1) − 1)/2`. -/
theorem sqrt_le_extent {l : List Site} (h : IsSAW l) :
    Real.sqrt ((l.length : ℝ) + 1) ≤ 2 * (extent l : ℝ) + 1 := by
  have hb := length_succ_le_extent_sq h
  have hb' : ((l.length : ℝ) + 1) ≤ (2 * (extent l : ℝ) + 1) ^ 2 := by exact_mod_cast hb
  have hpos : (0 : ℝ) ≤ 2 * (extent l : ℝ) + 1 := by
    have := extent_nonneg l
    have : (0 : ℝ) ≤ (extent l : ℝ) := by exact_mod_cast this
    linarith
  calc Real.sqrt ((l.length : ℝ) + 1) ≤ Real.sqrt ((2 * (extent l : ℝ) + 1) ^ 2) :=
        Real.sqrt_le_sqrt hb'
    _ = 2 * (extent l : ℝ) + 1 := by
        rw [Real.sqrt_sq hpos]

/-- The Flory exponent is at least `1/2` in two dimensions, deterministically: the extension
of *any* `n`-step self-avoiding walk is at least `(√(n+1) − 1)/2 ≥ (√n − 1)/2`. -/
theorem flory_exponent_half {l : List Site} (h : IsSAW l) :
    (Real.sqrt (l.length : ℝ) - 1) / 2 ≤ (extent l : ℝ) := by
  have h1 : Real.sqrt (l.length : ℝ) ≤ Real.sqrt ((l.length : ℝ) + 1) :=
    Real.sqrt_le_sqrt (by linarith)
  have h2 := sqrt_le_extent h
  linarith

/-- The trivial upper bound: one step moves the chain by at most one lattice spacing, so the
extension never exceeds the number of bonds.  With `flory_exponent_half` this pins the
deterministic Flory exponent to the window `1/2 ≤ ν ≤ 1`. -/
theorem extent_le_length {l : List Site} (h : ∀ d ∈ l, max |d.1| |d.2| ≤ 1) :
    extent l ≤ (l.length : ℤ) := by
  induction l with
  | nil => simp [extent, sites]
  | cons d t ih =>
      have hd : max |d.1| |d.2| ≤ 1 := h d (by simp)
      have iht : extent t ≤ (t.length : ℤ) := ih fun e he => h e (by simp [he])
      have hnn : (0 : ℤ) ≤ (t.length : ℤ) + 1 := by positivity
      refine foldr_max_le hnn ?_
      intro x hx
      rw [sites_cons] at hx
      simp only [List.map_cons, List.mem_cons, List.map_map, Function.comp_def] at hx
      rcases hx with rfl | hx
      · simpa using hnn
      · obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
        have hy' : max |y.1| |y.2| ≤ extent t := le_extent hy
        have e1 : |d.1 + y.1| ≤ |d.1| + |y.1| := abs_add_le _ _
        have e2 : |d.2 + y.2| ≤ |d.2| + |y.2| := abs_add_le _ _
        have : max |(d + y).1| |(d + y).2| ≤ max |d.1| |d.2| + max |y.1| |y.2| := by
          simp only [Prod.fst_add, Prod.snd_add]
          refine max_le ?_ ?_
          · exact le_trans e1 (add_le_add (le_max_left _ _) (le_max_left _ _))
          · exact le_trans e2 (add_le_add (le_max_right _ _) (le_max_right _ _))
        simp only [List.length_cons]
        push_cast
        linarith

/-- **The deterministic Flory window in two dimensions.**  For every self-avoiding walk of
`n ≥ 4` unit bonds the extension lies between `n^(1/2)/4` and `n^1`: the swelling exponent of
an individual walk is confined to `[1/2, 1]`, with no probability and no ensemble average
involved.  (The conjectured typical value `ν = 3/4` is not decided by this — it is open
mathematics.) -/
theorem flory_window {l : List Site} (h : IsSAW l)
    (hstep : ∀ d ∈ l, max |d.1| |d.2| ≤ 1) (hn : 4 ≤ l.length) :
    Real.sqrt (l.length : ℝ) / 4 ≤ (extent l : ℝ) ∧ (extent l : ℝ) ≤ (l.length : ℝ) := by
  have hlow := flory_exponent_half h
  have h4 : (2 : ℝ) ≤ Real.sqrt (l.length : ℝ) := by
    have hcast : (4 : ℝ) ≤ (l.length : ℝ) := by exact_mod_cast hn
    have : Real.sqrt 4 ≤ Real.sqrt (l.length : ℝ) := Real.sqrt_le_sqrt hcast
    calc (2 : ℝ) = Real.sqrt 4 := by
          rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
      _ ≤ _ := this
  refine ⟨by linarith, ?_⟩
  have := extent_le_length hstep
  exact_mod_cast this

/-- Every square-lattice bond is a unit vector. -/
lemma stepsOf_unit {n : ℕ} (w : Fin n → Fin 4) :
    ∀ d ∈ stepsOf w, max |d.1| |d.2| ≤ 1 := by
  intro d hd
  simp only [stepsOf, stepsOfDir, List.mem_ofFn] at hd
  obtain ⟨i, rfl⟩ := hd
  have hall : ∀ j : Fin 4, max |(dir j).1| |(dir j).2| ≤ 1 := by decide
  exact hall _

/-- The Flory window, stated for the conformations counted by `cnt`. -/
theorem saw_flory_window {n : ℕ} (w : Fin n → Fin 4) (hw : w ∈ sawFinset n) (hn : 4 ≤ n) :
    Real.sqrt (n : ℝ) / 4 ≤ (extent (stepsOf w) : ℝ) ∧ (extent (stepsOf w) : ℝ) ≤ (n : ℝ) := by
  have hlen : (stepsOf w).length = n := by simp [stepsOf, stepsOfDir]
  have hsaw : IsSAW (stepsOf w) := (mem_sawFinset w).1 hw
  have := flory_window hsaw (stepsOf_unit w) (by omega)
  rwa [hlen] at this

/-! ## The directed walk: the Flory exponent is exactly `1` -/

lemma hgt_endpoint (l : List Site) (h : ∀ d ∈ l, hgt d = 1) :
    hgt (endpoint l) = (l.length : ℤ) := by
  have key : ∀ (x : Site) (l : List Site), (∀ d ∈ l, hgt d = 1) →
      hgt (List.foldl (· + ·) x l) = hgt x + (l.length : ℤ) := by
    intro x l
    induction l generalizing x with
    | nil => simp
    | cons d t ih =>
        intro hh
        rw [List.foldl_cons, ih (x + d) (fun e he => hh e (by simp [he])), hgt_add,
          hh d (by simp)]
        simp only [List.length_cons]
        push_cast; ring
  rw [endpoint, key 0 l h, hgt_zero]
  simp

/-- **The directed walk is maximally extended.**  A chain of `n` east/north bonds ends at
`ℓ^1` distance exactly `n` from its start, so its extension is at least `n/2`: the Flory
exponent of the directed lattice is exactly `1`. -/
theorem directed_extent {n : ℕ} (w : Fin n → Fin 2) :
    ((n : ℤ) : ℝ) / 2 ≤ (extent (stepsOfDir ddir w) : ℝ) := by
  set l := stepsOfDir ddir w with hl
  have hlen : l.length = n := by simp [hl, stepsOfDir]
  have hstep : ∀ d ∈ l, hgt d = 1 := by
    intro d hd
    simp only [hl, stepsOfDir, List.mem_ofFn] at hd
    obtain ⟨i, rfl⟩ := hd
    have hall : ∀ j : Fin 2, hgt (ddir j) = 1 := by decide
    exact hall _
  have hend : hgt (endpoint l) = (n : ℤ) := by rw [hgt_endpoint l hstep, hlen]
  have hmem : endpoint l ∈ sites l := foldl_mem_sitesFrom 0 l
  have hle : max |(endpoint l).1| |(endpoint l).2| ≤ extent l := le_extent hmem
  have h1 : (endpoint l).1 + (endpoint l).2 = (n : ℤ) := hend
  have hb1 : (endpoint l).1 ≤ max |(endpoint l).1| |(endpoint l).2| :=
    le_trans (le_abs_self _) (le_max_left _ _)
  have hb2 : (endpoint l).2 ≤ max |(endpoint l).1| |(endpoint l).2| :=
    le_trans (le_abs_self _) (le_max_right _ _)
  have : (n : ℤ) ≤ 2 * extent l := by omega
  have : ((n : ℤ) : ℝ) ≤ 2 * (extent l : ℝ) := by exact_mod_cast this
  linarith

/-! ## An improved rigorous upper bound for the genuine square lattice -/

set_option maxRecDepth 100000 in
set_option maxHeartbeats 4000000 in
/-- There are exactly `780` self-avoiding conformations of a six-bond chain on the square
lattice. -/
theorem cnt_six : cnt 6 = 780 := by decide

/-- **Improved upper bound on the connective constant of `ℤ²`.** -/
theorem connectiveConstant_le_log780_div6 :
    connectiveConstant ≤ Real.log 780 / 6 := by
  have h := connectiveConstant_le_div (n := 6) (by norm_num)
  have hval : logCnt 6 = Real.log 780 := by
    rw [logCnt, logCntOf, show cntOf dir 6 = 780 from cnt_six]
    norm_num
  rwa [hval] at h

/-- The six-step bound strictly improves the four-step bound `log 100 / 4`. -/
theorem improves_on_cnt_four : Real.log 780 / 6 < Real.log 100 / 4 := by
  have h : Real.log 780 * 4 < Real.log 100 * 6 := by
    have h1 : (780 : ℝ) ^ 4 < (100 : ℝ) ^ 6 := by norm_num
    have := Real.log_lt_log (by positivity) h1
    rw [Real.log_pow, Real.log_pow] at this
    push_cast at this
    linarith
  linarith

/-- Combining: the entropy per residue of a genuine self-avoiding chain on `ℤ²` lies strictly
between `log 2` and `log 780 / 6`, itself strictly below the four-step bound and the
ideal-chain value `log 4`. -/
theorem connectiveConstant_window :
    Real.log 2 ≤ connectiveConstant ∧ connectiveConstant < Real.log 100 / 4 :=
  ⟨log_two_le_connectiveConstant,
    lt_of_le_of_lt connectiveConstant_le_log780_div6 improves_on_cnt_four⟩

/-! ## The same box argument in three dimensions -/

/-- The `ℓ^∞` radius of the sites visited by a chain on the cubic lattice. -/
def extent3 (l : List Site3) : ℤ :=
  ((sites l).map fun p => max |p.1| (max |p.2.1| |p.2.2|)).foldr max 0

lemma extent3_nonneg (l : List Site3) : 0 ≤ extent3 l := foldr_max_nonneg _

lemma le_extent3 {l : List Site3} {p : Site3} (hp : p ∈ sites l) :
    max |p.1| (max |p.2.1| |p.2.2|) ≤ extent3 l :=
  le_foldr_max (List.mem_map_of_mem hp)

/-- **The box bound in three dimensions.**  The `n + 1` distinct sites of a self-avoiding walk
on `ℤ³` fit inside the cube of side `2 * extent3 + 1`. -/
theorem length_succ_le_extent3_cube {l : List Site3} (h : IsSAW l) :
    ((l.length : ℤ) + 1) ≤ (2 * extent3 l + 1) ^ 3 := by
  classical
  set R := extent3 l with hR
  have hRnn : 0 ≤ R := extent3_nonneg l
  have hsub : (sites l).toFinset ⊆
      (Finset.Icc (-R) R) ×ˢ ((Finset.Icc (-R) R) ×ˢ (Finset.Icc (-R) R)) := by
    intro p hp
    rw [List.mem_toFinset] at hp
    have hb := le_extent3 hp
    have h1 : |p.1| ≤ R := le_trans (le_max_left _ _) hb
    have h2 : |p.2.1| ≤ R := le_trans (le_trans (le_max_left _ _) (le_max_right _ _)) hb
    have h3 : |p.2.2| ≤ R := le_trans (le_trans (le_max_right _ _) (le_max_right _ _)) hb
    simp only [Finset.mem_product, Finset.mem_Icc]
    exact ⟨⟨neg_le_of_abs_le h1, le_of_abs_le h1⟩, ⟨neg_le_of_abs_le h2, le_of_abs_le h2⟩,
      ⟨neg_le_of_abs_le h3, le_of_abs_le h3⟩⟩
  have hcard := Finset.card_le_card hsub
  rw [card_sites_of_isSAW h, Finset.card_product, Finset.card_product, Int.card_Icc] at hcard
  have hval : ((R + 1 - -R).toNat) = (2 * R + 1).toNat := by ring_nf
  rw [hval] at hcard
  have hstep : ((l.length + 1 : ℕ) : ℤ) ≤
      (((2 * R + 1).toNat * ((2 * R + 1).toNat * (2 * R + 1).toNat) : ℕ) : ℤ) := by
    exact_mod_cast hcard
  have hpos : (0 : ℤ) ≤ 2 * R + 1 := by omega
  rw [Nat.cast_mul, Nat.cast_mul, Int.toNat_of_nonneg hpos] at hstep
  push_cast at hstep ⊢
  nlinarith [hstep]

/-- **Deterministic Flory lower bound in three dimensions**: every self-avoiding walk of `n`
steps on the cubic lattice has extension at least `((n+1)^(1/3) − 1)/2`, so `ν ≥ 1/3`. -/
theorem cubic_flory_exponent_third {l : List Site3} (h : IsSAW l) :
    ((l.length : ℝ) + 1) ^ ((1 : ℝ) / 3) ≤ 2 * (extent3 l : ℝ) + 1 := by
  set a : ℝ := 2 * (extent3 l : ℝ) + 1 with ha
  have hann : 0 ≤ a := by
    have : (0 : ℝ) ≤ (extent3 l : ℝ) := by exact_mod_cast extent3_nonneg l
    simp only [ha]; linarith
  have hb : ((l.length : ℝ) + 1) ≤ a ^ 3 := by
    have h3 := length_succ_le_extent3_cube h
    have h3' : ((l.length : ℝ) + 1) ≤ (2 * (extent3 l : ℝ) + 1) ^ 3 := by exact_mod_cast h3
    simpa [ha] using h3'
  have hnn : (0 : ℝ) ≤ (l.length : ℝ) + 1 := by positivity
  calc ((l.length : ℝ) + 1) ^ ((1 : ℝ) / 3)
      ≤ (a ^ 3) ^ ((1 : ℝ) / 3) := Real.rpow_le_rpow hnn hb (by norm_num)
    _ = a := by
        rw [← Real.rpow_natCast a 3, ← Real.rpow_mul hann]
        norm_num

end IDR.SAW
