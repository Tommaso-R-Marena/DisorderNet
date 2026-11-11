/-
# Part XII.1  Excluded volume on the square lattice

Part IX treated excluded volume at the level of Flory's mean-field free energy
(`RequestProject.Flory`): a scaling exponent obtained by balancing an entropic spring against
a repulsive term.  That is the physicist's estimate, not a theorem about a chain that really
cannot overlap.  This file instantiates the exact lattice theory of
`RequestProject.LatticeWalk` on the square lattice and extracts the consequences for a model
of a disordered region.  (`RequestProject.CubicLattice` does the same in three dimensions,
where a real polypeptide lives; nothing below is special to two dimensions except the
explicit conformation counts and the two explicit walks.)

A conformation of an `n`-bond chain is a walk `w : Fin n → Fin 4`; the walk is self-avoiding
when the `n + 1` positions it visits are pairwise distinct, and `cnt n` counts the
self-avoiding conformations.

Main results.

* `cnt_submultiplicative` : `cnt (m + n) ≤ cnt m * cnt n` -- the conformational entropy of a
  chain is subadditive in its length -- and, by Fekete's lemma, the entropy per residue
  `mu = lim (log (cnt n))/n` exists (`tendsto_connectiveConstant`).
* `two_pow_le_cnt` : `2 ^ n ≤ cnt n`, hence `log 2 ≤ mu`
  (`log_two_le_connectiveConstant`): excluded volume does **not** collapse the ensemble, so by
  the capacity law of Part III a model that is exactly right still needs exponentially many
  components.
* `connectiveConstant_lt_log_four` : `mu < log 4`, strictly below the ideal-chain value.
  Excluded volume removes a fixed fraction of the conformational entropy of every residue, and
  `saw_fraction_tendsto_zero` restates this as: the self-avoiding conformations are an
  exponentially vanishing fraction of the freely jointed ones, so an ideal-chain generator is
  wrong about the *support* of the ensemble and not merely about its weights.
* `exists_trapped_walk` : there is a self-avoiding walk with **no** self-avoiding extension.
  Growing a conformation residue by residue can dead-end, so a sequential generator must be
  able to reject and backtrack; correctness cannot be checked one step at a time.
* `unbounded_memory` : for every context length `k` there are two self-avoiding walks whose
  last `k` steps agree, exactly one of which admits a given next step.  Self-avoidance is
  therefore not a finite-context property of the step sequence: an autoregressive generator
  with a bounded context window cannot be right.  This is the microscopic counterpart of the
  receptive-field law of Part III (`RequestProject.Invariance`).
-/
import Mathlib
import RequestProject.LatticeWalk

namespace IDR.SAW

open scoped BigOperators

/-- A position on the square lattice. -/
abbrev Site := ℤ × ℤ

/-- The four unit bond vectors: east, west, north, south. -/
def dir : Fin 4 → Site := ![(1, 0), (-1, 0), (0, 1), (0, -1)]

/-- The self-avoiding conformations of a chain of `n` bonds on the square lattice. -/
def sawFinset (n : ℕ) : Finset (Fin n → Fin 4) := sawFinsetOf dir n

/-- The number of self-avoiding conformations of a chain of `n` bonds. -/
def cnt (n : ℕ) : ℕ := cntOf dir n

/-- The step list of a walk given as a function on residue indices. -/
def stepsOf {n : ℕ} (w : Fin n → Fin 4) : List Site := stepsOfDir dir w

@[simp] lemma mem_sawFinset {n : ℕ} (w : Fin n → Fin 4) :
    w ∈ sawFinset n ↔ IsSAW (stepsOf w) := mem_sawFinsetOf dir w

/-- **Subadditivity of conformational entropy** on the square lattice. -/
theorem cnt_submultiplicative (m n : ℕ) : cnt (m + n) ≤ cnt m * cnt n :=
  cntOf_submultiplicative dir m n

/-! ## The chain still has exponentially many conformations -/

/-- The "height" functional `x + y`, strictly increasing along a directed walk. -/
def hgt (p : Site) : ℤ := p.1 + p.2

lemma hgt_add (a b : Site) : hgt (a + b) = hgt a + hgt b := by
  simp only [hgt, Prod.fst_add, Prod.snd_add]; ring

lemma hgt_zero : hgt 0 = 0 := by simp [hgt]

/-- **Excluded volume does not collapse the ensemble.**  Every walk that only steps east or
north is self-avoiding, so a chain of `n` bonds has at least `2 ^ n` conformations. -/
theorem two_pow_le_cnt (n : ℕ) : 2 ^ n ≤ cnt n := by
  refine pow_le_cntOf dir hgt hgt_add hgt_zero ![0, 2] ?_ ?_ n
  · decide
  · decide

lemma cnt_pos (n : ℕ) : 0 < cnt n :=
  lt_of_lt_of_le (Nat.two_pow_pos n) (two_pow_le_cnt n)

/-! ## The entropy per residue -/

/-- The conformational entropy of a chain of `n` bonds, in nats. -/
noncomputable def logCnt (n : ℕ) : ℝ := logCntOf dir n

/-- **The connective constant of the square lattice**: the conformational entropy per residue
of a self-avoiding chain. -/
noncomputable def connectiveConstant : ℝ := connectiveConstantOf dir cnt_pos

/-- The entropy per residue converges. -/
theorem tendsto_connectiveConstant :
    Filter.Tendsto (fun n => logCnt n / n) Filter.atTop (nhds connectiveConstant) :=
  tendsto_connectiveConstantOf cnt_pos

/-- Every chain length gives an upper bound on the entropy per residue. -/
theorem connectiveConstant_le_div {n : ℕ} (hn : n ≠ 0) :
    connectiveConstant ≤ logCnt n / n :=
  connectiveConstantOf_le_div cnt_pos hn

/-- There are exactly `100` self-avoiding conformations of a four-bond chain, against `256`
unrestricted ones. -/
theorem cnt_four : cnt 4 = 100 := by decide

/-- **Excluded volume costs a fixed amount of entropy per residue.**  The entropy per residue
of a self-avoiding chain is strictly below the ideal-chain value `log 4`. -/
theorem connectiveConstant_lt_log_four : connectiveConstant < Real.log 4 := by
  have h4 : connectiveConstant ≤ logCnt 4 / 4 := connectiveConstant_le_div (by norm_num)
  have hval : logCnt 4 = Real.log 100 := by
    rw [logCnt, logCntOf, show cntOf dir 4 = 100 from cnt_four]
    norm_num
  have hlt : Real.log 100 < Real.log 256 := Real.log_lt_log (by norm_num) (by norm_num)
  have h256 : Real.log 256 = 4 * Real.log 4 := by
    rw [show (256 : ℝ) = 4 ^ (4 : ℕ) by norm_num, Real.log_pow]
    norm_num
  rw [hval] at h4
  linarith [h4, hlt, h256]

/-- **Excluded volume does not collapse the ensemble.**  The entropy per residue is at least
`log 2`: a self-avoiding chain still has exponentially many conformations. -/
theorem log_two_le_connectiveConstant : Real.log 2 ≤ connectiveConstant := by
  have h := log_le_connectiveConstantOf (dir := dir) cnt_pos (r := 2) (by norm_num)
    two_pow_le_cnt
  simpa [connectiveConstant] using h

/-- The connective constant is strictly positive. -/
theorem connectiveConstant_pos : 0 < connectiveConstant :=
  lt_of_lt_of_le (Real.log_pos (by norm_num)) log_two_le_connectiveConstant

/-! ## Self-avoiding conformations are an exponentially small part of the ideal chain -/

/-- Iterating the cut bound: a chain of `4 m` bonds has at most `100 ^ m` conformations. -/
lemma cnt_four_mul (m : ℕ) : cnt (4 * m) ≤ 100 ^ m := by
  induction m with
  | zero => decide
  | succ k ih =>
      have hcut : cnt (4 * (k + 1)) ≤ cnt 4 * cnt (4 * k) := by
        have := cnt_submultiplicative 4 (4 * k)
        simpa [Nat.mul_succ, Nat.add_comm] using this
      calc cnt (4 * (k + 1)) ≤ cnt 4 * cnt (4 * k) := hcut
        _ ≤ 100 * 100 ^ k := by rw [cnt_four]; exact Nat.mul_le_mul_left _ ih
        _ = 100 ^ (k + 1) := by ring

/-- **An ideal-chain generator is wrong about the support.**  The self-avoiding conformations
are an exponentially vanishing fraction of the conformations of a freely jointed chain of the
same length. -/
theorem saw_fraction_le (m : ℕ) :
    (cnt (4 * m) : ℝ) / 4 ^ (4 * m) ≤ (25 / 64 : ℝ) ^ m := by
  have h1 : (cnt (4 * m) : ℝ) ≤ 100 ^ m := by exact_mod_cast cnt_four_mul m
  have h2 : (4 : ℝ) ^ (4 * m) = 256 ^ m := by rw [pow_mul]; norm_num
  rw [h2, div_le_iff₀ (by positivity)]
  calc (cnt (4 * m) : ℝ) ≤ 100 ^ m := h1
    _ = (25 / 64 : ℝ) ^ m * 256 ^ m := by rw [← mul_pow]; norm_num

theorem saw_fraction_tendsto_zero :
    Filter.Tendsto (fun m : ℕ => (cnt (4 * m) : ℝ) / 4 ^ (4 * m)) Filter.atTop (nhds 0) := by
  have hgeom : Filter.Tendsto (fun m : ℕ => (25 / 64 : ℝ) ^ m) Filter.atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  exact squeeze_zero (fun m => by positivity) (fun m => saw_fraction_le m) hgeom

/-! ## Growing a conformation: dead ends and unbounded memory -/

def eastStep : Site := (1, 0)
def westStep : Site := (-1, 0)
def northStep : Site := (0, 1)
def southStep : Site := (0, -1)

def straightWalk (k : ℕ) : List Site := List.replicate k westStep

def uturnWalk (k : ℕ) : List Site :=
  List.replicate k eastStep ++ northStep :: List.replicate k westStep

lemma sites_uturn (k : ℕ) :
    sites (uturnWalk k) =
      (List.range (k + 1)).map (fun i : ℕ => ((i : ℤ), (0 : ℤ))) ++
        (List.range (k + 1)).map (fun i : ℕ => ((k : ℤ) - i, (1 : ℤ))) := by
  have hfold : List.foldl (· + ·) (0 : Site) (List.replicate k eastStep) = ((k : ℤ), 0) := by
    rw [foldl_replicate]; simp [eastStep]
  rw [uturnWalk, sites, sitesFrom, List.scanl_append, hfold, ← sitesFrom, ← sitesFrom,
    sitesFrom_cons, List.tail_cons, sitesFrom_replicate, sitesFrom_replicate]
  congr 1
  · apply List.map_congr_left
    intro i _
    simp [eastStep]
  · apply List.map_congr_left
    intro i _
    simp [westStep, northStep]
    ring


lemma nodup_sites_uturn (k : ℕ) : IsSAW (uturnWalk k) := by
  rw [IsSAW, sites_uturn]
  refine List.nodup_append.2 ⟨?_, ?_, ?_⟩
  · exact (List.nodup_range).map (fun a b hab => by simpa using hab)
  · refine (List.nodup_range).map (fun a b hab => ?_)
    simp only [Prod.mk.injEq] at hab
    omega
  · rintro a ha b hb rfl
    obtain ⟨i, _, hi⟩ := List.mem_map.1 ha
    obtain ⟨j, _, hj⟩ := List.mem_map.1 hb
    rw [← hj] at hi
    simp only [Prod.mk.injEq] at hi
    omega

lemma zero_mem_sites_uturn (k : ℕ) : (0 : Site) ∈ sites (uturnWalk k) := by
  rw [sites_uturn]
  refine List.mem_append_left _ (List.mem_map.2 ⟨0, ?_, ?_⟩)
  · simp
  · simp [Prod.ext_iff]

lemma foldl_uturn (k : ℕ) : List.foldl (· + ·) (0 : Site) (uturnWalk k) = (0, 1) := by
  rw [uturnWalk, List.foldl_append, foldl_replicate, List.foldl_cons, foldl_replicate]
  simp [eastStep, westStep, northStep]

/-- The straight walk is self-avoiding, and stays so after a south step. -/
lemma isSAW_straight_south (k : ℕ) : IsSAW (straightWalk k ++ [southStep]) := by
  refine (directed_isSAW_of_hom (fun p => -(p.1 + p.2)) (fun a b => by
      simp only [Prod.fst_add, Prod.snd_add]; ring) (by simp) _ ?_).2
  intro d hd
  rcases List.mem_append.1 hd with hd | hd
  · simp only [straightWalk] at hd
    rw [List.eq_of_mem_replicate hd]; simp [westStep]
  · rw [List.mem_singleton.1 hd]; simp [southStep]

lemma isSAW_straight (k : ℕ) : IsSAW (straightWalk k) :=
  (directed_isSAW_of_hom (fun p => -(p.1 + p.2)) (fun a b => by
      simp only [Prod.fst_add, Prod.snd_add]; ring) (by simp) _ (by
    intro d hd
    simp only [straightWalk] at hd
    rw [List.eq_of_mem_replicate hd]; simp [westStep])).2

/-- The u-turn walk cannot be extended by a south step: the site below its endpoint is its own
starting point. -/
lemma not_isSAW_uturn_south (k : ℕ) : ¬ IsSAW (uturnWalk k ++ [southStep]) := by
  rw [IsSAW, sites_append_singleton, foldl_uturn]
  intro h
  have hmem : (0 : Site) ∈ sites (uturnWalk k) := zero_mem_sites_uturn k
  rw [List.nodup_append] at h
  have := h.2.2 0 hmem (((0 : ℤ), (1 : ℤ)) + southStep) (by simp)
  simp [southStep, Prod.ext_iff] at this


lemma length_straightWalk (k : ℕ) : (straightWalk k).length = k := by
  simp [straightWalk]

lemma length_uturnWalk (k : ℕ) : (uturnWalk k).length = 2 * k + 1 := by
  simp [uturnWalk]; ring

lemma drop_uturnWalk (k : ℕ) : (uturnWalk k).drop (k + 1) = straightWalk k := by
  rw [uturnWalk, ← List.drop_drop (i := 1) (j := k), List.drop_left' (by simp)]
  simp [straightWalk]

/-- The shortest trapped walk on the square lattice: east, north, north, west, west, south,
east. -/
def trappedWalk : List Site := [(1, 0), (0, 1), (0, 1), (-1, 0), (-1, 0), (0, -1), (1, 0)]

/-- **Growth can dead-end.**  There is a self-avoiding walk none of whose four extensions is
self-avoiding. -/
theorem exists_trapped_walk :
    IsSAW trappedWalk ∧ ∀ i : Fin 4, ¬ IsSAW (trappedWalk ++ [dir i]) := by
  decide

/-- **Self-avoidance is not a finite-context property.**  For every context length `k` there
are two self-avoiding conformations whose last `k` steps coincide, exactly one of which may be
continued by a south step. -/
theorem unbounded_memory (k : ℕ) :
    ∃ l₁ l₂ : List Site, IsSAW l₁ ∧ IsSAW l₂ ∧
      l₁.drop (l₁.length - k) = l₂.drop (l₂.length - k) ∧
      IsSAW (l₁ ++ [southStep]) ∧ ¬ IsSAW (l₂ ++ [southStep]) := by
  refine ⟨straightWalk k, uturnWalk k, isSAW_straight k, nodup_sites_uturn k, ?_,
    isSAW_straight_south k, not_isSAW_uturn_south k⟩
  rw [length_straightWalk, length_uturnWalk, Nat.sub_self,
    show 2 * k + 1 - k = k + 1 by omega, drop_uturnWalk, List.drop_zero]

end IDR.SAW
