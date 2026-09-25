/-
# Part XII.0  Walks on a lattice: the general theory of a chain that cannot overlap

This file develops, for an arbitrary lattice -- any abelian group `V` of positions together
with a finite set `dir : Fin q → V` of allowed bond vectors -- the combinatorics of chains
that cannot occupy the same position twice.  The square lattice
(`RequestProject.SelfAvoiding`) and the cubic lattice (`RequestProject.CubicLattice`, the case
of a real polypeptide) are the two instances used later.

* `sites` : the positions a chain visits, i.e. the partial sums of its bond vectors.
* `IsSAW` : the chain is self-avoiding -- the positions are pairwise distinct.
* `isSAW_append` : any piece of a self-avoiding chain is self-avoiding.  This is the exact
  content of the physical statement "excluded volume is inherited by subchains", and it makes
  the conformation count submultiplicative (`cntOf_submultiplicative`).
* `directed_isSAW_of_hom` : a chain whose bond vectors all increase some additive functional
  is automatically self-avoiding.  This gives the matching exponential lower bound
  `pow_le_cntOf` on the number of conformations.
* `connectiveConstantOf` : by Fekete's lemma the conformational entropy per residue
  `lim (log (cnt n))/n` exists, is bounded below by the directed count, and is bounded above
  by `log (cnt k)/k` at every single chain length `k` -- which is what makes a finite
  computation sufficient to prove that excluded volume costs entropy.
-/
import Mathlib

namespace IDR.SAW

open scoped BigOperators

section Walks

variable {V : Type*} [AddCommGroup V]

/-- The positions visited by a chain whose bond vectors are `l`, starting from `x`: the
`n + 1` partial sums. -/
def sitesFrom (x : V) (l : List V) : List V := l.scanl (· + ·) x

/-- The positions visited by a chain whose bond vectors are `l`, starting from the origin. -/
def sites (l : List V) : List V := sitesFrom 0 l

/-- A chain is *self-avoiding* when the positions it visits are pairwise distinct: no two
residues occupy the same place. -/
def IsSAW (l : List V) : Prop := (sites l).Nodup

instance [DecidableEq V] : DecidablePred (IsSAW (V := V)) := fun l => by
  unfold IsSAW; infer_instance

@[simp] lemma sitesFrom_nil (x : V) : sitesFrom x ([] : List V) = [x] := by
  simp [sitesFrom]

lemma sitesFrom_cons (x d : V) (l : List V) :
    sitesFrom x (d :: l) = x :: sitesFrom (x + d) l := by
  simp [sitesFrom, List.scanl_cons]

/-- Starting the chain at `x` translates every position by `x`. -/
lemma sitesFrom_eq_map (x : V) (l : List V) :
    sitesFrom x l = (sites l).map fun y => x + y := by
  induction l generalizing x with
  | nil => simp [sites]
  | cons d t ih =>
      rw [sitesFrom_cons, sites, sitesFrom_cons, ih (x + d), ih (0 + d)]
      simp [List.map_map, Function.comp_def, add_assoc]

/-- The position list is never empty and starts where the chain starts. -/
lemma sitesFrom_head (x : V) (l : List V) : ∃ t, sitesFrom x l = x :: t := by
  cases l with
  | nil => exact ⟨[], by simp⟩
  | cons d t => exact ⟨sitesFrom (x + d) t, sitesFrom_cons x d t⟩

/-- The far end of a chain is one of the positions it visits. -/
lemma foldl_mem_sitesFrom (x : V) (l : List V) :
    List.foldl (· + ·) x l ∈ sitesFrom x l := by
  induction l generalizing x with
  | nil => simp
  | cons d t ih =>
      rw [sitesFrom_cons]
      simp only [List.foldl_cons, List.mem_cons]
      exact Or.inr (ih (x + d))

/-- Translating a chain does not change whether it is self-avoiding. -/
lemma nodup_sitesFrom_iff (x : V) (l : List V) :
    (sitesFrom x l).Nodup ↔ IsSAW l := by
  have hinj : Function.Injective fun y : V => x + y := by
    intro a b hab; simpa using hab
  rw [sitesFrom_eq_map]
  exact ⟨fun h => h.of_map _, fun h => h.map hinj⟩

/-- **A piece of a self-avoiding chain is self-avoiding.**  Cutting at any residue leaves two
self-avoiding chains. -/
lemma isSAW_append {l₁ l₂ : List V} (h : IsSAW (l₁ ++ l₂)) : IsSAW l₁ ∧ IsSAW l₂ := by
  set x := List.foldl (· + ·) (0 : V) l₁ with hx
  have hsplit : sites (l₁ ++ l₂) = sites l₁ ++ (sitesFrom x l₂).tail := by
    simp [sites, sitesFrom, List.scanl_append, hx]
  rw [IsSAW, hsplit, List.nodup_append] at h
  obtain ⟨h1, h2, hcross⟩ := h
  refine ⟨h1, ?_⟩
  obtain ⟨t, ht⟩ := sitesFrom_head x l₂
  have hnd : (sitesFrom x l₂).Nodup := by
    rw [ht]
    refine List.nodup_cons.2 ⟨?_, ?_⟩
    · intro hmem
      exact hcross x (by rw [hx]; exact foldl_mem_sitesFrom 0 l₁) x (by rw [ht] at *; exact hmem)
        rfl
    · simpa [ht] using h2
  exact (nodup_sitesFrom_iff x l₂).1 hnd

/-- Adding one bond adds one position, at the far end. -/
lemma sites_append_singleton (l : List V) (d : V) :
    sites (l ++ [d]) = sites l ++ [List.foldl (· + ·) 0 l + d] := by
  simp [sites, sitesFrom, List.scanl_append]

/-- **A monotone chain is self-avoiding.**  If some additive functional `phi` increases
strictly along every bond, the chain never returns to a position it has left. -/
lemma directed_isSAW_of_hom (phi : V → ℤ) (hadd : ∀ a b, phi (a + b) = phi a + phi b)
    (hzero : phi 0 = 0) (l : List V) (h : ∀ d ∈ l, 0 < phi d) :
    (∀ y ∈ sites l, 0 ≤ phi y) ∧ IsSAW l := by
  induction l with
  | nil => exact ⟨by simp [sites, hzero], by simp [IsSAW, sites]⟩
  | cons d t ih =>
      have hd : 0 < phi d := h d (by simp)
      obtain ⟨hpos, hsaw⟩ := ih fun e he => h e (by simp [he])
      have hcons : sites (d :: t) = (0 : V) :: (sites t).map fun y => d + y := by
        rw [sites, sitesFrom_cons, sitesFrom_eq_map]
        simp
      constructor
      · intro y hy
        rw [hcons] at hy
        rcases List.mem_cons.1 hy with rfl | hy
        · simp [hzero]
        · obtain ⟨z, hz, rfl⟩ := List.mem_map.1 hy
          have hz' := hpos z hz
          rw [hadd]; linarith
      · rw [IsSAW, hcons]
        refine List.nodup_cons.2 ⟨?_, ?_⟩
        · intro hmem
          obtain ⟨z, hz, hzz⟩ := List.mem_map.1 hmem
          have h0 : phi (d + z) = 0 := by rw [hzz, hzero]
          have hz' := hpos z hz
          rw [hadd] at h0
          linarith
        · refine List.Nodup.map ?_ hsaw
          intro a b hab
          simpa using hab

/-! ### Chains of identical bonds -/

lemma foldl_replicate (x d : V) (k : ℕ) :
    List.foldl (· + ·) x (List.replicate k d) = x + k • d := by
  induction k generalizing x with
  | zero => simp
  | succ n ih =>
      rw [List.replicate_succ, List.foldl_cons, ih (x + d), succ_nsmul]
      abel

lemma sitesFrom_replicate (x d : V) (k : ℕ) :
    sitesFrom x (List.replicate k d) = (List.range (k + 1)).map fun i : ℕ => x + i • d := by
  induction k generalizing x with
  | zero => simp
  | succ n ih =>
      rw [List.replicate_succ, sitesFrom_cons, ih (x + d)]
      conv_rhs => rw [List.range_succ_eq_map]
      simp only [List.map_cons, List.map_map, Function.comp_def]
      congr 1
      · simp
      · apply List.map_congr_left
        intro i _
        rw [succ_nsmul]
        abel

end Walks

/-! ## Counting the conformations of a chain on a lattice -/

section Counting

variable {V : Type*} [AddCommGroup V] [DecidableEq V] {q : ℕ}

/-- The bond-vector list of a chain whose bonds are indexed by residue. -/
def stepsOfDir (dir : Fin q → V) {n : ℕ} (w : Fin n → Fin q) : List V :=
  List.ofFn fun i => dir (w i)

/-- The self-avoiding conformations of a chain of `n` bonds on the lattice `dir`. -/
def sawFinsetOf (dir : Fin q → V) (n : ℕ) : Finset (Fin n → Fin q) :=
  Finset.univ.filter fun w => IsSAW (stepsOfDir dir w)

@[simp] lemma mem_sawFinsetOf (dir : Fin q → V) {n : ℕ} (w : Fin n → Fin q) :
    w ∈ sawFinsetOf dir n ↔ IsSAW (stepsOfDir dir w) := by
  simp [sawFinsetOf]

/-- The number of self-avoiding conformations of a chain of `n` bonds. -/
def cntOf (dir : Fin q → V) (n : ℕ) : ℕ := (sawFinsetOf dir n).card

omit [AddCommGroup V] [DecidableEq V] in
lemma stepsOfDir_add (dir : Fin q → V) {m n : ℕ} (w : Fin (m + n) → Fin q) :
    stepsOfDir dir w
      = stepsOfDir dir (fun i : Fin m => w (Fin.castLE (Nat.le_add_right m n) i)) ++
        stepsOfDir dir (fun i : Fin n => w (Fin.natAdd m i)) := by
  simp [stepsOfDir, List.ofFn_add]

/-- **Subadditivity of conformational entropy.**  The conformation count is submultiplicative
in the chain length, because cutting a chain in two is injective and both pieces are
self-avoiding. -/
theorem cntOf_submultiplicative (dir : Fin q → V) (m n : ℕ) :
    cntOf dir (m + n) ≤ cntOf dir m * cntOf dir n := by
  classical
  have hcard : (sawFinsetOf dir m ×ˢ sawFinsetOf dir n).card = cntOf dir m * cntOf dir n := by
    simp [cntOf, Finset.card_product]
  rw [← hcard]
  refine Finset.card_le_card_of_injOn
    (fun w => (fun i : Fin m => w (Fin.castLE (Nat.le_add_right m n) i),
      fun i : Fin n => w (Fin.natAdd m i))) ?_ ?_
  · intro w hw
    rw [Finset.mem_coe, mem_sawFinsetOf, stepsOfDir_add] at hw
    obtain ⟨h1, h2⟩ := isSAW_append hw
    simp [h1, h2]
  · intro w _ w' _ h
    have h1 := congrArg Prod.fst h
    have h2 := congrArg Prod.snd h
    funext i
    induction i using Fin.addCases with
    | left i =>
        have := congrFun h1 i
        simpa [Fin.castAdd, Fin.castLE] using this
    | right i => exact congrFun h2 i

/-- **A directed subfamily gives an exponential lower bound.**  If `r` of the lattice's bond
vectors all increase one additive functional, then every one of the `r ^ n` chains built from
them is self-avoiding. -/
theorem pow_le_cntOf (dir : Fin q → V) (phi : V → ℤ)
    (hadd : ∀ a b, phi (a + b) = phi a + phi b) (hzero : phi 0 = 0) {r : ℕ}
    (sub : Fin r → Fin q) (hsub : Function.Injective sub)
    (hpos : ∀ i, 0 < phi (dir (sub i))) (n : ℕ) : r ^ n ≤ cntOf dir n := by
  classical
  have himg : (Finset.univ : Finset (Fin n → Fin r)).card = r ^ n := by simp
  rw [cntOf, ← himg]
  refine Finset.card_le_card_of_injOn (fun v => sub ∘ v) ?_ ?_
  · intro v _
    rw [Finset.mem_coe, mem_sawFinsetOf]
    refine (directed_isSAW_of_hom phi hadd hzero _ ?_).2
    intro d hd
    simp only [stepsOfDir, List.mem_ofFn] at hd
    obtain ⟨i, rfl⟩ := hd
    exact hpos (v i)
  · intro v _ v' _ h
    funext i
    exact hsub (congrFun h i)

lemma cntOf_pos_of_pow {dir : Fin q → V} {r : ℕ} (hr : 0 < r)
    (h : ∀ n, r ^ n ≤ cntOf dir n) (n : ℕ) : 0 < cntOf dir n :=
  lt_of_lt_of_le (Nat.pow_pos hr) (h n)

end Counting

/-! ## The entropy per residue -/

section Fekete

variable {V : Type*} [AddCommGroup V] [DecidableEq V] {q : ℕ}

/-- The conformational entropy of a chain of `n` bonds, in nats. -/
noncomputable def logCntOf (dir : Fin q → V) (n : ℕ) : ℝ := Real.log (cntOf dir n)

lemma logCntOf_nonneg {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) (n : ℕ) :
    0 ≤ logCntOf dir n :=
  Real.log_nonneg (by exact_mod_cast hdir n)

lemma logCntOf_subadditive {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) :
    Subadditive (logCntOf dir) := by
  intro m n
  have h := cntOf_submultiplicative dir m n
  calc logCntOf dir (m + n) = Real.log (cntOf dir (m + n) : ℝ) := rfl
    _ ≤ Real.log ((cntOf dir m : ℝ) * (cntOf dir n : ℝ)) := by
        refine Real.log_le_log (by exact_mod_cast hdir (m + n)) ?_
        exact_mod_cast h
    _ = logCntOf dir m + logCntOf dir n := by
        rw [Real.log_mul (by exact_mod_cast (hdir m).ne') (by exact_mod_cast (hdir n).ne')]
        rfl

lemma logCntOf_bddBelow {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) :
    BddBelow (Set.range fun n => logCntOf dir n / n) := by
  refine ⟨0, ?_⟩
  rintro x ⟨n, rfl⟩
  exact div_nonneg (logCntOf_nonneg hdir n) (Nat.cast_nonneg n)

/-- **The connective constant of a lattice**: the conformational entropy per residue of a
self-avoiding chain, which exists by Fekete's lemma. -/
noncomputable def connectiveConstantOf (dir : Fin q → V) (hdir : ∀ n, 0 < cntOf dir n) : ℝ :=
  (logCntOf_subadditive hdir).lim

theorem tendsto_connectiveConstantOf {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) :
    Filter.Tendsto (fun n => logCntOf dir n / n) Filter.atTop
      (nhds (connectiveConstantOf dir hdir)) :=
  (logCntOf_subadditive hdir).tendsto_lim (logCntOf_bddBelow hdir)

/-- Every single chain length gives an upper bound on the entropy per residue.  A finite
computation therefore proves that excluded volume costs entropy. -/
theorem connectiveConstantOf_le_div {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {n : ℕ}
    (hn : n ≠ 0) : connectiveConstantOf dir hdir ≤ logCntOf dir n / n :=
  Subadditive.lim_le_div (logCntOf_subadditive hdir) (logCntOf_bddBelow hdir) hn

/-- A directed subfamily of `r` bond vectors bounds the entropy per residue below by
`log r`. -/
theorem log_le_connectiveConstantOf {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {r : ℕ}
    (hr : 0 < r) (hpow : ∀ n, r ^ n ≤ cntOf dir n) :
    Real.log r ≤ connectiveConstantOf dir hdir := by
  refine ge_of_tendsto (tendsto_connectiveConstantOf hdir) ?_
  filter_upwards [Filter.eventually_gt_atTop 0] with n hn
  have hpow' : ((r : ℝ) ^ n) ≤ (cntOf dir n : ℝ) := by exact_mod_cast hpow n
  have hlog : (n : ℝ) * Real.log r ≤ logCntOf dir n := by
    have := Real.log_le_log (by positivity) hpow'
    rwa [Real.log_pow] at this
  rw [le_div_iff₀ (by exact_mod_cast hn)]
  linarith [hlog]

end Fekete

end IDR.SAW
