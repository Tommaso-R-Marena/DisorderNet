/-
# Part XL  Bridges: a lower bound on the entropy per residue from a finite count

Part XII bounded the conformational entropy per residue of a self-avoiding chain below by
`log 2`, using the two bond directions that increase a single coordinate.  That argument is
limited to *directed* chains and cannot see any conformation that ever steps sideways.  This file
supplies the standard device that removes the limitation — Hammersley's bridges — and turns a
single finite count into a lower bound on the entropy per residue.

A *bridge* (`IsBridge`) is a self-avoiding chain such that some additive functional `phi` of the
position is strictly positive at every site after the first and never exceeds its value at the
far end.  Two bridges concatenate to a bridge (`isBridge_append`): the first chain lives weakly
to the left of its endpoint and the second strictly to the right of its start, so the two can
never collide.  Bridge counts are therefore *super*multiplicative
(`brCntOf_supermultiplicative`), the exact opposite of the submultiplicativity of the full count,
and iterating gives `brCnt N ^ k ≤ brCnt (kN) ≤ cnt (kN)`.  Since the entropy per residue is a
limit, every single chain length now gives a *lower* bound as well as an upper one
(`log_brCntOf_div_le_connectiveConstantOf`).

On the square lattice the count of six-bond bridges is `101` (`brCnt_six`, by exhaustive
enumeration), which improves the lower bound from `log 2 = 0.693…` to
`(log 101)/6 = 0.769…` (`log_bridge_div_six_le_connectiveConstant`).  Together with
Part XXXIX this brackets the entropy per residue of a self-avoiding square-lattice chain in
`[(log 101)/6, log 3) ⊂ [0.769, 1.099)`, against the ideal-chain value `log 4 = 1.386`.

Design consequence: the entropy a disordered chain has is bounded on *both* sides by finite
computations, and both bounds are strictly interior to the ideal-chain value.  A generative model
of a disordered region has a definite, and definitely reduced, amount of conformational freedom
to reproduce.
-/
import Mathlib
import RequestProject.SelfAvoiding

namespace IDR.SAW

/-! ### The general theory of bridges -/

section BridgeTheory

variable {V : Type*} [AddCommGroup V]

/-- The far end of a chain, as a position. -/
def endpt (l : List V) : V := List.foldl (· + ·) 0 l

lemma foldl_add (x : V) (l : List V) : List.foldl (· + ·) x l = x + endpt l := by
  induction l generalizing x with
  | nil => simp [endpt]
  | cons d t ih =>
      have h1 : List.foldl (· + ·) x (d :: t) = x + d + endpt t := by
        rw [List.foldl_cons, ih (x + d)]
      have h2 : endpt (d :: t) = d + endpt t := by
        rw [endpt, List.foldl_cons, ih (0 + d)]
        abel
      rw [h1, h2]
      abel

@[simp] lemma endpt_nil : endpt ([] : List V) = 0 := by simp [endpt]

lemma endpt_append (l₁ l₂ : List V) : endpt (l₁ ++ l₂) = endpt l₁ + endpt l₂ := by
  rw [endpt, List.foldl_append, foldl_add, ← endpt]

/-- A chain is a *bridge* for the additive functional `phi` when it is self-avoiding, every
position after the first has `phi > 0`, and no position exceeds the far end. -/
def IsBridge (phi : V → ℤ) (l : List V) : Prop :=
  IsSAW l ∧ ∀ p ∈ (sites l).tail, 0 < phi p ∧ phi p ≤ phi (endpt l)

instance [DecidableEq V] (phi : V → ℤ) : DecidablePred (IsBridge phi) := fun l => by
  unfold IsBridge; infer_instance

lemma sites_eq_cons (l : List V) : sites l = 0 :: (sites l).tail := by
  obtain ⟨t, ht⟩ := sitesFrom_head (0 : V) l
  rw [sites, ht]
  simp

lemma endpt_mem_tail {l : List V} (hl : l ≠ []) : endpt l ∈ (sites l).tail := by
  cases l with
  | nil => exact absurd rfl hl
  | cons d t =>
      have hs : sites (d :: t) = (0:V) :: sitesFrom (0 + d) t := by
        rw [sites, sitesFrom_cons]
      rw [hs]
      simp only [List.tail_cons]
      have : endpt (d :: t) = List.foldl (· + ·) (0 + d) t := by
        rw [endpt, List.foldl_cons]
      rw [this]
      exact foldl_mem_sitesFrom (0 + d) t

lemma IsBridge.endpt_nonneg {phi : V → ℤ} (hzero : phi 0 = 0) {l : List V}
    (h : IsBridge phi l) : 0 ≤ phi (endpt l) := by
  by_cases hl : l = []
  · subst hl; simp [hzero]
  · exact le_of_lt (h.2 _ (endpt_mem_tail hl)).1

lemma IsBridge.le_endpt {phi : V → ℤ} (hzero : phi 0 = 0) {l : List V} (h : IsBridge phi l)
    {p : V} (hp : p ∈ sites l) : phi p ≤ phi (endpt l) := by
  rw [sites_eq_cons l] at hp
  rcases List.mem_cons.1 hp with rfl | hp
  · rw [hzero]; exact h.endpt_nonneg hzero
  · exact (h.2 p hp).2

/-- **Two bridges concatenate to a bridge.**  The first chain lies weakly below its far end in
`phi`, the second strictly above its start, so the two never meet. -/
theorem isBridge_append {phi : V → ℤ} (hadd : ∀ a b, phi (a + b) = phi a + phi b)
    (hzero : phi 0 = 0) {l₁ l₂ : List V} (h₁ : IsBridge phi l₁) (h₂ : IsBridge phi l₂) :
    IsBridge phi (l₁ ++ l₂) := by
  set e₁ : V := endpt l₁ with he₁
  have hsplit : sites (l₁ ++ l₂) = sites l₁ ++ (sitesFrom e₁ l₂).tail := by
    simp [sites, sitesFrom, List.scanl_append, he₁, endpt]
  have hmap : (sitesFrom e₁ l₂).tail = ((sites l₂).tail).map fun y => e₁ + y := by
    rw [sitesFrom_eq_map]
    conv_lhs => rw [sites_eq_cons l₂]
    simp
  have hshift : ∀ q ∈ (sitesFrom e₁ l₂).tail,
      phi e₁ < phi q ∧ phi q ≤ phi e₁ + phi (endpt l₂) := by
    intro q hq
    rw [hmap, List.mem_map] at hq
    obtain ⟨p, hp, rfl⟩ := hq
    obtain ⟨hp1, hp2⟩ := h₂.2 p hp
    rw [hadd]
    exact ⟨by linarith, by linarith⟩
  have hnodup : (sites (l₁ ++ l₂)).Nodup := by
    rw [hsplit, List.nodup_append]
    refine ⟨h₁.1, ?_, ?_⟩
    · rw [hmap]
      refine List.Nodup.map ?_ ?_
      · intro a b hab; simpa using hab
      · exact (sites_eq_cons l₂ ▸ h₂.1 : ((0:V) :: (sites l₂).tail).Nodup).of_cons
    · intro p hp q hq hpq
      subst hpq
      have h1 : phi p ≤ phi e₁ := h₁.le_endpt hzero hp
      have h2 := (hshift p hq).1
      linarith
  refine ⟨hnodup, ?_⟩
  have htail : (sites (l₁ ++ l₂)).tail
      = (sites l₁).tail ++ (sitesFrom e₁ l₂).tail := by
    rw [hsplit]
    conv_lhs => rw [sites_eq_cons l₁]
    simp
  have he₂ : 0 ≤ phi (endpt l₂) := h₂.endpt_nonneg hzero
  have he₁nn : 0 ≤ phi e₁ := h₁.endpt_nonneg hzero
  have hend : phi (endpt (l₁ ++ l₂)) = phi e₁ + phi (endpt l₂) := by
    rw [endpt_append, hadd, he₁]
  intro p hp
  rw [htail, List.mem_append] at hp
  rcases hp with hp | hp
  · obtain ⟨hp1, hp2⟩ := h₁.2 p hp
    rw [hend]
    exact ⟨hp1, by linarith⟩
  · obtain ⟨hp1, hp2⟩ := hshift p hp
    rw [hend]
    exact ⟨by linarith, hp2⟩

end BridgeTheory

/-! ### Counting bridges -/

section BridgeCounting

variable {V : Type*} [AddCommGroup V] [DecidableEq V] {q : ℕ}

/-- The bridge conformations of a chain of `n` bonds. -/
def bridgeFinsetOf (dir : Fin q → V) (phi : V → ℤ) (n : ℕ) : Finset (Fin n → Fin q) :=
  Finset.univ.filter fun w => IsBridge phi (stepsOfDir dir w)

/-- The number of bridge conformations of a chain of `n` bonds. -/
def brCntOf (dir : Fin q → V) (phi : V → ℤ) (n : ℕ) : ℕ := (bridgeFinsetOf dir phi n).card

@[simp] lemma mem_bridgeFinsetOf (dir : Fin q → V) (phi : V → ℤ) {n : ℕ} (w : Fin n → Fin q) :
    w ∈ bridgeFinsetOf dir phi n ↔ IsBridge phi (stepsOfDir dir w) := by
  simp [bridgeFinsetOf]

/-- Bridges are self-avoiding, so they are never more numerous than conformations. -/
theorem brCntOf_le_cntOf (dir : Fin q → V) (phi : V → ℤ) (n : ℕ) :
    brCntOf dir phi n ≤ cntOf dir n := by
  refine Finset.card_le_card ?_
  intro w hw
  rw [mem_bridgeFinsetOf] at hw
  rw [mem_sawFinsetOf]
  exact hw.1

omit [AddCommGroup V] [DecidableEq V] in
lemma stepsOfDir_append (dir : Fin q → V) {m n : ℕ} (w₁ : Fin m → Fin q) (w₂ : Fin n → Fin q) :
    stepsOfDir dir (Fin.append w₁ w₂) = stepsOfDir dir w₁ ++ stepsOfDir dir w₂ := by
  have h1 : (fun i : Fin m => Fin.append w₁ w₂ (Fin.castLE (Nat.le_add_right m n) i)) = w₁ := by
    funext i
    have hcast : Fin.castLE (Nat.le_add_right m n) i = Fin.castAdd n i := rfl
    rw [hcast, Fin.append_left]
  have h2 : (fun i : Fin n => Fin.append w₁ w₂ (Fin.natAdd m i)) = w₂ := by
    funext i
    rw [Fin.append_right]
  rw [stepsOfDir_add, h1, h2]

/-- **Bridge counts are supermultiplicative.**  Concatenating bridges is injective and lands in
the bridges of the total length. -/
theorem brCntOf_supermultiplicative (dir : Fin q → V) {phi : V → ℤ}
    (hadd : ∀ a b, phi (a + b) = phi a + phi b) (hzero : phi 0 = 0) (m n : ℕ) :
    brCntOf dir phi m * brCntOf dir phi n ≤ brCntOf dir phi (m + n) := by
  classical
  have hcard : (bridgeFinsetOf dir phi m ×ˢ bridgeFinsetOf dir phi n).card
      = brCntOf dir phi m * brCntOf dir phi n := by
    simp [brCntOf, Finset.card_product]
  rw [← hcard]
  refine Finset.card_le_card_of_injOn (fun p => Fin.append p.1 p.2) ?_ ?_
  · rintro ⟨w₁, w₂⟩ hw
    rw [Finset.mem_coe, Finset.mem_product, mem_bridgeFinsetOf, mem_bridgeFinsetOf] at hw
    rw [Finset.mem_coe, mem_bridgeFinsetOf, stepsOfDir_append]
    exact isBridge_append hadd hzero hw.1 hw.2
  · rintro ⟨w₁, w₂⟩ _ ⟨w₁', w₂'⟩ _ h
    have h1 : ∀ i, Fin.append w₁ w₂ (Fin.castAdd n i) = Fin.append w₁' w₂' (Fin.castAdd n i) :=
      fun i => congrFun h _
    have h2 : ∀ i, Fin.append w₁ w₂ (Fin.natAdd m i) = Fin.append w₁' w₂' (Fin.natAdd m i) :=
      fun i => congrFun h _
    simp only [Fin.append_left, Fin.append_right] at h1 h2
    exact Prod.ext (funext h1) (funext h2)

/-- Iterating: `k` bridges of `N` bonds concatenate to a bridge of `kN` bonds. -/
theorem pow_brCntOf_le (dir : Fin q → V) {phi : V → ℤ}
    (hadd : ∀ a b, phi (a + b) = phi a + phi b) (hzero : phi 0 = 0) (N k : ℕ) :
    brCntOf dir phi N ^ k ≤ brCntOf dir phi (k * N) := by
  induction k with
  | zero =>
      simp only [pow_zero, Nat.zero_mul]
      have hmem : (Fin.elim0 : Fin 0 → Fin q) ∈ bridgeFinsetOf dir phi 0 := by
        rw [mem_bridgeFinsetOf]
        refine ⟨?_, ?_⟩
        · simp [stepsOfDir, IsSAW, sites, sitesFrom]
        · intro p hp
          simp [stepsOfDir, sites, sitesFrom] at hp
      exact Finset.card_pos.mpr ⟨_, hmem⟩
  | succ k ih =>
      have hstep := brCntOf_supermultiplicative dir hadd hzero N (k * N)
      calc brCntOf dir phi N ^ (k + 1)
          = brCntOf dir phi N * brCntOf dir phi N ^ k := by ring
        _ ≤ brCntOf dir phi N * brCntOf dir phi (k * N) := by
            exact Nat.mul_le_mul_left _ ih
        _ ≤ brCntOf dir phi (N + k * N) := hstep
        _ = brCntOf dir phi ((k + 1) * N) := by ring_nf

end BridgeCounting

/-! ### A finite count bounds the entropy per residue from below -/

section BridgeLimit

variable {V : Type*} [AddCommGroup V] [DecidableEq V] {q : ℕ}

/-- **Every chain length gives a lower bound too.**  The entropy per residue is at least
`(log (number of `N`-bond bridges))/N`. -/
theorem log_brCntOf_div_le_connectiveConstantOf {dir : Fin q → V} {phi : V → ℤ}
    (hadd : ∀ a b, phi (a + b) = phi a + phi b) (hzero : phi 0 = 0)
    (hdir : ∀ n, 0 < cntOf dir n) {N : ℕ} (hN : N ≠ 0) (hbr : 0 < brCntOf dir phi N) :
    Real.log (brCntOf dir phi N) / N ≤ connectiveConstantOf dir hdir := by
  have hNpos : 0 < N := Nat.pos_of_ne_zero hN
  have htend := tendsto_connectiveConstantOf (dir := dir) hdir
  have hmap : Filter.Tendsto (fun k : ℕ => k * N) Filter.atTop Filter.atTop :=
    Filter.tendsto_atTop_atTop.2 fun b => ⟨b, fun a ha => le_trans ha (Nat.le_mul_of_pos_right a hNpos)⟩
  have htend2 : Filter.Tendsto (fun k : ℕ => logCntOf dir (k * N) / (k * N : ℕ))
      Filter.atTop (nhds (connectiveConstantOf dir hdir)) := htend.comp hmap
  refine ge_of_tendsto htend2 ?_
  filter_upwards [Filter.eventually_gt_atTop 0] with k hk
  have hpow : brCntOf dir phi N ^ k ≤ cntOf dir (k * N) :=
    le_trans (pow_brCntOf_le dir hadd hzero N k) (brCntOf_le_cntOf dir phi (k * N))
  have hpowR : ((brCntOf dir phi N : ℝ)) ^ k ≤ (cntOf dir (k * N) : ℝ) := by
    exact_mod_cast hpow
  have hbrR : (0:ℝ) < (brCntOf dir phi N : ℝ) := by exact_mod_cast hbr
  have hlog : (k : ℝ) * Real.log (brCntOf dir phi N) ≤ logCntOf dir (k * N) := by
    have := Real.log_le_log (by positivity) hpowR
    rwa [Real.log_pow] at this
  have hkN : (0:ℝ) < ((k * N : ℕ) : ℝ) := by
    have : 0 < k * N := Nat.mul_pos hk hNpos
    exact_mod_cast this
  have hNR : (0:ℝ) < (N : ℝ) := by exact_mod_cast hNpos
  rw [le_div_iff₀ hkN]
  have hcast : ((k * N : ℕ) : ℝ) = (k : ℝ) * (N : ℝ) := by push_cast; ring
  rw [hcast]
  have hsimp : Real.log (brCntOf dir phi N) / N * ((k : ℝ) * N)
      = (k : ℝ) * Real.log (brCntOf dir phi N) := by
    field_simp
  rw [hsimp]
  exact hlog

end BridgeLimit

/-! ### The square lattice: a six-bond count -/

/-- The horizontal coordinate, the functional whose bridges we count. -/
def xco (p : Site) : ℤ := p.1

lemma xco_add (a b : Site) : xco (a + b) = xco a + xco b := rfl

lemma xco_zero : xco 0 = 0 := rfl

/-- The bridge conformations of a chain of `n` bonds on the square lattice. -/
def brCnt (n : ℕ) : ℕ := brCntOf dir xco n

set_option maxRecDepth 100000 in
set_option maxHeartbeats 2000000 in
/-- There are exactly `101` six-bond bridges on the square lattice.  Proved by exhaustive kernel
enumeration of all `4⁶ = 4096` bond sequences. -/
theorem brCnt_six : brCnt 6 = 101 := by decide

/-- **A tighter lower bound on the entropy per residue.**  The conformational entropy per residue
of a self-avoiding chain on the square lattice is at least `(log 101)/6`. -/
theorem log_bridge_div_six_le_connectiveConstant :
    Real.log 101 / 6 ≤ connectiveConstant := by
  have h := log_brCntOf_div_le_connectiveConstantOf (dir := dir) (phi := xco)
    xco_add xco_zero cnt_pos (N := 6) (by norm_num) (by rw [← brCnt, brCnt_six]; norm_num)
  have hval : ((brCntOf dir xco 6 : ℕ) : ℝ) = 101 := by
    rw [show brCntOf dir xco 6 = 101 from brCnt_six]
    norm_num
  rw [hval] at h
  simpa [connectiveConstant] using h

/-- The new bound strictly improves the directed-chain bound `log 2`. -/
theorem log_two_lt_log_bridge_div_six : Real.log 2 < Real.log 101 / 6 := by
  have h64 : Real.log 64 = 6 * Real.log 2 := by
    rw [show (64 : ℝ) = 2 ^ (6 : ℕ) by norm_num, Real.log_pow]
    norm_num
  have hlt : Real.log 64 < Real.log 101 := Real.log_lt_log (by norm_num) (by norm_num)
  rw [lt_div_iff₀ (by norm_num : (0:ℝ) < 6)]
  linarith

end IDR.SAW
