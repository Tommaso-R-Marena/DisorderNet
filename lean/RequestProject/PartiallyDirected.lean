/-
# Part CXVI  A second exactly solvable chain, and a sharper lower bound on the connective constant

Part CXIV solved the *directed* chain exactly (only east and north bonds: every walk is
self-avoiding, the count is `2 ^ n`, the entropy per residue is exactly `log 2`) and Part CXV
gave the rigorous bracket `log 251 / 7 ≤ μ ≤ log 780 / 6` for the genuine square lattice.

This file solves a second, strictly richer model exactly: the **partially directed chain**, in
which a bond may point east, north or south — every direction except backwards — subject only
to the rule that a north bond is never immediately followed by a south bond, or vice versa.
This is the standard exactly solvable model of a chain with a preferred axis (a chain under
tension, or one threaded through a pore); the physics of it is developed with a stretching
force in `RequestProject.StretchedChain`.

Two things are proved.

* **The model is exactly solvable and its conformations are genuinely self-avoiding.**
  `adm_isSAW` : every partially directed conformation is self-avoiding — the geometric heart of
  the file.  `pdCnt_bounds` : the number of conformations of an `n`-bond partially directed
  chain satisfies `λ ^ n ≤ pdCnt n ≤ √2 · λ ^ n` with `λ = 1 + √2`, so
  `pd_entropy_per_residue` : its entropy per residue is *exactly* `log (1 + √2)`.

* **A sharper rigorous lower bound for the square lattice.**  Partially directed conformations
  are self-avoiding conformations, so `log_one_add_sqrt_two_le_connectiveConstant` :
  `log (1 + √2) ≤ μ`.  Numerically `0.8813…`, against the bridge bound `log 251 / 7 = 0.7896…`
  of Part CXV (`improves_on_bridge_bound`), giving the new bracket
  `connectiveConstant_bracket_pd` : `log (1 + √2) ≤ μ ≤ log 780 / 6`, i.e.
  `2.4142… ≤ e^μ ≤ 3.0295…`.

The design reading is the one of Part XII, sharpened: the conformational entropy of a
disordered region is at least `log (1 + √2) ≈ 0.88` nats per residue *even after* excluded
volume is imposed exactly, so the number of conformations a model must be able to represent is
at least `2.414 ^ n`; and the partially directed chain shows that this much entropy survives
even when the chain is forbidden ever to turn back on itself.
-/
import Mathlib
import RequestProject.LatticeWalk
import RequestProject.SelfAvoiding
import RequestProject.ConnectiveLower

namespace IDR.SAW
namespace PD

open scoped BigOperators

/-! ## The model -/

/-- A bond of a partially directed chain: `0` east, `1` north, `2` south. -/
abbrev Letter := Fin 3

/-- The bond vector of a letter. -/
def stepOf : Letter → Site
  | 0 => (1, 0)
  | 1 => (0, 1)
  | 2 => (0, -1)

/-- The encoding of a partially directed bond as a square-lattice bond. -/
def enc : Letter → Fin 4
  | 0 => 0
  | 1 => 2
  | 2 => 3

lemma dir_enc (a : Letter) : dir (enc a) = stepOf a := by
  fin_cases a <;> rfl

/-- Two consecutive bonds are allowed when they are not opposite vertical bonds. -/
def ok (a b : Letter) : Prop := ¬ (a = 1 ∧ b = 2) ∧ ¬ (a = 2 ∧ b = 1)

instance : DecidableRel ok := fun a b =>
  decidable_of_iff (¬ (a = 1 ∧ b = 2) ∧ ¬ (a = 2 ∧ b = 1)) Iff.rfl

/-- A partially directed conformation: a word in which no north bond is followed by a south
bond and no south bond by a north bond. -/
def Adm (l : List Letter) : Prop := List.IsChain ok l

/-! ## Partially directed conformations are self-avoiding -/

lemma sites_cons (d : Site) (t : List Site) :
    sites (d :: t) = (0 : Site) :: (sites t).map (fun y => d + y) := by
  rw [sites, sitesFrom_cons, sitesFrom_eq_map]
  simp

lemma stepOf_fst_nonneg : ∀ a : Letter, 0 ≤ (stepOf a).1 := by decide

/-- The chain never moves west, so it never reaches a negative abscissa. -/
lemma x_nonneg (l : List Letter) : ∀ p ∈ sites (l.map stepOf), 0 ≤ p.1 := by
  induction l with
  | nil => intro p hp; simp [sites] at hp; simp [hp]
  | cons a t ih =>
      intro p hp
      rw [List.map_cons, sites_cons] at hp
      rcases List.mem_cons.1 hp with rfl | hp
      · simp
      · obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hp
        have h1 := stepOf_fst_nonneg a
        have h2 := ih y hy
        simp only [Prod.fst_add]
        linarith

/-- If the chain does not start with a south bond, then on the initial column it stays weakly
above the origin. -/
lemma y_nonneg_of_head (l : List Letter) (hl : Adm l) (hhead : ∀ b ∈ l.head?, b ≠ 2) :
    ∀ p ∈ sites (l.map stepOf), p.1 = 0 → 0 ≤ p.2 := by
  induction l with
  | nil => intro p hp _; simp [sites] at hp; simp [hp]
  | cons a t ih =>
      intro p hp hx
      rw [List.map_cons, sites_cons] at hp
      have hat : Adm t := (List.isChain_cons.1 hl).2
      have hlink := (List.isChain_cons.1 hl).1
      rcases List.mem_cons.1 hp with rfl | hp
      · simp
      · obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hp
        have hy0 := x_nonneg t y hy
        have ha : a ≠ 2 := hhead a (by simp)
        rcases a with ⟨i, hi⟩
        interval_cases i
        · exfalso
          simp only [stepOf, Prod.fst_add] at hx
          linarith
        · have hhead' : ∀ b ∈ t.head?, b ≠ 2 := by
            intro b hb hb2
            have := hlink b hb
            rw [hb2] at this
            exact this.1 ⟨rfl, rfl⟩
          have hx' : y.1 = 0 := by
            simp only [stepOf, Prod.fst_add] at hx
            norm_num at hx
            exact hx
          have hyy := ih hat hhead' y hy hx'
          simp only [stepOf, Prod.snd_add]
          linarith
        · exact absurd rfl ha

/-- Mirror image of `y_nonneg_of_head`: a chain that does not start with a north bond stays
weakly below the origin on the initial column. -/
lemma y_nonpos_of_head (l : List Letter) (hl : Adm l) (hhead : ∀ b ∈ l.head?, b ≠ 1) :
    ∀ p ∈ sites (l.map stepOf), p.1 = 0 → p.2 ≤ 0 := by
  induction l with
  | nil => intro p hp _; simp [sites] at hp; simp [hp]
  | cons a t ih =>
      intro p hp hx
      rw [List.map_cons, sites_cons] at hp
      have hat : Adm t := (List.isChain_cons.1 hl).2
      have hlink := (List.isChain_cons.1 hl).1
      rcases List.mem_cons.1 hp with rfl | hp
      · simp
      · obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hp
        have hy0 := x_nonneg t y hy
        have ha : a ≠ 1 := hhead a (by simp)
        rcases a with ⟨i, hi⟩
        interval_cases i
        · exfalso
          simp only [stepOf, Prod.fst_add] at hx
          linarith
        · exact absurd rfl ha
        · have hhead' : ∀ b ∈ t.head?, b ≠ 1 := by
            intro b hb hb1
            have := hlink b hb
            rw [hb1] at this
            exact this.2 ⟨rfl, rfl⟩
          have hx' : y.1 = 0 := by
            simp only [stepOf, Prod.fst_add] at hx
            norm_num at hx
            exact hx
          have hyy := ih hat hhead' y hy hx'
          simp only [stepOf, Prod.snd_add]
          linarith

/-- **Partially directed conformations are self-avoiding.**  A chain that never steps west, and
never reverses a vertical bond, never revisits a site. -/
theorem adm_isSAW (l : List Letter) (hl : Adm l) : IsSAW (l.map stepOf) := by
  induction l with
  | nil => simp [IsSAW, sites]
  | cons a t ih =>
      have hat : Adm t := (List.isChain_cons.1 hl).2
      have hlink := (List.isChain_cons.1 hl).1
      have iht := ih hat
      rw [List.map_cons, IsSAW, sites_cons]
      refine List.nodup_cons.2 ⟨?_, ?_⟩
      · intro hmem
        obtain ⟨y, hy, hzero⟩ := List.mem_map.1 hmem
        have hy0 := x_nonneg t y hy
        have hfst := congrArg Prod.fst hzero
        have hsnd := congrArg Prod.snd hzero
        simp only [Prod.fst_add, Prod.snd_add, Prod.fst_zero, Prod.snd_zero] at hfst hsnd
        rcases a with ⟨i, hi⟩
        interval_cases i
        · simp only [stepOf] at hfst
          linarith
        · have hhead' : ∀ b ∈ t.head?, b ≠ 2 := by
            intro b hb hb2
            have := hlink b hb
            rw [hb2] at this
            exact this.1 ⟨rfl, rfl⟩
          have hx : y.1 = 0 := by
            simp only [stepOf] at hfst
            norm_num at hfst
            exact hfst
          have hy' : y.2 = -1 := by
            simp only [stepOf] at hsnd
            linarith
          have := y_nonneg_of_head t hat hhead' y hy hx
          rw [hy'] at this
          norm_num at this
        · have hhead' : ∀ b ∈ t.head?, b ≠ 1 := by
            intro b hb hb1
            have := hlink b hb
            rw [hb1] at this
            exact this.2 ⟨rfl, rfl⟩
          have hx : y.1 = 0 := by
            simp only [stepOf] at hfst
            norm_num at hfst
            exact hfst
          have hy' : y.2 = 1 := by
            simp only [stepOf] at hsnd
            linarith
          have := y_nonpos_of_head t hat hhead' y hy hx
          rw [hy'] at this
          norm_num at this
      · exact iht.map (fun x y hxy => by simpa using hxy)

/-! ## Counting the conformations -/

/-- The partially directed conformations of `n` bonds that may follow a bond `p`. -/
def wordsFrom : ℕ → Letter → Finset (List Letter)
  | 0, _ => {[]}
  | (n + 1), p =>
      (Finset.univ.filter fun a => ok p a).biUnion fun a => (wordsFrom n a).image (fun w => a :: w)

lemma mem_wordsFrom_zero (p : Letter) (w : List Letter) : w ∈ wordsFrom 0 p ↔ w = [] := by
  simp [wordsFrom]

lemma length_of_mem_wordsFrom : ∀ (n : ℕ) (p : Letter) (w : List Letter),
    w ∈ wordsFrom n p → w.length = n := by
  intro n
  induction n with
  | zero => intro p w hw; simp [(mem_wordsFrom_zero p w).1 hw]
  | succ n ih =>
      intro p w hw
      rw [wordsFrom, Finset.mem_biUnion] at hw
      obtain ⟨a, _, hw⟩ := hw
      obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hw
      simp [ih a v hv]

/-- Every word counted is admissible, and compatible with the preceding bond. -/
lemma adm_of_mem_wordsFrom : ∀ (n : ℕ) (p : Letter) (w : List Letter),
    w ∈ wordsFrom n p → Adm w ∧ ∀ b ∈ w.head?, ok p b := by
  intro n
  induction n with
  | zero =>
      intro p w hw
      rw [(mem_wordsFrom_zero p w).1 hw]
      exact ⟨List.isChain_nil, by simp⟩
  | succ n ih =>
      intro p w hw
      rw [wordsFrom, Finset.mem_biUnion] at hw
      obtain ⟨a, ha, hw⟩ := hw
      obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hw
      obtain ⟨hadm, hhead⟩ := ih a v hv
      refine ⟨List.isChain_cons.2 ⟨hhead, hadm⟩, ?_⟩
      intro b hb
      simp only [List.head?_cons, Option.mem_def, Option.some.injEq] at hb
      subst hb
      simpa using ha

lemma card_wordsFrom_succ (n : ℕ) (p : Letter) :
    (wordsFrom (n + 1) p).card
      = ∑ a ∈ Finset.univ.filter (fun a => ok p a), (wordsFrom n a).card := by
  rw [wordsFrom, Finset.card_biUnion]
  · refine Finset.sum_congr rfl fun a _ => ?_
    exact Finset.card_image_of_injective _ (fun x y h => by simpa using h)
  · intro a _ b _ hab
    simp only [Finset.disjoint_left, Finset.mem_image]
    rintro w ⟨x, _, rfl⟩ ⟨y, _, hEq⟩
    apply hab
    have := congrArg List.head? hEq
    simpa using this.symm

/-- The number of partially directed conformations of an `n`-bond chain. -/
def pdCnt (n : ℕ) : ℕ := (wordsFrom n 0).card

/-- The number of such conformations that may follow a north bond. -/
def pdN (n : ℕ) : ℕ := (wordsFrom n 1).card

/-- The number of such conformations that may follow a south bond. -/
def pdS (n : ℕ) : ℕ := (wordsFrom n 2).card

lemma pdCnt_succ (n : ℕ) : pdCnt (n + 1) = pdCnt n + pdN n + pdS n := by
  have h : (Finset.univ.filter fun a : Letter => ok 0 a) = Finset.univ := by decide
  rw [pdCnt, card_wordsFrom_succ, h, Fin.sum_univ_three]
  rfl

lemma pdN_succ (n : ℕ) : pdN (n + 1) = pdCnt n + pdN n := by
  have h : (Finset.univ.filter fun a : Letter => ok 1 a) = {0, 1} := by decide
  rw [pdN, card_wordsFrom_succ, h, Finset.sum_pair (by decide : (0 : Letter) ≠ 1)]
  rfl

lemma pdS_succ (n : ℕ) : pdS (n + 1) = pdCnt n + pdS n := by
  have h : (Finset.univ.filter fun a : Letter => ok 2 a) = {0, 2} := by decide
  rw [pdS, card_wordsFrom_succ, h, Finset.sum_pair (by decide : (0 : Letter) ≠ 2)]
  rfl

@[simp] lemma pdCnt_zero : pdCnt 0 = 1 := by simp [pdCnt, wordsFrom]
@[simp] lemma pdN_zero : pdN 0 = 1 := by simp [pdN, wordsFrom]
@[simp] lemma pdS_zero : pdS 0 = 1 := by simp [pdS, wordsFrom]

/-! ## The exact growth constant `1 + √2` -/

/-- The growth constant of the partially directed chain. -/
noncomputable def lam : ℝ := 1 + Real.sqrt 2

lemma sqrt_two_sq : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)

lemma sqrt_two_pos : 0 < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)

lemma lam_pos : 0 < lam := by
  have := sqrt_two_pos; unfold lam; linarith

lemma lam_sq : lam * lam = 2 * lam + 1 := by
  unfold lam
  nlinarith [sqrt_two_sq]

/-- **Two-sided exact bounds on the conformation count.**  The partially directed chain has
between `λ ^ n` and `√2 · λ ^ n` conformations, `λ = 1 + √2`. -/
theorem pdCnt_bounds (n : ℕ) :
    lam ^ n ≤ (pdCnt n : ℝ) ∧ (pdCnt n : ℝ) ≤ (lam - 1) * lam ^ n ∧
      ((lam - 1) / 2) * lam ^ n ≤ (pdN n : ℝ) ∧ (pdN n : ℝ) ≤ lam ^ n ∧
      ((lam - 1) / 2) * lam ^ n ≤ (pdS n : ℝ) ∧ (pdS n : ℝ) ≤ lam ^ n := by
  have hs := sqrt_two_sq
  have hspos := sqrt_two_pos
  have hs1 : 1 < Real.sqrt 2 := by nlinarith
  have hs2 : Real.sqrt 2 < 3 / 2 := by nlinarith
  have hlam : lam = 1 + Real.sqrt 2 := rfl
  have hsq := lam_sq
  induction n with
  | zero =>
      refine ⟨by simp, ?_, ?_, ?_, ?_, ?_⟩ <;> simp <;> rw [hlam] <;> linarith
  | succ n ih =>
      obtain ⟨h1, h2, h3, h4, h5, h6⟩ := ih
      have hp : (0:ℝ) < lam ^ n := pow_pos lam_pos n
      have hpow : lam ^ (n + 1) = lam ^ n * lam := by ring
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
      · rw [pdCnt_succ]; push_cast; rw [hpow]; nlinarith [h1, h3, h5, hp, hsq]
      · rw [pdCnt_succ]; push_cast; rw [hpow]; nlinarith [h2, h4, h6, hp, hsq]
      · rw [pdN_succ]; push_cast; rw [hpow]; nlinarith [h1, h3, hp, hsq]
      · rw [pdN_succ]; push_cast; rw [hpow]; nlinarith [h2, h4, hp, hsq]
      · rw [pdS_succ]; push_cast; rw [hpow]; nlinarith [h1, h5, hp, hsq]
      · rw [pdS_succ]; push_cast; rw [hpow]; nlinarith [h2, h6, hp, hsq]

theorem lam_pow_le_pdCnt (n : ℕ) : lam ^ n ≤ (pdCnt n : ℝ) := (pdCnt_bounds n).1

theorem pdCnt_le_sqrt_two_mul (n : ℕ) : (pdCnt n : ℝ) ≤ Real.sqrt 2 * lam ^ n := by
  have h := (pdCnt_bounds n).2.1
  have : lam - 1 = Real.sqrt 2 := by simp [lam]
  rwa [this] at h

lemma pdCnt_pos (n : ℕ) : 0 < pdCnt n := by
  have h := lam_pow_le_pdCnt n
  have h0 : (0:ℝ) < lam ^ n := pow_pos lam_pos n
  have : (0:ℝ) < (pdCnt n : ℝ) := lt_of_lt_of_le h0 h
  exact_mod_cast this

/-- **The partially directed chain is exactly solvable**: its conformational entropy per
residue is exactly `log (1 + √2)`. -/
theorem pd_entropy_per_residue :
    Filter.Tendsto (fun n : ℕ => Real.log (pdCnt n) / n) Filter.atTop
      (nhds (Real.log (1 + Real.sqrt 2))) := by
  have hlam : Real.log lam = Real.log (1 + Real.sqrt 2) := rfl
  have hlow : ∀ n : ℕ, 0 < n → Real.log lam ≤ Real.log (pdCnt n) / n := by
    intro n hn
    have h := lam_pow_le_pdCnt n
    have hpos : (0:ℝ) < lam ^ n := pow_pos lam_pos n
    have hh := Real.log_le_log hpos h
    rw [Real.log_pow] at hh
    rw [le_div_iff₀ (by exact_mod_cast hn)]
    linarith
  have hhigh : ∀ n : ℕ, 0 < n →
      Real.log (pdCnt n) / n ≤ Real.log lam + Real.log (Real.sqrt 2) / n := by
    intro n hn
    have h := pdCnt_le_sqrt_two_mul n
    have hpos : (0:ℝ) < (pdCnt n : ℝ) := by exact_mod_cast pdCnt_pos n
    have hh := Real.log_le_log hpos h
    rw [Real.log_mul (ne_of_gt sqrt_two_pos) (ne_of_gt (pow_pos lam_pos n)),
      Real.log_pow] at hh
    have hnpos : (0:ℝ) < n := by exact_mod_cast hn
    rw [div_le_iff₀ hnpos]
    have : Real.log (Real.sqrt 2) / n * n = Real.log (Real.sqrt 2) := by
      field_simp
    nlinarith [hh]
  have hconst : Filter.Tendsto
      (fun n : ℕ => Real.log lam + Real.log (Real.sqrt 2) / n) Filter.atTop
      (nhds (Real.log lam)) := by
    have h0 : Filter.Tendsto (fun n : ℕ => Real.log (Real.sqrt 2) / n) Filter.atTop (nhds 0) :=
      tendsto_const_div_atTop_nhds_zero_nat _
    simpa using tendsto_const_nhds.add h0
  rw [← hlam]
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hconst ?_ ?_
  · filter_upwards [Filter.eventually_gt_atTop 0] with n hn using hlow n hn
  · filter_upwards [Filter.eventually_gt_atTop 0] with n hn using hhigh n hn

/-! ## The lower bound for the square lattice -/

/-- A partially directed conformation, read as a square-lattice conformation. -/
def toWalk (n : ℕ) (w : List Letter) : Fin n → Fin 4 := fun i => enc (w.getD i 0)

lemma stepsOf_toWalk {n : ℕ} {w : List Letter} (hw : w.length = n) :
    stepsOf (toWalk n w) = w.map stepOf := by
  have hlen : (stepsOf (toWalk n w)).length = (w.map stepOf).length := by
    simp [stepsOf, stepsOfDir, hw]
  refine List.ext_getElem hlen ?_
  intro i h1 h2
  have hi : i < n := by simpa [stepsOf, stepsOfDir] using h1
  simp only [stepsOf, stepsOfDir, List.getElem_ofFn, List.getElem_map, toWalk]
  rw [dir_enc]
  congr 1
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]
  rfl

theorem pdCnt_le_cnt (n : ℕ) : pdCnt n ≤ cnt n := by
  classical
  rw [pdCnt, cnt, cntOf]
  refine Finset.card_le_card_of_injOn (toWalk n) ?_ ?_
  · intro w hw
    rw [Finset.mem_coe] at hw
    obtain ⟨hadm, _⟩ := adm_of_mem_wordsFrom n 0 w hw
    have hlen := length_of_mem_wordsFrom n 0 w hw
    simp only [Finset.mem_coe, mem_sawFinsetOf]
    rw [show stepsOfDir dir (toWalk n w) = stepsOf (toWalk n w) from rfl,
      stepsOf_toWalk hlen]
    exact adm_isSAW w hadm
  · intro w hw v hv h
    rw [Finset.mem_coe] at hw hv
    have hlw := length_of_mem_wordsFrom n 0 w hw
    have hlv := length_of_mem_wordsFrom n 0 v hv
    refine List.ext_getElem (by rw [hlw, hlv]) ?_
    intro i h1 h2
    have hi : i < n := by omega
    have hfun := congrFun h ⟨i, hi⟩
    simp only [toWalk] at hfun
    have hw' : w.getD i 0 = w[i] := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; rfl
    have hv' : v.getD i 0 = v[i] := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; rfl
    rw [hw', hv'] at hfun
    have henc : Function.Injective enc := by decide
    exact henc hfun

/-- **A sharper rigorous lower bound on the connective constant of the square lattice.**
The conformational entropy per residue of a self-avoiding chain is at least `log (1 + √2)`. -/
theorem log_one_add_sqrt_two_le_connectiveConstant :
    Real.log (1 + Real.sqrt 2) ≤ connectiveConstant := by
  refine ge_of_tendsto tendsto_connectiveConstant ?_
  filter_upwards [Filter.eventually_gt_atTop 0] with n hn
  have hpow : lam ^ n ≤ (cnt n : ℝ) :=
    le_trans (lam_pow_le_pdCnt n) (by exact_mod_cast pdCnt_le_cnt n)
  have hpos : (0:ℝ) < lam ^ n := pow_pos lam_pos n
  have hlog := Real.log_le_log hpos hpow
  rw [Real.log_pow, show Real.log lam = Real.log (1 + Real.sqrt 2) from rfl] at hlog
  have hval : logCnt n = Real.log (cnt n) := rfl
  rw [le_div_iff₀ (by exact_mod_cast hn), hval]
  linarith

/-- The new bound strictly improves the bridge bound `log 251 / 7` of Part CXV. -/
theorem improves_on_bridge_bound : Real.log 251 / 7 < Real.log (1 + Real.sqrt 2) := by
  have hs1 : (1.414 : ℝ) < Real.sqrt 2 := by
    nlinarith [sqrt_two_sq, sqrt_two_pos]
  have hbig : (251 : ℝ) < (1 + Real.sqrt 2) ^ 7 := by
    have h1 : (2.414 : ℝ) ≤ 1 + Real.sqrt 2 := by linarith
    have h2 : (251 : ℝ) < (2.414 : ℝ) ^ 7 := by norm_num
    calc (251:ℝ) < (2.414:ℝ) ^ 7 := h2
      _ ≤ (1 + Real.sqrt 2) ^ 7 := by
          exact pow_le_pow_left₀ (by norm_num) h1 7
  have hlt : Real.log 251 < Real.log ((1 + Real.sqrt 2) ^ 7) :=
    Real.log_lt_log (by norm_num) hbig
  rw [Real.log_pow] at hlt
  rw [div_lt_iff₀ (by norm_num : (0:ℝ) < 7)]
  push_cast at hlt
  linarith

/-- **The improved bracket.**  `log (1 + √2) ≤ μ ≤ log 780 / 6`, i.e. `0.881… ≤ μ ≤ 1.109…`. -/
theorem connectiveConstant_bracket_pd :
    Real.log (1 + Real.sqrt 2) ≤ connectiveConstant ∧ connectiveConstant ≤ Real.log 780 / 6 :=
  ⟨log_one_add_sqrt_two_le_connectiveConstant, connectiveConstant_le_log780_div6⟩

end PD
end IDR.SAW
