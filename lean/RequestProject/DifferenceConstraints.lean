/-
# Systems of strict difference constraints and their cycle criterion

The question "can a per-protein bias make every cross-protein comparison come out right?" is a
question about a system of strict difference constraints

  `c k l < b k − b l`  for all `k ≠ l`.

This file proves the exact criterion, from scratch: **such a system is feasible if and only if
every closed walk has strictly negative total weight** (`feasible_iff_noNonnegCycle`).  Since the
condition ranges only over closed walks, and a maximum-weight walk can be taken simple, it is
decided by a negative-cycle detection (Bellman–Ford / Floyd–Warshall) in `O(K³)` arithmetic
operations on the `K × K` matrix `c` — no search over the `K!` orders of the proteins.

The proof of the hard direction is by elimination of one vertex at a time: removing `x` from the
vertex set, every constraint `(k, l)` is replaced by the stronger of itself and the shortcut
through `x`, and every cycle of the reduced system expands to a cycle of the original one
(`wsum_expand`).  The reduced system is feasible by induction; `x` is then placed in the open
interval left free by the shortcut constraints.
-/
import Mathlib

set_option autoImplicit false

namespace IDR.DiffConstraints

open List

variable {G : Type*}

/-! ## Walks and their weights -/

/-- The weight of a walk presented as its list of vertices: `Σ c (v i) (v (i+1))`. -/
def wsum (c : G → G → ℝ) : List G → ℝ
  | [] => 0
  | [_] => 0
  | x :: y :: t => c x y + wsum c (y :: t)

@[simp] lemma wsum_nil (c : G → G → ℝ) : wsum c ([] : List G) = 0 := rfl

@[simp] lemma wsum_singleton (c : G → G → ℝ) (x : G) : wsum c [x] = 0 := rfl

@[simp] lemma wsum_cons_cons (c : G → G → ℝ) (x y : G) (t : List G) :
    wsum c (x :: y :: t) = c x y + wsum c (y :: t) := rfl

/-- Feasibility of the strict difference system on a set `S` of vertices. -/
def Feasible (S : Finset G) (c : G → G → ℝ) : Prop :=
  ∃ b : G → ℝ, ∀ k ∈ S, ∀ l ∈ S, k ≠ l → c k l < b k - b l

/-- No closed walk inside `S` has non-negative weight.  (Walks are required to make genuine
steps: consecutive vertices differ.) -/
def NoNonnegCycleOn (S : Finset G) (c : G → G → ℝ) : Prop :=
  ∀ L : List G, (∀ v ∈ L, v ∈ S) → L.IsChain (· ≠ ·) → 2 ≤ L.length →
    L.head? = L.getLast? → wsum c L < 0

/-! ## Feasible ⇒ every cycle is negative -/

lemma wsum_lt_of_feasible {c : G → G → ℝ} {b : G → ℝ} {S : Finset G}
    (hb : ∀ k ∈ S, ∀ l ∈ S, k ≠ l → c k l < b k - b l) :
    ∀ (t : List G) (x y : G), x ∈ S → y ∈ S → (∀ v ∈ t, v ∈ S) →
      (x :: y :: t).IsChain (· ≠ ·) →
      wsum c (x :: y :: t) < b x - b ((y :: t).getLast (by simp)) := by
  intro t
  induction t with
  | nil =>
      intro x y hx hy _ h
      have hxy : x ≠ y := (List.isChain_cons_cons.mp h).1
      simpa using hb x hx y hy hxy
  | cons z t ih =>
      intro x y hx hy hmem h
      have hxy : x ≠ y := (List.isChain_cons_cons.mp h).1
      have hz : z ∈ S := hmem z (by simp)
      have hmem' : ∀ v ∈ t, v ∈ S := fun v hv => hmem v (by simp [hv])
      have hrest : (y :: z :: t).IsChain (· ≠ ·) := (List.isChain_cons_cons.mp h).2
      have hih := ih y z hy hz hmem' hrest
      have h1 : c x y < b x - b y := hb x hx y hy hxy
      have hlast : ((y :: z :: t).getLast (by simp)) = ((z :: t).getLast (by simp)) := by
        simp [List.getLast_cons]
      rw [wsum_cons_cons, hlast]
      linarith

/-- If the system is feasible then every closed walk has strictly negative weight. -/
theorem noNonnegCycle_of_feasible {S : Finset G} {c : G → G → ℝ} (h : Feasible S c) :
    NoNonnegCycleOn S c := by
  obtain ⟨b, hb⟩ := h
  intro L hmem hchain hlen hcyc
  match L, hlen with
  | x :: y :: t, _ =>
      have hx : x ∈ S := hmem x (by simp)
      have hy : y ∈ S := hmem y (by simp)
      have hmem' : ∀ v ∈ t, v ∈ S := fun v hv => hmem v (by simp [hv])
      have key := wsum_lt_of_feasible hb t x y hx hy hmem' hchain
      have hlast : (x :: y :: t).getLast? = some ((y :: t).getLast (by simp)) := by
        rw [List.getLast?_cons_cons]
        exact List.getLast?_eq_some_getLast _
      have hhead : (x :: y :: t).head? = some x := rfl
      rw [hhead, hlast] at hcyc
      have : x = (y :: t).getLast (by simp) := Option.some.inj hcyc
      rw [← this] at key
      simpa using key

/-! ## Vertex elimination -/

/-- The reduced weight after eliminating `x`: the stronger of the direct constraint and the
constraint obtained by going through `x`. -/
noncomputable def shortcut (c : G → G → ℝ) (x : G) : G → G → ℝ :=
  fun a b => max (c a b) (c a x + c x b)

/-- Expand a walk of the reduced system into a walk of the original one, inserting `x` wherever
the reduced weight came from the shortcut. -/
noncomputable def expand (c : G → G → ℝ) (x : G) : List G → List G
  | [] => []
  | [a] => [a]
  | a :: b :: t =>
      if c a b < c a x + c x b then a :: x :: expand c x (b :: t) else a :: expand c x (b :: t)

lemma expand_head? (c : G → G → ℝ) (x : G) (L : List G) :
    (expand c x L).head? = L.head? := by
  match L with
  | [] => rfl
  | [_] => rfl
  | a :: b :: t =>
      unfold expand
      split <;> rfl

lemma expand_ne_nil (c : G → G → ℝ) (x : G) {L : List G} (h : L ≠ []) :
    expand c x L ≠ [] := by
  intro hcon
  have := expand_head? c x L
  rw [hcon] at this
  cases L with
  | nil => exact h rfl
  | cons a t => simp at this

lemma getLast?_cons_of_ne_nil (a : G) {M : List G} (h : M ≠ []) :
    (a :: M).getLast? = M.getLast? := by
  cases M with
  | nil => exact absurd rfl h
  | cons b t => exact List.getLast?_cons_cons

lemma expand_getLast? (c : G → G → ℝ) (x : G) (L : List G) :
    (expand c x L).getLast? = L.getLast? := by
  induction L with
  | nil => rfl
  | cons a t ih =>
      cases t with
      | nil => rfl
      | cons b t' =>
          have hne : expand c x (b :: t') ≠ [] := expand_ne_nil c x (by simp)
          unfold expand
          split
          · rw [getLast?_cons_of_ne_nil a (by simp), getLast?_cons_of_ne_nil x hne, ih,
              List.getLast?_cons_cons]
          · rw [getLast?_cons_of_ne_nil a hne, ih, List.getLast?_cons_cons]

lemma expand_mem (c : G → G → ℝ) (x : G) (L : List G) :
    ∀ v ∈ expand c x L, v = x ∨ v ∈ L := by
  induction L with
  | nil => intro v hv; simp [expand] at hv
  | cons a t ih =>
      cases t with
      | nil => intro v hv; simp [expand] at hv; simp [hv]
      | cons b t' =>
          intro v hv
          unfold expand at hv
          split at hv
          · rcases List.mem_cons.mp hv with h | h
            · exact Or.inr (by simp [h])
            · rcases List.mem_cons.mp h with h' | h'
              · exact Or.inl h'
              · rcases ih v h' with h'' | h''
                · exact Or.inl h''
                · exact Or.inr (by simp [h''])
          · rcases List.mem_cons.mp hv with h | h
            · exact Or.inr (by simp [h])
            · rcases ih v h with h'' | h''
              · exact Or.inl h''
              · exact Or.inr (by simp [h''])

lemma expand_length (c : G → G → ℝ) (x : G) (L : List G) :
    L.length ≤ (expand c x L).length := by
  induction L with
  | nil => simp [expand]
  | cons a t ih =>
      cases t with
      | nil => simp [expand]
      | cons b t' =>
          simp only [List.length_cons] at ih ⊢
          unfold expand
          split
          · simp only [List.length_cons]
            omega
          · simp only [List.length_cons]
            omega

lemma expand_chain' (c : G → G → ℝ) (x : G) (L : List G)
    (hchain : L.IsChain (· ≠ ·)) (hx : ∀ v ∈ L, v ≠ x) :
    (expand c x L).IsChain (· ≠ ·) := by
  induction L with
  | nil => simp [expand]
  | cons a t ih =>
      cases t with
      | nil => simp [expand]
      | cons b t' =>
          have hab : a ≠ b := (List.isChain_cons_cons.mp hchain).1
          have hrest : (b :: t').IsChain (· ≠ ·) := (List.isChain_cons_cons.mp hchain).2
          have hx' : ∀ v ∈ (b :: t'), v ≠ x := fun v hv => hx v (by simp [hv])
          have hih := ih hrest hx'
          have hhead : (expand c x (b :: t')).head? = some b := by
            rw [expand_head?]; rfl
          obtain ⟨r, hr⟩ : ∃ r, expand c x (b :: t') = b :: r := by
            cases hE : expand c x (b :: t') with
            | nil => rw [hE] at hhead; simp at hhead
            | cons u r =>
                rw [hE] at hhead
                simp only [List.head?_cons, Option.some.injEq] at hhead
                exact ⟨r, by rw [hhead]⟩
          have hax : a ≠ x := hx a (by simp)
          have hxb : x ≠ b := fun h => (hx b (by simp)) h.symm
          unfold expand
          split
          · rw [hr] at hih ⊢
            refine List.isChain_cons_cons.mpr ⟨hax, List.isChain_cons_cons.mpr ⟨hxb, hih⟩⟩
          · rw [hr] at hih ⊢
            exact List.isChain_cons_cons.mpr ⟨hab, hih⟩

lemma wsum_expand (c : G → G → ℝ) (x : G) (L : List G) :
    wsum c (expand c x L) = wsum (shortcut c x) L := by
  induction L with
  | nil => simp [expand]
  | cons a t ih =>
      cases t with
      | nil => simp [expand]
      | cons b t' =>
          have hhead : (expand c x (b :: t')).head? = some b := by
            rw [expand_head?]; rfl
          obtain ⟨r, hr⟩ : ∃ r, expand c x (b :: t') = b :: r := by
            cases hE : expand c x (b :: t') with
            | nil => rw [hE] at hhead; simp at hhead
            | cons u r =>
                rw [hE] at hhead
                simp only [List.head?_cons, Option.some.injEq] at hhead
                exact ⟨r, by rw [hhead]⟩
          rw [hr] at ih
          unfold expand
          split <;> rename_i hcond
          · rw [hr]
            rw [wsum_cons_cons, wsum_cons_cons, ih, wsum_cons_cons, shortcut,
              max_eq_right hcond.le]
            ring
          · rw [hr]
            rw [wsum_cons_cons, ih, wsum_cons_cons, shortcut,
              max_eq_left (not_lt.mp hcond)]

/-! ## Every cycle negative ⇒ feasible -/

theorem feasible_of_noNonnegCycle [DecidableEq G] [Fintype G] :
    ∀ (S : Finset G) (c : G → G → ℝ), NoNonnegCycleOn S c → Feasible S c := by
  intro S
  induction S using Finset.strongInduction with
  | _ S ih =>
    intro c hcyc
    rcases S.eq_empty_or_nonempty with rfl | ⟨x, hxS⟩
    · exact ⟨fun _ => 0, by simp⟩
    · set S' := S.erase x with hS'
      have hsub : S' ⊂ S := Finset.erase_ssubset hxS
      have hcyc' : NoNonnegCycleOn S' (shortcut c x) := by
        intro L hmem hchain hlen hhl
        have hxL : ∀ v ∈ L, v ≠ x := by
          intro v hv
          have := hmem v hv
          rw [hS'] at this
          exact (Finset.mem_erase.mp this).1
        have hmemS : ∀ v ∈ expand c x L, v ∈ S := by
          intro v hv
          rcases expand_mem c x L v hv with h | h
          · exact h ▸ hxS
          · exact Finset.mem_of_mem_erase (hmem v h)
        have hchainE : (expand c x L).IsChain (· ≠ ·) := expand_chain' c x L hchain hxL
        have hlenE : 2 ≤ (expand c x L).length := le_trans hlen (expand_length c x L)
        have hhlE : (expand c x L).head? = (expand c x L).getLast? := by
          rw [expand_head?, expand_getLast?]; exact hhl
        have := hcyc (expand c x L) hmemS hchainE hlenE hhlE
        rwa [wsum_expand] at this
      obtain ⟨b', hb'⟩ := ih S' hsub (shortcut c x) hcyc'
      -- place `x`
      have hgap : ∀ l ∈ S', ∀ m ∈ S', b' l + c x l < b' m - c m x := by
        intro l hl m hm
        by_cases hlm : l = m
        · subst hlm
          -- the two-cycle `x → l → x`
          have hxl : x ≠ l := fun h => (Finset.mem_erase.mp hl).1 h.symm
          have hcycle : wsum c [x, l, x] < 0 := by
            refine hcyc [x, l, x] ?_ ?_ (by simp) (by simp)
            · intro v hv
              simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
              rcases hv with rfl | rfl | rfl
              · exact hxS
              · exact Finset.mem_of_mem_erase hl
              · exact hxS
            · simp [List.isChain_cons_cons, hxl, Ne.symm hxl]
          simp only [wsum_cons_cons, wsum_singleton, add_zero] at hcycle
          linarith
        · have hml : (shortcut c x) m l < b' m - b' l := hb' m hm l hl (Ne.symm hlm)
          have : c m x + c x l ≤ (shortcut c x) m l := le_max_right _ _
          linarith
      -- choose a value for `x` strictly between the two finite bounds
      rcases S'.eq_empty_or_nonempty with hemp | hne
      · refine ⟨Function.update b' x 0, ?_⟩
        intro k hk l hl hkl
        exfalso
        have hkx : k = x := by
          by_contra h
          have : k ∈ S' := Finset.mem_erase.mpr ⟨h, hk⟩
          rw [hemp] at this; simp at this
        have hlx : l = x := by
          by_contra h
          have : l ∈ S' := Finset.mem_erase.mpr ⟨h, hl⟩
          rw [hemp] at this; simp at this
        exact hkl (hkx.trans hlx.symm)
      · set lo := S'.sup' hne (fun l => b' l + c x l) with hlo
        set hi := S'.inf' hne (fun m => b' m - c m x) with hhi
        have hlohi : lo < hi := by
          rw [hlo, hhi]
          rw [Finset.sup'_lt_iff]
          intro l hl
          rw [Finset.lt_inf'_iff]
          intro m hm
          exact hgap l hl m hm
        refine ⟨Function.update b' x ((lo + hi) / 2), ?_⟩
        intro k hk l hl hkl
        by_cases hkx : k = x
        · subst hkx
          have hlS' : l ∈ S' := Finset.mem_erase.mpr ⟨fun h => hkl h.symm, hl⟩
          have hle : b' l + c k l ≤ lo := by
            rw [hlo]; exact Finset.le_sup' (fun l => b' l + c k l) hlS'
          have h1 : Function.update b' k ((lo + hi) / 2) k = (lo + hi) / 2 := by simp
          have h2 : Function.update b' k ((lo + hi) / 2) l = b' l := by
            rw [Function.update_of_ne (fun h => hkl h.symm)]
          rw [h1, h2]
          linarith
        · by_cases hlx : l = x
          · subst hlx
            have hkS' : k ∈ S' := Finset.mem_erase.mpr ⟨hkx, hk⟩
            have hge : hi ≤ b' k - c k l := by
              rw [hhi]; exact Finset.inf'_le (fun m => b' m - c m l) hkS'
            have h1 : Function.update b' l ((lo + hi) / 2) l = (lo + hi) / 2 := by simp
            have h2 : Function.update b' l ((lo + hi) / 2) k = b' k := by
              rw [Function.update_of_ne hkx]
            rw [h1, h2]
            linarith
          · have hkS' : k ∈ S' := Finset.mem_erase.mpr ⟨hkx, hk⟩
            have hlS' : l ∈ S' := Finset.mem_erase.mpr ⟨hlx, hl⟩
            have h1 : Function.update b' x ((lo + hi) / 2) k = b' k := by
              rw [Function.update_of_ne hkx]
            have h2 : Function.update b' x ((lo + hi) / 2) l = b' l := by
              rw [Function.update_of_ne hlx]
            rw [h1, h2]
            have := hb' k hkS' l hlS' hkl
            have hle : c k l ≤ (shortcut c x) k l := le_max_left _ _
            linarith

/-- **The cycle criterion for strict difference constraints.**  A system `c k l < b k − b l` over
a finite vertex set is feasible if and only if every closed walk has strictly negative weight. -/
theorem feasible_iff_noNonnegCycle [DecidableEq G] [Fintype G] (S : Finset G) (c : G → G → ℝ) :
    Feasible S c ↔ NoNonnegCycleOn S c :=
  ⟨noNonnegCycle_of_feasible, feasible_of_noNonnegCycle S c⟩

end IDR.DiffConstraints
