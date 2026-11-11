/-
# Part LXVI  Umbrella sampling: what window data determine, and what they never can

A free-energy profile along a reaction coordinate -- radius of gyration, end-to-end distance,
a contact order -- is almost never measured directly.  It is stitched together from *windows*:
a bias `V k` is added to confine the run to a region `S k`, the biased histogram is recorded,
and the pieces are recombined (WHAM, MBAR, umbrella integration) into one profile.  For a
disordered region the profile *is* the result: the reported populations of compact and expanded
states, the barrier between them, the free energy of a bound-like conformer.

This file treats the recombination step exactly, on a finite conformation space.  A window is a
support `S` together with a bias `V`; its data is the biased conditional
`winDist pi V S x = pi x exp(-V x)/Z_S` supported on `S`.

* `winDist_unbias` -- reweighting the window data by `exp(V)` returns exactly the conditional
  distribution of the target on `S`.  Within a window there is no bias left.
* `winDist_smul` -- but the data are invariant under rescaling the target: a window fixes the
  *shape* on its own support and nothing else, which is why WHAM returns free-energy offsets
  only up to a global constant.
* `free_energy_unidentifiable` -- the sharp negative.  If the conformation space splits into `A`
  and its complement so that **no window straddles the split**, then for *every* positive ratio
  `r` there is a target with the prescribed ratio of populations reproducing every window
  histogram exactly.  The relative free energy of the two regions is not underdetermined by a
  little: it is completely undetermined, and no amount of sampling inside the windows helps.
* `umbrella_identifiable` -- the matching positive result.  If consecutive windows overlap and
  the windows cover the space, then window data determine the target up to a single positive
  constant, hence determine every population ratio and every free-energy difference.  Overlap of
  the window supports is therefore exactly the right condition: necessary by the theorem above,
  sufficient by this one.
* `overlap_missed_ge`, `half_of_runs_blind`, `thin_overlap_example` -- and overlap on paper is
  not overlap in the data.  If the shared region carries probability `m`, a run of `N` frames
  misses it entirely with probability at least `1 - N m`, so for `N <= 1/(2m)` at least half of
  all runs return data whose *observed* supports do not straddle -- and then the previous theorem
  applies to what was actually recorded.  A window overlap that is never visited is no overlap.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace Umbrella

open Finset

variable {n : ℕ}

/-! ## Windows -/

/-- The normalisation of a biased window. -/
noncomputable def winZ (pi V : Fin n → ℝ) (S : Finset (Fin n)) : ℝ :=
  ∑ y ∈ S, pi y * Real.exp (-V y)

/-- The data returned by an umbrella window: the biased distribution restricted to `S`. -/
noncomputable def winDist (pi V : Fin n → ℝ) (S : Finset (Fin n)) (x : Fin n) : ℝ :=
  if x ∈ S then pi x * Real.exp (-V x) / winZ pi V S else 0

lemma winZ_pos {pi : Fin n → ℝ} (hpi : ∀ x, 0 < pi x) (V : Fin n → ℝ) {S : Finset (Fin n)}
    (hS : S.Nonempty) : 0 < winZ pi V S :=
  Finset.sum_pos (fun y _ => mul_pos (hpi y) (Real.exp_pos _)) hS

lemma winDist_mem {pi V : Fin n → ℝ} {S : Finset (Fin n)} {x : Fin n} (hx : x ∈ S) :
    winDist pi V S x = pi x * Real.exp (-V x) / winZ pi V S := by
  rw [winDist, if_pos hx]

/-- **Within a window there is no bias left.**  Reweighting the window data by `exp(V)` returns
exactly the conditional distribution of the target on the window support. -/
theorem winDist_unbias {pi : Fin n → ℝ} (hpi : ∀ x, 0 < pi x) (V : Fin n → ℝ)
    {S : Finset (Fin n)} (hS : S.Nonempty) {x : Fin n} (hx : x ∈ S) :
    winDist pi V S x * Real.exp (V x) / (∑ y ∈ S, winDist pi V S y * Real.exp (V y))
      = pi x / ∑ y ∈ S, pi y := by
  have hZ := winZ_pos hpi V hS
  have hs : 0 < ∑ y ∈ S, pi y := Finset.sum_pos (fun y _ => hpi y) hS
  have hZ' : winZ pi V S ≠ 0 := hZ.ne'
  have hs' : (∑ y ∈ S, pi y) ≠ 0 := hs.ne'
  have hterm : ∀ y ∈ S, winDist pi V S y * Real.exp (V y) = pi y / winZ pi V S := by
    intro y hy
    rw [winDist_mem hy]
    field_simp
    rw [mul_assoc, ← Real.exp_add]
    simp
  rw [Finset.sum_congr rfl hterm, hterm x hx, ← Finset.sum_div]
  field_simp

/-- A window fixes the shape of the target on its own support and nothing more: rescaling the
target leaves every window histogram unchanged. -/
theorem winDist_smul {pi : Fin n → ℝ} (V : Fin n → ℝ) {S : Finset (Fin n)} {c : ℝ} (hc : 0 < c) :
    winDist (fun x => c * pi x) V S = winDist pi V S := by
  funext x
  unfold winDist winZ
  by_cases hx : x ∈ S
  · rw [if_pos hx, if_pos hx]
    have : ∑ y ∈ S, c * pi y * Real.exp (-V y) = c * ∑ y ∈ S, pi y * Real.exp (-V y) := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl (fun y _ => by ring)
    rw [this]
    rw [show c * pi x * Real.exp (-V x) = c * (pi x * Real.exp (-V x)) by ring]
    rw [mul_div_mul_left _ _ hc.ne']
  · rw [if_neg hx, if_neg hx]

/-- If two targets agree up to a positive factor on a window support, the window cannot tell
them apart. -/
theorem winDist_congr_of_prop {pi q : Fin n → ℝ} (V : Fin n → ℝ) {S : Finset (Fin n)} {c : ℝ}
    (hc : 0 < c) (h : ∀ x ∈ S, q x = c * pi x) : winDist q V S = winDist pi V S := by
  have : winDist q V S = winDist (fun x => c * pi x) V S := by
    funext x
    unfold winDist winZ
    by_cases hx : x ∈ S
    · rw [if_pos hx, if_pos hx, h x hx,
        Finset.sum_congr rfl (fun y hy => by rw [h y hy])]
    · rw [if_neg hx, if_neg hx]
  rw [this, winDist_smul V hc]

/-! ## Without a straddling window the relative free energy is arbitrary -/

/-- Tilting the target by a factor `c` on the region `A`. -/
noncomputable def tilt (c : ℝ) (A : Finset (Fin n)) (pi : Fin n → ℝ) (x : Fin n) : ℝ :=
  if x ∈ A then c * pi x else pi x

lemma tilt_pos {c : ℝ} (hc : 0 < c) {A : Finset (Fin n)} {pi : Fin n → ℝ} (hpi : ∀ x, 0 < pi x)
    (x : Fin n) : 0 < tilt c A pi x := by
  unfold tilt
  split
  · exact mul_pos hc (hpi x)
  · exact hpi x

lemma tilt_sum_mem {c : ℝ} {A : Finset (Fin n)} {pi : Fin n → ℝ} :
    ∑ x ∈ A, tilt c A pi x = c * ∑ x ∈ A, pi x := by
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun x hx => by rw [tilt, if_pos hx])

lemma tilt_sum_not_mem {c : ℝ} {A : Finset (Fin n)} {pi : Fin n → ℝ} :
    ∑ x ∈ Aᶜ, tilt c A pi x = ∑ x ∈ Aᶜ, pi x :=
  Finset.sum_congr rfl (fun x hx => by
    rw [tilt, if_neg (Finset.mem_compl.mp hx)])

/-- **The relative free energy of two regions never straddled by a window is completely
undetermined.**  For every prescribed ratio `r` of populations there is a target with that ratio
reproducing every window histogram exactly. -/
theorem free_energy_unidentifiable {pi : Fin n → ℝ} (hpi : ∀ x, 0 < pi x)
    {A : Finset (Fin n)} (hA : A.Nonempty) (hAc : Aᶜ.Nonempty)
    (S : ℕ → Finset (Fin n)) (V : ℕ → Fin n → ℝ) (K : ℕ)
    (hstr : ∀ k, k ≤ K → (S k ⊆ A ∨ Disjoint (S k) A)) {r : ℝ} (hr : 0 < r) :
    ∃ q : Fin n → ℝ, (∀ x, 0 < q x) ∧
      (∀ k, k ≤ K → winDist q (V k) (S k) = winDist pi (V k) (S k)) ∧
      (∑ x ∈ A, q x) / (∑ x ∈ Aᶜ, q x) = r := by
  have hsA : 0 < ∑ x ∈ A, pi x := Finset.sum_pos (fun x _ => hpi x) hA
  have hsAc : 0 < ∑ x ∈ Aᶜ, pi x := Finset.sum_pos (fun x _ => hpi x) hAc
  set c : ℝ := r * (∑ x ∈ Aᶜ, pi x) / (∑ x ∈ A, pi x) with hcdef
  have hc : 0 < c := by
    rw [hcdef]
    exact div_pos (mul_pos hr hsAc) hsA
  refine ⟨tilt c A pi, fun x => tilt_pos hc hpi x, ?_, ?_⟩
  · intro k hk
    rcases hstr k hk with hsub | hdisj
    · refine winDist_congr_of_prop (V k) hc (fun x hx => ?_)
      rw [tilt, if_pos (hsub hx)]
    · refine winDist_congr_of_prop (V k) one_pos (fun x hx => ?_)
      have hxA : x ∉ A := fun hxA => (Finset.disjoint_left.mp hdisj hx) hxA
      rw [tilt, if_neg hxA, one_mul]
  · rw [tilt_sum_mem, tilt_sum_not_mem, hcdef]
    field_simp

/-! ## With overlapping windows the target is determined -/

/-- Agreement of two targets on one window forces them to be proportional there. -/
theorem prop_of_winDist_eq {pi q : Fin n → ℝ} (hpi : ∀ x, 0 < pi x) (hq : ∀ x, 0 < q x)
    {S : Finset (Fin n)} (hS : S.Nonempty) {V : Fin n → ℝ}
    (h : winDist q V S = winDist pi V S) :
    ∃ c, 0 < c ∧ ∀ x ∈ S, q x = c * pi x := by
  have hZp := winZ_pos hpi V hS
  have hZq := winZ_pos hq V hS
  refine ⟨winZ q V S / winZ pi V S, div_pos hZq hZp, fun x hx => ?_⟩
  have hx' := congrFun h x
  rw [winDist_mem hx, winDist_mem hx] at hx'
  have hexp : Real.exp (-V x) ≠ 0 := (Real.exp_pos _).ne'
  field_simp at hx' ⊢
  nlinarith [hx', hZp, hZq, Real.exp_pos (-V x)]

/-- **Overlapping windows determine the target up to one constant.**  If consecutive window
supports meet and the windows cover the conformation space, two targets with the same window
histograms are proportional -- hence give the same populations, and the same free-energy
differences. -/
theorem umbrella_identifiable {pi q : Fin n → ℝ} (hpi : ∀ x, 0 < pi x) (hq : ∀ x, 0 < q x)
    (S : ℕ → Finset (Fin n)) (V : ℕ → Fin n → ℝ) (K : ℕ)
    (hns : ∀ k, k ≤ K → (S k).Nonempty)
    (hov : ∀ k, k < K → ((S k) ∩ (S (k + 1))).Nonempty)
    (hcov : ∀ x : Fin n, ∃ k, k ≤ K ∧ x ∈ S k)
    (hdata : ∀ k, k ≤ K → winDist q (V k) (S k) = winDist pi (V k) (S k)) :
    ∃ c, 0 < c ∧ ∀ x, q x = c * pi x := by
  obtain ⟨c, hc, hc0⟩ := prop_of_winDist_eq hpi hq (hns 0 (Nat.zero_le K)) (hdata 0 (Nat.zero_le K))
  refine ⟨c, hc, fun x => ?_⟩
  have key : ∀ k, k ≤ K → ∀ y ∈ S k, q y = c * pi y := by
    intro k
    induction k with
    | zero => intro _ y hy; exact hc0 y hy
    | succ m ih =>
        intro hm y hy
        have hmK : m ≤ K := Nat.le_of_succ_le hm
        obtain ⟨cm, hcm, hcm'⟩ :=
          prop_of_winDist_eq hpi hq (hns (m + 1) hm) (hdata (m + 1) hm)
        obtain ⟨z, hz⟩ := hov m (by omega)
        have hz1 : z ∈ S m := (Finset.mem_inter.mp hz).1
        have hz2 : z ∈ S (m + 1) := (Finset.mem_inter.mp hz).2
        have h1 : q z = c * pi z := ih hmK z hz1
        have h2 : q z = cm * pi z := hcm' z hz2
        have hceq : c = cm := by
          have hpz := hpi z
          have : c * pi z = cm * pi z := by rw [← h1, h2]
          exact mul_right_cancel₀ hpz.ne' this
        rw [hcm' y hy, hceq]
  obtain ⟨k, hk, hxk⟩ := hcov x
  exact key k hk x hxk

/-- Proportional targets have identical populations, so the free energy of every region is
determined. -/
theorem population_determined {pi q : Fin n → ℝ} {c : ℝ} (hc : 0 < c)
    (h : ∀ x, q x = c * pi x) (A : Finset (Fin n)) :
    (∑ x ∈ A, q x) / (∑ x, q x) = (∑ x ∈ A, pi x) / (∑ x, pi x) := by
  have hsum : ∀ B : Finset (Fin n), ∑ x ∈ B, q x = c * ∑ x ∈ B, pi x := by
    intro B
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun x _ => h x)
  rw [hsum A, hsum Finset.univ]
  rw [mul_div_mul_left _ _ hc.ne']

/-! ## Overlap in the data, not on paper -/

/-- A region of probability `m` is missed by all `N` frames of a run with probability at least
`1 - N m`. -/
theorem overlap_missed_ge {m : ℝ} (hm1 : m ≤ 1) (N : ℕ) :
    1 - N * m ≤ (1 - m) ^ N := by
  have h := one_add_mul_le_pow (a := -m) (by linarith) N
  calc 1 - (N : ℝ) * m = 1 + N * (-m) := by ring
    _ ≤ (1 + -m) ^ N := h
    _ = (1 - m) ^ N := by ring_nf

/-- So a run shorter than `1/(2m)` frames misses the shared region entirely in at least half of
all runs. -/
theorem half_of_runs_blind {m : ℝ} (hm1 : m ≤ 1) {N : ℕ} (hN : (N : ℝ) * m ≤ 1/2) :
    (1:ℝ)/2 ≤ (1 - m) ^ N := by
  have := overlap_missed_ge hm1 N
  linarith

/-- **A thin overlap is no overlap.**  Two windows `{0,1}` and `{1,2}` do overlap, but if no
frame of either run visits state `1` the recorded supports are `{0}` and `{2}`, and then the
relative free energy of `{0,1}` against `{2}` takes every value consistent with the data. -/
theorem thin_overlap_example {pi : Fin 3 → ℝ} (hpi : ∀ x, 0 < pi x) (V : ℕ → Fin 3 → ℝ)
    {r : ℝ} (hr : 0 < r) :
    ∃ q : Fin 3 → ℝ, (∀ x, 0 < q x) ∧
      winDist q (V 0) {0} = winDist pi (V 0) {0} ∧
      winDist q (V 1) {2} = winDist pi (V 1) {2} ∧
      (∑ x ∈ ({0, 1} : Finset (Fin 3)), q x) / (∑ x ∈ ({0, 1} : Finset (Fin 3))ᶜ, q x) = r := by
  have hA : ({0, 1} : Finset (Fin 3)).Nonempty := ⟨0, by decide⟩
  have hAc : (({0, 1} : Finset (Fin 3))ᶜ).Nonempty := ⟨2, by decide⟩
  obtain ⟨q, hq, hdata, hratio⟩ :=
    free_energy_unidentifiable hpi hA hAc
      (fun k => if k = 0 then ({0} : Finset (Fin 3)) else ({2} : Finset (Fin 3))) V 1
      (fun k _ => by
        by_cases hk : k = 0
        · exact Or.inl (by simp [hk])
        · exact Or.inr (by simp [hk])) hr
  refine ⟨q, hq, ?_, ?_, hratio⟩
  · simpa using hdata 0 (Nat.zero_le 1)
  · simpa using hdata 1 (le_refl 1)

end Umbrella

end IDR
