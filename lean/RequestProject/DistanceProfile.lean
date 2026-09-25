/-
# Part CXL  The internal-distance profile is identifiable without a chain model

Parts CXXXVIII–CXXXIX read the distance law of a charged disordered region off a salt titration
inside a *parametric* family: first the ideal chain `R(d) = b√d`, then the polymer scaling law
`R(d) = b·d^ν`.  Both times the model had to declare a family and could at best calibrate its two
parameters.  A real disordered region has no reason to obey either law exactly: local stiffness,
proline content, transient helices and charge blocks all deform the internal-distance profile
`R(d)` (the mean spatial distance between residues `d` apart) away from any two-parameter form,
and it is `R` itself, not a fitted exponent, that a model of the region is supposed to reproduce.

This part drops the family entirely.  The titration curve of a region with internal-distance
profile `R` and charge-correlation profile `c` is

  `kCurve 1 N R c κ = ∑_{d=1}^{N−1} c d · e^{−κ·R d} / R d`,

the screened Coulomb reading of the correlations with *no* assumption on `R` beyond positivity and
monotonicity — distances grow with sequence separation.  Three results.

* **Non-parametric identification.**  `internal_distance_profile_identifiable` — if the charge
  correlation profile is known and vanishes nowhere, and both candidate distance profiles are
  positive and strictly increasing, then agreement of the titration curves at every ionic strength
  forces the two distance profiles to agree at *every* separation.  So the distance law is not a
  modelling choice that has to be declared and defended: given an independent measurement of the
  correlations, the whole profile `R` is a measurable object.  The proof is an induction on
  separation driven by the slowest-rate lemma: the shortest separation carries the slowest
  screening rate, so it is determined first; once determined its term cancels identically and the
  next separation becomes the slowest, and so on.

* **What monotonicity is doing.**  `swap_rates_confound` — it is not a technical hypothesis.  If
  two separations carry *equal* correlations, exchanging their distances leaves the titration
  curve unchanged at every ionic strength.  Identification of `R` therefore holds up to
  permutations of separations with equal correlation, and it is exactly the physical requirement
  that distance increase with separation that removes them.

* **The parametric results are corollaries.**  `polymer_params_of_rates` — matching rates at
  separations `1` and `2` already forces the bond length and the swelling exponent of Part CXXXIX,
  so the non-parametric theorem contains the parametric ones, and shows what they were really
  using: two active short separations and a monotone law.

Design consequence, closing the sequence of Parts CXXXIV–CXL: a model of a charged disordered
region should report its internal-distance profile as *data*, calibrated separation by separation
against a complete titration with an independently measured correlation profile, and should quote
a parametric law (`b√d`, `b·d^ν`) only as a summary of that profile, never as an assumption that
the data were not allowed to contradict.
-/
import Mathlib
import RequestProject.SwellingExponent

set_option autoImplicit false

namespace IDR
namespace DistanceProfile

open Finset

/-! ## 1. The titration curve of an arbitrary internal-distance profile -/

/-- The titration curve of a region whose mean internal distance at sequence separation `d` is
`a d` and whose charge correlation at that separation is `c d`, read over separations
`m ≤ d < N`.  No chain model is assumed: `a` is an arbitrary profile. -/
noncomputable def kCurve (m N : ℕ) (a c : ℕ → ℝ) (kappa : ℝ) : ℝ :=
  ∑ d ∈ Ico m N, c d * (Real.exp (-(kappa * a d)) / a d)

/-- The polymer law of Part CXXXIX is the special case `a d = b·d^ν`. -/
lemma kCurve_eq_swellCurve (N : ℕ) (b nu kappa : ℝ) (c : ℕ → ℝ) :
    kCurve 1 N (fun d => b * (d : ℝ) ^ nu) c kappa
      = SwellingExponent.swellCurve N b nu kappa c := by
  rw [kCurve, SwellingExponent.swellCurve, SwellingExponent.swellTail]
  exact Finset.sum_congr rfl fun d _ => by rw [SwellingExponent.skern]; ring

/-- Peeling off the shortest separation. -/
lemma kCurve_split {m N : ℕ} (hmN : m < N) (a c : ℕ → ℝ) (kappa : ℝ) :
    kCurve m N a c kappa
      = c m * (Real.exp (-(kappa * a m)) / a m) + kCurve (m + 1) N a c kappa := by
  rw [kCurve, kCurve]
  exact Finset.sum_eq_sum_Ico_succ_bot hmN _

/-! ## 2. The slowest separation is determined first -/

/-- The screening rates of two competing distance profiles, indexed so that separation `d` of the
first has index `d` and separation `d` of the second has index `N + d`. -/
noncomputable def gRate (N : ℕ) (a a' : ℕ → ℝ) : ℕ → ℝ :=
  fun j => if j < N then a j else a' (j - N)

/-- The amplitudes of the difference of the two models, in the same indexing. -/
noncomputable def gAmp (N : ℕ) (a a' c c' : ℕ → ℝ) : ℕ → ℝ :=
  fun j => if j < N then c j / a j else -(c' (j - N) / a' (j - N))

/-- The difference of the titration curves of two distance profiles is an exponential sum in the
ionic strength. -/
theorem kCurve_sub_eq_expSum (m N : ℕ) (a a' c c' : ℕ → ℝ) (kappa : ℝ) :
    kCurve m N a c kappa - kCurve m N a' c' kappa
      = DistanceLaw.expSum ((Ico m N) ∪ (Ico m N).image (fun d => N + d))
          (gAmp N a a' c c') (gRate N a a') kappa := by
  classical
  have hdisj : Disjoint (Ico m N) ((Ico m N).image (fun d => N + d)) := by
    rw [Finset.disjoint_left]
    intro x hx hx'
    rw [Finset.mem_Ico] at hx
    simp only [Finset.mem_image, Finset.mem_Ico] at hx'
    obtain ⟨e, he, rfl⟩ := hx'
    omega
  have e1 : ∑ d ∈ Ico m N,
      gAmp N a a' c c' d * Real.exp (-(kappa * gRate N a a' d)) = kCurve m N a c kappa := by
    rw [kCurve]
    refine Finset.sum_congr rfl fun d hd => ?_
    have h1 : d < N := (Finset.mem_Ico.1 hd).2
    simp only [gAmp, gRate, if_pos h1]
    ring
  have e2 : ∑ d ∈ Ico m N,
      gAmp N a a' c c' (N + d) * Real.exp (-(kappa * gRate N a a' (N + d)))
        = -kCurve m N a' c' kappa := by
    rw [kCurve, ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun d _ => ?_
    have h2 : ¬ (N + d < N) := by omega
    simp only [gAmp, gRate, if_neg h2, Nat.add_sub_cancel_left]
    ring
  rw [DistanceLaw.expSum, Finset.sum_union hdisj,
    Finset.sum_image (by intro x _ y _ h; dsimp only at h; omega), e1, e2]
  ring

/-- **The slowest rate in play cannot be cancelled.**  If separation `m` of the first profile is
strictly closer than every other separation in play — its own longer separations and all
separations of the competing profile — then agreement of the curves at every ionic strength forces
its charge correlation to vanish. -/
theorem kCurve_min_rate {m N : ℕ} (hmN : m < N) {a a' c c' : ℕ → ℝ} {kappa0 : ℝ}
    (ham : 0 < a m)
    (hmin : ∀ d, m ≤ d → d < N → d ≠ m → a m < a d)
    (hmin' : ∀ d, m ≤ d → d < N → a m < a' d)
    (h : ∀ kappa, kappa0 ≤ kappa → kCurve m N a c kappa = kCurve m N a' c' kappa) :
    c m = 0 := by
  classical
  set S : Finset ℕ := (Ico m N) ∪ (Ico m N).image (fun d => N + d) with hS
  have hzero : ∀ kappa, kappa0 ≤ kappa →
      DistanceLaw.expSum S (gAmp N a a' c c') (gRate N a a') kappa = 0 := by
    intro k hk
    rw [hS, ← kCurve_sub_eq_expSum, h k hk, sub_self]
  have hmS : m ∈ S := Finset.mem_union_left _ (Finset.mem_Ico.2 ⟨le_refl m, hmN⟩)
  have hratem : gRate N a a' m = a m := by simp [gRate, hmN]
  have hmin_all : ∀ j ∈ S, j ≠ m → gRate N a a' m < gRate N a a' j := by
    intro j hj hne
    rw [hratem]
    rcases Finset.mem_union.1 hj with hj1 | hj2
    · rw [Finset.mem_Ico] at hj1
      have hval : gRate N a a' j = a j := by simp [gRate, hj1.2]
      rw [hval]
      exact hmin j hj1.1 hj1.2 hne
    · simp only [Finset.mem_image, Finset.mem_Ico] at hj2
      obtain ⟨e, he, rfl⟩ := hj2
      have hlt2 : ¬ (N + e < N) := by omega
      have hval : gRate N a a' (N + e) = a' e := by
        simp [gRate, hlt2]
      rw [hval]
      exact hmin' e he.1 he.2
  have hamp := DistanceLaw.expSum_min_rate_zero hmS hmin_all hzero
  have hampval : gAmp N a a' c c' m = c m / a m := by simp [gAmp, hmN]
  rw [hampval, div_eq_zero_iff] at hamp
  rcases hamp with h1 | h2
  · exact h1
  · exact absurd h2 (ne_of_gt ham)

/-! ## 3. Non-parametric identification of the distance profile -/

section Identification

variable {N : ℕ} {a a' c : ℕ → ℝ} {kappa0 : ℝ}

/-- The induction that drives the identification: knowing the two profiles agree below separation
`m`, their curves read from separation `m` agree, and the separation-`m` distances agree too. -/
theorem rate_prefix
    (hpos : ∀ d, 1 ≤ d → d < N → 0 < a d)
    (hpos' : ∀ d, 1 ≤ d → d < N → 0 < a' d)
    (hmono : ∀ d e, 1 ≤ d → d < e → e < N → a d < a e)
    (hmono' : ∀ d e, 1 ≤ d → d < e → e < N → a' d < a' e)
    (hc : ∀ d, 1 ≤ d → d < N → c d ≠ 0)
    (h : ∀ kappa, kappa0 ≤ kappa → kCurve 1 N a c kappa = kCurve 1 N a' c kappa) :
    ∀ m, 1 ≤ m → m ≤ N →
      (∀ d, 1 ≤ d → d < m → a d = a' d) ∧
      (∀ kappa, kappa0 ≤ kappa → kCurve m N a c kappa = kCurve m N a' c kappa) := by
  intro m
  induction m with
  | zero => omega
  | succ n ih =>
    intro _ hsucc
    rcases Nat.eq_or_lt_of_le (Nat.one_le_iff_ne_zero.2 (Nat.succ_ne_zero n)) with hn1 | hn1
    · -- `n + 1 = 1`
      have hn : n = 0 := by omega
      subst hn
      refine ⟨fun d hd1 hd2 => by omega, h⟩
    · -- `n ≥ 1`
      have hn1' : 1 ≤ n := by omega
      have hnN : n < N := by omega
      obtain ⟨hpre, htail⟩ := ih hn1' (by omega)
      -- the two separation-`n` distances agree
      have hkey : a n = a' n := by
        rcases lt_trichotomy (a n) (a' n) with hlt | heq | hgt
        · exfalso
          refine hc n hn1' hnN (kCurve_min_rate hnN (hpos n hn1' hnN) ?_ ?_ htail)
          · intro d hd hdN hne
            exact hmono n d hn1' (by omega) hdN
          · intro d hd hdN
            rcases eq_or_lt_of_le hd with rfl | hlt2
            · exact hlt
            · exact lt_trans hlt (hmono' n d hn1' hlt2 hdN)
        · exact heq
        · exfalso
          refine hc n hn1' hnN
            (kCurve_min_rate hnN (hpos' n hn1' hnN) ?_ ?_ (fun k hk => (htail k hk).symm))
          · intro d hd hdN hne
            exact hmono' n d hn1' (by omega) hdN
          · intro d hd hdN
            rcases eq_or_lt_of_le hd with rfl | hlt2
            · exact hgt
            · exact lt_trans hgt (hmono n d hn1' hlt2 hdN)
      refine ⟨fun d hd1 hd2 => ?_, fun k hk => ?_⟩
      · rcases Nat.lt_or_ge d n with hlt | hge
        · exact hpre d hd1 hlt
        · have : d = n := by omega
          subst this
          exact hkey
      · have hsplit := htail k hk
        rw [kCurve_split hnN a c k, kCurve_split hnN a' c k, hkey] at hsplit
        linarith

/-- **The internal-distance profile is identifiable, with no chain model assumed.**  If the charge
correlation profile is known independently and vanishes at no separation, and both candidate
distance profiles are positive and strictly increasing in sequence separation, then agreement of
the titration curves at every ionic strength forces the profiles to agree at every separation.
The distance law is data, not an assumption. -/
theorem internal_distance_profile_identifiable
    (hpos : ∀ d, 1 ≤ d → d < N → 0 < a d)
    (hpos' : ∀ d, 1 ≤ d → d < N → 0 < a' d)
    (hmono : ∀ d e, 1 ≤ d → d < e → e < N → a d < a e)
    (hmono' : ∀ d e, 1 ≤ d → d < e → e < N → a' d < a' e)
    (hc : ∀ d, 1 ≤ d → d < N → c d ≠ 0)
    (h : ∀ kappa, kappa0 ≤ kappa → kCurve 1 N a c kappa = kCurve 1 N a' c kappa) :
    ∀ d, 1 ≤ d → d < N → a d = a' d := by
  intro d hd1 hdN
  exact (rate_prefix hpos hpos' hmono hmono' hc h N (by omega) (le_refl N)).1 d hd1 hdN

end Identification

/-! ## 4. What monotonicity is doing: equal correlations hide a swap -/

/-- The distance profile obtained by exchanging the distances assigned to separations `i` and
`j`. -/
noncomputable def swapRate (a : ℕ → ℝ) (i j : ℕ) : ℕ → ℝ :=
  fun d => if d = i then a j else if d = j then a i else a d

/-- **Separations carrying equal correlations can have their distances exchanged.**  If two
separations carry the same charge correlation, the titration curve at every ionic strength is
unchanged by swapping the distances assigned to them.  Monotonicity of the distance profile is
therefore not a technical hypothesis of
`internal_distance_profile_identifiable`: it is what removes this degeneracy. -/
theorem swap_rates_confound {N : ℕ} (a c : ℕ → ℝ) {i j : ℕ} (hi : i ∈ Ico 1 N) (hj : j ∈ Ico 1 N)
    (hij : i ≠ j) (hc : c i = c j) (kappa : ℝ) :
    kCurve 1 N a c kappa = kCurve 1 N (swapRate a i j) c kappa := by
  classical
  set f : ℕ → ℝ := fun d => c d * (Real.exp (-(kappa * a d)) / a d) with hf
  set g : ℕ → ℝ := fun d => c d * (Real.exp (-(kappa * swapRate a i j d)) / swapRate a i j d)
    with hg
  have hsub : ({i, j} : Finset ℕ) ⊆ Ico 1 N := by
    intro x hx
    rcases Finset.mem_insert.1 hx with rfl | hx'
    · exact hi
    · rw [Finset.mem_singleton] at hx'
      subst hx'
      exact hj
  have hoff : ∀ x ∈ Ico 1 N, x ∉ ({i, j} : Finset ℕ) → f x - g x = 0 := by
    intro x _ hx
    have hxi : x ≠ i := by
      intro hcon; exact hx (by simp [hcon])
    have hxj : x ≠ j := by
      intro hcon; exact hx (by simp [hcon])
    simp [hf, hg, swapRate, hxi, hxj]
  have hpair : ∑ x ∈ ({i, j} : Finset ℕ), (f x - g x) = 0 := by
    rw [Finset.sum_pair hij]
    have hgi : g i = c i * (Real.exp (-(kappa * a j)) / a j) := by
      simp [hg, swapRate]
    have hgj : g j = c j * (Real.exp (-(kappa * a i)) / a i) := by
      simp [hg, swapRate, Ne.symm hij]
    rw [hgi, hgj]
    simp only [hf]
    rw [hc]
    ring
  have hsum : ∑ x ∈ Ico 1 N, (f x - g x) = 0 := by
    rw [← Finset.sum_subset hsub hoff, hpair]
  have : kCurve 1 N a c kappa - kCurve 1 N (swapRate a i j) c kappa = 0 := by
    rw [kCurve, kCurve, ← Finset.sum_sub_distrib]
    exact hsum
  linarith

/-! ## 5. The parametric laws are corollaries -/

/-- **Matching rates at the two shortest separations pins the polymer law.**  If two polymer
distance laws `b·d^ν` and `b'·d^{ν'}` agree at separations `1` and `2`, they have the same bond
length and the same swelling exponent.  With
`internal_distance_profile_identifiable` this recovers Part CXXXIX's identification theorem
without any assumption that the region obeys a scaling law at all. -/
theorem polymer_params_of_rates {b b' nu nu' : ℝ} (hb : 0 < b)
    (h1 : b * ((1 : ℕ) : ℝ) ^ nu = b' * ((1 : ℕ) : ℝ) ^ nu')
    (h2 : b * ((2 : ℕ) : ℝ) ^ nu = b' * ((2 : ℕ) : ℝ) ^ nu') :
    b = b' ∧ nu = nu' := by
  have hone : ((1 : ℕ) : ℝ) ^ nu = 1 := by simp
  have hone' : ((1 : ℕ) : ℝ) ^ nu' = 1 := by simp
  have hbb : b = b' := by
    rw [hone, hone', mul_one, mul_one] at h1
    exact h1
  subst hbb
  refine ⟨rfl, ?_⟩
  have hcast : ((2 : ℕ) : ℝ) = (2 : ℝ) := by norm_num
  rw [hcast] at h2
  have h2' : (2 : ℝ) ^ nu = (2 : ℝ) ^ nu' := mul_left_cancel₀ (ne_of_gt hb) h2
  rcases lt_trichotomy nu nu' with hlt | heq | hgt
  · exact absurd h2' (ne_of_lt ((Real.rpow_lt_rpow_left_iff (by norm_num)).2 hlt))
  · exact heq
  · exact absurd h2'.symm (ne_of_lt ((Real.rpow_lt_rpow_left_iff (by norm_num)).2 hgt))

/-- **The distance-profile law.**  With an independently measured, nowhere-vanishing correlation
profile: a complete titration determines the internal distance of every sequence separation, for
arbitrary positive strictly increasing distance profiles; the monotonicity requirement is sharp,
since separations with equal correlations may exchange their distances undetectably; and a
parametric law, when one is quoted, is pinned by the two shortest separations. -/
theorem distance_profile_law {N : ℕ} {c : ℕ → ℝ} {kappa0 : ℝ}
    (hc : ∀ d, 1 ≤ d → d < N → c d ≠ 0) :
    (∀ a a' : ℕ → ℝ,
        (∀ d, 1 ≤ d → d < N → 0 < a d) → (∀ d, 1 ≤ d → d < N → 0 < a' d) →
        (∀ d e, 1 ≤ d → d < e → e < N → a d < a e) →
        (∀ d e, 1 ≤ d → d < e → e < N → a' d < a' e) →
        (∀ kappa, kappa0 ≤ kappa → kCurve 1 N a c kappa = kCurve 1 N a' c kappa) →
        ∀ d, 1 ≤ d → d < N → a d = a' d) ∧
    (∀ (a : ℕ → ℝ) (i j : ℕ), i ∈ Ico 1 N → j ∈ Ico 1 N → i ≠ j → c i = c j →
        ∀ kappa, kCurve 1 N a c kappa = kCurve 1 N (swapRate a i j) c kappa) ∧
    (∀ b b' nu nu' : ℝ, 0 < b →
        b * ((1 : ℕ) : ℝ) ^ nu = b' * ((1 : ℕ) : ℝ) ^ nu' →
        b * ((2 : ℕ) : ℝ) ^ nu = b' * ((2 : ℕ) : ℝ) ^ nu' → b = b' ∧ nu = nu') :=
  ⟨fun _ _ hpos hpos' hmono hmono' h =>
      internal_distance_profile_identifiable hpos hpos' hmono hmono' hc h,
   fun a _ _ hi hj hij hcij kappa => swap_rates_confound a c hi hj hij hcij kappa,
   fun _ _ _ _ hb h1 h2 => polymer_params_of_rates hb h1 h2⟩

end DistanceProfile
end IDR
