/-
# Epistatic spikes: when one mutation costs everything

The mutational model of `RequestProject.SequenceEntropyLimit` assumes a site-Lipschitz energy:
one substitution moves every conformational energy by at most `L`.  Real landscapes are not like
that.  Break a buried salt bridge, or push a methyl group into a packed core, and `ΔΔG` jumps by
several kcal/mol in a single step — a non-linear spike, invisible to any additive (site
independent) model, and concentrated on a handful of positions.

This file quantifies the spike and repairs the model.

* `IDR.Spike.bridge` — the salt-bridge energy: `-J` when both partners are present, `0`
  otherwise.
* `IDR.Spike.bridge_spike` — **one mutation, the whole bridge**: there is a sequence and a single
  substitution whose `ΔΔG` is exactly `J`.
* `IDR.Spike.bridge_not_siteLip` — consequently the landscape is **not** site-Lipschitz with any
  constant below `J`: the hypothesis `SiteLip L` is false on real force fields with `L` small.
* `IDR.Spike.additive_error_ge` — **no additive model can fit it**: every site-independent
  energy `c + ∑ᵢ aᵢ(sᵢ)` is off by at least `J/4` on one of the four sequences of the two-site
  cycle.  The error is a lower bound, not an artefact of fitting.
* `IDR.Spike.splitLip_wdist` — the repair: energies that are `L`-Lipschitz **off** a set `Bad` of
  spike sites and only `Smax`-bounded on `Bad` obey the weighted Hamming bound
  `|E s − E t| ≤ ∑_{mismatches} cost`, `cost = Smax` on `Bad` and `L` elsewhere.
* `IDR.Spike.ordered_needs_minority_with_spikes`,
  `IDR.Spike.disordered_of_low_entropy_with_spikes` — **the entropy floor survives spikes**, with
  an explicit correction `2·Smax·|Bad|`.  A low-complexity tract can only be rescued by epistasis
  if the number of spike sites is at least
  `(kT·log(M−1) − S₀ − 2·L·κ·N)/(2·Smax)` (`spike_sites_needed`): a single dramatic contact is
  never enough, however dramatic it is.
-/
import Mathlib
import RequestProject.FreeEnergy
import RequestProject.SequenceEntropyCore
import RequestProject.SequenceEntropyLimit

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR

namespace Spike

open Finset
open SeqLimit
open scoped Classical

variable {N q M : ℕ}

/-! ## A salt bridge is a two-body, non-additive, spiky term -/

/-- The salt-bridge energy: the contact pays `-J` when both partners are present at the two
positions, and nothing otherwise. -/
noncomputable def bridge (J : ℝ) (i0 i1 : Fin N) (a b : Fin q) (s : Seq N q) : ℝ :=
  if s i0 = a ∧ s i1 = b then -J else 0

/-- **A single mutation can cost the whole contact.**  Removing one partner of the bridge changes
the energy by exactly `J`, however small the per-site scale of the rest of the force field. -/
theorem bridge_spike {J : ℝ} {i0 i1 : Fin N} (hne : i0 ≠ i1) {a b : Fin q} (hab : a ≠ b) :
    ∃ (s : Seq N q) (i : Fin N) (c : Fin q),
      |bridge J i0 i1 a b (Function.update s i c) - bridge J i0 i1 a b s| = |J| := by
  refine ⟨Function.update (fun _ => a) i1 b, i1, a, ?_⟩
  have h0 : (Function.update (fun _ => a) i1 b : Seq N q) i0 = a := by
    simp [Function.update_of_ne hne]
  have h1 : (Function.update (fun _ => a) i1 b : Seq N q) i1 = b := by
    rw [Function.update_self]
  have hupd : Function.update (Function.update (fun _ => a) i1 b : Seq N q) i1 a
      = (fun _ => a : Seq N q) := by
    funext i
    by_cases h : i = i1
    · subst h; simp
    · simp [Function.update_of_ne h]
  rw [hupd]
  have hbase : bridge J i0 i1 a b (fun _ => a : Seq N q) = 0 := by
    unfold bridge
    rw [if_neg]
    rintro ⟨-, h⟩
    exact hab h
  have hfull : bridge J i0 i1 a b (Function.update (fun _ => a) i1 b : Seq N q) = -J := by
    unfold bridge
    rw [if_pos ⟨h0, h1⟩]
  rw [hbase, hfull]
  simp

/-- **A spiky landscape is not site-Lipschitz.**  With a bridge of depth `J`, no Lipschitz
constant below `J` is valid; the smooth mutational model of the entropy ceiling is simply false
for contact energies. -/
theorem bridge_not_siteLip {J L : ℝ} (hJ : L < J) {i0 i1 : Fin N} (hne : i0 ≠ i1) {a b : Fin q}
    (hab : a ≠ b) (hM : 0 < M) :
    ¬ SiteLip L (fun (s : Seq N q) (_ : Fin M) => bridge J i0 i1 a b s) := by
  intro hlip
  obtain ⟨s, i, c, hspec⟩ := bridge_spike (N := N) (q := q) hne hab
  have := hlip s i c ⟨0, hM⟩
  simp only at this
  rw [hspec] at this
  linarith [le_abs_self J]

/-! ## No additive model can represent a contact -/

/-- A site-independent (additive) energy model. -/
def Additive (f : Seq N q → ℝ) : Prop :=
  ∃ (c : ℝ) (a : Fin N → Fin q → ℝ), ∀ s, f s = c + ∑ i, a i (s i)

/-- The four corners of a two-site mutational cycle. -/
noncomputable def corner (s : Seq N q) (i0 i1 : Fin N) (x y : Fin q) : Seq N q :=
  Function.update (Function.update s i0 x) i1 y

lemma corner_apply_i0 {s : Seq N q} {i0 i1 : Fin N} (hne : i0 ≠ i1) (x y : Fin q) :
    corner s i0 i1 x y i0 = x := by
  unfold corner
  rw [Function.update_of_ne hne, Function.update_self]

lemma corner_apply_i1 {s : Seq N q} {i0 i1 : Fin N} (x y : Fin q) :
    corner s i0 i1 x y i1 = y := by
  unfold corner
  rw [Function.update_self]

lemma corner_apply_other {s : Seq N q} {i0 i1 i : Fin N} (h0 : i ≠ i0) (h1 : i ≠ i1)
    (x y : Fin q) : corner s i0 i1 x y i = s i := by
  unfold corner
  rw [Function.update_of_ne h1, Function.update_of_ne h0]

/-- An additive model splits into a constant plus one term per mutated site: its two-site
mutational cycle vanishes identically. -/
theorem additive_cycle_zero {f : Seq N q → ℝ} (hf : Additive f) (s : Seq N q) {i0 i1 : Fin N}
    (hne : i0 ≠ i1) (x0 x1 y0 y1 : Fin q) :
    f (corner s i0 i1 x1 y1) + f (corner s i0 i1 x0 y0)
      - f (corner s i0 i1 x1 y0) - f (corner s i0 i1 x0 y1) = 0 := by
  obtain ⟨c, a, hrep⟩ := hf
  have hsum : ∀ x y : Fin q, ∑ i, a i (corner s i0 i1 x y i)
      = a i0 x + a i1 y + ∑ i ∈ (Finset.univ.erase i0).erase i1, a i (s i) := by
    intro x y
    have h1mem : i1 ∈ Finset.univ.erase i0 := by
      simp [Finset.mem_erase, Ne.symm hne]
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ i0),
      ← Finset.add_sum_erase _ _ h1mem]
    have hrest : ∑ i ∈ (Finset.univ.erase i0).erase i1, a i (corner s i0 i1 x y i)
        = ∑ i ∈ (Finset.univ.erase i0).erase i1, a i (s i) := by
      refine Finset.sum_congr rfl fun i hi => ?_
      rw [Finset.mem_erase] at hi
      obtain ⟨hi1, hi0⟩ := hi
      rw [Finset.mem_erase] at hi0
      rw [corner_apply_other hi0.1 hi1]
    rw [hrest, corner_apply_i0 hne, corner_apply_i1]
    ring
  rw [hrep, hrep, hrep, hrep, hsum, hsum, hsum, hsum]
  ring

/-- **The cycle of a salt bridge is the depth of the bridge.** -/
theorem bridge_cycle {J : ℝ} (s : Seq N q) {i0 i1 : Fin N} (hne : i0 ≠ i1) {a b : Fin q}
    (ha : a ≠ b) :
    bridge J i0 i1 a b (corner s i0 i1 a b) + bridge J i0 i1 a b (corner s i0 i1 b a)
      - bridge J i0 i1 a b (corner s i0 i1 a a) - bridge J i0 i1 a b (corner s i0 i1 b b) = -J := by
  have e1 : bridge J i0 i1 a b (corner s i0 i1 a b) = -J := by
    unfold bridge
    rw [if_pos ⟨corner_apply_i0 hne a b, corner_apply_i1 a b⟩]
  have e2 : bridge J i0 i1 a b (corner s i0 i1 b a) = 0 := by
    unfold bridge
    rw [if_neg]
    rintro ⟨h, -⟩
    exact ha (by rw [← corner_apply_i0 hne b a, h])
  have e3 : bridge J i0 i1 a b (corner s i0 i1 a a) = 0 := by
    unfold bridge
    rw [if_neg]
    rintro ⟨-, h⟩
    exact ha (by rw [← corner_apply_i1 (s := s) (i0 := i0) (i1 := i1) a a, h])
  have e4 : bridge J i0 i1 a b (corner s i0 i1 b b) = 0 := by
    unfold bridge
    rw [if_neg]
    rintro ⟨h, -⟩
    exact ha (by rw [← corner_apply_i0 hne b b, h])
  rw [e1, e2, e3, e4]
  ring

/-- **Additive models are wrong by at least a quarter of the contact.**  Any site-independent
energy misfits one of the four sequences of the bridge cycle by `|J|/4` — the epistatic term is
not absorbable into single-site parameters, however they are fitted. -/
theorem additive_error_ge {J : ℝ} {f : Seq N q → ℝ} (hf : Additive f) (s : Seq N q)
    {i0 i1 : Fin N} (hne : i0 ≠ i1) {a b : Fin q} (hab : a ≠ b) :
    ∃ t : Seq N q, |J| / 4 ≤ |f t - bridge J i0 i1 a b t| := by
  by_contra hcon
  push_neg at hcon
  have h1 := hcon (corner s i0 i1 a b)
  have h2 := hcon (corner s i0 i1 b a)
  have h3 := hcon (corner s i0 i1 a a)
  have h4 := hcon (corner s i0 i1 b b)
  have hcycf := additive_cycle_zero hf s hne b a a b
  have hcycE := bridge_cycle (J := J) s hne hab
  have b1 := abs_lt.1 h1
  have b2 := abs_lt.1 h2
  have b3 := abs_lt.1 h3
  have b4 := abs_lt.1 h4
  rcases abs_cases J with ⟨hJv, -⟩ | ⟨hJv, -⟩ <;>
    linarith [b1.1, b1.2, b2.1, b2.2, b3.1, b3.2, b4.1, b4.2]

/-! ## Rebuilding the mutational bound with a sparse set of spike sites -/

/-- The mismatch set of two sequences. -/
noncomputable def mism (s t : Seq N q) : Finset (Fin N) :=
  Finset.univ.filter fun i => s i ≠ t i

lemma mism_card (s t : Seq N q) : (mism s t).card = hdist s t := rfl

lemma mism_update {s t : Seq N q} {i : Fin N} (hi : s i ≠ t i) :
    mism (Function.update s i (t i)) t = (mism s t).erase i := by
  ext k
  by_cases h : k = i
  · subst h; simp [mism]
  · simp [mism, h, Finset.mem_erase]

/-- The per-site mutational cost: `Smax` at a spike site, `L` elsewhere. -/
noncomputable def cost (Bad : Finset (Fin N)) (L Smax : ℝ) (i : Fin N) : ℝ :=
  if i ∈ Bad then Smax else L

/-- The weighted Hamming distance: each mismatch is charged its own site's cost. -/
noncomputable def wdist (Bad : Finset (Fin N)) (L Smax : ℝ) (s t : Seq N q) : ℝ :=
  ∑ i ∈ mism s t, cost Bad L Smax i

/-- **Energies with sparse spikes**: substitutions cost at most `L` outside `Bad` and at most
`Smax` inside it. -/
def SplitLip (Bad : Finset (Fin N)) (L Smax : ℝ) (E : Seq N q → Fin M → ℝ) : Prop :=
  ∀ (s : Seq N q) (i : Fin N) (c : Fin q) (j : Fin M),
    |E (Function.update s i c) j - E s j| ≤ cost Bad L Smax i

/-- **The weighted Lipschitz bound.**  A landscape with spikes confined to `Bad` still obeys a
Lipschitz estimate — in the weighted Hamming metric. -/
theorem splitLip_wdist {Bad : Finset (Fin N)} {L Smax : ℝ} {E : Seq N q → Fin M → ℝ}
    (hlip : SplitLip Bad L Smax E) :
    ∀ (n : ℕ) (s t : Seq N q), hdist s t = n → ∀ j, |E s j - E t j| ≤ wdist Bad L Smax s t := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro s t hn j
    rcases Nat.eq_zero_or_pos n with rfl | hpos
    · have hst : s = t := eq_of_hdist_eq_zero hn
      subst hst
      have : mism s s = (∅ : Finset (Fin N)) := by
        ext k; simp [mism]
      simp [wdist, this]
    · have hdpos : 0 < hdist s t := by omega
      obtain ⟨i, hi⟩ : ∃ i, s i ≠ t i := by
        obtain ⟨i, hmem⟩ := Finset.card_pos.1 hdpos
        exact ⟨i, by simpa [mism] using hmem⟩
      set s' : Seq N q := Function.update s i (t i) with hs'
      have hstep : hdist s' t + 1 = hdist s t := hdist_update_succ hi
      have hm : hdist s' t = n - 1 := by omega
      have h1 : |E s j - E s' j| ≤ cost Bad L Smax i := by
        rw [abs_sub_comm]
        exact hlip s i (t i) j
      have h2 : |E s' j - E t j| ≤ wdist Bad L Smax s' t := ih (n - 1) (by omega) s' t hm j
      have hsplit : wdist Bad L Smax s t = cost Bad L Smax i + wdist Bad L Smax s' t := by
        unfold wdist
        rw [hs', mism_update hi, Finset.add_sum_erase _ _ ?_]
        exact Finset.mem_filter.2 ⟨Finset.mem_univ i, hi⟩
      have habs : |E s j - E t j| ≤ |E s j - E s' j| + |E s' j - E t j| := abs_sub_le _ _ _
      rw [hsplit]
      linarith

/-- The weighted distance is bounded by "smooth cost times all mismatches plus spike cost times
the number of spike sites". -/
theorem wdist_le {Bad : Finset (Fin N)} {L Smax : ℝ} (hL : 0 ≤ L) (hS : 0 ≤ Smax)
    (s t : Seq N q) :
    wdist Bad L Smax s t ≤ L * (hdist s t : ℝ) + Smax * (Bad.card : ℝ) := by
  have hsplit : wdist Bad L Smax s t
      = (∑ i ∈ (mism s t).filter (fun i => i ∈ Bad), cost Bad L Smax i)
        + ∑ i ∈ (mism s t).filter (fun i => i ∉ Bad), cost Bad L Smax i := by
    unfold wdist
    rw [← Finset.sum_filter_add_sum_filter_not (mism s t) (fun i => i ∈ Bad)]
  have hbad : (∑ i ∈ (mism s t).filter (fun i => i ∈ Bad), cost Bad L Smax i)
      ≤ Smax * (Bad.card : ℝ) := by
    have hsub : (mism s t).filter (fun i => i ∈ Bad) ⊆ Bad := by
      intro i hi
      exact (Finset.mem_filter.1 hi).2
    have hval : ∀ i ∈ (mism s t).filter (fun i => i ∈ Bad), cost Bad L Smax i = Smax := by
      intro i hi
      unfold cost
      rw [if_pos (Finset.mem_filter.1 hi).2]
    rw [Finset.sum_congr rfl hval, Finset.sum_const, nsmul_eq_mul]
    have hcard : (((mism s t).filter (fun i => i ∈ Bad)).card : ℝ) ≤ (Bad.card : ℝ) := by
      exact_mod_cast Finset.card_le_card hsub
    calc (((mism s t).filter (fun i => i ∈ Bad)).card : ℝ) * Smax
        ≤ (Bad.card : ℝ) * Smax := by exact mul_le_mul_of_nonneg_right hcard hS
      _ = Smax * (Bad.card : ℝ) := by ring
  have hgood : (∑ i ∈ (mism s t).filter (fun i => i ∉ Bad), cost Bad L Smax i)
      ≤ L * (hdist s t : ℝ) := by
    have hval : ∀ i ∈ (mism s t).filter (fun i => i ∉ Bad), cost Bad L Smax i = L := by
      intro i hi
      unfold cost
      rw [if_neg (Finset.mem_filter.1 hi).2]
    rw [Finset.sum_congr rfl hval, Finset.sum_const, nsmul_eq_mul]
    have hcard : ((((mism s t).filter (fun i => i ∉ Bad)).card : ℝ)) ≤ (hdist s t : ℝ) := by
      have := Finset.card_le_card (Finset.filter_subset (fun i => i ∉ Bad) (mism s t))
      rw [mism_card] at this
      exact_mod_cast this
    calc ((((mism s t).filter (fun i => i ∉ Bad)).card : ℝ)) * L
        ≤ (hdist s t : ℝ) * L := mul_le_mul_of_nonneg_right hcard hL
      _ = L * (hdist s t : ℝ) := by ring
  rw [hsplit]
  linarith

/-- **Order requires heterogeneity, spikes included.**  With smooth constant `L`, spike sites
`Bad` of individual size at most `Smax`, and homopolymer spread at most `S₀`, an ordered sequence
still needs a minority count of at least `(kT·log(M−1) − S₀ − 2·Smax·|Bad|)/(2L)`. -/
theorem ordered_needs_minority_with_spikes {beta L Smax S0 : ℝ} (hbeta : 0 < beta) (hM : 1 < M)
    (hL : 0 ≤ L) (hS : 0 ≤ Smax) {Bad : Finset (Fin N)} {E : Seq N q → Fin M → ℝ}
    (hlip : SplitLip Bad L Smax E) (c : Fin q)
    (hflat : ∀ j j' : Fin M, E (fun _ => c) j - E (fun _ => c) j' ≤ S0)
    {s : Seq N q} (h : Ordered beta E s) :
    Real.log ((M : ℝ) - 1) / beta
      ≤ S0 + 2 * L * (minorityCount s c : ℝ) + 2 * Smax * (Bad.card : ℝ) := by
  have hne : (Finset.univ : Finset (Fin M)).Nonempty := ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  obtain ⟨jmax, -, hjmax⟩ := Finset.exists_max_image Finset.univ (E s) hne
  obtain ⟨jmin, -, hjmin⟩ := Finset.exists_min_image Finset.univ (E s) hne
  have hspread := ordered_needs_spread hbeta hM (fun j => hjmax j (Finset.mem_univ j))
    (fun j => hjmin j (Finset.mem_univ j)) h
  have hd : hdist s (fun _ => c) = minorityCount s c := rfl
  have hbound : ∀ j, |E s j - E (fun _ => c) j|
      ≤ L * (minorityCount s c : ℝ) + Smax * (Bad.card : ℝ) := by
    intro j
    have h1 := splitLip_wdist hlip (hdist s (fun _ => c)) s (fun _ => c) rfl j
    have h2 := wdist_le (Bad := Bad) (L := L) (Smax := Smax) hL hS s (fun _ => c)
    rw [hd] at h2
    linarith
  have hA := (abs_le.1 (hbound jmax)).2
  have hB := (abs_le.1 (hbound jmin)).1
  have hC : E (fun _ => c) jmax - E (fun _ => c) jmin ≤ S0 := hflat jmax jmin
  linarith

/-- **The entropy floor survives epistasis.**  A low-complexity tract is still forced to be
disordered — the spikes only add the term `2·Smax·|Bad|` to the budget. -/
theorem disordered_of_low_entropy_with_spikes {beta L Smax S0 kappa : ℝ} (hbeta : 0 < beta)
    (hM : 1 < M) (hN : 0 < N) (hL : 0 ≤ L) (hS : 0 ≤ Smax) {Bad : Finset (Fin N)}
    {E : Seq N q → Fin M → ℝ} (hlip : SplitLip Bad L Smax E)
    (hflat : ∀ (c : Fin q) (j j' : Fin M), E (fun _ => c) j - E (fun _ => c) j' ≤ S0)
    (hk : kappa ≤ 1 / 2)
    (hthr : S0 + 2 * L * (kappa * N) + 2 * Smax * (Bad.card : ℝ)
      < Real.log ((M : ℝ) - 1) / beta)
    {s : Seq N q} (hlow : SeqEnt.H (comp s) < kappa * Real.log 2) :
    ¬ Ordered beta E s := by
  intro hord
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hnn := comp_nonneg s
  have hsum := comp_sum_one hN s
  have hqpos : 0 < q := Fin.pos_iff_nonempty.2 ⟨s ⟨0, hN⟩⟩
  obtain ⟨c, -, hc⟩ := Finset.exists_max_image Finset.univ (comp s) ⟨⟨0, hqpos⟩,
    Finset.mem_univ _⟩
  set del : ℝ := ∑ a ∈ Finset.univ.erase c, comp s a with hdel
  have hdelval : del = (minorityCount s c : ℝ) / N := by rw [hdel, sum_comp_erase hN s c]
  have hdel0 : 0 ≤ del := by rw [hdelval]; positivity
  have hmin_entropy : -Real.log (comp s c) ≤ SeqEnt.H (comp s) :=
    SeqEnt.H_ge_neg_log_max hnn hsum (fun a => hc a (Finset.mem_univ a))
  have hlog2 : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  have hHlt : SeqEnt.H (comp s) < Real.log 2 := by nlinarith
  have hcpos : 1 / 2 < comp s c := by
    by_contra hcon
    push_neg at hcon
    have hcnn : 0 ≤ comp s c := hnn c
    rcases eq_or_lt_of_le hcnn with h0 | hpos
    · have hall : ∀ a, comp s a ≤ 0 := by
        intro a
        have hle := hc a (Finset.mem_univ a)
        rw [← h0] at hle
        exact hle
      have hnp : ∑ a, comp s a ≤ 0 := Finset.sum_nonpos fun a _ => hall a
      linarith
    · have hlt : Real.log (comp s c) ≤ Real.log (1/2) := Real.log_le_log hpos hcon
      have h12 : Real.log (1/2) = -Real.log 2 := by rw [one_div, Real.log_inv]
      rw [h12] at hlt
      linarith
  have hdelhalf : del < 1 / 2 := by
    have hsplit : comp s c + del = 1 := by
      rw [hdel, Finset.add_sum_erase _ _ (Finset.mem_univ c)]
      exact hsum
    linarith
  have hminor : del * Real.log 2 ≤ SeqEnt.H (comp s) :=
    SeqEnt.H_ge_minority hnn hsum hdel hdelhalf.le
  have hdelk : del < kappa := by
    have := lt_of_le_of_lt hminor hlow
    exact lt_of_mul_lt_mul_right (by linarith) hlog2.le
  have hcount : (minorityCount s c : ℝ) < kappa * N := by
    rw [hdelval] at hdelk
    calc (minorityCount s c : ℝ) = ((minorityCount s c : ℝ) / N) * N := by field_simp
      _ < kappa * N := mul_lt_mul_of_pos_right hdelk hN0
  have hgap := ordered_needs_minority_with_spikes hbeta hM hL hS hlip c (hflat c) hord
  nlinarith [hgap, hcount, hthr, hL]

/-- **How much epistasis it takes to rescue a repeat.**  If a low-complexity sequence *is*
ordered, the number of spike sites is bounded below: one dramatic contact cannot fold a
homopolymeric tract; the landscape needs `(kT·log(M−1) − S₀ − 2·L·κ·N)/(2·Smax)` of them. -/
theorem spike_sites_needed {beta L Smax S0 kappa : ℝ} (hbeta : 0 < beta) (hM : 1 < M)
    (hN : 0 < N) (hL : 0 ≤ L) (hS : 0 < Smax) {Bad : Finset (Fin N)}
    {E : Seq N q → Fin M → ℝ} (hlip : SplitLip Bad L Smax E)
    (hflat : ∀ (c : Fin q) (j j' : Fin M), E (fun _ => c) j - E (fun _ => c) j' ≤ S0)
    (hk : kappa ≤ 1 / 2)
    {s : Seq N q} (hlow : SeqEnt.H (comp s) < kappa * Real.log 2) (hord : Ordered beta E s) :
    (Real.log ((M : ℝ) - 1) / beta - S0 - 2 * L * (kappa * N)) / (2 * Smax)
      ≤ (Bad.card : ℝ) := by
  by_contra hcon
  push_neg at hcon
  have hpos : (0 : ℝ) < 2 * Smax := by linarith
  have hthr : S0 + 2 * L * (kappa * N) + 2 * Smax * (Bad.card : ℝ)
      < Real.log ((M : ℝ) - 1) / beta := by
    rw [lt_div_iff₀ hpos] at hcon
    linarith
  exact disordered_of_low_entropy_with_spikes hbeta hM hN hL hS.le hlip hflat hk hthr hlow hord

end Spike

end IDR
