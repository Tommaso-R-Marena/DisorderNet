/-
# Part XII  Excluded volume: the design laws of a chain that cannot overlap

Part IX described excluded volume through Flory's mean-field free energy; Part XII.1
(`RequestProject.SelfAvoiding`) describes it exactly, by counting the conformations a chain
that really cannot overlap actually has.  This file joins that count to the capacity theory of
Part III and bundles the consequences.

* `sawEns` is the athermal ensemble of a self-avoiding chain of `n` bonds: the uniform
  distribution on its `cnt n` conformations.
* `saw_exact_capacity` : a model that is *exactly* right about a self-avoiding chain carries
  at least `cnt n ≥ 2 ^ n` components.
* `saw_capacity_lower_bound` : accuracy `eps` in the population metric of Part III still costs
  `2 ^ n (1 - eps)` components.  Excluded volume, which is often invoked as the reason a
  disordered region is "less floppy than it looks", does not reduce the capacity requirement
  below exponential.
* `excluded_volume_design_laws` : the five clauses -- entropy per residue exists and lies
  strictly between `log 2` and `log 4`; capacity is exponential; an ideal-chain generator
  misses the support by an exponentially large factor; conformational growth can dead-end;
  and self-avoidance has unbounded memory, so no bounded context window suffices.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Statistics
import RequestProject.ModelNature
import RequestProject.Metric
import RequestProject.FreeEnergy
import RequestProject.SelfAvoiding
import RequestProject.CubicLattice

namespace IDR

open scoped Classical

namespace SAW

section Generic

variable {V : Type*} [AddCommGroup V] [DecidableEq V] {q : ℕ}

/-- An enumeration of the self-avoiding conformations of a chain of `n` bonds on the lattice
`dir`. -/
noncomputable def enumOf (dir : Fin q → V) (n : ℕ) : Fin (cntOf dir n) → (Fin n → Fin q) :=
  fun j => ((sawFinsetOf dir n).equivFin.symm j : Fin n → Fin q)

lemma enumOf_injective (dir : Fin q → V) (n : ℕ) : Function.Injective (enumOf dir n) := by
  intro a b hab
  have : (sawFinsetOf dir n).equivFin.symm a = (sawFinsetOf dir n).equivFin.symm b :=
    Subtype.ext hab
  simpa using this

lemma enumOf_isSAW (dir : Fin q → V) {n : ℕ} (j : Fin (cntOf dir n)) :
    IsSAW (stepsOfDir dir (enumOf dir n j)) := by
  have : ((sawFinsetOf dir n).equivFin.symm j : Fin n → Fin q) ∈ sawFinsetOf dir n :=
    ((sawFinsetOf dir n).equivFin.symm j).2
  simpa [enumOf] using (mem_sawFinsetOf dir _).1 this

/-- **The athermal self-avoiding chain**: the uniform ensemble on the conformations of an
`n`-bond chain that cannot overlap itself. -/
noncomputable def sawEnsOf (dir : Fin q → V) (hdir : ∀ n, 0 < cntOf dir n) (n : ℕ) :
    Ens (Fin n → Fin q) := unif (hdir n) (enumOf dir n)

lemma sawEnsOf_prob {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {n : ℕ}
    (j : Fin (cntOf dir n)) : (sawEnsOf dir hdir n).prob (enumOf dir n j) = 1 / cntOf dir n :=
  unif_prob_eq (hdir n) (enumOf_injective dir n) j

/-- **Exact correctness costs one component per conformation.** -/
theorem exact_capacity_of {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {n : ℕ}
    {M : Ens (Fin n → Fin q)} (h : M.Same (sawEnsOf dir hdir n)) : cntOf dir n ≤ M.card := by
  refine Ens.card_le_of_same h (enumOf dir n) (enumOf_injective dir n) fun l => ?_
  rw [sawEnsOf_prob hdir l]
  have : (0 : ℝ) < (cntOf dir n : ℝ) := by exact_mod_cast hdir n
  positivity

/-- **Accuracy `eps` costs `cnt n (1 - eps)` components.** -/
theorem capacity_lower_bound_of {dir : Fin q → V} (hdir : ∀ n, 0 < cntOf dir n) {n k : ℕ}
    {M : Ens (Fin n → Fin q)} (hM : M.card ≤ k) {eps : ℝ}
    (h : ApproxSame eps M (sawEnsOf dir hdir n)) :
    (cntOf dir n : ℝ) * (1 - eps) ≤ k := by
  have hcpos : (0 : ℝ) < (cntOf dir n : ℝ) := by exact_mod_cast hdir n
  have hbound : (cntOf dir n : ℝ) - eps / (1 / (cntOf dir n : ℝ)) ≤ k :=
    Ens.capacity_lower_bound hM (enumOf_injective dir n) (by positivity)
      (fun l => le_of_eq (sawEnsOf_prob hdir l).symm) h
  have hdiv : eps / (1 / (cntOf dir n : ℝ)) = eps * cntOf dir n := by field_simp
  rw [hdiv] at hbound
  nlinarith [hbound]

end Generic

/-- The athermal ensemble of the two-dimensional self-avoiding chain. -/
noncomputable def sawEns (n : ℕ) : Ens (Fin n → Fin 4) := sawEnsOf dir cnt_pos n

lemma sawEns_prob {n : ℕ} (j : Fin (cnt n)) :
    (sawEns n).prob (enumOf dir n j) = 1 / cnt n :=
  sawEnsOf_prob cnt_pos j

/-- **Exact correctness costs `cnt n ≥ 2 ^ n` components.** -/
theorem saw_exact_capacity {n : ℕ} {M : Ens (Fin n → Fin 4)} (h : M.Same (sawEns n)) :
    2 ^ n ≤ M.card :=
  le_trans (two_pow_le_cnt n) (exact_capacity_of cnt_pos h)

/-- **Excluded volume does not lower the capacity requirement.**  A model of at most `k`
components that reproduces the self-avoiding chain to accuracy `eps` in the population metric
must have `k ≥ 2 ^ n (1 - eps)`: exponential in the length of the region, exactly as for the
ideal chain of Part IV. -/
theorem saw_capacity_lower_bound {n k : ℕ} {M : Ens (Fin n → Fin 4)} (hM : M.card ≤ k)
    {eps : ℝ} (h : ApproxSame eps M (sawEns n)) :
    (2 : ℝ) ^ n * (1 - eps) ≤ k := by
  have hfac : (cnt n : ℝ) * (1 - eps) ≤ k := capacity_lower_bound_of cnt_pos hM h
  rcases le_or_gt eps 1 with heps | heps
  · have h2 : (2 : ℝ) ^ n ≤ (cnt n : ℝ) := by exact_mod_cast two_pow_le_cnt n
    nlinarith [h2, hfac, sub_nonneg.2 heps]
  · have hneg : (2 : ℝ) ^ n * (1 - eps) ≤ 0 := by
      have h1 : (1 : ℝ) - eps < 0 := by linarith
      nlinarith [pow_pos (by norm_num : (0:ℝ) < 2) n]
    linarith [hneg, Nat.cast_nonneg (α := ℝ) k]

/-- The athermal ensemble of the three-dimensional self-avoiding chain. -/
noncomputable def sawEns3 (n : ℕ) : Ens (Fin n → Fin 6) := sawEnsOf dir3 cnt3_pos n

/-- **The capacity law in three dimensions**, where a polypeptide actually lives: accuracy
`eps` costs `3 ^ n (1 - eps)` components. -/
theorem saw3_capacity_lower_bound {n k : ℕ} {M : Ens (Fin n → Fin 6)} (hM : M.card ≤ k)
    {eps : ℝ} (h : ApproxSame eps M (sawEns3 n)) :
    (3 : ℝ) ^ n * (1 - eps) ≤ k := by
  have hfac : (cnt3 n : ℝ) * (1 - eps) ≤ k := capacity_lower_bound_of cnt3_pos hM h
  rcases le_or_gt eps 1 with heps | heps
  · have h2 : (3 : ℝ) ^ n ≤ (cnt3 n : ℝ) := by exact_mod_cast three_pow_le_cnt3 n
    nlinarith [h2, hfac, sub_nonneg.2 heps]
  · have hneg : (3 : ℝ) ^ n * (1 - eps) ≤ 0 := by
      have h1 : (1 : ℝ) - eps < 0 := by linarith
      nlinarith [pow_pos (by norm_num : (0:ℝ) < 3) n]
    linarith [hneg, Nat.cast_nonneg (α := ℝ) k]

/-- **Excluded volume does not remove the extensive folding gap.**  Index the self-avoiding
conformations of an `n`-bond chain by `Fin (cnt n)` and give them energies `U`.  If one
conformation is to carry half the Boltzmann population while all its competitors have energy
at most `Uu`, the force field must supply a gap of at least `(1/beta)·log (cnt n - 1)`, and
`cnt n ≥ 2 ^ n`: an energy linear in the length of the region, exactly as for the ideal chain
of Part IV. -/
theorem saw_folding_gap {n : ℕ} (hn : 0 < n) {beta : ℝ} (hbeta : 0 < beta)
    (U : Fin (cnt n) → ℝ) (j0 : Fin (cnt n)) (Uu : ℝ)
    (hU : ∀ j, j ≠ j0 → U j ≤ Uu) (hhalf : 1 / 2 ≤ FreeEnergy.boltz beta U j0) :
    Real.log ((cnt n - 1 : ℕ) : ℝ) / beta ≤ Uu - U j0 := by
  have hpos : 0 < cnt n := cnt_pos n
  have hcard2 : 2 ≤ cnt n := by
    have h1 : 2 ^ 1 ≤ 2 ^ n := Nat.pow_le_pow_right (by norm_num) hn
    have h2 := two_pow_le_cnt n
    omega
  set D : Finset (Fin (cnt n)) := Finset.univ.erase j0 with hD
  have hDcard : D.card = cnt n - 1 := by
    rw [hD, Finset.card_erase_of_mem (Finset.mem_univ j0)]
    simp
  have hDne : D.Nonempty := by
    rw [← Finset.card_pos, hDcard]
    omega
  have hj0 : j0 ∉ D := by simp [hD]
  have h := FreeEnergy.folded_needs_entropic_gap hbeta hpos U j0 D hDne hj0 Uu
    (fun j hj => hU j (Finset.ne_of_mem_erase hj)) hhalf
  rwa [hDcard] at h

end SAW

/-- **The design laws of excluded volume.**

1. The conformational entropy per residue of a self-avoiding chain exists (Fekete's lemma
   applied to the exact submultiplicativity of the conformation count).
2. It lies strictly between the ideal-chain value `log 4` and `log 2`: self-avoidance costs a
   fixed amount of entropy per residue but leaves exponentially many conformations.
3. Accordingly the capacity of a model accurate to `eps` still grows exponentially with the
   length of the region.
4. An ideal-chain generator is wrong about the support: the self-avoiding conformations are an
   exponentially vanishing fraction of the freely jointed ones.
5. Conformational growth can dead-end (`exists_trapped_walk`), and self-avoidance is not a
   finite-context property (`unbounded_memory`), so a generator that appends residues using a
   bounded window of previous steps, without rejection, cannot be correct.
6. Only an energy gap growing linearly with the length of the region could make one
   conformation dominant, so excluded volume alone never turns a disordered region into a
   folded one.
7. And none of this is an artefact of two dimensions: on the cubic lattice the entropy per
   residue lies strictly between `log 3` and `log 6`, the capacity law is again exponential,
   and the ideal chain again misses the support. -/
theorem excluded_volume_design_laws :
    -- 1. the entropy per residue exists
    Filter.Tendsto (fun n => SAW.logCnt n / n) Filter.atTop (nhds SAW.connectiveConstant) ∧
    -- 2. and is strictly between log 2 and log 4
    (Real.log 2 ≤ SAW.connectiveConstant ∧ SAW.connectiveConstant < Real.log 4) ∧
    -- 3. capacity is exponential in the length of the region
    (∀ (n k : ℕ) (M : Ens (Fin n → Fin 4)), M.card ≤ k → ∀ eps : ℝ,
        ApproxSame eps M (SAW.sawEns n) → (2 : ℝ) ^ n * (1 - eps) ≤ k) ∧
    -- 4. an ideal-chain generator misses the support
    Filter.Tendsto (fun m : ℕ => (SAW.cnt (4 * m) : ℝ) / 4 ^ (4 * m)) Filter.atTop (nhds 0) ∧
    -- 5. growth dead-ends, and no bounded context window decides self-avoidance
    ((SAW.IsSAW SAW.trappedWalk ∧ ∀ i : Fin 4, ¬ SAW.IsSAW (SAW.trappedWalk ++ [SAW.dir i])) ∧
      ∀ k : ℕ, ∃ l₁ l₂ : List SAW.Site, SAW.IsSAW l₁ ∧ SAW.IsSAW l₂ ∧
        l₁.drop (l₁.length - k) = l₂.drop (l₂.length - k) ∧
        SAW.IsSAW (l₁ ++ [SAW.southStep]) ∧ ¬ SAW.IsSAW (l₂ ++ [SAW.southStep])) ∧
    -- 6. and only an extensive energy gap could make one conformation dominant
    (∀ (n : ℕ), 0 < n → ∀ beta : ℝ, 0 < beta → ∀ (U : Fin (SAW.cnt n) → ℝ)
        (j0 : Fin (SAW.cnt n)) (Uu : ℝ), (∀ j, j ≠ j0 → U j ≤ Uu) →
        1 / 2 ≤ FreeEnergy.boltz beta U j0 →
        Real.log ((SAW.cnt n - 1 : ℕ) : ℝ) / beta ≤ Uu - U j0) ∧
    -- 7. and all of this holds in three dimensions, where a polypeptide lives
    (Filter.Tendsto (fun n => SAW.logCnt3 n / n) Filter.atTop (nhds SAW.connectiveConstant3) ∧
      (Real.log 3 ≤ SAW.connectiveConstant3 ∧ SAW.connectiveConstant3 < Real.log 6) ∧
      (∀ (n k : ℕ) (M : Ens (Fin n → Fin 6)), M.card ≤ k → ∀ eps : ℝ,
        ApproxSame eps M (SAW.sawEns3 n) → (3 : ℝ) ^ n * (1 - eps) ≤ k) ∧
      Filter.Tendsto (fun m : ℕ => (SAW.cnt3 (2 * m) : ℝ) / 6 ^ (2 * m)) Filter.atTop
        (nhds 0)) := by
  refine ⟨SAW.tendsto_connectiveConstant,
    ⟨SAW.log_two_le_connectiveConstant, SAW.connectiveConstant_lt_log_four⟩,
    fun n k M hM eps h => SAW.saw_capacity_lower_bound hM h,
    SAW.saw_fraction_tendsto_zero,
    ⟨SAW.exists_trapped_walk, SAW.unbounded_memory⟩,
    fun _ hn _ hbeta U j0 Uu hU hhalf => SAW.saw_folding_gap hn hbeta U j0 Uu hU hhalf,
    SAW.tendsto_connectiveConstant3,
    ⟨SAW.log_three_le_connectiveConstant3, SAW.connectiveConstant3_lt_log_six⟩,
    fun _ _ _ hM _ h => SAW.saw3_capacity_lower_bound hM h,
    SAW.saw3_fraction_tendsto_zero⟩

end IDR
