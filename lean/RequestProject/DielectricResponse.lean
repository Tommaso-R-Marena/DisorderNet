/-
# Part CXXI  The exact dielectric response of a disordered polyampholyte

Part CXX computed the mean squared dipole moment of a disordered region exactly and showed
that it is an extremal functional of the *order* of the charges.  This file turns the
observable into a thermodynamic one: it switches on a uniform electric field, couples it to
the dipole, and **solves the resulting model exactly**.

Because the dipole moment of an ideal chain is a linear form in the independent bond
variables -- `dipole_eq_linear`, with the coefficient of bond `i` equal to minus the charge
of the first `i+1` residues -- the partition function factorises completely:

    `Zdip_prod`:   Z(u) = ∏_{i<N} 2 cosh(u b · (prefix charge through residue i)).

That is a closed-form, sequence-resolved free energy for a charged disordered region in a
field, with no approximation of any kind: the conformational sum has been performed.  Its
logarithm, `logZ_eq`, is a sum of `log cosh` terms, one per bond, and the transfer of
sequence information into thermodynamics is complete and exact.

Two elementary bounds on `log (2 cosh)` then sandwich the field free energy between
`|u b| ∑_k |prefix charge|` and that plus `N log 2` (`logZ_lower`, `logZ_upper`), so the
free energy gained from the field is, up to a bounded additive term, `|u b|` times the
**total absolute prefix charge** of the sequence -- a second exact sequence functional.

Evaluating it on the two extremal patterns of Part CXX gives the conclusion:

* `logZ_alt_le` -- the perfectly mixed sequence `+-+-...` gains only `u b t + N log 2`;
* `logZ_blk_ge` -- the diblock `++...+--...-` gains at least `u b t(t+1)/2`;
* `dielectric_amplification` -- hence the diblock's field free energy exceeds the
  alternating one by at least `u b (t(t+1)/2 - t) - N log 2`, a gap growing like the
  *square* of the length of the region.

A disordered region therefore has a dielectric response that is not a property of its amino
acid composition at all: two regions with identical charge content differ in their coupling
to an electric field by a factor that grows without bound with their length.
-/
import Mathlib
import RequestProject.ChargePatterningExact

namespace IDR
namespace Charge

open Finset
open scoped Classical

variable {N : ℕ}

/-! ## 1. Independent bonds factorise -/

/-- The conformational sum of a product of one-bond factors factorises. -/
lemma sum_prod_bool (g : Fin N → Bool → ℝ) :
    ∑ s : Chain N, ∏ i, g i (s i) = ∏ i, (g i true + g i false) := by
  have h := Finset.prod_univ_sum (fun _ : Fin N => (Finset.univ : Finset Bool)) g
  simp only [Fintype.piFinset_univ] at h
  rw [← h]
  exact Finset.prod_congr rfl (fun i _ => by simp)

/-- **The dipole is a linear form in the bonds.**  Bond `i` enters the dipole moment with
coefficient minus the charge of the first `i+1` residues. -/
lemma dipole_eq_linear (q : ℕ → ℝ) (b : ℝ) (s : Chain N) (hQ : pre q (N + 1) = 0) :
    dipole q b s = ∑ i : Fin N, (-(pre q ((i : ℕ) + 1))) * bondVec b s i := by
  have h1 : ∀ k ∈ range (N + 1), q k * pos b s k
      = ∑ i : Fin N, (if (i : ℕ) < k then q k * bondVec b s i else 0) := by
    intro k _
    rw [pos, Finset.sum_filter, Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by split <;> simp
  have hinner : ∀ i : Fin N, ∑ k ∈ range (N + 1), (if (i : ℕ) < k then q k * bondVec b s i else 0)
      = (-(pre q ((i : ℕ) + 1))) * bondVec b s i := by
    intro i
    have hfil : ∑ k ∈ range (N + 1), (if (i : ℕ) < k then q k else 0)
        = pre q (N + 1) - pre q ((i : ℕ) + 1) := by
      rw [← Finset.sum_filter]
      have hset : (range (N + 1)).filter (fun k => (i : ℕ) < k) = Finset.Ico ((i : ℕ) + 1) (N + 1) := by
        ext x
        simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico]
        omega
      rw [hset, Finset.sum_Ico_eq_sub _ (by omega)]
      rfl
    calc ∑ k ∈ range (N + 1), (if (i : ℕ) < k then q k * bondVec b s i else 0)
        = ∑ k ∈ range (N + 1), (if (i : ℕ) < k then q k else 0) * bondVec b s i := by
          exact Finset.sum_congr rfl fun k _ => by split <;> simp
      _ = (pre q (N + 1) - pre q ((i : ℕ) + 1)) * bondVec b s i := by
          rw [← Finset.sum_mul, hfil]
      _ = (-(pre q ((i : ℕ) + 1))) * bondVec b s i := by rw [hQ]; ring
  rw [dipole, Finset.sum_congr rfl h1, Finset.sum_comm]
  exact Finset.sum_congr rfl fun i _ => hinner i

/-! ## 2. The partition function in a field -/

/-- The partition function of the chain in a uniform field coupling to the dipole, at
reduced field strength `u`. -/
noncomputable def Zdip (q : ℕ → ℝ) (b u : ℝ) (N : ℕ) : ℝ :=
  ∑ s : Chain N, Real.exp (u * dipole (N := N) q b s)

/-- **The exact solution.**  The field partition function of a neutral disordered
polyampholyte factorises into one `2 cosh` per bond, with the prefix charges as
couplings. -/
theorem Zdip_prod (q : ℕ → ℝ) (b u : ℝ) (hQ : pre q (N + 1) = 0) :
    Zdip q b u N = ∏ i : Fin N, (2 * Real.cosh (u * b * pre q ((i : ℕ) + 1))) := by
  have hstep : ∀ s : Chain N, Real.exp (u * dipole (N := N) q b s)
      = ∏ i : Fin N, Real.exp (u * ((-(pre q ((i : ℕ) + 1))) * bondVec b s i)) := by
    intro s
    rw [dipole_eq_linear q b s hQ, Finset.mul_sum, Real.exp_sum]
  rw [Zdip, Finset.sum_congr rfl (fun s (_ : s ∈ (univ : Finset (Chain N))) => hstep s)]
  have hfun : ∀ (s : Chain N) (i : Fin N),
      Real.exp (u * ((-(pre q ((i : ℕ) + 1))) * bondVec b s i))
        = (fun (i : Fin N) (β : Bool) =>
            Real.exp (u * ((-(pre q ((i : ℕ) + 1))) * (if β then b else -b)))) i (s i) := by
    intro s i
    rfl
  have hkey := sum_prod_bool (N := N) (fun (i : Fin N) (β : Bool) =>
    Real.exp (u * ((-(pre q ((i : ℕ) + 1))) * (if β then b else -b))))
  rw [Finset.sum_congr rfl (fun s (_ : s ∈ (univ : Finset (Chain N))) =>
    Finset.prod_congr rfl (fun i _ => hfun s i)), hkey]
  refine Finset.prod_congr rfl fun i _ => ?_
  have hb1 : (if (true : Bool) then b else -b) = b := by simp
  have hb2 : (if (false : Bool) then b else -b) = -b := by simp
  rw [hb1, hb2, Real.cosh_eq]
  have h1 : u * ((-(pre q ((i : ℕ) + 1))) * b) = -(u * b * pre q ((i : ℕ) + 1)) := by ring
  have h2 : u * ((-(pre q ((i : ℕ) + 1))) * -b) = u * b * pre q ((i : ℕ) + 1) := by ring
  rw [h1, h2]
  ring

lemma cosh_pos' (x : ℝ) : 0 < 2 * Real.cosh x := by
  have := Real.cosh_pos x
  linarith

/-- The exact free energy: one `log (2 cosh)` per bond. -/
theorem logZ_eq (q : ℕ → ℝ) (b u : ℝ) (hQ : pre q (N + 1) = 0) :
    Real.log (Zdip q b u N)
      = ∑ i : Fin N, Real.log (2 * Real.cosh (u * b * pre q ((i : ℕ) + 1))) := by
  rw [Zdip_prod q b u hQ, Real.log_prod]
  intro i _
  exact ne_of_gt (cosh_pos' _)

/-! ## 3. Bounds on `log (2 cosh)` -/

lemma log_two_cosh_ge (x : ℝ) : |x| ≤ Real.log (2 * Real.cosh x) := by
  have hcosh : 2 * Real.cosh x = Real.exp x + Real.exp (-x) := by
    rw [Real.cosh_eq]; ring
  have hle : Real.exp |x| ≤ 2 * Real.cosh x := by
    rw [hcosh]
    rcases abs_cases x with ⟨h, _⟩ | ⟨h, _⟩
    · rw [h]; linarith [Real.exp_pos (-x)]
    · rw [h]; linarith [Real.exp_pos x]
  calc |x| = Real.log (Real.exp |x|) := (Real.log_exp _).symm
    _ ≤ Real.log (2 * Real.cosh x) := Real.log_le_log (Real.exp_pos _) hle

lemma log_two_cosh_le (x : ℝ) : Real.log (2 * Real.cosh x) ≤ Real.log 2 + |x| := by
  have hcosh : 2 * Real.cosh x = Real.exp x + Real.exp (-x) := by
    rw [Real.cosh_eq]; ring
  have h1 : Real.exp x ≤ Real.exp |x| := Real.exp_le_exp.2 (le_abs_self x)
  have h2 : Real.exp (-x) ≤ Real.exp |x| := Real.exp_le_exp.2 (neg_le_abs x)
  have hle : 2 * Real.cosh x ≤ 2 * Real.exp |x| := by rw [hcosh]; linarith
  calc Real.log (2 * Real.cosh x) ≤ Real.log (2 * Real.exp |x|) :=
        Real.log_le_log (cosh_pos' x) hle
    _ = Real.log 2 + |x| := by
        rw [Real.log_mul (by norm_num) (ne_of_gt (Real.exp_pos _)), Real.log_exp]

/-! ## 4. The field free energy is the total absolute prefix charge -/

/-- The total absolute prefix charge of the sequence. -/
noncomputable def absPre (q : ℕ → ℝ) (m : ℕ) : ℝ := ∑ k ∈ range m, |pre q k|

lemma sum_abs_shift (q : ℕ → ℝ) (b u : ℝ) :
    ∑ i : Fin N, |u * b * pre q ((i : ℕ) + 1)| = |u * b| * absPre q (N + 1) := by
  have h1 : ∑ i : Fin N, |u * b * pre q ((i : ℕ) + 1)|
      = ∑ k ∈ range N, |u * b * pre q (k + 1)| :=
    Fin.sum_univ_eq_sum_range (fun k => |u * b * pre q (k + 1)|) N
  have h2 : ∑ k ∈ range (N + 1), |u * b * pre q k|
      = ∑ k ∈ range N, |u * b * pre q (k + 1)| + |u * b * pre q 0| :=
    Finset.sum_range_succ' (fun k => |u * b * pre q k|) N
  have h3 : pre q 0 = 0 := by simp [pre]
  rw [h1]
  have h4 : ∑ k ∈ range (N + 1), |u * b * pre q k| = |u * b| * absPre q (N + 1) := by
    rw [absPre, Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => abs_mul _ _
  rw [← h4, h2, h3]
  simp

/-- **Lower bound on the field free energy.** -/
theorem logZ_lower (q : ℕ → ℝ) (b u : ℝ) (hQ : pre q (N + 1) = 0) :
    |u * b| * absPre q (N + 1) ≤ Real.log (Zdip q b u N) := by
  rw [logZ_eq q b u hQ, ← sum_abs_shift (N := N) q b u]
  exact Finset.sum_le_sum fun i _ => log_two_cosh_ge _

/-- **Upper bound on the field free energy.** -/
theorem logZ_upper (q : ℕ → ℝ) (b u : ℝ) (hQ : pre q (N + 1) = 0) :
    Real.log (Zdip q b u N) ≤ (N : ℝ) * Real.log 2 + |u * b| * absPre q (N + 1) := by
  rw [logZ_eq q b u hQ, ← sum_abs_shift (N := N) q b u]
  calc ∑ i : Fin N, Real.log (2 * Real.cosh (u * b * pre q ((i : ℕ) + 1)))
      ≤ ∑ i : Fin N, (Real.log 2 + |u * b * pre q ((i : ℕ) + 1)|) :=
        Finset.sum_le_sum fun i _ => log_two_cosh_le _
    _ = (N : ℝ) * Real.log 2 + ∑ i : Fin N, |u * b * pre q ((i : ℕ) + 1)| := by
        rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]

/-! ## 5. The two extremal patterns -/

lemma absPre_cast (q : ℕ → ℤ) (m : ℕ) :
    absPre (fun i => (q i : ℝ)) m = ((∑ k ∈ range m, |preZ q k| : ℤ) : ℝ) := by
  rw [absPre]
  push_cast
  exact Finset.sum_congr rfl fun k _ => by rw [pre_cast]

/-- The perfectly mixed sequence has total absolute prefix charge `m/2`. -/
theorem absPre_alt (m : ℕ) : ∑ k ∈ range m, |preZ alt k| = ((m / 2 : ℕ) : ℤ) := by
  have hterm : ∀ k ∈ range m, |preZ alt k| = if ¬ Even k then (1 : ℤ) else 0 := by
    intro k _
    rw [preZ_alt]
    by_cases h : Even k <;> simp [h]
  rw [Finset.sum_congr rfl hterm, Finset.sum_ite, Finset.sum_const, Finset.sum_const,
    card_odd_range]
  simp

lemma two_sum_range_id (n : ℕ) : 2 * ∑ k ∈ range (n + 1), (k : ℤ) = n * (n + 1) := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_range_succ, mul_add, ih]
      push_cast
      ring

/-- The diblock has total absolute prefix charge at least `t(t+1)/2`. -/
theorem absPre_blk_ge (t : ℕ) :
    (t : ℤ) * (t + 1) ≤ 2 * ∑ k ∈ range (2 * t), |preZ (blk t) k| := by
  rcases Nat.eq_zero_or_pos t with ht | ht
  · subst ht; simp
  have habs : ∀ k ∈ range (2 * t), |preZ (blk t) k| = ((min k (2 * t - k) : ℕ) : ℤ) := by
    intro k hk
    rw [preZ_blk t k (le_of_lt (Finset.mem_range.1 hk))]
    exact abs_of_nonneg (by positivity)
  have hsub : range (t + 1) ⊆ range (2 * t) := by
    intro x hx
    simp only [Finset.mem_range] at hx ⊢
    omega
  have hsmall : ∀ k ∈ range (t + 1), (k : ℤ) = ((min k (2 * t - k) : ℕ) : ℤ) := by
    intro k hk
    have hkt : k ≤ t := by have := Finset.mem_range.1 hk; omega
    have hmin : min k (2 * t - k) = k := by omega
    rw [hmin]
  have hle : ∑ k ∈ range (t + 1), (k : ℤ) ≤ ∑ k ∈ range (2 * t), ((min k (2 * t - k) : ℕ) : ℤ) :=
    calc ∑ k ∈ range (t + 1), (k : ℤ)
        = ∑ k ∈ range (t + 1), ((min k (2 * t - k) : ℕ) : ℤ) := Finset.sum_congr rfl hsmall
      _ ≤ ∑ k ∈ range (2 * t), ((min k (2 * t - k) : ℕ) : ℤ) :=
          Finset.sum_le_sum_of_subset_of_nonneg hsub (fun k _ _ => by positivity)
  rw [Finset.sum_congr rfl habs, ← two_sum_range_id t]
  omega

/-- **The mixed sequence responds only linearly.** -/
theorem logZ_alt_le {N t : ℕ} (hm : N + 1 = 2 * t) (b u : ℝ) :
    Real.log (Zdip (fun k => (alt k : ℝ)) b u N)
      ≤ (N : ℝ) * Real.log 2 + |u * b| * (t : ℝ) := by
  have heven : Even (N + 1) := ⟨t, by omega⟩
  have hQ : pre (fun k => (alt k : ℝ)) (N + 1) = 0 := by
    rw [pre_cast, alt_neutral heven]; simp
  have hup := logZ_upper (N := N) (fun k => (alt k : ℝ)) b u hQ
  have hval : absPre (fun k => (alt k : ℝ)) (N + 1) = (t : ℝ) := by
    rw [absPre_cast, absPre_alt]
    have hlow : ((N + 1) / 2 : ℕ) = t := by omega
    rw [hlow]
    push_cast
    ring
  rwa [hval] at hup

/-- **The diblock responds quadratically.** -/
theorem logZ_blk_ge {N t : ℕ} (hm : N + 1 = 2 * t) (b u : ℝ) :
    |u * b| * ((t : ℝ) * (t + 1) / 2) ≤ Real.log (Zdip (fun k => (blk t k : ℝ)) b u N) := by
  have hQ : pre (fun k => (blk t k : ℝ)) (N + 1) = 0 := by
    rw [pre_cast, hm, blk_neutral t]; simp
  have hlow := logZ_lower (N := N) (fun k => (blk t k : ℝ)) b u hQ
  have hval : (t : ℝ) * (t + 1) / 2 ≤ absPre (fun k => (blk t k : ℝ)) (N + 1) := by
    rw [absPre_cast, hm]
    have h := absPre_blk_ge t
    have hR : ((t : ℤ) : ℝ) * (((t : ℤ) : ℝ) + 1)
        ≤ 2 * ((∑ k ∈ range (2 * t), |preZ (blk t) k| : ℤ) : ℝ) := by
      exact_mod_cast h
    push_cast at hR ⊢
    linarith
  have habs : (0 : ℝ) ≤ |u * b| := abs_nonneg _
  calc |u * b| * ((t : ℝ) * (t + 1) / 2)
      ≤ |u * b| * absPre (fun k => (blk t k : ℝ)) (N + 1) := by
        exact mul_le_mul_of_nonneg_left hval habs
    _ ≤ Real.log (Zdip (fun k => (blk t k : ℝ)) b u N) := hlow

/-- **Charge patterning amplifies the dielectric response quadratically.**  At fixed charge
composition, the field free energy of the diblock exceeds that of the perfectly mixed
sequence by at least `|u b| (t(t+1)/2 - t) - N log 2`: a gap growing like the square of the
length of the disordered region, against a fixed entropic offset. -/
theorem dielectric_amplification {N t : ℕ} (hm : N + 1 = 2 * t) (b u : ℝ) :
    |u * b| * ((t : ℝ) * (t + 1) / 2 - t) - (N : ℝ) * Real.log 2
      ≤ Real.log (Zdip (fun k => (blk t k : ℝ)) b u N)
        - Real.log (Zdip (fun k => (alt k : ℝ)) b u N) := by
  have h1 := logZ_alt_le hm b u
  have h2 := logZ_blk_ge hm b u
  linarith

end Charge
end IDR
