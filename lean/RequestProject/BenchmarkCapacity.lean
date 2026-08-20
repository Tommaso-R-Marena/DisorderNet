/-
# The capacity of a noisy benchmark, in the abstract

A benchmark reports a score for each method.  The score it reports is not the quantity anyone
cares about: it is the quantity of interest plus an error coming from the imperfection of the
reference labels.  If the reported scores of two methods differ by less than the size of that
error, the benchmark has not separated them -- and no reanalysis of the same data can, because
the data are compatible with either ordering.

The consequence is a *capacity*: a benchmark whose score range is `R` and whose resolution
(the smallest score difference it can certify) is `c` can place at most a fixed number of
methods in a certified total order, no matter how many methods enter it.  This file proves
that number, in three settings, together with matching instances showing it is attained.

* `Separated`, `capacityNat`, `card_le_capacityNat`, `capacityNat_attained` -- **the integer
  core.**  A set of integer scores in `{0, …, N}` that are pairwise more than `ν` apart has at
  most `N / (ν + 1) + 1` elements, and some such set has exactly that many.
* `SeparatedR`, `spread_lt`, `capacityReal`, `card_le_capacityReal`,
  `capacityReal_attained` -- **the real-valued statement.**  A set of real scores in `[0, R]`
  that are pairwise more than `c` apart has at most `⌈R / c⌉` elements (at least one), and
  this is attained.  `c` is whatever resolution the analyst can justify; the theorem does not
  care where it comes from.
* `benchCapacity`, `card_le_benchCapacity`, `benchCapacity_attained` -- **the benchmark
  form.**  A benchmark with `n` targets scored in `[0,1]` (so scores live on the grid of
  denominator `n`), annotation error rate `eps` and effect size `delta` can rank at most

      `n / (⌊max delta (2 * eps) * n⌋ + 1) + 1`

  methods, and there is a family of methods of exactly that size which it does rank.  The
  factor two on `eps` is the two-sided price of label noise (see `RequestProject.LabelNoise`);
  `delta` is the smallest difference the community agrees is worth calling a difference.
* `benchCapacity_le_ceil`, `benchCapacity_noise_only` -- the readable consequences: the
  capacity never exceeds `⌈1 / max delta (2 eps)⌉`, and with `delta = 0` it is governed by the
  annotation error rate alone: roughly `1/(2 eps)` methods, whatever `n` is.

Nothing here is specific to protein disorder, or to biology.  The input is a range, a
resolution and a grid; every benchmark with noisy labels has all three.
-/
import Mathlib

set_option autoImplicit false

namespace IDR
namespace BenchCapacity

open Finset

/-! ## 1.  The integer core

Scores are counts (residues wrong, targets missed, …), so they are natural numbers bounded by
some `N`, and "certified different" means "differ by more than `ν`". -/

/-- A set of integer scores is `ν`-separated when distinct members differ by more than `ν`. -/
def Separated (nu : ℕ) (S : Finset ℕ) : Prop :=
  ∀ a ∈ S, ∀ b ∈ S, a < b → a + nu < b

/-- The capacity of an integer score range `{0, …, N}` at resolution `nu`. -/
def capacityNat (N nu : ℕ) : ℕ := N / (nu + 1) + 1

/-- **The capacity bound.**  At most `N / (nu + 1) + 1` scores in `{0, …, N}` can be pairwise
separated by more than `nu`. -/
theorem card_le_capacityNat {N nu : ℕ} {S : Finset ℕ}
    (hmem : ∀ a ∈ S, a ≤ N) (hsep : Separated nu S) :
    S.card ≤ capacityNat N nu := by
  classical
  have hinj : Set.InjOn (fun a => a / (nu + 1)) S := by
    intro a ha b hb hab
    by_contra hne
    rcases lt_or_gt_of_ne hne with h | h
    · have := hsep a ha b hb h
      have : a / (nu + 1) + 1 ≤ b / (nu + 1) := by
        have hle : a + (nu + 1) ≤ b := by omega
        calc a / (nu + 1) + 1 = (a + (nu + 1)) / (nu + 1) := by
              rw [Nat.add_div_right _ (Nat.succ_pos nu)]
          _ ≤ b / (nu + 1) := Nat.div_le_div_right hle
      simp only at hab
      omega
    · have := hsep b hb a ha h
      have : b / (nu + 1) + 1 ≤ a / (nu + 1) := by
        have hle : b + (nu + 1) ≤ a := by omega
        calc b / (nu + 1) + 1 = (b + (nu + 1)) / (nu + 1) := by
              rw [Nat.add_div_right _ (Nat.succ_pos nu)]
          _ ≤ a / (nu + 1) := Nat.div_le_div_right hle
      simp only at hab
      omega
  have himg : (S.image (fun a => a / (nu + 1))) ⊆ Finset.range (N / (nu + 1) + 1) := by
    intro x hx
    simp only [Finset.mem_image] at hx
    obtain ⟨a, ha, rfl⟩ := hx
    simp only [Finset.mem_range]
    have : a / (nu + 1) ≤ N / (nu + 1) := Nat.div_le_div_right (hmem a ha)
    omega
  have hcard : S.card = (S.image (fun a => a / (nu + 1))).card :=
    (Finset.card_image_of_injOn hinj).symm
  rw [hcard, capacityNat]
  calc (S.image (fun a => a / (nu + 1))).card ≤ (Finset.range (N / (nu + 1) + 1)).card :=
        Finset.card_le_card himg
    _ = N / (nu + 1) + 1 := Finset.card_range _

/-- **The capacity is attained.**  The arithmetic progression of step `nu + 1` inside
`{0, …, N}` is `nu`-separated and has exactly `capacityNat N nu` elements. -/
theorem capacityNat_attained (N nu : ℕ) :
    ∃ S : Finset ℕ, (∀ a ∈ S, a ≤ N) ∧ Separated nu S ∧ S.card = capacityNat N nu := by
  classical
  refine ⟨(Finset.range (N / (nu + 1) + 1)).image (fun i => i * (nu + 1)), ?_, ?_, ?_⟩
  · intro a ha
    simp only [Finset.mem_image, Finset.mem_range] at ha
    obtain ⟨i, hi, rfl⟩ := ha
    have hi' : i ≤ N / (nu + 1) := by omega
    calc i * (nu + 1) ≤ (N / (nu + 1)) * (nu + 1) := Nat.mul_le_mul_right _ hi'
      _ ≤ N := Nat.div_mul_le_self _ _
  · intro a ha b hb hab
    simp only [Finset.mem_image, Finset.mem_range] at ha hb
    obtain ⟨i, _, rfl⟩ := ha
    obtain ⟨j, _, rfl⟩ := hb
    have hij : i < j := by
      by_contra h
      push_neg at h
      exact absurd (Nat.mul_le_mul_right _ h) (not_le.mpr hab)
    have : i + 1 ≤ j := hij
    calc i * (nu + 1) + nu < (i + 1) * (nu + 1) := by ring_nf; omega
      _ ≤ j * (nu + 1) := Nat.mul_le_mul_right _ this
  · rw [Finset.card_image_of_injective _ (fun x y h => by
      have : (0:ℕ) < nu + 1 := Nat.succ_pos _
      exact Nat.eq_of_mul_eq_mul_right this h), Finset.card_range, capacityNat]

/-! ## 2.  The real-valued statement -/

/-- A set of real scores is `c`-separated when distinct members differ by more than `c`. -/
def SeparatedR (c : ℝ) (S : Finset ℝ) : Prop :=
  ∀ a ∈ S, ∀ b ∈ S, a < b → a + c < b

/-- **Spread lemma.**  A `c`-separated set of at least two reals spans a range of strictly more
than `(k - 1) * c`, where `k` is its cardinality. -/
theorem spread_lt {c : ℝ} {S : Finset ℝ} (h2 : 2 ≤ S.card) (hsep : SeparatedR c S)
    {lo hi : ℝ} (hlo : ∀ a ∈ S, lo ≤ a) (hhi : ∀ a ∈ S, a ≤ hi) :
    ((S.card : ℝ) - 1) * c < hi - lo := by
  classical
  obtain ⟨k, hk⟩ : ∃ k, S.card = k := ⟨S.card, rfl⟩
  have hkpos : 2 ≤ k := hk ▸ h2
  have hne : S.Nonempty := Finset.card_pos.mp (by omega)
  let f := S.orderIsoOfFin hk
  have hmono : StrictMono f := fun i j hij => (f.lt_iff_lt).mpr hij
  have hmemf : ∀ i : Fin k, (f i : ℝ) ∈ S := fun i => (f i).2
  have hstep : ∀ i : ℕ, ∀ h : i < k, (lo : ℝ) + (i : ℝ) * c ≤ (f ⟨i, h⟩ : ℝ) := by
    intro i
    induction i with
    | zero => intro h; simpa using hlo _ (hmemf ⟨0, h⟩)
    | succ n ih =>
      intro h
      have hn : n < k := by omega
      have h1 : (lo : ℝ) + (n : ℝ) * c ≤ (f ⟨n, hn⟩ : ℝ) := ih hn
      have hlt : (f ⟨n, hn⟩ : ℝ) < (f ⟨n + 1, h⟩ : ℝ) := hmono (by simp [Fin.lt_def])
      have := hsep _ (hmemf ⟨n, hn⟩) _ (hmemf ⟨n + 1, h⟩) hlt
      push_cast
      linarith
  -- the last step is strict
  have hlastlt : (lo : ℝ) + ((k - 1 : ℕ) : ℝ) * c < (f ⟨k - 1, by omega⟩ : ℝ) := by
    have hn : k - 2 < k := by omega
    have h1 : (lo : ℝ) + ((k - 2 : ℕ) : ℝ) * c ≤ (f ⟨k - 2, hn⟩ : ℝ) := hstep (k - 2) hn
    have hidx : (⟨k - 1, by omega⟩ : Fin k) = ⟨k - 2 + 1, by omega⟩ := by
      apply Fin.ext; simp; omega
    have hlt : (f ⟨k - 2, hn⟩ : ℝ) < (f ⟨k - 2 + 1, by omega⟩ : ℝ) := hmono (by simp [Fin.lt_def])
    have hsp := hsep _ (hmemf ⟨k - 2, hn⟩) _ (hmemf ⟨k - 2 + 1, by omega⟩) hlt
    have hcast : ((k - 1 : ℕ) : ℝ) = ((k - 2 : ℕ) : ℝ) + 1 := by
      have h2' : (2:ℕ) ≤ k := hkpos
      push_cast [Nat.cast_sub (show (1:ℕ) ≤ k by omega), Nat.cast_sub h2']
      ring
    rw [hidx, hcast]
    nlinarith [hsp, h1]
  have hub := hhi _ (hmemf ⟨k - 1, by omega⟩)
  have hcast : ((k : ℝ) - 1) = ((k - 1 : ℕ) : ℝ) := by
    have : (1:ℕ) ≤ k := by omega
    push_cast [Nat.cast_sub this]
    ring
  rw [hk, hcast]
  linarith

/-- The capacity of a real score range `[0, R]` at resolution `c`. -/
noncomputable def capacityReal (R c : ℝ) : ℕ := max 1 ⌈R / c⌉₊

/-- **The real capacity bound.**  At most `max 1 ⌈R / c⌉` scores in `[0, R]` can be pairwise
separated by more than `c`. -/
theorem card_le_capacityReal {R c : ℝ} {S : Finset ℝ} (hc : 0 < c)
    (hmem : ∀ a ∈ S, a ∈ Set.Icc (0:ℝ) R) (hsep : SeparatedR c S) :
    S.card ≤ capacityReal R c := by
  classical
  rcases lt_or_ge S.card 2 with hsmall | h2
  · have : S.card ≤ 1 := by omega
    exact this.trans (le_max_left _ _)
  · have hspread : ((S.card : ℝ) - 1) * c < R - 0 :=
      spread_lt h2 hsep (fun a ha => (hmem a ha).1) (fun a ha => (hmem a ha).2)
    have hlt : ((S.card : ℝ) - 1) < R / c := by
      rw [lt_div_iff₀ hc]; linarith
    have hcast : (((S.card - 1 : ℕ)) : ℝ) = (S.card : ℝ) - 1 := by
      have h1 : (1:ℕ) ≤ S.card := by omega
      push_cast [Nat.cast_sub h1]; ring
    have hceil : (S.card - 1 : ℕ) < ⌈R / c⌉₊ := Nat.lt_ceil.mpr (by rw [hcast]; exact hlt)
    have hmax : ⌈R / c⌉₊ ≤ capacityReal R c := le_max_right _ _
    omega

/-- **The real capacity is attained**, by an arithmetic progression inside `[0, R]`. -/
theorem capacityReal_attained {R c : ℝ} (hc : 0 < c) (hR : 0 ≤ R) :
    ∃ S : Finset ℝ, (∀ a ∈ S, a ∈ Set.Icc (0:ℝ) R) ∧ SeparatedR c S ∧
      S.card = capacityReal R c := by
  classical
  set k := capacityReal R c with hk
  have hk1 : 1 ≤ k := le_max_left _ _
  by_cases hk2 : k = 1
  · refine ⟨{0}, ?_, ?_, ?_⟩
    · intro a ha; simp only [Finset.mem_singleton] at ha; subst ha; exact ⟨le_refl _, hR⟩
    · intro a ha b hb hab
      simp only [Finset.mem_singleton] at ha hb
      subst ha; subst hb; exact absurd hab (lt_irrefl _)
    · rw [hk2]; simp
  · have hkgt : 2 ≤ k := by omega
    have hceil : k = ⌈R / c⌉₊ := by
      have hdef : k = max 1 ⌈R / c⌉₊ := hk
      omega
    have hkm1 : ((k : ℝ) - 1) * c < R := by
      have hlt : ((k - 1 : ℕ) : ℝ) < R / c := Nat.lt_ceil.mp (by omega)
      have hcast : ((k - 1 : ℕ) : ℝ) = (k : ℝ) - 1 := by
        have : (1:ℕ) ≤ k := hk1
        push_cast [Nat.cast_sub this]; ring
      rw [hcast] at hlt
      calc ((k : ℝ) - 1) * c < (R / c) * c := mul_lt_mul_of_pos_right hlt hc
        _ = R := by field_simp
    set d : ℝ := R / ((k : ℝ) - 1) with hd
    have hkr : (0:ℝ) < (k : ℝ) - 1 := by
      have : (2:ℝ) ≤ (k : ℝ) := by exact_mod_cast hkgt
      linarith
    have hdc : c < d := by
      rw [hd, lt_div_iff₀ hkr]
      linarith [hkm1]
    have hdpos : 0 < d := lt_trans hc hdc
    have hinj : Function.Injective (fun i : ℕ => (i : ℝ) * d) := by
      intro x y hxy
      have := mul_right_cancel₀ (ne_of_gt hdpos) hxy
      exact_mod_cast this
    refine ⟨(Finset.range k).image (fun i : ℕ => (i : ℝ) * d), ?_, ?_, ?_⟩
    · intro a ha
      simp only [Finset.mem_image] at ha
      obtain ⟨i, hi, rfl⟩ := ha
      have hik : i < k := Finset.mem_range.mp hi
      refine ⟨mul_nonneg (Nat.cast_nonneg i) hdpos.le, ?_⟩
      have hile : ((i : ℝ)) ≤ (k : ℝ) - 1 := by
        have h1 : (i : ℝ) ≤ ((k - 1 : ℕ) : ℝ) := by
          exact_mod_cast Nat.le_sub_one_of_lt hik
        have hcast : ((k - 1 : ℕ) : ℝ) = (k : ℝ) - 1 := by
          have : (1:ℕ) ≤ k := hk1
          push_cast [Nat.cast_sub this]; ring
        linarith [hcast ▸ h1]
      calc (i : ℝ) * d ≤ ((k : ℝ) - 1) * d := mul_le_mul_of_nonneg_right hile hdpos.le
        _ = R := by rw [hd]; field_simp
    · intro a ha b hb hab
      simp only [Finset.mem_image] at ha hb
      obtain ⟨i, _, rfl⟩ := ha
      obtain ⟨j, _, rfl⟩ := hb
      have hij : (i : ℝ) < (j : ℝ) := by
        by_contra h
        push_neg at h
        exact absurd (mul_le_mul_of_nonneg_right h hdpos.le) (not_le.mpr hab)
      have hij' : (i : ℝ) + 1 ≤ (j : ℝ) := by
        have hn : i < j := by exact_mod_cast hij
        have : i + 1 ≤ j := hn
        exact_mod_cast this
      calc (i : ℝ) * d + c < (i : ℝ) * d + d := by linarith
        _ = ((i : ℝ) + 1) * d := by ring
        _ ≤ (j : ℝ) * d := mul_le_mul_of_nonneg_right hij' hdpos.le
    · rw [Finset.card_image_of_injective _ hinj]
      exact Finset.card_range k


/-! ## 3.  The benchmark form: `n` targets, annotation error rate `eps`, effect size `delta`

A benchmark scores each method in `[0,1]` by averaging over its `n` targets, so the attainable
scores are the multiples of `1/n`.  Two ingredients set the smallest difference it can certify:

* the **annotation error rate** `eps` -- the fraction of the reference labels that are wrong.
  A measured score can be wrong by `eps` in either direction, so a *comparison* can be wrong by
  `2 * eps`, and a gap must exceed `2 * eps` to be certified (`IDR.LabelNoise.ranking_certified`
  is the underlying inequality, proved there for Hamming scores);
* the **effect size** `delta` -- the smallest difference the field agrees is worth calling a
  difference.

The resolution of the benchmark is the larger of the two.  -/

/-- The **resolution** of a benchmark: the smallest score difference it can certify, given an
annotation error rate `eps` and an effect size `delta`. -/
def resolution (eps delta : ℝ) : ℝ := max delta (2 * eps)

lemma resolution_nonneg {eps delta : ℝ} (hdelta : 0 ≤ delta) :
    0 ≤ resolution eps delta :=
  le_max_of_le_left hdelta

/-- The **capacity of a benchmark**: the largest number of methods that a benchmark with `n`
targets, annotation error rate `eps` and effect size `delta` can place in a certified order. -/
noncomputable def benchCapacity (n : ℕ) (eps delta : ℝ) : ℕ :=
  n / (⌊resolution eps delta * n⌋₊ + 1) + 1

/-- A finite family `M` of methods, scored by the number `score i ≤ n` of targets they get
right, is **resolved** by the benchmark when every two of them differ in normalised score by
more than the resolution `c`. -/
def Resolves {iota : Type*} (n : ℕ) (c : ℝ) (score : iota → ℕ) (M : Finset iota) : Prop :=
  (∀ i ∈ M, score i ≤ n) ∧
    ∀ i ∈ M, ∀ j ∈ M, i ≠ j →
      ((score i : ℝ) / n + c < (score j : ℝ) / n ∨ (score j : ℝ) / n + c < (score i : ℝ) / n)

/-- **The capacity theorem for a benchmark.**  No matter how many methods are entered, a
benchmark with `n` targets, annotation error rate `eps` and effect size `delta` resolves at
most `benchCapacity n eps delta` of them. -/
theorem card_le_benchCapacity {iota : Type*} {n : ℕ} {eps delta : ℝ} {score : iota → ℕ}
    {M : Finset iota} (hn : 0 < n) (hdelta : 0 ≤ delta)
    (hres : Resolves n (resolution eps delta) score M) :
    M.card ≤ benchCapacity n eps delta := by
  classical
  obtain ⟨hbd, hsep⟩ := hres
  set c : ℝ := resolution eps delta with hc
  have hc0 : 0 ≤ c := resolution_nonneg hdelta
  have hnR : (0:ℝ) < (n : ℝ) := by exact_mod_cast hn
  -- separation of raw integer scores
  have hgap : ∀ i ∈ M, ∀ j ∈ M, score i < score j → score i + ⌊c * n⌋₊ < score j := by
    intro i hi j hj hij
    have hne : i ≠ j := by
      rintro rfl; exact absurd hij (lt_irrefl _)
    have hor := hsep i hi j hj hne
    have hlt : (score i : ℝ) / n + c < (score j : ℝ) / n := by
      rcases hor with h | h
      · exact h
      · exfalso
        have : (score i : ℝ) < (score j : ℝ) := by exact_mod_cast hij
        have h1 : (score i : ℝ) / n < (score j : ℝ) / n := div_lt_div_of_pos_right this hnR
        linarith
    have hmul : (score i : ℝ) + c * n < (score j : ℝ) := by
      have e1 : (score i : ℝ) / n * n = (score i : ℝ) := div_mul_cancel₀ _ (ne_of_gt hnR)
      have e2 : (score j : ℝ) / n * n = (score j : ℝ) := div_mul_cancel₀ _ (ne_of_gt hnR)
      have hm := mul_lt_mul_of_pos_right hlt hnR
      nlinarith [hm, e1, e2]
    have hdiff : c * (n : ℝ) < ((score j - score i : ℕ) : ℝ) := by
      have hle : score i ≤ score j := hij.le
      push_cast [Nat.cast_sub hle]
      linarith
    have hfloor : ⌊c * (n : ℝ)⌋₊ < score j - score i :=
      Nat.floor_lt (by positivity) |>.mpr hdiff
    omega
  -- pass to the finset of scores
  have hinj : Set.InjOn score M := by
    intro i hi j hj hij
    by_contra hne
    rcases hsep i hi j hj hne with h | h <;> rw [hij] at h <;> linarith
  have himg : ∀ a ∈ M.image score, a ≤ n := by
    intro a ha
    simp only [Finset.mem_image] at ha
    obtain ⟨i, hi, rfl⟩ := ha
    exact hbd i hi
  have hsepimg : Separated ⌊c * (n : ℝ)⌋₊ (M.image score) := by
    intro a ha b hb hab
    simp only [Finset.mem_image] at ha hb
    obtain ⟨i, hi, rfl⟩ := ha
    obtain ⟨j, hj, rfl⟩ := hb
    exact hgap i hi j hj hab
  have := card_le_capacityNat himg hsepimg
  rw [Finset.card_image_of_injOn hinj] at this
  simpa [benchCapacity, capacityNat, hc] using this

/-- **The capacity is attained.**  There is a family of exactly `benchCapacity n eps delta`
methods that the benchmark does resolve, so the bound is the exact capacity, not an estimate. -/
theorem benchCapacity_attained (n : ℕ) {eps delta : ℝ} (hn : 0 < n) (hdelta : 0 ≤ delta) :
    ∃ (M : Finset ℕ) (score : ℕ → ℕ), Resolves n (resolution eps delta) score M ∧
      M.card = benchCapacity n eps delta := by
  classical
  set c : ℝ := resolution eps delta with hc
  have hc0 : 0 ≤ c := resolution_nonneg hdelta
  have hnR : (0:ℝ) < (n : ℝ) := by exact_mod_cast hn
  obtain ⟨S, hbd, hsep, hcard⟩ := capacityNat_attained n ⌊c * (n : ℝ)⌋₊
  refine ⟨S, id, ⟨hbd, ?_⟩, ?_⟩
  · intro i hi j hj hij
    have hnat : ∀ a b : ℕ, a + ⌊c * (n : ℝ)⌋₊ < b → (a : ℝ) / n + c < (b : ℝ) / n := by
      intro a b hab
      have h1 : c * (n : ℝ) < ((b - a : ℕ) : ℝ) := by
        have : ⌊c * (n:ℝ)⌋₊ < b - a := by omega
        exact lt_of_lt_of_le (Nat.lt_floor_add_one _) (by exact_mod_cast this)
      have hle : a ≤ b := by omega
      have h2 : c * (n : ℝ) < (b : ℝ) - (a : ℝ) := by
        push_cast [Nat.cast_sub hle] at h1
        linarith
      rw [div_add' _ _ _ (ne_of_gt hnR), div_lt_div_iff_of_pos_right hnR]
      linarith
    rcases lt_trichotomy i j with h | h | h
    · exact Or.inl (hnat i j (hsep i hi j hj h))
    · exact absurd h hij
    · exact Or.inr (hnat j i (hsep j hj i hi h))
  · simpa [benchCapacity, capacityNat, hc] using hcard

/-- **The capacity never exceeds `⌈1 / resolution⌉`**, whatever the number of targets: refining
the grid does not buy resolution that the labels do not have. -/
theorem benchCapacity_le_ceil (n : ℕ) {eps delta : ℝ} (hn : 0 < n)
    (hdelta : 0 ≤ delta) (hpos : 0 < resolution eps delta) :
    benchCapacity n eps delta ≤ max 1 ⌈1 / resolution eps delta⌉₊ := by
  classical
  set c : ℝ := resolution eps delta with hc
  have hnR : (0:ℝ) < (n : ℝ) := by exact_mod_cast hn
  obtain ⟨M, score, ⟨hbd, hsep⟩, hcard⟩ := benchCapacity_attained n hn hdelta
  -- the normalised scores of the attaining family form a `c`-separated subset of `[0,1]`
  have hinj : Set.InjOn (fun i => (score i : ℝ) / n) M := by
    intro i hi j hj hij
    by_contra hne
    rcases hsep i hi j hj hne with h | h <;> simp only at hij <;> rw [hij] at h <;> linarith
  have hmem : ∀ a ∈ M.image (fun i => (score i : ℝ) / n), a ∈ Set.Icc (0:ℝ) 1 := by
    intro a ha
    simp only [Finset.mem_image] at ha
    obtain ⟨i, hi, rfl⟩ := ha
    refine ⟨by positivity, ?_⟩
    rw [div_le_one hnR]
    exact_mod_cast hbd i hi
  have hsepR : SeparatedR c (M.image (fun i => (score i : ℝ) / n)) := by
    intro a ha b hb hab
    simp only [Finset.mem_image] at ha hb
    obtain ⟨i, hi, rfl⟩ := ha
    obtain ⟨j, hj, rfl⟩ := hb
    have hne : i ≠ j := by rintro rfl; exact absurd hab (lt_irrefl _)
    rcases hsep i hi j hj hne with h | h
    · exact h
    · linarith
  have hb := card_le_capacityReal hpos hmem hsepR
  rw [Finset.card_image_of_injOn hinj, hcard] at hb
  simpa [capacityReal] using hb

/-- **More label noise, fewer methods.**  Capacity is antitone in the resolution. -/
theorem benchCapacity_antitone (n : ℕ) {eps delta eps' delta' : ℝ}
    (h : resolution eps delta ≤ resolution eps' delta') :
    benchCapacity n eps' delta' ≤ benchCapacity n eps delta := by
  have hn : (0:ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have hfl : ⌊resolution eps delta * n⌋₊ ≤ ⌊resolution eps' delta' * n⌋₊ :=
    Nat.floor_mono (mul_le_mul_of_nonneg_right h hn)
  have hdiv : n / (⌊resolution eps' delta' * n⌋₊ + 1) ≤ n / (⌊resolution eps delta * n⌋₊ + 1) :=
    Nat.div_le_div_left (by omega) (Nat.succ_pos _)
  simpa [benchCapacity] using hdiv

/-- **Labels half wrong rank nothing.**  If the resolution reaches the whole score range -- in
particular if the annotation error rate reaches `1/2` -- the capacity is one: the benchmark
cannot separate any two methods at all. -/
theorem benchCapacity_eq_one_of_resolution_ge_one (n : ℕ) {eps delta : ℝ}
    (h : 1 ≤ resolution eps delta) : benchCapacity n eps delta = 1 := by
  have hn : (0:ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have hge : (n : ℝ) ≤ resolution eps delta * n := by nlinarith
  have hfl : n ≤ ⌊resolution eps delta * n⌋₊ := by
    exact Nat.le_floor (by exact_mod_cast hge)
  have hdiv : n / (⌊resolution eps delta * n⌋₊ + 1) = 0 :=
    Nat.div_eq_of_lt (by omega)
  simp [benchCapacity, hdiv]

/-- **The label-noise reading.**  With no effect-size floor (`delta = 0`) the capacity is set by
the annotation error rate alone: a benchmark whose labels are `eps` wrong can rank about
`1 / (2 eps)` methods and no more, however many targets it collects. -/
theorem benchCapacity_noise_only (n : ℕ) {eps : ℝ} (hn : 0 < n) (heps : 0 < eps) :
    benchCapacity n eps 0 ≤ max 1 ⌈1 / (2 * eps)⌉₊ := by
  have hres : resolution eps 0 = 2 * eps := by
    simp only [resolution]
    exact max_eq_right (by linarith)
  have := benchCapacity_le_ceil n hn le_rfl (by rw [hres]; linarith)
  rwa [hres] at this

end BenchCapacity
end IDR
