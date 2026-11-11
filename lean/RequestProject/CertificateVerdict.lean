/-
# Part CLII  The verdict: what one single-molecule experiment proves about a disordered region

Parts CXLVI–CLI each certify one thing from one measured number.  This part states them as a single
theorem about a single experiment: a labelled disordered region, a burst-efficiency histogram taken
with a photon budget `n`, and a contour bound on the internal distance.  Nothing is assumed about
the chain beyond the geometry of the label and the bound; in particular no polymer model, no
Gaussian chain, no fitted distribution, and no reference structure appear.

`single_experiment_verdict` bundles, for any chosen distance `s`:

1. **A compact population** — at least `(Ē − E(s))/(1 − E(s))` of the ensemble is more compact
   than `s` (Part CXLVII);
2. **An expanded population** — at least `(E(s) − Ē)/E(s)` of it is more expanded than `s`
   (Part CXLVII);
3. **Heterogeneity beyond shot noise** — the ensemble variance of the efficiency is at least the
   histogram variance minus `1/(4n)` (Part CLI);
4. **Conformational entropy** — at least `(Var(histogram) − 1/(4n))²/2` nats (Parts CL and CLI);
5. **A free-energy cost of ordering** — at least `k_B T` times that entropy, for any binding event
   that renders the region a single conformation (Part CL).

`verdict_is_sharp` records the other half: each of these is the best that can be claimed.  The
compact population bound is attained by an explicit two-state ensemble, and a histogram no broader
than shot noise is produced by a homogeneous ensemble, so no positive heterogeneity — and hence no
positive entropy — can be certified below the shot-noise ceiling.

This is the measurement-side counterpart of the model-side verdict established earlier in the
development: a model of an intrinsically disordered region must output an ensemble, no finite
collection of experiments identifies one, and what an experiment does deliver is the certified
interval above — which the model must respect, and beyond which it cannot be tested.
-/
import Mathlib
import RequestProject.FretCertificate
import RequestProject.ShotNoise

set_option autoImplicit false

namespace IDR
namespace CertificateVerdict

open Finset IDR.PopulationCertificate IDR.EntropyFloor IDR.ShotNoise IDR.SaxsFret
open IDR.FretCertificate

variable {N : ℕ}

/-- **The verdict of a single single-molecule experiment.**  Populations on both sides of a chosen
distance, heterogeneity beyond the shot-noise ceiling, a conformational entropy floor, and the
free-energy cost of ordering — all certified simultaneously, with no model of the chain. -/
theorem single_experiment_verdict {w r : Fin N → ℝ} {R0 s kT : ℝ} {n : ℕ}
    (hR : 0 < R0) (hs : 0 < s) (hn : 0 < n) (hkT : 0 ≤ kT)
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) (hr : ∀ j, 0 ≤ r j)
    (hbroad : 1 / (4 * n) ≤ burstVar w (fun j => eff R0 (r j)) n
        (wmean w (fun j => eff R0 (r j)))) :
    -- 1. compact population
    (wmean w (fun j => eff R0 (r j)) - eff R0 s) / (1 - eff R0 s) ≤ FretCertificate.popLt w r s ∧
    -- 2. expanded population
    (eff R0 s - wmean w (fun j => eff R0 (r j))) / eff R0 s ≤ popGtDist w r s ∧
    -- 3. heterogeneity beyond shot noise
    burstVar w (fun j => eff R0 (r j)) n (wmean w (fun j => eff R0 (r j))) - 1 / (4 * n)
        ≤ wvar w (fun j => eff R0 (r j)) ∧
    -- 4. conformational entropy floor
    (burstVar w (fun j => eff R0 (r j)) n (wmean w (fun j => eff R0 (r j))) - 1 / (4 * n)) ^ 2 / 2
        ≤ ent w ∧
    -- 5. free-energy cost of ordering the region completely
    kT * (wvar w (fun j => eff R0 (r j)) ^ 2 / (2 * (1:ℝ) ^ 4)) ≤ kT * ent w := by
  have hE0 : ∀ j, 0 ≤ eff R0 (r j) := fun j => (eff_pos hR).le
  have hE1 : ∀ j, eff R0 (r j) ≤ 1 := fun _ => eff_le_one hR
  refine ⟨compact_population_certificate hR hs hw hsum hr,
    expanded_population_certificate hR hs hw hsum hr,
    ensemble_var_lower (E := fun j => eff R0 (r j)) hn hw hsum,
    entropy_floor_from_burst_histogram (E := fun j => eff R0 (r j)) hn hw hsum hE0 hE1 hbroad,
    ordering_cost_lower_bound (x := fun j => eff R0 (r j)) one_pos hkT hw hsum hE0 hE1⟩

/-- **The verdict is sharp on both sides.**  The compact-population certificate is attained by an
explicit two-state ensemble, and a homogeneous ensemble produces a histogram of width exactly the
shot noise, so nothing positive can be certified about heterogeneity — or therefore about
conformational entropy — below the shot-noise ceiling. -/
theorem verdict_is_sharp {R0 s p : ℝ} {n : ℕ} (hR : 0 < R0) (hs : 0 < s) (hn : 0 < n) :
    (∀ q : ℝ, 0 ≤ q → q ≤ 1 → ∃ w r : Fin 2 → ℝ, (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧
        (∀ j, 0 ≤ r j) ∧ FretCertificate.popLt w r s = q ∧
        (wmean w (fun j => eff R0 (r j)) - eff R0 s) / (1 - eff R0 s) = FretCertificate.popLt w r s) ∧
      (∃ w E : Fin 1 → ℝ, (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ (∀ j, E j = p) ∧
        wvar w E = 0 ∧ burstVar w E n (wmean w E) = p * (1 - p) / n) :=
  ⟨fun _ hq0 hq1 => compact_certificate_sharp hR hs hq0 hq1,
   homogeneous_histogram_has_width (p := p) hn⟩

end CertificateVerdict
end IDR
