"""
Training–eval structure-signal contamination audit (paper honesty layer).

If hallucination upweighting or pLDDT features were on during training, the
post-AF rescue claim is partly circular. Reports must surface this explicitly.
"""

from __future__ import annotations

from typing import Any, Optional


def audit_structure_signal_contamination(cfg: Any = None, **overrides) -> dict:
    """
    Summarize whether structure pLDDT entered training in ways that inflate
    labeled hallucination rescue / ΔAUC vs inverse-pLDDT.
    """
    def _get(name: str, default):
        if name in overrides and overrides[name] is not None:
            return overrides[name]
        if cfg is None:
            return default
        return getattr(cfg, name, default)

    use_hw = bool(_get("use_hallucination_weighting", False))
    use_feat = bool(_get("use_plddt_features", False))
    split_method = str(_get("split_method", "protein"))
    hall_w = float(_get("hallucination_weight", 1.0))
    high_plddt = float(_get("high_plddt_threshold", 70.0))
    plddt_dim = int(_get("plddt_feature_dim", 0) or 0)
    homology_id = float(_get("homology_min_identity", 0.4))

    if use_feat and use_hw:
        risk = "high"
        caveat = (
            "Training used both pLDDT input features and hallucination upweighting; "
            "labeled rescue / ΔAUC vs inverse-pLDDT must be interpreted cautiously "
            "and accompanied by a no-structure-training ablation."
        )
    elif use_feat or use_hw:
        risk = "medium"
        caveat = (
            "Training used structure-derived signal "
            f"({'pLDDT features' if use_feat else 'hallucination weighting'}); "
            "report a matched ablation with both flags off before strong claims."
        )
    else:
        risk = "low"
        caveat = (
            "Training did not use pLDDT features or hallucination weighting; "
            "structure distrust evaluation is cleaner."
        )

    return {
        "use_hallucination_weighting": use_hw,
        "hallucination_weight": hall_w if use_hw else None,
        "high_plddt_threshold": high_plddt,
        "use_plddt_features": use_feat,
        "plddt_feature_dim": plddt_dim if use_feat else None,
        "split_method": split_method,
        "homology_min_identity": homology_id if split_method == "homology" else None,
        "risk_tier": risk,
        "paper_caveat": caveat,
        "required_ablation": (
            None if risk == "low"
            else {
                "use_hallucination_weighting": False,
                "use_plddt_features": False,
                "note": "Re-run CV / eval with both flags False for companion table",
            }
        ),
    }


def attach_contamination_flags(report: dict, cfg: Any = None, **overrides) -> dict:
    """Mutate/return report with ``training_contamination`` attached."""
    report = dict(report)
    report["training_contamination"] = audit_structure_signal_contamination(cfg, **overrides)
    return report
