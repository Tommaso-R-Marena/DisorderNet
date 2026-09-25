"""Hugging Face Space: what can this benchmark actually resolve?

Paste a benchmark's size and annotation error rate and get back the number of
methods it can place in a certified order, how many comparisons it cannot
decide, and what a statistic with the pairing structure would buy. Every figure
comes from the same library the paper's tables do.
"""

from __future__ import annotations

import gradio as gr

from disordernet import (
    assess, capacity_over_range, imbalance_factor, rates,
)
from disordernet.cli import PUBLISHED

ACCENT = "#0F766E"

INTRO = """
# What can your benchmark resolve?

A benchmark that ranks methods on annotations of finite accuracy has a
**capacity**: the number of methods it can place in a certified order, however
many enter and however the data are analysed. Past it the ordering is not a
function of the observations — for two close methods there are two ground
truths consistent with everything recorded, one favouring each.

Everything here is computed by the [`disordernet`](https://pypi.org/project/disordernet/)
package from statements that are machine-checked in Lean 4.
"""


def verdict(n_methods, eps, prevalence, lo, hi, eps_pair):
    if eps <= 0 or eps >= 1:
        return "Annotation error rate must be between 0 and 1."
    kappa = 1.0
    if 0 < prevalence < 1:
        kappa = imbalance_factor(round(prevalence * 10_000),
                                 round((1 - prevalence) * 10_000))
    v = assess(int(n_methods), float(eps), kappa=kappa,
               eps_pair=(float(eps_pair) if eps_pair and eps_pair > 0 else None))
    out = [str(v), ""]
    if hi > lo:
        out.append(f"Over the score range published methods actually occupy "
                   f"[{lo}, {hi}]: **{capacity_over_range(eps, lo, hi)}**.")
    if kappa > 1.0:
        out.append(f"Class prevalence {prevalence:.2f} gives an imbalance "
                   f"factor κ = {kappa:.3f}, which is carried into the pairwise "
                   f"bound (no balance hypothesis is assumed).")
    if v.n_methods > v.capacity:
        out.append(f"\n**This benchmark is over capacity by "
                   f"{v.over_capacity_by:.0f}×.** At least "
                   f"{v.unresolvable:,} of its {v.total_comparisons:,} "
                   f"pairwise comparisons are undecidable from these data by "
                   f"anyone — not unresolved by a particular analysis.")
    else:
        out.append("\nThis benchmark is within capacity: the labels can "
                   "support the ranking it prints.")
    return "\n".join(out)


def measure(truth_text, annot_text):
    import numpy as np
    try:
        t = np.array([int(x) for x in truth_text.replace(",", " ").split()])
        a = np.array([int(x) for x in annot_text.replace(",", " ").split()])
    except ValueError:
        return "Both boxes take 0/1 values separated by spaces or commas."
    if t.size != a.size or t.size == 0:
        return "The two annotations must cover the same items."
    r = rates(t, a)
    return (f"```\n{r}\n```\n\n"
            f"- label-flip rate **{r.eps_label:.4f}** "
            f"({r.n_disagreements} of {r.n_items} items)\n"
            f"- pairwise discordance **{r.eps_pair:.2e}** "
            f"({r.discordant_pairs} of {r.comparable_pairs} comparable pairs)\n"
            f"- a pair reverses only when both members flip in opposite "
            f"directions, which is why the second rate is smaller by "
            f"{r.ratio:,.0f}×")


def build() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal"),
                   title="Benchmark capacity") as demo:
        gr.Markdown(INTRO)
        with gr.Tab("Capacity"):
            with gr.Row():
                with gr.Column():
                    n = gr.Number(117, label="Methods entered", precision=0)
                    e = gr.Number(0.0801, label="Annotation error rate ε")
                    p = gr.Slider(0, 1, 0.5, step=0.01,
                                  label="Positive-class prevalence (for κ)")
                    with gr.Accordion("Optional", open=False):
                        lo = gr.Number(0.0, label="Published score range, low")
                        hi = gr.Number(0.0, label="…high")
                        ep = gr.Number(0.0, label="Measured pairwise rate "
                                                  "(0 = use the bound)")
                    go = gr.Button("Assess", variant="primary")
                out = gr.Markdown()
            go.click(verdict, [n, e, p, lo, hi, ep], out)
            gr.Examples(
                [[c[2] and 1060 or 117, c[3], 0.5, 0.0, 0.0, 0.0]
                 for c in PUBLISHED[:4]],
                [n, e, p, lo, hi, ep],
                label="Published benchmarks (validated label-error rates)")
        with gr.Tab("Measure your own rate"):
            gr.Markdown("Two annotations of the same items, 0/1. Where an item "
                        "has been annotated twice, both rates the theory needs "
                        "can be measured directly.")
            with gr.Row():
                t1 = gr.Textbox(label="Annotation A (or the truth)",
                                value="1 1 1 0 0 0 0 0")
                t2 = gr.Textbox(label="Annotation B", value="1 1 0 1 0 0 0 0")
            mb = gr.Button("Measure", variant="primary")
            mo = gr.Markdown()
            mb.click(measure, [t1, t2], mo)
        with gr.Tab("Published benchmarks"):
            gr.Markdown("Capacities from the validated label-error rates of "
                        "Northcutt, Athalye & Mueller (NeurIPS 2021). Rates are "
                        "lower bounds, so capacities are upper bounds.")
            from disordernet import capacity as _cap
            gr.Dataframe(
                value=[[b, m, f"{s:,}", f"{x:.4f}", _cap(x)]
                       for b, m, s, x in PUBLISHED],
                headers=["benchmark", "modality", "items", "ε", "capacity"],
                interactive=False)
    return demo


if __name__ == "__main__":
    build().launch()
