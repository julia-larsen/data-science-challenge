from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .analyze import canonicalize_step

SUMMARY_PATH_DEFAULT = "data/analysis_summary.json"
EXTRACTED_PATH_DEFAULT = "data/extracted.jsonl"
OUT_DIR_DEFAULT = "data/figures"


def load_summary(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_bar_chart(
    *,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    color: str = "#2A6F97",
    annotate: bool = True,
) -> None:
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=25, ha="right")

    if annotate:
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def chart_timing_distribution(summary: Dict, out_dir: Path) -> Path:
    timing = summary["qa"]["timing_distribution"]
    labels = list(timing.keys())
    values = [timing[k] for k in labels]
    output_path = out_dir / "biologic_timing_distribution.png"
    save_bar_chart(
        labels=labels,
        values=values,
        title="Biologic Timing Distribution",
        xlabel="Biologic Timing",
        ylabel="Patient Count",
        output_path=output_path,
        color="#1D4E89",
    )
    return output_path


def chart_reasons_not_on_biologic(summary: Dict, out_dir: Path) -> Path:
    reason_ranked = summary["business_questions"]["q2_reasons_not_on_biologic"][
        "reason_ranked"
    ]
    labels = [str(item[0]) for item in reason_ranked]
    values = [int(item[1]) for item in reason_ranked]
    output_path = out_dir / "reasons_not_on_biologic.png"
    save_bar_chart(
        labels=labels,
        values=values,
        title="Reasons Patients Are Not On Biologic",
        xlabel="Reason",
        ylabel="Count",
        output_path=output_path,
        color="#B85C38",
    )
    return output_path


def chart_top_treatments(summary: Dict, out_dir: Path, top_n: int = 8) -> Path:
    treatments = summary["business_questions"]["q3_pre_biologic_treatments"][
        "top_in_not_current_cohort"
    ]
    top = treatments[:top_n]
    labels = [str(item[0]) for item in top]
    values = [int(item[1]) for item in top]
    output_path = out_dir / "top_pre_biologic_treatments_not_current.png"
    save_bar_chart(
        labels=labels,
        values=values,
        title="Top Pre-Biologic Treatments (Not-Current Cohort)",
        xlabel="Treatment",
        ylabel="Mentions",
        output_path=output_path,
        color="#5E8C61",
    )
    return output_path


def chart_referral_steps(summary: Dict, out_dir: Path) -> Path:
    dist_raw = summary["business_questions"]["q4_referral_pathway"][
        "step_count_distribution"
    ]
    ordered = sorted((int(k), int(v)) for k, v in dist_raw.items())
    labels = [str(k) for k, _ in ordered]
    values = [v for _, v in ordered]
    output_path = out_dir / "referral_step_distribution.png"
    save_bar_chart(
        labels=labels,
        values=values,
        title="Referral Step Count Distribution",
        xlabel="Referral Steps",
        ylabel="Patient Count",
        output_path=output_path,
        color="#6C5B7B",
    )
    return output_path


def chart_churn_distribution(summary: Dict, out_dir: Path) -> Path:
    churn = summary["limitations"]["churn_suspected_distribution"]
    labels = ["true", "false", "unknown"]
    values = [int(churn.get(label, 0)) for label in labels]
    output_path = out_dir / "churn_suspected_distribution.png"
    save_bar_chart(
        labels=labels,
        values=values,
        title="Churn-Suspected Distribution",
        xlabel="churn_suspected",
        ylabel="Patient Count",
        output_path=output_path,
        color="#C06C84",
    )
    return output_path


def _dedupe_consecutive(steps: Sequence[str]) -> List[str]:
    if not steps:
        return []
    deduped = [steps[0]]
    for step in steps[1:]:
        if step != deduped[-1]:
            deduped.append(step)
    return deduped


def chart_referral_pathway_sankey(rows: List[Dict], out_dir: Path) -> List[Path]:
    transition_counter: Counter[Tuple[str, str]] = Counter()
    for row in rows:
        raw_steps = row.get("referral_pathway_steps") or []
        steps = [canonicalize_step(step) for step in raw_steps if str(step).strip()]
        steps = _dedupe_consecutive(steps)
        if len(steps) < 2:
            continue
        for source, target in zip(steps, steps[1:]):
            transition_counter[(source, target)] += 1

    if not transition_counter:
        return []

    top_edges = transition_counter.most_common(25)
    nodes: List[str] = []
    for (source, target), _ in top_edges:
        if source not in nodes:
            nodes.append(source)
        if target not in nodes:
            nodes.append(target)

    node_to_idx = {name: idx for idx, name in enumerate(nodes)}
    sources = [node_to_idx[source] for (source, _), _count in top_edges]
    targets = [node_to_idx[target] for (_source, target), _count in top_edges]
    values = [int(count) for (_edge, count) in top_edges]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    label=nodes,
                    pad=18,
                    thickness=18,
                    color="#6C5B7B",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(42,111,151,0.35)",
                ),
            )
        ]
    )
    fig.update_layout(
        title_text="Referral Pathway Sankey (Top Transition Flows)",
        font=dict(size=12),
        margin=dict(l=10, r=10, t=45, b=10),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "referral_pathway_sankey.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    outputs = [html_path]

    png_path = out_dir / "referral_pathway_sankey.png"
    try:
        fig.write_image(str(png_path), width=1200, height=680, scale=2)
        outputs.append(png_path)
    except Exception as e:
        print(f"Could not export Sankey PNG ({type(e).__name__}): {e}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate chart outputs from analysis_summary.json"
    )
    p.add_argument("--summary", default=SUMMARY_PATH_DEFAULT)
    p.add_argument("--extracted", default=EXTRACTED_PATH_DEFAULT)
    p.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = load_summary(args.summary)
    extracted_rows = load_jsonl(args.extracted)
    out_dir = Path(args.out_dir)

    outputs = [
        chart_timing_distribution(summary, out_dir),
        chart_reasons_not_on_biologic(summary, out_dir),
        chart_top_treatments(summary, out_dir),
        chart_referral_steps(summary, out_dir),
        chart_churn_distribution(summary, out_dir),
    ]
    outputs.extend(chart_referral_pathway_sankey(extracted_rows, out_dir))

    print("Generated charts:")
    for path in outputs:
        print(path.as_posix())


if __name__ == "__main__":
    main()
