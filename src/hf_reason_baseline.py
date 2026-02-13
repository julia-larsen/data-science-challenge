from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .analyze import load_jsonl, normalize_reason

INTERVIEWS_PATH_DEFAULT = "data/interviews.json"
EXTRACTED_PATH_DEFAULT = "data/extracted.jsonl"
OUTPUT_PATH_DEFAULT = "data/hf_reason_baseline.json"
MODEL_DEFAULT = "valhalla/distilbart-mnli-12-1"
SWEEP_THRESHOLDS_DEFAULT = "0.45,0.55,0.65"

REASON_LABELS: List[str] = [
    "insurance",
    "cost",
    "side_effect_fear",
    "needle_fear",
    "doctor_advice",
    "monitoring_burden",
    "access_delay",
    "other",
]


def load_interviews(path: str) -> Dict[str, str]:
    with open(path) as f:
        rows = json.load(f)
    return {
        str(row["patient_id"]): str(row["interview_transcript"])
        for row in rows
        if "patient_id" in row and "interview_transcript" in row
    }


def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def parse_thresholds(raw: str) -> List[float]:
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    # Keep deterministic order and unique values.
    unique_sorted = sorted(set(values))
    return unique_sorted


def load_zero_shot_pipeline(model: str) -> Any:
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "transformers is not installed in the active interpreter. "
            "Activate .venv and run: pip install -r requirements.txt"
        ) from e

    return pipeline(
        "zero-shot-classification",
        model=model,
        device=-1,
    )


def predict_scores(
    *,
    rows: List[Dict[str, Any]],
    interview_map: Dict[str, str],
    classifier: Any,
    max_chars: int,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for row in rows:
        patient_id = str(row["patient_id"])
        transcript = interview_map.get(patient_id, "")[:max_chars]
        result = classifier(
            transcript,
            candidate_labels=REASON_LABELS,
            multi_label=True,
            hypothesis_template="The patient is not on biologic because of {}.",
        )

        scores_by_label = {
            label: float(score)
            for label, score in zip(result["labels"], result["scores"])
        }
        gemini_labels = sorted(
            {
                normalize_reason(reason)
                for reason in row.get("reasons_not_on_biologic", [])
            }
        )

        scored.append(
            {
                "patient_id": patient_id,
                "biologic_timing": row.get("biologic_timing"),
                "gemini_labels": gemini_labels,
                "scores_by_label": scores_by_label,
            }
        )
    return scored


def evaluate_threshold(
    *,
    scored_rows: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    patient_results: List[Dict[str, Any]] = []
    hf_reason_counter: Counter[str] = Counter()
    gemini_reason_counter: Counter[str] = Counter()
    jaccard_scores: List[float] = []
    exact_matches = 0

    label_metrics: Dict[str, Dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0} for label in REASON_LABELS
    }

    for row in scored_rows:
        scores_by_label = row["scores_by_label"]
        hf_labels = sorted(
            [label for label in REASON_LABELS if scores_by_label.get(label, 0.0) >= threshold],
            key=lambda x: scores_by_label.get(x, 0.0),
            reverse=True,
        )
        gemini_labels = list(row["gemini_labels"])

        if set(hf_labels) == set(gemini_labels):
            exact_matches += 1

        score = jaccard_similarity(gemini_labels, hf_labels)
        jaccard_scores.append(score)

        for label in hf_labels:
            hf_reason_counter[label] += 1
        for label in gemini_labels:
            gemini_reason_counter[label] += 1

        gemini_set = set(gemini_labels)
        hf_set = set(hf_labels)
        for label in REASON_LABELS:
            if label in gemini_set and label in hf_set:
                label_metrics[label]["tp"] += 1
            elif label in hf_set and label not in gemini_set:
                label_metrics[label]["fp"] += 1
            elif label in gemini_set and label not in hf_set:
                label_metrics[label]["fn"] += 1

        patient_results.append(
            {
                "patient_id": row["patient_id"],
                "biologic_timing": row["biologic_timing"],
                "gemini_reasons": gemini_labels,
                "hf_predicted_reasons": hf_labels,
                "hf_scores_by_label": {
                    label: round(float(scores_by_label.get(label, 0.0)), 4)
                    for label in REASON_LABELS
                },
                "agreement_jaccard": round(score, 4),
            }
        )

    per_label_metrics: Dict[str, Dict[str, float]] = {}
    for label, c in label_metrics.items():
        tp = c["tp"]
        fp = c["fp"]
        fn = c["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_label_metrics[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    n = len(scored_rows)
    avg_labels = round((sum(len(r["hf_predicted_reasons"]) for r in patient_results) / n), 4) if n else 0.0

    return {
        "threshold": threshold,
        "exact_set_match_count": exact_matches,
        "exact_set_match_rate": round(exact_matches / n, 4) if n else 0.0,
        "mean_jaccard_similarity": round(sum(jaccard_scores) / n, 4) if n else 0.0,
        "avg_hf_labels_per_patient": avg_labels,
        "aggregate_counts": {
            "hf_reason_counts": dict(hf_reason_counter),
            "gemini_reason_counts": dict(gemini_reason_counter),
        },
        "per_label": per_label_metrics,
        "patient_results": patient_results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Hugging Face zero-shot baseline for reasons_not_on_biologic."
    )
    p.add_argument("--interviews", default=INTERVIEWS_PATH_DEFAULT)
    p.add_argument("--extracted", default=EXTRACTED_PATH_DEFAULT)
    p.add_argument("--out", default=OUTPUT_PATH_DEFAULT)
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--sweep-thresholds", default=SWEEP_THRESHOLDS_DEFAULT)
    p.add_argument("--max-chars", type=int, default=2200)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    interview_map = load_interviews(args.interviews)
    extracted_rows = load_jsonl(args.extracted)
    not_current = [row for row in extracted_rows if row.get("biologic_timing") != "current"]

    classifier = load_zero_shot_pipeline(args.model)
    scored_rows = predict_scores(
        rows=not_current,
        interview_map=interview_map,
        classifier=classifier,
        max_chars=args.max_chars,
    )

    selected = evaluate_threshold(scored_rows=scored_rows, threshold=args.threshold)

    sweep_thresholds = parse_thresholds(args.sweep_thresholds)
    if args.threshold not in sweep_thresholds:
        sweep_thresholds = sorted(set(sweep_thresholds + [args.threshold]))

    sweep_rows = []
    for threshold in sweep_thresholds:
        e = evaluate_threshold(scored_rows=scored_rows, threshold=threshold)
        sweep_rows.append(
            {
                "threshold": threshold,
                "exact_set_match_rate": e["exact_set_match_rate"],
                "mean_jaccard_similarity": e["mean_jaccard_similarity"],
                "avg_hf_labels_per_patient": e["avg_hf_labels_per_patient"],
            }
        )

    summary = {
        "config": {
            "model": args.model,
            "threshold": args.threshold,
            "sweep_thresholds": sweep_thresholds,
            "max_chars": args.max_chars,
            "cohort_definition": "biologic_timing != 'current'",
            "candidate_reason_labels": REASON_LABELS,
        },
        "dataset": {
            "not_current_records": len(not_current),
            "extracted_source": args.extracted,
            "interviews_source": args.interviews,
        },
        "agreement_with_gemini": {
            "exact_set_match_count": selected["exact_set_match_count"],
            "exact_set_match_rate": selected["exact_set_match_rate"],
            "mean_jaccard_similarity": selected["mean_jaccard_similarity"],
            "per_label": selected["per_label"],
        },
        "aggregate_counts": selected["aggregate_counts"],
        "threshold_sweep": sweep_rows,
        "patient_results": selected["patient_results"],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Not-current records processed: {len(not_current)}")
    print(
        "Exact set agreement with Gemini: "
        f"{summary['agreement_with_gemini']['exact_set_match_count']}/{len(not_current)}"
    )
    print(
        "Mean Jaccard similarity: "
        f"{summary['agreement_with_gemini']['mean_jaccard_similarity']}"
    )
    print("Threshold sweep:")
    for row in summary["threshold_sweep"]:
        print(
            f"  t={row['threshold']}: "
            f"exact={row['exact_set_match_rate']}, "
            f"jaccard={row['mean_jaccard_similarity']}, "
            f"avg_labels={row['avg_hf_labels_per_patient']}"
        )
    print(f"Wrote baseline output: {args.out}")


if __name__ == "__main__":
    main()
