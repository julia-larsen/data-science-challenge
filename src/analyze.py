from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

INPUT_PATH_DEFAULT = "data/extracted.jsonl"
OUTPUT_PATH_DEFAULT = "data/analysis_summary.json"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, list) and len(value) == 0)


def missing_rate(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    missing = sum(1 for row in rows if is_missing(row.get(field)))
    total = len(rows)
    return {
        "missing_count": missing,
        "total_count": total,
        "missing_rate": round(missing / total, 4) if total else 0.0,
    }


def normalize_reason(reason: str) -> str:
    r = reason.strip().lower()
    if not r:
        return "other"
    if "insurance" in r:
        return "insurance"
    if "cost" in r or "financial" in r or "expensive" in r or "afford" in r:
        return "cost"
    if "side effect" in r or "infection risk" in r or "risk" in r:
        return "side_effect_fear"
    if "needle" in r or "inject" in r:
        return "needle_fear"
    if "doctor" in r or "physician" in r or "specialist advised" in r:
        return "doctor_advice"
    if "monitor" in r or "blood test" in r or "lab" in r:
        return "monitoring_burden"
    if "wait" in r or "delay" in r or "access" in r:
        return "access_delay"
    # Keep known normalized labels as-is.
    if r in {
        "insurance",
        "cost",
        "side_effect_fear",
        "needle_fear",
        "doctor_advice",
        "monitoring_burden",
        "access_delay",
        "other",
    }:
        return r
    return "other"


def normalize_treatment(treatment: str) -> str:
    t = " ".join(treatment.strip().lower().split())
    if not t:
        return t
    mapping = {
        "mesalamine suppositories": "mesalamine",
        "oral mesalamine": "mesalamine",
        "prednisone taper": "prednisone",
        "6-mercaptopurine": "6-mp",
    }
    return mapping.get(t, t)


def canonicalize_step(step: str) -> str:
    s = step.strip().lower()
    if not s:
        return "Unknown"
    if (
        "general practitioner" in s
        or "primary care" in s
        or "family doctor" in s
        or "pcp" in s
        or s == "gp"
    ):
        return "GP/Primary Care"
    if "gastro" in s or s == "gi" or "gi " in s or " gi" in s:
        return "Gastroenterologist"
    if "emergency" in s or "er" == s or "er " in s or " er" in s:
        return "Emergency Department"
    if "rheumatologist" in s:
        return "Rheumatologist"
    if "endocrinologist" in s:
        return "Endocrinologist"
    if "specialist" in s:
        return "Specialist"
    if "doctor" in s:
        return "Doctor"
    return step.strip()


def referral_step_count(row: Dict[str, Any]) -> Optional[int]:
    explicit = row.get("referral_steps_count")
    if explicit is not None:
        return int(explicit)
    steps = row.get("referral_pathway_steps") or []
    if steps:
        return len(steps)
    return None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    fields = [
        "age",
        "gender",
        "location",
        "occupation",
        "years_with_crohns",
        "biologic_use",
        "biologic_timing",
        "biologic_names",
        "planned_biologic_names",
        "non_biologic_treatments",
        "reasons_not_on_biologic",
        "referral_pathway_steps",
        "referral_steps_count",
    ]

    missingness = {field: missing_rate(rows, field) for field in fields}

    consistency_issues: List[Dict[str, str]] = []
    for row in rows:
        patient_id = str(row.get("patient_id"))
        biologic_use = row.get("biologic_use")
        biologic_timing = row.get("biologic_timing")
        reasons = row.get("reasons_not_on_biologic") or []
        if biologic_timing == "current" and biologic_use is not True:
            consistency_issues.append(
                {
                    "patient_id": patient_id,
                    "issue": "biologic_timing=current but biologic_use!=true",
                }
            )
        if (
            biologic_timing in {"past", "planned", "considering", "never"}
            and biologic_use is not False
        ):
            consistency_issues.append(
                {
                    "patient_id": patient_id,
                    "issue": f"biologic_timing={biologic_timing} but biologic_use!=false",
                }
            )
        if biologic_use is True and len(reasons) > 0:
            consistency_issues.append(
                {
                    "patient_id": patient_id,
                    "issue": "biologic_use=true with non-empty reasons_not_on_biologic",
                }
            )

    current_count = sum(1 for row in rows if row.get("biologic_timing") == "current")
    known_timing_rows = [
        row for row in rows if row.get("biologic_timing") not in {None, "unknown"}
    ]
    known_timing_count = len(known_timing_rows)

    not_current_rows = [row for row in rows if row.get("biologic_timing") != "current"]

    reason_counter: Counter[str] = Counter()
    for row in not_current_rows:
        for reason in row.get("reasons_not_on_biologic") or []:
            reason_counter[normalize_reason(reason)] += 1

    treatment_counter_not_current: Counter[str] = Counter()
    treatment_counter_all: Counter[str] = Counter()
    for row in rows:
        for treatment in row.get("non_biologic_treatments") or []:
            normalized = normalize_treatment(treatment)
            if normalized:
                treatment_counter_all[normalized] += 1
    for row in not_current_rows:
        for treatment in row.get("non_biologic_treatments") or []:
            normalized = normalize_treatment(treatment)
            if normalized:
                treatment_counter_not_current[normalized] += 1

    step_counts = [
        count
        for count in (referral_step_count(row) for row in rows)
        if count is not None
    ]
    step_counter = Counter(step_counts)

    pathway_counter: Counter[str] = Counter()
    for row in rows:
        steps = row.get("referral_pathway_steps") or []
        if not steps:
            continue
        canonical_steps = [canonicalize_step(step) for step in steps]
        pathway_counter[" -> ".join(canonical_steps)] += 1

    churn_counter = Counter(row.get("meta", {}).get("churn_suspected") for row in rows)
    missing_fields_sizes = [
        len(row.get("meta", {}).get("missing_fields", [])) for row in rows
    ]

    summary = {
        "dataset": {
            "records": total,
            "source": INPUT_PATH_DEFAULT,
        },
        "assumptions": {
            "biologic_on_treatment_definition": "biologic_timing == 'current'",
            "unknown_timing_handling": "excluded from known-denominator sensitivity calculation",
            "not_on_biologic_cohort_definition": "all rows where biologic_timing != 'current'",
        },
        "qa": {
            "missingness": missingness,
            "timing_distribution": dict(
                Counter(row.get("biologic_timing") for row in rows)
            ),
            "consistency_issue_count": len(consistency_issues),
            "consistency_issues": consistency_issues,
        },
        "business_questions": {
            "q1_biologic_prevalence": {
                "current_count": current_count,
                "total_count": total,
                "percent_of_all": round((current_count / total) * 100, 2)
                if total
                else 0.0,
                "known_timing_count": known_timing_count,
                "percent_of_known_timing": round(
                    (current_count / known_timing_count) * 100, 2
                )
                if known_timing_count
                else 0.0,
            },
            "q2_reasons_not_on_biologic": {
                "not_current_count": len(not_current_rows),
                "reason_counts": dict(reason_counter),
                "reason_ranked": reason_counter.most_common(),
            },
            "q3_pre_biologic_treatments": {
                "top_in_not_current_cohort": treatment_counter_not_current.most_common(
                    15
                ),
                "top_in_all_patients": treatment_counter_all.most_common(15),
            },
            "q4_referral_pathway": {
                "records_with_known_steps": len(step_counts),
                "step_count_distribution": dict(step_counter),
                "median_step_count": statistics.median(step_counts)
                if step_counts
                else None,
                "most_common_pathways": pathway_counter.most_common(10),
            },
        },
        "limitations": {
            "churn_suspected_distribution": {
                "true": churn_counter.get(True, 0),
                "false": churn_counter.get(False, 0),
                "unknown": churn_counter.get(None, 0),
            },
            "missing_fields_per_record": {
                "average": round(sum(missing_fields_sizes) / total, 2)
                if total
                else 0.0,
                "median": statistics.median(missing_fields_sizes)
                if missing_fields_sizes
                else 0.0,
                "max": max(missing_fields_sizes) if missing_fields_sizes else 0,
                "min": min(missing_fields_sizes) if missing_fields_sizes else 0,
            },
        },
    }
    return summary


def write_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze extracted Crohn's interview data and produce summary metrics."
    )
    parser.add_argument("--in", dest="input_path", default=INPUT_PATH_DEFAULT)
    parser.add_argument("--out", dest="output_path", default=OUTPUT_PATH_DEFAULT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = load_jsonl(args.input_path)
    summary = summarize(rows)
    write_json(args.output_path, summary)

    q1 = summary["business_questions"]["q1_biologic_prevalence"]
    print(f"Records: {summary['dataset']['records']}")
    print(
        "Biologic current prevalence: "
        f"{q1['current_count']}/{q1['total_count']} ({q1['percent_of_all']}%)"
    )
    print(f"Wrote summary to: {args.output_path}")


if __name__ == "__main__":
    main()
