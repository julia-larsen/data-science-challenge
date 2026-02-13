from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Set

from .llm import extract_with_gemini_with_metrics
from .schema import ExtractionMeta, PatientExtraction

DATA_PATH_DEFAULT = "data/interviews.json"
OUTPUT_PATH_DEFAULT = "data/extracted.jsonl"
METRICS_PATH_DEFAULT = "data/extraction_metrics.jsonl"


def iter_interviews(path: str) -> Iterable[dict]:
    with open(path) as f:
        data = json.load(f)
    for row in data:
        yield row


def load_completed_ids(path: str) -> Set[str]:
    completed: Set[str] = set()
    if not os.path.exists(path):
        return completed
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "patient_id" in obj:
                    completed.add(obj["patient_id"])
            except json.JSONDecodeError:
                continue
    return completed


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_metrics_event(path: str, event: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a") as metrics:
        metrics.write(json.dumps(event))
        metrics.write("\n")


def extract_all(
    *,
    data_path: str,
    output_path: str,
    model: str,
    dry_run: bool,
    max_retries: int,
    base_sleep_s: float,
    min_interval_s: float,
    timeout_s: float,
    limit: Optional[int],
    skip_existing: bool,
    metrics_path: str,
) -> None:
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    completed_ids = load_completed_ids(output_path) if skip_existing else set()
    mode = "a" if (skip_existing and os.path.exists(output_path)) else "w"

    run_started_at = utc_now_iso()
    run_event = {
        "event": "run_started",
        "at": run_started_at,
        "data_path": data_path,
        "output_path": output_path,
        "model": model,
        "dry_run": dry_run,
        "max_retries": max_retries,
        "base_sleep_s": base_sleep_s,
        "min_interval_s": min_interval_s,
        "timeout_s": timeout_s,
        "limit": limit,
        "skip_existing": skip_existing,
    }
    write_metrics_event(metrics_path, run_event)

    written = 0
    skipped_existing = 0
    with open(output_path, mode) as out:
        for row in iter_interviews(data_path):
            patient_id = row["patient_id"]
            transcript = row["interview_transcript"]

            if skip_existing and patient_id in completed_ids:
                skipped_existing += 1
                write_metrics_event(
                    metrics_path,
                    {
                        "event": "patient_skipped_existing",
                        "at": utc_now_iso(),
                        "patient_id": patient_id,
                    },
                )
                continue

            print(f"Processing {patient_id}...", flush=True)
            patient_start = time.perf_counter()
            write_metrics_event(
                metrics_path,
                {
                    "event": "patient_started",
                    "at": utc_now_iso(),
                    "patient_id": patient_id,
                    "dry_run": dry_run,
                },
            )

            llm_metrics: Dict[str, Any] = {}
            attempts = 0
            if dry_run:
                extracted = PatientExtraction(
                    patient_id=patient_id,
                    meta=ExtractionMeta(
                        uncertainty_notes="Dry-run; LLM not invoked.",
                        missing_fields=[
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
                        ],
                    ),
                )
            else:
                while True:
                    try:
                        attempts += 1
                        extracted, llm_metrics = extract_with_gemini_with_metrics(
                            patient_id=patient_id,
                            transcript=transcript,
                            model=model,
                            timeout_s=timeout_s,
                        )
                        break
                    except Exception as e:
                        if attempts > max_retries:
                            elapsed_s = round(time.perf_counter() - patient_start, 3)
                            write_metrics_event(
                                metrics_path,
                                {
                                    "event": "patient_failed",
                                    "at": utc_now_iso(),
                                    "patient_id": patient_id,
                                    "attempts": attempts,
                                    "error_type": type(e).__name__,
                                    "error_message": str(e)[:300],
                                    "elapsed_s": elapsed_s,
                                },
                            )
                            raise

                        msg = str(e)
                        sleep_s = base_sleep_s * (2 ** (attempts - 1))
                        retry_match = re.search(
                            r"retryDelay\"\\s*:\\s*\"(\\d+)s\"", msg
                        )
                        if retry_match:
                            sleep_s = max(sleep_s, float(retry_match.group(1)))

                        print(
                            f"Retrying {patient_id} (attempt {attempts}/{max_retries}) "
                            f"after error: {msg[:200]}... sleeping {sleep_s:.1f}s",
                            flush=True,
                        )
                        write_metrics_event(
                            metrics_path,
                            {
                                "event": "patient_retry",
                                "at": utc_now_iso(),
                                "patient_id": patient_id,
                                "attempt": attempts,
                                "max_retries": max_retries,
                                "error_type": type(e).__name__,
                                "error_message": msg[:300],
                                "sleep_s": round(sleep_s, 3),
                            },
                        )
                        time.sleep(sleep_s)

            out.write(extracted.model_dump_json())
            out.write("\n")
            out.flush()

            elapsed_s = round(time.perf_counter() - patient_start, 3)
            write_metrics_event(
                metrics_path,
                {
                    "event": "patient_succeeded",
                    "at": utc_now_iso(),
                    "patient_id": patient_id,
                    "dry_run": dry_run,
                    "attempts": attempts if not dry_run else 0,
                    "elapsed_s": elapsed_s,
                    "missing_fields_count": len(extracted.meta.missing_fields),
                    "biologic_timing": extracted.biologic_timing,
                    "llm": llm_metrics,
                },
            )

            if not dry_run and min_interval_s > 0:
                time.sleep(min_interval_s)

            written += 1
            if limit is not None and written >= limit:
                print(f"Reached limit {limit}.", flush=True)
                break

    write_metrics_event(
        metrics_path,
        {
            "event": "run_completed",
            "at": utc_now_iso(),
            "written": written,
            "skipped_existing": skipped_existing,
            "dry_run": dry_run,
            "output_path": output_path,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract structured data from transcripts")
    p.add_argument("--data", default=DATA_PATH_DEFAULT)
    p.add_argument("--out", default=OUTPUT_PATH_DEFAULT)
    p.add_argument("--model", default="gemini/gemini-2.0-flash-lite")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--base-sleep-s", type=float, default=2.0)
    p.add_argument("--min-interval-s", type=float, default=3.0)
    p.add_argument("--timeout-s", type=float, default=60.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--metrics-out", default=METRICS_PATH_DEFAULT)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    extract_all(
        data_path=args.data,
        output_path=args.out,
        model=args.model,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        base_sleep_s=args.base_sleep_s,
        min_interval_s=args.min_interval_s,
        timeout_s=args.timeout_s,
        limit=args.limit,
        skip_existing=args.skip_existing,
        metrics_path=args.metrics_out,
    )


if __name__ == "__main__":
    main()
