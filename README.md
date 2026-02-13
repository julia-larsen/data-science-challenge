# mama health - Data Scientist Challenge Solution

## Executive Summary

- Current biologic prevalence: `36/50` (`72.0%`), or `73.47%` on known-timing records.
- Main barriers for not-on-biologic cohort (`n=14`): `insurance`, `cost`, `side_effect_fear`.
- Typical referral depth (where inferable): median `2` steps.
- Churn/incompleteness is material (`8` churn-suspected, high missingness in referral/location fields), so findings are directional.

## Reproducibility

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run extraction (Gemini)

```bash
set -a && source .env && set +a
.venv/bin/python -m src.extract --out data/extracted.jsonl --model gemini/gemini-2.0-flash-lite --metrics-out data/extraction_metrics.jsonl
```

### 3. Run analysis

```bash
.venv/bin/python -m src.analyze --in data/extracted.jsonl --out data/analysis_summary.json
```

### 4. Generate charts

```bash
.venv/bin/python -m src.plot_analysis --summary data/analysis_summary.json --extracted data/extracted.jsonl --out-dir data/figures
```

### 5. Run Hugging Face OS-model baseline (optional)

```bash
.venv/bin/python -m src.hf_reason_baseline --interviews data/interviews.json --extracted data/extracted.jsonl --out data/hf_reason_baseline.json --model valhalla/distilbart-mnli-12-1 --threshold 0.55 --sweep-thresholds 0.45,0.55,0.65
```

Primary outputs:
- `data/extracted.jsonl`
- `data/extraction_metrics.jsonl` (generated locally; git-ignored)
- `data/analysis_summary.json`
- `data/hf_reason_baseline.json`
- `data/figures/*.png`
- `data/figures/referral_pathway_sankey.html`

Run timestamp for results below: February 13, 2026.

## Pre-Analysis QA And Assumptions

Data QA checks were run before analysis:
- Records analyzed: 50
- Schema consistency issues found: 0
- Biologic timing distribution: `current=36`, `considering=10`, `planned=1`, `past=1`, `never=1`, `unknown=1`

Important missingness:
- `location`: 98% missing
- `referral_steps_count`: 66% missing
- `referral_pathway_steps`: 40% missing
- `gender`: 12% missing

Assumptions used for business calculations:
- "On biologic" is defined as `biologic_timing == "current"`.
- Sensitivity view excludes `biologic_timing == "unknown"` from denominator.
- "Not on biologic" cohort includes all records where `biologic_timing != "current"`.

## Business Questions

### 1. What percentage of patients appear to be on a biologic treatment?

- `36/50` are currently on biologic treatment (`72.0%`).
- Sensitivity on known-timing records: `36/49` (`73.47%`).

### 2. For patients not on a biologic, what are the primary reasons?

Not-on-biologic cohort size: `14` patients.

Reason counts (non-mutually-exclusive):
- `insurance`: 5 (`35.71%` of not-current cohort)
- `cost`: 4 (`28.57%`)
- `side_effect_fear`: 4 (`28.57%`)
- `needle_fear`: 1 (`7.14%`)
- `doctor_advice`: 1 (`7.14%`)

Interpretation:
- Access barriers (insurance + cost) are the strongest observed blockers.
- Safety concern (side-effect fear) is similarly prominent.

### 3. What other treatments are commonly discussed/tried before biologic?

Top treatments in not-on-biologic cohort:
- `mesalamine` (15 mentions)
- `prednisone` (11)
- `budesonide` (5)
- `methotrexate` (4)
- `sulfasalazine` (3)
- `azathioprine` (3)

Interpretation:
- Patients commonly cycle through anti-inflammatory and immunomodulator options before biologic uptake.

### 4. Typical referral pathway and number of steps from GP to specialist

Referral step coverage:
- 30 records had enough data to infer step count.
- Median steps: `2`

Step distribution:
- 1 step: 11 records
- 2 steps: 10 records
- 3 steps: 8 records
- 4 steps: 1 record

Most common normalized pathways (top observed):
- `Gastroenterologist` (7)
- `Emergency Department -> Gastroenterologist` (3)
- `Doctor` (3)
- `GP/Primary Care -> Gastroenterologist` (2)
- `Doctor -> Gastroenterologist` (2)

Interpretation:
- A two-step pathway is typical when pathway detail is present.
- Many transcripts are compressed/incomplete and only mention one provider, which likely understates true referral complexity.

## Data Limitations (Churn and Incompleteness)

Churn metadata:
- `churn_suspected=true`: 8
- `churn_suspected=false`: 27
- `churn_suspected=unknown`: 15

Missingness burden:
- Average missing fields per patient: `4.38`
- Median missing fields per patient: `4`

Impact on interpretation:
- Referral and geography insights are least reliable because pathway/location fields are often absent.
- For not-on-biologic reasons, absence of reason text can reflect missing narrative rather than no barrier.
- Incomplete interviews can bias estimates toward clearer and more complete journeys.

Stakeholder communication guidance:
- Treat percentages as directional for this sample, not population estimates.
- Prioritize decisions that remain robust under missing-data sensitivity checks.
- Collect follow-up data on access barriers and referral detail before commercial commitments.

## Methodology

1. LLM extraction:
- Used `litellm` with Gemini (`gemini/gemini-2.0-flash-lite`) to convert transcripts to structured JSON.
- Prompt was hardened for uncertainty, negation, contradiction handling, and biologic time-frame semantics (`current/past/planned/considering/never/unknown`).

2. Schema and validation:
- Pydantic schema includes timeline-specific biologic fields (`biologic_timing`, `planned_biologic_names`).
- Coherence validators enforce consistency between timing and current-use flags.
- Evidence quotes are clipped to max 25 words.

3. Analysis:
- `src/analyze.py` performs QA checks, computes business metrics, and writes `data/analysis_summary.json`.
- Reason labels and treatment names are lightly normalized for aggregate counts.

4. Testing:
- Unit tests in `tests/test_schema.py` cover future intent, ambiguity/missing-field handling, quote-length enforcement, and timing/use consistency.

## Optional Engineering Extras Implemented

### 1. Sankey Diagram For Referral Pathways

- Added in `src/plot_analysis.py`
- Uses transition counts between normalized referral steps
- Outputs:
  - `data/figures/referral_pathway_sankey.png`
  - `data/figures/referral_pathway_sankey.html`

### 2. Dockerized Workflow

- Added `Dockerfile` and `.dockerignore`

Build:

```bash
docker build -t mama-health-challenge .
```

Run analysis inside container:

```bash
docker run --rm -v "$PWD/data:/app/data" mama-health-challenge
```

Run extraction inside container (with API key):

```bash
docker run --rm \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -v "$PWD/data:/app/data" \
  mama-health-challenge \
  python -m src.extract --out data/extracted.jsonl --model gemini/gemini-2.0-flash-lite --metrics-out data/extraction_metrics.jsonl
```

### 3. LLM Extraction Observability

- Added structured run/patient telemetry in `src/extract.py`
- Metrics are appended as JSONL events to `data/extraction_metrics.jsonl`
- Captures:
  - run start/completion metadata
  - per-patient start/success/failure
  - retry events with error type/message and backoff
  - elapsed time, missing-fields count, biologic timing
  - provider response model, finish reason, token usage (when available)

### 4. Hugging Face OS-Model Baseline

- Added `src/hf_reason_baseline.py` using Hugging Face zero-shot classification (`valhalla/distilbart-mnli-12-1`)
- Cohort analyzed: patients with `biologic_timing != "current"` (14 records)
- Labels: `insurance`, `cost`, `side_effect_fear`, `needle_fear`, `doctor_advice`, `monitoring_burden`, `access_delay`, `other`
- Output: `data/hf_reason_baseline.json`

Observed agreement with Gemini-extracted reasons:
- Exact set match: `2/14` (`14.29%`)
- Mean Jaccard similarity: `0.2756`

Threshold sweep (same model/data):

| Threshold | Exact Set Match Rate | Mean Jaccard | Avg HF Labels/Patient |
|---|---:|---:|---:|
| 0.45 | 0.1429 | 0.2787 | 4.7857 |
| 0.55 | 0.1429 | 0.2756 | 3.2143 |
| 0.65 | 0.1429 | 0.1994 | 1.8571 |

Interpretation:
- This baseline is useful as an independent open-source check, but over-predicts labels in this setting.
- The Gemini structured extraction remains the primary signal; HF baseline is a robustness sanity check.

## Visualizations

### Biologic Timing Distribution

![Biologic timing distribution](data/figures/biologic_timing_distribution.png)

### Reasons Not On Biologic

![Reasons not on biologic](data/figures/reasons_not_on_biologic.png)

### Top Pre-Biologic Treatments (Not-Current Cohort)

![Top pre-biologic treatments](data/figures/top_pre_biologic_treatments_not_current.png)

### Referral Step Distribution

![Referral step distribution](data/figures/referral_step_distribution.png)

### Referral Pathway Sankey

![Referral pathway sankey](data/figures/referral_pathway_sankey.png)

[Open interactive Sankey HTML](data/figures/referral_pathway_sankey.html)

### Churn Suspected Distribution

![Churn suspected distribution](data/figures/churn_suspected_distribution.png)
