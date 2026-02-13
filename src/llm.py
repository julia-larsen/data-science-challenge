from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from litellm import completion

from .schema import PatientExtraction

SYSTEM_PROMPT = (
    "You extract structured data from Crohn's patient interview transcripts.\n"
    "Return ONLY valid JSON that matches the provided schema.\n"
    "Use conservative extraction: never guess facts not stated.\n"
    "Temporal anchor policy:\n"
    "- Interpret all status fields as of the interview moment, not today's date.\n"
    "- Keep current/past/future mentions separate using biologic_timing.\n"
    "Uncertainty policy:\n"
    "- If not stated: use null (scalars) or [] (lists).\n"
    "- If explicitly denied (e.g., 'never used biologic'), encode the semantic value "
    "(e.g., biologic_timing='never') instead of null.\n"
    "- Populate meta.missing_fields for every unknown field.\n"
    "- Set meta.churn_suspected=true only when transcript appears incomplete "
    "(abrupt ending, forgotten details, unresolved journey).\n"
    "- Use meta.uncertainty_notes for contradictions or ambiguity.\n"
    "Biologic timeline policy:\n"
    "- biologic_use means currently on biologic now.\n"
    "- If transcript says future intent (e.g., 'will start', 'about to begin', "
    "'next month'), set biologic_timing='planned' and biologic_use=false.\n"
    "- If actively discussing but undecided, set biologic_timing='considering'.\n"
    "- If prior biologic use but not current, set biologic_timing='past' and "
    "biologic_use=false.\n"
    "- If currently taking/injecting biologic, set biologic_timing='current' and "
    "biologic_use=true.\n"
    "- Respect negations: phrases like 'not on anymore', 'haven't started', "
    "'stopped', or 'never took' must not be mapped to current use.\n"
    "Contradiction policy:\n"
    "- If transcript contains conflicting statements, choose the most conservative "
    "interpretation and document the conflict in meta.uncertainty_notes.\n"
    "Reason normalization policy:\n"
    "- Normalize reasons_not_on_biologic to these labels when applicable: "
    "cost, insurance, needle_fear, side_effect_fear, doctor_advice, "
    "monitoring_burden, access_delay, other.\n"
    "- Use short snake_case labels from this set, not long prose.\n"
    "Referral pathway policy:\n"
    "- Include only explicitly stated care-provider steps.\n"
    "- Keep steps chronological and deduplicate repeated providers.\n"
    "Evidence gating policy:\n"
    "- Only populate a field when supported by transcript evidence.\n"
    "- If unsupported, leave null/[] and include field in meta.missing_fields.\n"
    "Evidence policy:\n"
    "- evidence_quotes must be direct snippets from transcript, max 25 words each.\n"
    "- Provide up to 5 quotes only."
)


def _extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(text[start : end + 1])


def _to_response_dict(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()  # type: ignore[return-value]
    return {}


def _extract_usage(resp_dict: Dict[str, Any]) -> Dict[str, Optional[int]]:
    usage = resp_dict.get("usage")
    if not isinstance(usage, dict):
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    return {
        "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
        "completion_tokens": int(completion_tokens)
        if isinstance(completion_tokens, int)
        else None,
        "total_tokens": int(total_tokens) if isinstance(total_tokens, int) else None,
    }


def extract_with_gemini_with_metrics(
    *,
    patient_id: str,
    transcript: str,
    model: str = "gemini/gemini-2.0-flash",
    timeout_s: float = 60.0,
) -> Tuple[PatientExtraction, Dict[str, Any]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    user_prompt = {
        "patient_id": patient_id,
        "transcript": transcript,
        "schema": PatientExtraction.model_json_schema(),
    }

    response = completion(
        model=model,
        api_key=api_key,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt)},
        ],
        temperature=0.2,
        timeout=timeout_s,
    )

    resp_dict = _to_response_dict(response)
    choices = resp_dict.get("choices")
    first_choice: Dict[str, Any] = {}
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        first_choice = choices[0]

    message = first_choice.get("message", {})
    if not isinstance(message, dict):
        message = {}
    content: Optional[str] = message.get("content")
    if not content:
        raise ValueError("LLM response missing content")

    payload = _extract_json(content)
    extraction = PatientExtraction.model_validate(payload)

    metrics = {
        "provider_response_model": resp_dict.get("model"),
        "requested_model": model,
        "finish_reason": first_choice.get("finish_reason"),
        "usage": _extract_usage(resp_dict),
    }
    return extraction, metrics


def extract_with_gemini(
    *,
    patient_id: str,
    transcript: str,
    model: str = "gemini/gemini-2.0-flash",
    timeout_s: float = 60.0,
) -> PatientExtraction:
    extraction, _ = extract_with_gemini_with_metrics(
        patient_id=patient_id,
        transcript=transcript,
        model=model,
        timeout_s=timeout_s,
    )
    return extraction
