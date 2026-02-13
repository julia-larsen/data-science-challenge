from src.schema import ExtractionMeta, PatientExtraction


def test_future_biologic_intent_maps_to_planned_fields() -> None:
    extraction = PatientExtraction(
        patient_id="P999",
        biologic_timing="planned",
        biologic_names=["Humira"],
        reasons_not_on_biologic=["fear of injections"],
    )

    assert extraction.biologic_use is False
    assert extraction.biologic_names == []
    assert extraction.planned_biologic_names == ["Humira"]
    assert "biologic_timing" not in extraction.meta.missing_fields


def test_ambiguous_payload_autofills_missing_fields() -> None:
    ambiguous_meta = ExtractionMeta.model_validate(
        {"uncertainty_notes": "Patient was vague and forgot medication names."}
    )
    extraction = PatientExtraction(
        patient_id="P998",
        meta=ambiguous_meta,
    )

    assert extraction.meta.uncertainty_notes is not None
    assert "age" in extraction.meta.missing_fields
    assert "biologic_use" in extraction.meta.missing_fields
    assert "non_biologic_treatments" in extraction.meta.missing_fields


def test_long_evidence_quotes_are_truncated() -> None:
    quote = " ".join(["word"] * 40)
    extraction = PatientExtraction(patient_id="P997", evidence_quotes=[quote])

    assert len(extraction.evidence_quotes[0].split()) == 25


def test_negation_like_timeline_forces_not_current() -> None:
    extraction = PatientExtraction(
        patient_id="P996",
        biologic_use=True,
        biologic_timing="past",
        biologic_names=["Remicade"],
    )

    assert extraction.biologic_use is False
    assert extraction.biologic_timing == "past"
    assert extraction.biologic_names == ["Remicade"]


def test_current_use_clears_not_on_biologic_reasons() -> None:
    extraction = PatientExtraction(
        patient_id="P995",
        biologic_timing="current",
        reasons_not_on_biologic=["insurance", "needle_fear"],
    )

    assert extraction.biologic_use is True
    assert extraction.reasons_not_on_biologic == []
