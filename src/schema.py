from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_EVIDENCE_WORDS = 25


class ExtractionMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    churn_suspected: Optional[bool] = Field(
        default=None,
        description="True if the transcript seems incomplete due to patient churn.",
    )
    uncertainty_notes: Optional[str] = Field(
        default=None,
        description="Free-text note about ambiguity or missing info.",
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Field names that could not be confidently extracted.",
    )


class PatientExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(description="Patient identifier from source data")

    # Socio-demographic
    age: Optional[int] = Field(default=None, description="Age in years, if stated")
    gender: Optional[str] = Field(
        default=None, description="Gender identity as stated by the patient"
    )
    location: Optional[str] = Field(
        default=None, description="City/region/location if mentioned"
    )
    occupation: Optional[str] = Field(
        default=None, description="Job or role if mentioned"
    )

    # Clinical timeline / treatments
    years_with_crohns: Optional[float] = Field(
        default=None, description="Approx years since Crohn's diagnosis"
    )
    biologic_use: Optional[bool] = Field(
        default=None,
        description=(
            "Whether patient appears to currently be on a biologic."
            " For future intent, use biologic_timing='planned' or 'considering'."
        ),
    )
    biologic_timing: Optional[
        Literal["current", "past", "planned", "considering", "never", "unknown"]
    ] = Field(
        default=None,
        description=(
            "Timeline status for biologics: current use, past use, future plan, "
            "consideration, never used, or unknown."
        ),
    )
    biologic_names: List[str] = Field(
        default_factory=list,
        description="Biologic(s) currently or previously used.",
    )
    planned_biologic_names: List[str] = Field(
        default_factory=list,
        description="Biologic(s) mentioned as planned or intended future treatment.",
    )
    non_biologic_treatments: List[str] = Field(
        default_factory=list, description="Non-biologic treatments tried or discussed"
    )
    reasons_not_on_biologic: List[str] = Field(
        default_factory=list, description="Reasons for not using biologic"
    )

    # Referral pathway
    referral_pathway_steps: List[str] = Field(
        default_factory=list,
        description="Ordered care pathway steps, e.g., GP -> GI -> Specialist",
    )
    referral_steps_count: Optional[int] = Field(
        default=None, description="Number of steps in referral pathway"
    )

    # Evidence & notes
    evidence_quotes: List[str] = Field(
        default_factory=list,
        description=f"Short supporting quotes (<= {MAX_EVIDENCE_WORDS} words each)",
    )
    meta: ExtractionMeta = Field(default_factory=ExtractionMeta)

    @field_validator("evidence_quotes")
    @classmethod
    def _normalize_quotes(cls, quotes: List[str]) -> List[str]:
        normalized: List[str] = []
        for quote in quotes:
            words = quote.split()
            if len(words) > MAX_EVIDENCE_WORDS:
                normalized.append(" ".join(words[:MAX_EVIDENCE_WORDS]))
            else:
                normalized.append(quote)
        return normalized

    @model_validator(mode="after")
    def _enforce_coherence(self) -> "PatientExtraction":
        # Align biologic_use with timeline semantics when possible.
        if self.biologic_timing == "current":
            self.biologic_use = True
        elif self.biologic_timing in {"past", "planned", "considering", "never"}:
            self.biologic_use = False

        # If LLM placed planned biologics in biologic_names, move them.
        if self.biologic_timing in {"planned", "considering"} and self.biologic_names:
            self.planned_biologic_names = sorted(
                set(self.planned_biologic_names + self.biologic_names)
            )
            self.biologic_names = []

        # If currently on a biologic, reasons_not_on_biologic should be empty.
        if self.biologic_use is True and self.reasons_not_on_biologic:
            self.reasons_not_on_biologic = []

        # Auto-populate missing fields for better uncertainty accounting.
        inferred_missing: List[str] = []
        fields_to_track = {
            "age": self.age,
            "gender": self.gender,
            "location": self.location,
            "occupation": self.occupation,
            "years_with_crohns": self.years_with_crohns,
            "biologic_use": self.biologic_use,
            "biologic_timing": self.biologic_timing,
            "biologic_names": self.biologic_names,
            "planned_biologic_names": self.planned_biologic_names,
            "non_biologic_treatments": self.non_biologic_treatments,
            "reasons_not_on_biologic": self.reasons_not_on_biologic,
            "referral_pathway_steps": self.referral_pathway_steps,
            "referral_steps_count": self.referral_steps_count,
        }

        for field_name, value in fields_to_track.items():
            if value is None:
                inferred_missing.append(field_name)
            elif isinstance(value, list) and len(value) == 0:
                inferred_missing.append(field_name)

        merged_missing = sorted(set(self.meta.missing_fields + inferred_missing))
        self.meta.missing_fields = merged_missing
        return self
