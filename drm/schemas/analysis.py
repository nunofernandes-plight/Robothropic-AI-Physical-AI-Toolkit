"""Stage 2 output: one StageAnalysis per stage, across the three axes."""
from pydantic import Field, model_validator

from .common import (Confidence, HumanInLoop, InterventionType, LatencyClass,
                     LayerPlacement, Reversibility, Scale, Strict)


class Intervention(Strict):  # Axis 1: where can an agent act?
    primary: InterventionType
    secondary: list[InterventionType] = []
    suitability: Scale  # 1 = poor fit for AI, 5 = strong fit
    rationale: str


class Integration(Strict):  # Axis 2: what does connecting AI here require?
    data_required: list[str]
    interfaces: list[str]
    latency: LatencyClass
    human_in_loop: HumanInLoop
    effort: Scale  # 1 = trivial, 5 = major engineering
    rationale: str


class FailureMode(Strict):
    description: str
    severity: Scale
    reversibility: Reversibility
    detectable_before_harm: bool


class Risk(Strict):  # Axis 3: what can go wrong?
    failure_modes: list[FailureMode]
    regulatory_exposure: Scale
    regulatory_note: str = ""

    @model_validator(mode="after")
    def _need_a_failure_mode(self):
        if not self.failure_modes:
            raise ValueError("at least one failure mode is required")
        return self


class StageAnalysis(Strict):
    stage_id: int
    intervention: Intervention
    integration: Integration
    risk: Risk
    layer_placement: LayerPlacement
    confidence: Confidence
    assumptions: list[str] = []

    @model_validator(mode="after")
    def _engineering_consistency(self):
        # Rule 1: severe + irreversible failure needs a human before action.
        severe_irreversible = any(
            f.severity >= 4 and f.reversibility == Reversibility.IRREVERSIBLE
            for f in self.risk.failure_modes)
        if severe_irreversible and self.integration.human_in_loop in (
                HumanInLoop.NONE, HumanInLoop.REVIEW_AFTER):
            raise ValueError(
                "severe irreversible failure mode requires approve_before or human_only")
        # Rule 2: no AI role cannot also be a strong AI fit.
        if self.intervention.primary == InterventionType.NONE and self.intervention.suitability > 2:
            raise ValueError("intervention 'none' with high suitability is contradictory")
        return self


class AnalystOutput(Strict):
    """What the Analyst returns: exactly one StageAnalysis per stage."""
    analyses: list[StageAnalysis] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_stage_ids(self):
        ids = [a.stage_id for a in self.analyses]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate stage_id in analyses")
        return self


