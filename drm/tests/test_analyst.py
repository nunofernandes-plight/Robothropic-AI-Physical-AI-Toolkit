"""Analyst tests. A fake client stands in for the API: no key, no network."""
import json
import os
from pathlib import Path

import pytest

from drm.backend import analyst, bridge, config
from drm.schemas import (ApprovedProcessModel, Constraints, HumanOversight, ProcessModel, Resolution)

FIXTURE = Path(__file__).parents[1] / "examples" / "optical_coating_line.processmodel.json"
PROCESS = ProcessModel.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def approved():
    return ApprovedProcessModel(
        process=PROCESS, approved_by="N. Fernandes",
        resolutions=[Resolution(question=q, answer="settled") for q in PROCESS.ambiguities])


def entry(stage_id, hil="review_after", suit=3, sev=2, rev="partial"):
    return {"stage_id": stage_id,
            "intervention": {"primary": "perception", "secondary": [], "suitability": suit, "rationale": "r"},
            "integration": {"data_required": ["images"], "interfaces": ["camera"], "latency": "batch",
                            "human_in_loop": hil, "effort": 3, "rationale": "r"},
            "risk": {"failure_modes": [{"description": "f", "severity": sev, "reversibility": rev,
                                        "detectable_before_harm": True}],
                     "regulatory_exposure": 2, "regulatory_note": ""},
            "layer_placement": "slow", "confidence": "medium", "assumptions": []}


def reply(ids=None, **kw):
    ids = ids if ids is not None else [s.id for s in PROCESS.stages]
    return json.dumps({"analyses": [entry(i, **kw) for i in ids]})


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]


class FakeClient:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp(self.outputs.pop(0))


# ---------- prompts ----------
def test_system_prompt_has_anchors_schema_and_constraint_text():
    p = analyst.build_analyst_system_prompt(Constraints(human_oversight=HumanOversight.MANDATORY, notes=["no night shifts"]))
    for phrase in ("suitability (fit for an AI agent)", "regulatory_exposure", "JSON Schema", "MANDATORY", "no night shifts"):
        assert phrase in p
    assert "RECOMMENDED" in analyst.build_analyst_system_prompt(Constraints(human_oversight=HumanOversight.RECOMMENDED))
    assert "No human-oversight constraint" in analyst.build_analyst_system_prompt()


def test_user_prompt_includes_model_and_resolutions_only_when_approved():
    assert "<resolutions>" not in analyst.build_analyst_user_prompt(PROCESS)
    u = analyst.build_analyst_user_prompt(approved())
    assert "<process_model>" in u and "<resolutions>" in u and "settled" in u


def test_provisional_until_approved():
    assert analyst.is_provisional(PROCESS) is True
    assert analyst.is_provisional(approved()) is False


# ---------- parsing and checks ----------
def test_valid_reply_accepted():
    out = analyst.parse_and_check_analysis(reply(), PROCESS)
    assert len(out.analyses) == 10


def test_missing_extra_and_duplicate_stages_rejected():
    with pytest.raises(ValueError):
        analyst.parse_and_check_analysis(reply(ids=list(range(1, 10))), PROCESS)
    with pytest.raises(ValueError):
        analyst.parse_and_check_analysis(reply(ids=list(range(1, 12))), PROCESS)
    with pytest.raises(ValueError):
        analyst.parse_and_check_analysis(reply(ids=[1, 1] + list(range(3, 11))), PROCESS)


def test_scale_out_of_range_rejected():
    with pytest.raises(ValueError):
        analyst.parse_and_check_analysis(reply(suit=6), PROCESS)


def test_mandatory_oversight_rejects_none_but_unspecified_accepts_it():
    mandatory = Constraints(human_oversight=HumanOversight.MANDATORY)
    with pytest.raises(ValueError, match="mandatory"):
        analyst.parse_and_check_analysis(reply(hil="none"), PROCESS, mandatory)
    analyst.parse_and_check_analysis(reply(hil="review_after"), PROCESS, mandatory)
    analyst.parse_and_check_analysis(reply(hil="none"), PROCESS, Constraints())


def test_severe_irreversible_still_needs_a_human_before_action():
    with pytest.raises(ValueError):
        analyst.parse_and_check_analysis(reply(hil="none", sev=5, rev="irreversible"), PROCESS)


# ---------- live path with a stand-in client ----------
def test_analyse_success_and_request_shape():
    client = FakeClient([reply()])
    out = analyst.analyse(PROCESS, client=client)
    assert len(out.analyses) == 10
    call = client.calls[0]
    assert call["model"] == config.MODEL and call["output_config"] == {"effort": config.EFFORT}
    assert "temperature" not in call and "tool_choice" not in call


def test_analyse_retries_then_succeeds_and_gives_up():
    client = FakeClient(["not json", reply()])
    analyst.analyse(PROCESS, client=client)
    assert len(client.calls) == 2 and "validation_error" in client.calls[1]["messages"][0]["content"]
    with pytest.raises(analyst.AnalysisError):
        analyst.analyse(PROCESS, client=FakeClient(["nope", "nope"]), max_retries=1)


# ---------- bridge commands ----------
def test_load_process_detects_approved(tmp_path):
    f = tmp_path / "a.json"
    f.write_text(approved().model_dump_json(), encoding="utf-8")
    assert isinstance(bridge.load_process(str(f)), ApprovedProcessModel)
    assert isinstance(bridge.load_process(str(FIXTURE)), ProcessModel)


def test_bridge_analysis_flow(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    assert bridge.main(["bridge", "analysis-prompt", str(FIXTURE), "--oversight", "mandatory"]) == 0
    assert "MANDATORY" in Path("bridge_analysis_prompt.txt").read_text(encoding="utf-8")

    monkeypatch.setattr(bridge, "read_clipboard", lambda: reply())
    assert bridge.main(["bridge", "analysis-check", str(FIXTURE), "--clipboard", "--oversight", "mandatory"]) == 0
    out = capsys.readouterr().out
    assert "VALID" in out and "PROVISIONAL" in out and "Ranking" in out

    monkeypatch.setattr(bridge, "read_clipboard", lambda: reply(hil="none"))
    assert bridge.main(["bridge", "analysis-check", str(FIXTURE), "--clipboard", "--oversight", "mandatory"]) == 1
    assert "mandatory" in capsys.readouterr().out and os.path.exists("bridge_analysis_retry_prompt.txt")


