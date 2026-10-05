"""HTTP API tests: verdicts, replay, conflict, invalid model, health."""

import importlib
import json
import os
import tempfile
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_STORE", str(tmp_path / "evidence.json"))
    monkeypatch.setenv("HEALTH_ACK", "alive")
    from app import main
    importlib.reload(main)
    return TestClient(main.app)


MODEL = {
    "audit_id": "API-1",
    "locations": ["armed", "cooling", "open", "done"],
    "clocks": ["x", "y"],
    "initial_location": "armed",
    "final_locations": ["done"],
    "transitions": [
        {"id": "t_cooldown", "source": "armed", "target": "cooling",
         "event": "cmd",
         "guards": [{"clock": "x", "lower": 0, "upper": 4}],
         "resets": ["y"]},
        {"id": "t_open", "source": "armed", "target": "open", "event": "cmd",
         "guards": [{"clock": "x", "lower": 5, "upper": 10}],
         "resets": ["y"]},
        {"id": "t_ack_c", "source": "cooling", "target": "done",
         "event": "ack",
         "guards": [{"clock": "y", "lower": 1, "upper": 3}], "resets": []},
        {"id": "t_ack_o", "source": "open", "target": "done", "event": "ack",
         "guards": [{"clock": "y", "lower": 1, "upper": 3}], "resets": []},
    ],
}


def payload(events):
    return {"model": MODEL, "events": events}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_gap_submission_rejected(client):
    r = client.post("/api/reviews", json=payload([
        {"event": "cmd", "relative_lower": 4, "relative_upper": 6},
        {"event": "ack", "relative_lower": 2, "relative_upper": 2}]))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "rejected"
    assert body["earliest_event_index"] == 0
    assert body["stored"]["fingerprint"]
    assert len(body["steps"]) == 1


def test_freezing_submission(client):
    r = client.post("/api/reviews", json=payload([
        {"event": "cmd", "relative_lower": 5, "relative_upper": 10},
        {"event": "ack", "relative_lower": 1, "relative_upper": 3}]))
    assert r.status_code == 200
    assert r.json()["status"] == "frozen"


def test_semantic_equivalent_retransmission_replays(client):
    p = payload([{"event": "cmd", "relative_lower": 5,
                  "relative_upper": 10}])
    r1 = client.post("/api/reviews", json=p)
    b1 = r1.json()
    assert b1["status"] == "rejected"  # not final after one event
    # reorder JSON keys + whitespace -> semantically identical bytes content
    p2 = {"events": list(reversed(p["events"])),
          "model": dict(reversed(list(p["model"].items())))}
    # NOTE events order must stay identical; rebuild canonically instead
    p2 = {"model": {k: MODEL[k] for k in reversed(list(MODEL))},
          "events": p["events"]}
    r2 = client.post("/api/reviews", json=p2)
    b2 = r2.json()
    assert b2["status"] == "rejected"
    assert b2["replay"]["semantically_equivalent_retransmission"] is True
    assert b2["replay"]["original_fingerprint"] == b1["stored"]["fingerprint"]
    # same verdict/evidence content
    assert b2["reason"] == b1["reason"]
    assert b2["earliest_event_index"] == b1["earliest_event_index"]


def test_same_id_different_content_conflicts_and_keeps_original(client):
    p = payload([{"event": "cmd", "relative_lower": 5,
                  "relative_upper": 10}])
    b1 = client.post("/api/reviews", json=p).json()
    p2 = payload([{"event": "cmd", "relative_lower": 0,
                   "relative_upper": 4}])
    p2["model"] = {**MODEL}
    r = client.post("/api/reviews", json=p2)
    assert r.status_code == 409
    b2 = r.json()
    assert b2["status"] == "conflict"
    assert b2["conflict"]["original_fingerprint"] == b1["stored"]["fingerprint"]
    assert b2["conflict"]["incoming_fingerprint"] != \
        b1["stored"]["fingerprint"]
    assert b2["conflict"]["original_evidence"]["status"] == b1["status"]
    # GET still returns the original retained evidence
    g = client.get("/api/reviews/API-1").json()
    assert g["status"] == b1["status"]


def test_invalid_overlap_model_reported(client):
    bad = {**MODEL, "transitions": [
        dict(t) for t in MODEL["transitions"]]}
    bad["transitions"][0]["guards"][0]["upper"] = 5  # [0,5] vs [5,10]
    r = client.post("/api/reviews", json={"model": bad, "events": [
        {"event": "cmd", "relative_lower": 5, "relative_upper": 5}]})
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "invalid_model"
    assert body["error"]["details"]["code"] == "overlapping_guards"


ZERO_MODEL = {
    "audit_id": "ZERO-E400",
    "locations": ["s", "f"],
    "clocks": ["x"],
    "initial_location": "s",
    "final_locations": ["f"],
    "transitions": [
        {"id": "t_e", "source": "s", "target": "f", "event": "e",
         "guards": [{"clock": "x", "lower": 0, "upper": 0}],
         "resets": []},
    ],
}


def test_tiny_positive_sci_notation_window_rejected_not_frozen(client):
    """A raw JSON numeric literal 1e-400 is a strictly positive rational.

    The zero-width guard x in [0,0] must therefore reject the capture, and
    the value must survive JSON parsing exactly instead of underflowing to
    float 0.0 (which previously produced a false ``frozen`` verdict).
    """
    # Build the request with *numeric* 1e-400 literals exactly as a client
    # would submit them (json.dumps would re-encode a Python float as 0.0).
    raw = (
        '{"model": ' + json.dumps(ZERO_MODEL) + ', "events": [{"event": "e", '
        '"relative_lower": 1e-400, "relative_upper": 1e-400}]}')
    assert "1e-400" in raw  # literal really is a JSON number, not a string
    r = client.post("/api/reviews", content=raw,
                    headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "rejected", body
    assert body["earliest_event_index"] == 0

    # the exact positive window is preserved in the audit evidence
    win = body["steps"][0]["relative_window"]
    lo = Fraction(win["lower"]["numerator"], win["lower"]["denominator"])
    assert lo == Fraction(1, 10 ** 400)
    assert lo > 0
    assert win["lower"]["decimal"] == "1e-400"

    # witness: the only possible clock value is x == 1e-400, which misses
    # the zero guard; it must not be rendered/sampled as 0.
    x = body["failure"]["clock_values"]["x"]
    xv = Fraction(x["numerator"], x["denominator"])
    assert xv == Fraction(1, 10 ** 400)
    assert x["decimal"] == "1e-400"
    assert body["failure"]["kind"] == "uncovered_time"
    names = {g["transition"] for g in
             body["failure"]["blocking_guards"]}
    assert names == {"t_e"}


def test_exact_zero_window_still_freezes(client):
    """Regression guard: the true zero window keeps the existing semantics."""
    r = client.post("/api/reviews", json={
        "model": ZERO_MODEL,
        "events": [{"event": "e", "relative_lower": 0,
                    "relative_upper": 0}]})
    assert r.status_code == 200
    assert r.json()["status"] == "frozen"


def test_jsonx_preserves_tiny_literals_exactly():
    from app.jsonx import loads
    parsed = loads('{"a": 1e-400, "b": 0, "c": 2.5, "d": 7, "e": "1e-400"}')
    assert parsed["a"] == "1e-400"
    assert Fraction(parsed["a"]) == Fraction(1, 10 ** 400)
    assert parsed["b"] == 0 and isinstance(parsed["b"], int)
    assert parsed["c"] == "2.5"
    assert parsed["d"] == 7 and isinstance(parsed["d"], int)
    assert parsed["e"] == "1e-400"  # strings stay strings
    with pytest.raises(json.JSONDecodeError):
        loads('{"a": NaN}')


def test_bad_json_and_missing_fields(client):
    r = client.post("/api/reviews", content="not json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400
    r = client.post("/api/reviews", json={"events": []})
    assert r.status_code == 422


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "联锁捕获复核" in r.text
