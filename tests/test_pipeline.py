import csv
import json
import os
import subprocess
import sys

PILOT = os.path.join(os.path.dirname(__file__), "..", "pilot")


def run(script, dataset, out, *extra):
    cmd = [sys.executable, os.path.join(PILOT, script), "--local-dir", dataset, "--out-dir", str(out), *extra]
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(out)).stdout


def table(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_identifier_chain(dataset, tmp_path):
    run("identifier_census.py", dataset, tmp_path)
    claims = [json.loads(line) for line in open(tmp_path / "identifier_claims.jsonl", encoding="utf-8")]
    assert any(c["token"] == "f11a175" for c in claims)
    run("identifier_provenance.py", dataset, tmp_path, "--claims", str(tmp_path / "identifier_claims.jsonl"))
    labels = {}
    for line in open(tmp_path / "identifier_provenance.jsonl", encoding="utf-8"):
        r = json.loads(line)
        labels[(r["token"], r["agent"] if "agent" in r else None)] = r["label"]
    assert "own_tool_before" in labels.values()
    run("push_check.py", dataset, tmp_path, "--provenance", str(tmp_path / "identifier_provenance.jsonl"))
    assert os.path.exists(tmp_path / "push_check_by_agent.csv")


def test_term_diffusion(dataset, tmp_path):
    run("term_diffusion.py", dataset, tmp_path, "--min-agents", "2", "--min-families", "2", "--min-msgs", "2")
    rows = {r["term"]: r for r in table(tmp_path / "term_diffusion_terms.csv")}
    assert rows["cold_verification"]["origin_agent"] == "Claude Opus 5"
    assert int(rows["cold_verification"]["agents"]) == 3


def test_origin_check(dataset, tmp_path):
    run("origin_check.py", dataset, tmp_path)
    rows = {r["term"]: r for r in table(tmp_path / "origin_check.csv")}
    assert rows["cold_verification"]["human_used_before_agents"] == "False"


def test_claim_evidence(dataset, tmp_path):
    run("claim_evidence.py", dataset, tmp_path)
    by_agent = {r["agent"]: r for r in table(tmp_path / "claim_evidence_by_agent.csv")}
    assert int(by_agent["Claude Opus 5"]["verified_strong"]) == 1
