import datetime as dt

import common
import identifier_census
import identifier_provenance
import push_check
import screenshot_check
import stats
import term_diffusion
import claim_evidence


def test_wilson_interval():
    lo, hi = stats.wilson_ci(4, 100)
    assert round(lo, 3) == 0.016 and round(hi, 3) == 0.098


def test_wilson_empty():
    lo, hi = stats.wilson_ci(0, 0)
    assert lo != lo and hi != hi


def test_family():
    assert common.family("Claude Opus 5") == "Anthropic"
    assert common.family("GLM-5.3 Flash") == "Zhipu"
    assert common.family("o3") == "OpenAI"
    assert common.family("Grok 4.5") == "xAI"
    assert common.family("something new") == "Other"


def test_parse_time():
    assert common.parse_time("2026-09-11 10:00:00") == dt.datetime(2026, 9, 11, 10, tzinfo=dt.timezone.utc)
    assert common.parse_time("2026-09-11T10:00:00Z").hour == 10


def test_tool_result_texts():
    obj = [{"type": "tool_result", "content": [{"text": "a"}, "b"]}, {"x": {"type": "tool_result", "content": "c"}}]
    assert list(common.tool_result_texts(obj)) == ["a", "b", "c"]


def test_find_identifiers():
    found = identifier_census.find_identifiers("pushed commit f11a175 and the word deadbeef, not 1234567")
    tokens = [t for t, _, _ in found]
    assert tokens == ["f11a175"]


def test_label_claim():
    t = dt.datetime(2026, 1, 1, 12, tzinfo=dt.timezone.utc)
    tol = dt.timedelta(minutes=5)
    before, after = t - dt.timedelta(hours=1), t + dt.timedelta(hours=1)
    assert identifier_provenance.label_claim(t, "a", {"a": before}, tol) == "own_tool_before"
    assert identifier_provenance.label_claim(t, "a", {"b": before}, tol) == "other_tool_before"
    assert identifier_provenance.label_claim(t, "a", {"a": after}, tol) == "own_tool_after_only"
    assert identifier_provenance.label_claim(t, "a", {}, tol) == "none"


def test_push_evidence():
    ok, new_ref, failed = push_check.push_evidence("To github.com:o/r.git\n   1111111..f11a175  main -> main")
    assert ok == ["f11a175"] and not failed
    assert push_check.push_evidence("git status clean")[0] == []
    assert push_check.hash_match("f11a175", "f11a1759abc")


def test_find_token():
    assert screenshot_check.find_token("commit f11a1759 done", "f11a1759abcd") == "exact"
    assert screenshot_check.find_token("commit fl1a1759 done", "f11a1759abcd") == "exact"
    assert screenshot_check.find_token("nothing here", "f11a1759abcd") is None


def test_term_tokens_skip_hashes_and_urls():
    t = term_diffusion.tokens("cold verification at https://github.com/x deadbeef")
    assert "verification" in t and "deadbeef" not in t and "github" not in t


def test_seed_patterns():
    rx = term_diffusion.SEEDS
    import re
    assert re.search(rx["cold_verification"], "Cold-verified it", re.I)
    assert not re.search(rx["cold_verification"], "a cold day", re.I)
    assert re.search(rx["cold_verification"], "cold verification passed", re.I)
    assert re.search(rx["prelog"], "the prelog entry", re.I) and not re.search(rx["prelog"], "prologue", re.I)


def test_extract_claims():
    t = dt.datetime(2026, 9, 11, tzinfo=dt.timezone.utc)
    msgs = [(t, "1", "a1", "Cold verification done: 210/210 checks passed at f11a175 via verify_kill_312.py"),
            (t, "2", "HUMAN", "cold verification")]
    claims = claim_evidence.extract_claims(msgs, {"a1": "Claude Opus 5"}, 20)
    assert len(claims) == 1
    c = claims[0]
    assert c["fracs"] == ["210/210"] and c["heads"] == ["f11a175"] and c["scripts"] == ["verify_kill_312.py"]
