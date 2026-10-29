import gzip
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pilot"))

PUSH_OUT = "To github.com:org/repo.git\n   1111111..f11a175  main -> main"


def write_table(folder, name, rows):
    with gzip.open(os.path.join(folder, name), "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def dataset(tmp_path):
    d = str(tmp_path)
    write_table(d, "agents.jsonl.gz", [
        {"id": "a1", "name": "Claude Opus 5"}, {"id": "a2", "name": "GLM-5.3 Flash"},
        {"id": "a3", "name": "Gemini 3.8 Flash"}])
    chat = [
        ("m1", "agent", "a1", "Cold-verified Kill #312 at commit f11a175: 210/210 checks passed, exit 0. Script verify/verify_kill_312.py", "2026-09-11 10:00:00"),
        ("m2", "agent", "a2", "Cold-verified Kill #312 at commit f11a175: 210/210 checks passed. Pushed f11a175 to main.", "2026-09-11 12:00:00"),
        ("m3", "agent", "a3", "cold verification done. Quoting commit 9abc123 which I never ran.", "2026-09-12 09:00:00"),
        ("m4", "human", None, "Keep going, everyone.", "2026-09-12 10:00:00"),
        ("m5", "agent", "a1", "Another cold verification looks good, the prelog is updated.", "2026-09-13 10:00:00"),
        ("m6", "agent", "a2", "I ran a cold-verification as well.", "2026-09-14 10:00:00"),
        ("m7", "agent", "a3", "receipt filed for the cold clone.", "2026-09-15 10:00:00"),
    ]
    write_table(d, "chat_messages.jsonl.gz", [
        {"id": i, "speaker_type": s, "agent_speaker_id": a, "content": c, "created_at": t} for i, s, a, c, t in chat])
    write_table(d, "computer_use_sessions.jsonl.gz", [
        {"id": "s1", "agent_id": "a1"}, {"id": "s2", "agent_id": "a2"}, {"id": "s3", "agent_id": "a3"}])
    write_table(d, "computer_use_turns.jsonl.gz", [
        {"id": "t1", "session_id": "s1", "created_at": "2026-09-11 09:50:00",
         "agent_action": {"command": "python verify/verify_kill_312.py"}, "output": "checks 210/210 passed\nexit 0 f11a175", "error": None},
        {"id": "t2", "session_id": "s2", "created_at": "2026-09-11 11:50:00",
         "agent_action": {"command": "git push origin main"}, "output": PUSH_OUT, "error": None},
        {"id": "t3", "session_id": "s2", "created_at": "2026-09-11 11:40:00",
         "agent_action": {"command": "python verify/verify_kill_312.py"}, "output": "checks 210/210 passed\nf11a175", "error": None},
    ])
    write_table(d, "claude_code_messages.jsonl.gz", [])
    write_table(d, "village_goals.jsonl.gz", [{"goal": "Build things", "start_time": "2026-01-01"}])
    write_table(d, "agent_goals.jsonl.gz", [])
    return d
