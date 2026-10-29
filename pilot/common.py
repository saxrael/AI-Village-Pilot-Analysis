"""Shared helpers: dataset access, timestamps, model families, chat loading."""
import datetime as dt
import gzip
import json
import os
import re

try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads

REPO = "aidigestorg/ai-village"

_FAMILIES = (
    ("Anthropic", re.compile(r"claude|opus|sonnet|haiku|fable")),
    ("OpenAI", re.compile(r"gpt|^o[134]\b")),
    ("Google", re.compile(r"gemini|gemma")),
    ("DeepSeek", re.compile(r"deepseek")),
    ("Zhipu", re.compile(r"glm")),
    ("Moonshot", re.compile(r"kimi|fine-tuned|leader")),
    ("xAI", re.compile(r"grok")),
    ("Meta", re.compile(r"muse|llama")),
)


def family(name):
    n = (name or "").lower()
    for label, rx in _FAMILIES:
        if rx.search(n):
            return label
    return "Other"


def get_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_API_KEY")
    if tok:
        return tok
    for loader in ("kaggle", "colab"):
        try:
            if loader == "kaggle":
                from kaggle_secrets import UserSecretsClient
                return UserSecretsClient().get_secret("HF_TOKEN")
            from google.colab import userdata
            return userdata.get("HF_TOKEN")
        except Exception:  # noqa: BLE001
            continue
    return None


def open_stream(name, local_dir=None, token=None):
    """Text stream of a gzipped table, from a local folder or streamed from the Hub."""
    if local_dir:
        return gzip.open(os.path.join(local_dir, name), "rt", encoding="utf-8")
    from huggingface_hub import HfFileSystem
    raw = HfFileSystem(token=token).open(f"datasets/{REPO}/{name}", "rb")
    return gzip.open(raw, "rt", encoding="utf-8")


def read_rows(name, local_dir=None, token=None):
    with open_stream(name, local_dir, token) as f:
        for line in f:
            yield loads(line)


def require_token(local_dir):
    if local_dir:
        return None
    token = get_token()
    if not token:
        raise SystemExit("No token found. Set HF_TOKEN in the environment or the notebook secrets.")
    return token


def parse_time(s):
    s = str(s)
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def as_text(x):
    if not x:
        return ""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def tool_result_texts(obj):
    """Text of every tool_result block in a Claude Code message."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except ValueError:
            return
    if isinstance(obj, dict):
        if obj.get("type") == "tool_result":
            c = obj.get("content")
            if isinstance(c, str):
                yield c
            elif isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        yield item["text"]
                    elif isinstance(item, str):
                        yield item
            return
        for v in obj.values():
            yield from tool_result_texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from tool_result_texts(v)


def agent_names(local_dir=None, token=None):
    return {r["id"]: r.get("name") for r in read_rows("agents.jsonl.gz", local_dir, token)}


def load_chat(local_dir=None, token=None):
    """(names, messages). Messages are (time, id, speaker, text), time-sorted; speaker is 'HUMAN' for non-agents."""
    names = agent_names(local_dir, token)
    msgs = []
    for r in read_rows("chat_messages.jsonl.gz", local_dir, token):
        if not r.get("content"):
            continue
        is_agent = r.get("speaker_type") == "agent" and r.get("agent_speaker_id")
        msgs.append((parse_time(r["created_at"]), str(r["id"]),
                     r["agent_speaker_id"] if is_agent else "HUMAN", r["content"]))
    msgs.sort(key=lambda m: (m[0], m[1]))
    return names, msgs


def week_of(t):
    d = t.date()
    return (d - dt.timedelta(days=d.weekday())).isoformat()
