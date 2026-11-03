#!/usr/bin/env python3
"""CI lint for docs/ALERT_WORDING.md: user-facing alert text must be descriptive, not prescriptive.

Scans (1) string literals in alert-generating modules (docstrings and comments are skipped,
because they are developer documentation, not alert text) and (2) every string value in the
sample alert JSONL. Exit code 1 on any banned phrase.

The wordlist mirrors ALERT_WORDING.md rules 1-3. If you change the guide, change BANNED too.
"""
from __future__ import annotations
import ast, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Rule 1: actions. Rule 2: advice verbs / urgency. Rule 3: predictions / promises.
BANNED = [
    "buy", "sell", "hold", "exit", "add", "trim", "rotate into", "go long", "go short",
    "take profit",
    "should", "recommend", "consider buying", "act now", "opportunity", "don't miss",
    "must act",
    "will rise", "guaranteed", "safe", "can't lose",
]
PATTERNS = [(w, re.compile(r"(?<![A-Za-z_])" + re.escape(w).replace(r"\ ", r"\s+") + r"(?![A-Za-z_])", re.I))
            for w in BANNED]

# Alert text sources. Add new alert-producing modules here.
PY_SOURCES = ["src/riaguard/explain.py"]
JSONL_SOURCES = ["docs/reports/sample_alerts.jsonl"]
# JSON keys whose values are machine identifiers, not prose shown to readers.
JSON_SKIP_KEYS = {"config_hash", "ts", "logged_at", "engine_version"}


def find_banned(text: str) -> list[str]:
    return [w for w, p in PATTERNS if p.search(text)]


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def scan_python(src: str, name: str = "<src>") -> list[str]:
    tree = ast.parse(src); skip = _docstring_ids(tree); out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            for w in find_banned(node.value):
                out.append(f"{name}:{node.lineno}: '{w}' in {node.value!r}")
    return out


def _strings(obj, key=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in JSON_SKIP_KEYS: yield from _strings(v, k)
    elif isinstance(obj, list):
        for v in obj: yield from _strings(v, key)
    elif isinstance(obj, str):
        yield obj


def scan_jsonl(text: str, name: str = "<jsonl>") -> list[str]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip(): continue
        for s in _strings(json.loads(line)):
            for w in find_banned(s):
                out.append(f"{name}:{n}: '{w}' in {s!r}")
    return out


def main(root: Path = ROOT) -> int:
    problems = []
    for rel in PY_SOURCES:
        p = root / rel
        if p.exists(): problems += scan_python(p.read_text(encoding="utf-8"), rel)
    for rel in JSONL_SOURCES:
        p = root / rel
        if p.exists(): problems += scan_jsonl(p.read_text(encoding="utf-8"), rel)
    for msg in problems:
        print(f"::error::{msg} (prescriptive wording, see docs/ALERT_WORDING.md)")
    if not problems: print("alert wording OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
