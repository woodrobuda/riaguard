"""Simple command-line entry point for the Python-native Khoros tools."""





from __future__ import annotations
import argparse
import json
from pathlib import Path
from .workspace_runtime import HistoricalWorkspaceParser, WorkspaceExecutor, example_registry
from .phase21_numeric import run_phase21_checks, inventory, corpus_available, ROOT


def cmd_inspect(path: Path) -> int:
    """Read a workspace and print its main building blocks."""
    ws = HistoricalWorkspaceParser().parse(path)
    print(f"Glyphs: {len(ws.glyphs)}")
    print(f"Connections: {len(ws.connections)}")
    print(f"Variables: {len(ws.variables)}")
    for g in ws.glyphs:
        print(f"  {g.ident}: {g.toolbox}:{g.oname}  command={g.command!r}")
    return 0


def cmd_run(path: Path) -> int:
    """Run a workspace with the native operator registry."""
    hws = HistoricalWorkspaceParser().parse(path)
    ws = WorkspaceExecutor(example_registry()).execute(hws)
    print(f"Workspace completed: {path}")
    print(f"Nodes: {len(ws.nodes)}")
    return 0


def cmd_check() -> int:
    """Run the Phase 21 numerical compatibility checks."""
    results = run_phase21_checks()
    print(json.dumps(results, indent=2, default=str))
    skipped = sum(r["status"] == "SKIP" for r in results)
    if skipped:
        print(f"note: {skipped} check(s) skipped: historical corpus not found at {ROOT} (set KHOROS_SOURCE_ROOT)")
    return 0 if all(r["status"] in ("PASS", "SKIP") for r in results) else 1


def cmd_inventory() -> int:
    """Show the recovered historical DATAMANIP test inventory."""
    if not corpus_available():
        print(f"historical corpus not found at {ROOT}; set KHOROS_SOURCE_ROOT to enable the inventory")
        print(json.dumps({"available": False}, indent=2))
        return 0
    summary, _ = inventory()
    print(json.dumps(summary.__dict__, indent=2))
    return 0


def cmd_author(action: str, spec: Path, out: Path | None, do_verify: bool) -> int:
    """Authoring: validate / build (.wk) / explain (markdown) / verify a declarative spec."""
    from . import authoring as A
    try:
        rep = A.validate(spec)
    except Exception as e:  # unreadable spec
        print(f"ERROR: cannot read spec {spec}: {e}"); return 2
    if action == "validate":
        print(rep.summary()); return 0 if rep.ok else 1
    if not rep.ok:
        print(rep.summary()); return 1
    if action == "build":
        target = out or spec.with_suffix(".wk"); target.write_text(A.build_wk(spec, rep)); print(f"wrote {target}"); return 0
    if action == "verify":
        v = A.verify(spec); v.pop("outputs")
        print(json.dumps(v, indent=2, default=str)); return 0 if v["passed"] else 1
    if action == "explain":
        ver = None
        if do_verify:
            ver = A.verify(spec); ver.pop("outputs")
        text = A.explain(spec, rep, ver)
        if out: out.write_text(text); print(f"wrote {out}")
        else: print(text)
        return 0 if ver is None or ver["passed"] else 1
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Python-native Khoros/Cantata compatibility tools")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="show the contents of a workspace")
    p.add_argument("workspace", type=Path)
    p.set_defaults(func=lambda a: cmd_inspect(a.workspace))
    p = sub.add_parser("run", help="run a supported workspace")
    p.add_argument("workspace", type=Path)
    p.set_defaults(func=lambda a: cmd_run(a.workspace))
    sub.add_parser("check", help="run the Phase 21 numerical checks").set_defaults(func=lambda a: cmd_check())
    sub.add_parser("inventory", help="show the historical test inventory").set_defaults(func=lambda a: cmd_inventory())
    p = sub.add_parser("author", help="validate, build, explain or verify a workspace spec (YAML/JSON)")
    p.add_argument("action", choices=["validate", "build", "explain", "verify"])
    p.add_argument("spec", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--verify", action="store_true", help="explain: include a synthetic stream==batch + look-ahead check")
    p.set_defaults(func=lambda a: cmd_author(a.action, a.spec, a.output, a.verify))
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
