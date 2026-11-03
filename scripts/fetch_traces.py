"""Download the public traces pinned in configs/trace_manifest.json into <traces_dir> and verify sha256.

Usage: python scripts/fetch_traces.py [--verify]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.server import load_local  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="only check existing files")
    a = ap.parse_args()
    root = Path(load_local()["traces_dir"])
    ok = True
    for name, e in json.loads((REPO / "configs" / "trace_manifest.json").read_text(encoding="utf-8"))["traces"].items():
        path = root / e["path"]
        if not a.verify and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            r = httpx.get(e["url"], follow_redirects=True, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
        got = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        good = got == e["sha256"]
        ok &= good
        print(f"{name:24s} {'OK' if good else 'MISMATCH'}  {path}  {got}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
