"""Print (and optionally save as JSON) the environment facts that go into docs/environment.md and manifests.

Usage: python scripts/check_env.py [--hash-model] [--out env.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.env import environment  # noqa: E402
from runner.server import load_local  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-model", action="store_true", help="compute model sha256 (~10 s, then cached)")
    ap.add_argument("--out")
    a = ap.parse_args()
    env = environment(load_local(), hash_model=a.hash_model)
    print(json.dumps(env, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(env, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
