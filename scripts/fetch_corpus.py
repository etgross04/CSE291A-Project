"""Download a fixed set of Project Gutenberg books, strip the Gutenberg header/footer, and write
<corpus_dir>/corpus.txt plus configs/corpus_manifest.json (per-book raw + cleaned sha256, combined sha256).

Gutenberg files can change over time, so the manifest hashes are the reference. If a later fetch produces
different hashes, keep the original files and investigate rather than overwriting the manifest.

Usage: python scripts/fetch_corpus.py [--verify]    # --verify: re-hash local files against the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.server import REPO, load_local  # noqa: E402

BOOKS = [  # (gutenberg id, title)
    (1342, "Pride and Prejudice"),
    (2701, "Moby Dick"),
    (98, "A Tale of Two Cities"),
    (84, "Frankenstein"),
    (1661, "The Adventures of Sherlock Holmes"),
    (2600, "War and Peace"),
]
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
MANIFEST = REPO / "configs" / "corpus_manifest.json"
SEP = "\n\n"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def strip_gutenberg(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.startswith("*** START OF"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("*** END OF"))
    return "\n".join(lines[start + 1:end]).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    corpus_dir = Path(load_local().get("corpus_dir", "C:/llm/corpus"))
    raw_dir = corpus_dir / "raw"

    if a.verify:
        m = json.loads(MANIFEST.read_text())
        got = sha((corpus_dir / "corpus.txt").read_bytes())
        print("corpus.txt", "OK" if got == m["corpus_sha256"] else f"MISMATCH {got}")
        sys.exit(0 if got == m["corpus_sha256"] else 1)

    raw_dir.mkdir(parents=True, exist_ok=True)
    books, cleaned = [], []
    with httpx.Client(timeout=120, follow_redirects=True) as c:
        for gid, title in BOOKS:
            p = raw_dir / f"pg{gid}.txt"
            if not p.exists():
                r = c.get(URL.format(id=gid))
                r.raise_for_status()
                p.write_bytes(r.content)
            raw = p.read_bytes()
            text = strip_gutenberg(raw.decode("utf-8-sig"))
            cleaned.append(text)
            books.append({"id": gid, "title": title, "url": URL.format(id=gid), "raw_sha256": sha(raw),
                          "clean_sha256": sha(text.encode()), "clean_chars": len(text)})
            print(f"{gid:>5} {title:<36} {len(text):>9,} chars")

    corpus = SEP.join(cleaned).encode("utf-8")
    (corpus_dir / "corpus.txt").write_bytes(corpus)
    manifest = {"books_in_order": books, "separator": SEP, "corpus_chars": len(corpus.decode()),
                "corpus_sha256": sha(corpus), "path": str(corpus_dir / "corpus.txt")}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\ncorpus.txt {len(corpus):,} bytes sha256={manifest['corpus_sha256']}\nmanifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
