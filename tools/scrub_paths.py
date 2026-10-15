#!/usr/bin/env python
"""Strip this machine's absolute paths and user name out of the tracked text files.

The user name is derived at runtime from the environment, so this script never
contains the literal string it is removing.

Two complications this handles:

* Markdown code spans contain *escaped* separators, so a Windows path can appear
  as ``C:\\Users\\<name>/.cache\\torch`` -- a mix of doubled backslashes and
  forward slashes.  A plain substring replace misses those; a separator-agnostic
  regex does not.  (This is exactly what defeated the first attempt.)
* PowerShell 5.1 writes redirected output as GBK on this locale, so byte-level
  decode/re-encode must follow the file's actual encoding, or the logs corrupt.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # <root>\tools -> <root>
USER = Path.home().name                                # e.g. the Windows account name

SEP = r"[\\/]+"

# The mojibake this user name turns into when its UTF-8 bytes are decoded as GBK.
#
# Derived rather than hardcoded: a hardcoded version had a wrong code point and
# silently matched nothing.  The decode is also done *with the following
# separator included*, because GBK pairs bytes across the boundary -- the
# separator byte gets absorbed into the final mojibake character
# (9 name bytes + 1 separator byte -> 5 GBK characters, not 4).  Variants are
# emitted longest-first so the widest match wins.
_moji_variants: list[str] = []
for _tail in ("\\\\", "\\", "/", ""):
    try:
        _m = (USER + _tail).encode("utf-8").decode("gbk", errors="ignore")
    except Exception:  # pragma: no cover - locale dependent
        continue
    if _m and _m not in _moji_variants:
        _moji_variants.append(_m)
_moji_variants.sort(key=len, reverse=True)

# Longest / most specific first.
RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"C:" + SEP + r"Users" + SEP + re.escape(USER) + SEP + r"Documents"
            + SEP + r"deepseek-harness" + SEP + r"default-workspace" + SEP + r"3dgs",
            re.IGNORECASE,
        ),
        "<work>",
    ),
    (
        re.compile(r"C:" + SEP + r"Users" + SEP + re.escape(USER), re.IGNORECASE),
        "<home>",
    ),
]

for _m in _moji_variants:
    RULES.append((re.compile(re.escape(_m)), "<mojibake>"))

# Any remaining standalone mention of the account name.
RULES.append((re.compile(re.escape(USER)), "<user>"))

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".ply", ".bin", ".pth", ".zip", ".ico"}
ENCODINGS = ("utf-8", "gbk")


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True)
    if out.returncode != 0:
        print("git ls-files failed:", out.stderr.decode("utf-8", "replace"), file=sys.stderr)
        sys.exit(1)
    for line in out.stdout.decode("utf-8", "replace").splitlines():
        if line.strip():
            yield ROOT / line


def decode(data: bytes):
    """Return (text, encoding) for the first encoding that decodes cleanly."""
    for enc in ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return None, None


def main() -> int:
    files_changed = 0
    total = 0
    for path in tracked_files():
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        text, enc = decode(data)
        if text is None:
            continue

        hits = 0
        for pattern, repl in RULES:
            text, n = pattern.subn(repl, text)
            hits += n

        if hits:
            path.write_bytes(text.encode(enc))
            print(f"  {hits:4d}  {path.relative_to(ROOT)}  [{enc}]")
            files_changed += 1
            total += hits

    print(f"\n{files_changed} file(s) changed, {total} replacement(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
