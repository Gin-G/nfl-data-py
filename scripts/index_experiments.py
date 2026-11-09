#!/usr/bin/env python3
"""Regenerate the index at the top of EXPERIMENTS.md.

The ledger is ~6,000 lines and growing, which makes it the most valuable thing
in the repo and the least likely to be read. The index exists so an avenue that
has already been measured and abandoned can be spotted without opening the
file — several good-sounding ideas in here are already recorded as failures.

Idempotent: it replaces an existing "## Index" section rather than adding a
second one. Run it after adding an entry.

    python scripts/index_experiments.py [--check]

`--check` exits non-zero if the index is stale, for a pre-commit hook or CI.
"""
import argparse
import pathlib
import re
import sys

LEDGER = pathlib.Path(__file__).resolve().parent.parent / "EXPERIMENTS.md"

# Words the entries already use for their own verdicts. Deliberately
# conservative: an unmarked entry costs a click, a WRONGLY marked one could
# send someone past a result that does not say what the tag claims.
CLOSED = re.compile(
    r"\bnegative\b|all negative|did not help|no signal|\bnull\b|not a property|"
    r"\binconclusive\b|\breverted\b|\brevert\b|is a wash|worse than", re.I)
SHIPPED = re.compile(r"\bshipped\b|is the default|beats naive|\bwired\b", re.I)

#: How many lines of an entry to read when classifying it.
SCAN_LINES = 40


def slug(text: str) -> str:
    return re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", text.lower()).strip())


def build(lines: list[str]) -> list[str]:
    entries = [(i, l[3:].strip()) for i, l in enumerate(lines)
               if l.startswith("## ") and l[3:].strip() != "Index"]
    out = [
        "## Index",
        "",
        "Auto-generated from the `##` headings; regenerate with",
        "`python scripts/index_experiments.py` after adding an entry. **closed**",
        "marks an avenue measured and abandoned — read it before proposing the",
        "idea again.",
        "",
    ]
    for n, (i, title) in enumerate(entries):
        end = entries[n + 1][0] if n + 1 < len(entries) else len(lines)
        body = "\n".join(lines[i:end][:SCAN_LINES])
        tag = ("closed" if CLOSED.search(body) else
               "shipped" if SHIPPED.search(body) else "")
        out.append(f"- [{title}](#{slug(title)})" + (f" — **{tag}**" if tag else ""))
    out.append("")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the index is out of date")
    args = parser.parse_args()

    lines = LEDGER.read_text().split("\n")
    # Strip any existing index: from its heading to the next "## ".
    start = next((i for i, l in enumerate(lines) if l.strip() == "## Index"), None)
    if start is not None:
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].startswith("## ")), len(lines))
        lines = lines[:start] + lines[end:]

    first = next((i for i, l in enumerate(lines) if l.startswith("## ")), len(lines))
    updated = lines[:first] + build(lines) + lines[first:]
    new_text = "\n".join(updated)

    if args.check:
        if new_text != LEDGER.read_text():
            print("EXPERIMENTS.md index is stale; run "
                  "python scripts/index_experiments.py", file=sys.stderr)
            return 1
        print("index is current")
        return 0

    LEDGER.write_text(new_text)
    print(f"indexed {sum(1 for l in updated if l.startswith('- ['))} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
