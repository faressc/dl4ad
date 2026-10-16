#!/usr/bin/env python3
"""Find every occurrence of the given dates in the tracked files of the repo.

Each date is searched in the formats used across slides, notebooks and
README (DD.MM.YYYY, D.M.YYYY, ISO, "16 October 2026", "October 16, 2026",
"16. Oktober 2026"). Binary files and slides/assets/ (huge single-line SVGs)
are skipped.

Example:
    python find_dates.py 2026-04-15 2026-04-26 2026-09-30
"""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import date

MONTHS_EN: list[str] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTHS_DE: list[str] = [
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
]


def patterns(d: date) -> list[str]:
    en, de = MONTHS_EN[d.month - 1], MONTHS_DE[d.month - 1]
    return [
        rf"\b0?{d.day}\.0?{d.month}\.{d.year}\b",
        rf"\b{d:%Y-%m-%d}\b",
        rf"\b0?{d.day}\.? ({en}|{de}) {d.year}\b",
        rf"\b{en} 0?{d.day}(st|nd|rd|th)?,? {d.year}\b",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dates",
        nargs="+",
        type=date.fromisoformat,
        help="Dates to search for (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    found_any = False
    for d in args.dates:
        regex = "|".join(f"({p})" for p in patterns(d))
        result = subprocess.run(
            [
                "git",
                "grep",
                "-I",
                "-n",
                "-P",
                regex,
                "--",
                ".",
                ":!slides/assets",
                ":!dist",
                ":!node_modules",
            ],
            capture_output=True,
            text=True,
        )
        hits: list[str] = []
        for line in result.stdout.splitlines():
            location, _, text = line.partition(":")
            line_number, _, text = text.partition(":")
            match = re.search(regex, text)
            if match:
                # Show only a window around the match; slide lines can be very long.
                start = max(match.start() - 80, 0)
                snippet = text[start : match.end() + 80].strip()
                hits.append(f"{location}:{line_number}: {snippet}")
        print(f"{d:%d.%m.%Y}: {len(hits)} hit(s)")
        for hit in hits:
            print(f"    {hit}")
        found_any |= bool(hits)
    raise SystemExit(0 if found_any else 1)


if __name__ == "__main__":
    main()
