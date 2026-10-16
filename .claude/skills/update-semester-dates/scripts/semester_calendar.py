#!/usr/bin/env python3
"""Compute the weekly lecture slots of a TU Berlin semester.

Prints every occurrence of the lecture weekday inside the Vorlesungszeit,
marking the ones that fall on a Berlin public holiday or inside a
lecture-free break, plus the Sunday after each slot (deadlines in this
course are on Sundays). Optionally prints the ISIS project-group selection
window for a given proposal deadline.

Berlin's recurring public holidays are computed here; one-off holidays
(e.g. 08.05.2025) must be found by web research and passed via
--extra-holiday.

Example (WS 2026/27, Fridays 14:15-15:45):
    python semester_calendar.py \
        --lecture-start 2026-10-12 --lecture-end 2027-02-13 \
        --break 2026-12-21:2027-01-02 --weekday fri \
        --lecture-end-time 15:45 --proposal-deadline 2026-11-22
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

WEEKDAYS: dict[str, int] = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}
WEEKDAY_NAMES: list[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    weekday_offset = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * weekday_offset) // 451
    month, day = divmod(h + weekday_offset - 7 * m + 114, 31)
    return date(year, month, day + 1)


def berlin_holidays(year: int) -> dict[date, str]:
    """Recurring gesetzliche Feiertage in Berlin (state law since 2019)."""
    easter = easter_sunday(year)
    return {
        date(year, 1, 1): "Neujahr",
        date(year, 3, 8): "Internationaler Frauentag",
        easter - timedelta(days=2): "Karfreitag",
        easter + timedelta(days=1): "Ostermontag",
        date(year, 5, 1): "Tag der Arbeit",
        easter + timedelta(days=39): "Christi Himmelfahrt",
        easter + timedelta(days=50): "Pfingstmontag",
        date(year, 10, 3): "Tag der Deutschen Einheit",
        date(year, 12, 25): "1. Weihnachtstag",
        date(year, 12, 26): "2. Weihnachtstag",
    }


def fmt(d: date) -> str:
    return f"{d:%d.%m.%Y} ({WEEKDAY_NAMES[d.weekday()][:3]})"


def parse_range(text: str) -> tuple[date, date]:
    start, end = text.split(":")
    return date.fromisoformat(start), date.fromisoformat(end)


def parse_extra(text: str) -> tuple[date, str]:
    day, _, name = text.partition(":")
    return date.fromisoformat(day), name or "extra holiday"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lecture-start",
        type=date.fromisoformat,
        required=True,
        help="First day of the Vorlesungszeit (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--lecture-end",
        type=date.fromisoformat,
        required=True,
        help="Last day of the Vorlesungszeit (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--weekday", choices=WEEKDAYS, required=True, help="Weekday of the lecture"
    )
    parser.add_argument(
        "--first-lecture",
        type=date.fromisoformat,
        help="First lecture if it is not the first weekday of the Vorlesungszeit",
    )
    parser.add_argument(
        "--break",
        dest="breaks",
        type=parse_range,
        action="append",
        default=[],
        help="Lecture-free range START:END (repeatable)",
    )
    parser.add_argument(
        "--extra-holiday",
        type=parse_extra,
        action="append",
        default=[],
        help="One-off holiday YYYY-MM-DD:Name (repeatable)",
    )
    parser.add_argument(
        "--proposal-deadline",
        type=date.fromisoformat,
        help="Project proposal deadline (a Sunday) to compute the ISIS group selection window",
    )
    parser.add_argument(
        "--lecture-end-time",
        default="end of lecture",
        help="Lecture end time, used for the group selection opening (e.g. 15:45)",
    )
    parser.add_argument(
        "--moses-deadline",
        type=date.fromisoformat,
        help="MOSES registration deadline; prints the latest Sunday that is at least 2 days before it",
    )
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=date.today(),
        help="Reference date for marking past dates (default: today)",
    )
    args = parser.parse_args()

    holidays: dict[date, str] = {}
    for year in range(args.lecture_start.year, args.lecture_end.year + 1):
        holidays.update(berlin_holidays(year))
    holidays.update(dict(args.extra_holiday))

    def skip_reason(d: date) -> str | None:
        reasons: list[str] = []
        if d in holidays:
            reasons.append(holidays[d])
        for start, end in args.breaks:
            if start <= d <= end:
                reasons.append(f"lecture-free {start:%d.%m.}-{end:%d.%m.%Y}")
        return ", ".join(reasons) or None

    def past(d: date) -> str:
        return "  (PAST)" if d < args.today else "  (TODAY)" if d == args.today else ""

    print("Berlin public holidays inside the Vorlesungszeit:")
    for d in sorted(holidays):
        if args.lecture_start <= d <= args.lecture_end:
            print(f"  {fmt(d)}  {holidays[d]}")
    print()

    target = WEEKDAYS[args.weekday]
    day = args.first_lecture or args.lecture_start
    day += timedelta(days=(target - day.weekday()) % 7)
    slots: list[date] = []
    print(f"{WEEKDAY_NAMES[target]} lecture slots (following Sunday in brackets):")
    while day <= args.lecture_end:
        reason = skip_reason(day)
        sunday = day + timedelta(days=6 - day.weekday())
        if reason:
            print(f"   --  {fmt(day)}  SKIPPED: {reason}")
        else:
            slots.append(day)
            print(f"  {len(slots):>3}  {fmt(day)}  [Sun {sunday:%d.%m.%Y}]{past(day)}")
        day += timedelta(days=7)
    print(f"\n{len(slots)} usable lecture slots.")

    if args.moses_deadline:
        # "At least 2 days before" = calendar-day difference >= 2 (Sun -> Tue is OK, Sun -> Mon is not).
        latest = args.moses_deadline - timedelta(days=2)
        latest -= timedelta(days=(latest.weekday() - 6) % 7)
        gap = (args.moses_deadline - latest).days
        print(
            f"\nMOSES registration deadline: {fmt(args.moses_deadline)}{past(args.moses_deadline)}"
        )
        print(
            f"  Latest valid course application deadline (Sunday): {fmt(latest)}, gap {gap} days{past(latest)}"
        )

    if args.proposal_deadline:
        deadline = args.proposal_deadline
        if deadline.weekday() != 6:
            print(f"\nWARNING: proposal deadline {fmt(deadline)} is not a Sunday.")
        # Week before the deadline week: Monday..Sunday ending 7 days before the deadline's Sunday.
        deadline_sunday = deadline + timedelta(days=6 - deadline.weekday())
        week_before_monday = deadline_sunday - timedelta(days=13)
        opening = week_before_monday + timedelta(days=target)
        print("\nISIS project group selection window:")
        reason = skip_reason(opening)
        if reason:
            print(
                f"  WARNING: {fmt(opening)} has no lecture ({reason}); ask which day to open instead."
            )
        print(f"  Opens:  {fmt(opening)}, {args.lecture_end_time}{past(opening)}")
        print(f"  Closes: {fmt(deadline)}, 23:59{past(deadline)}")


if __name__ == "__main__":
    main()
