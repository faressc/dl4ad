---
name: update-semester-dates
description: Roll the DL4AD course over to the next TU Berlin semester. Research the official Vorlesungszeit and Berlin public holidays, collect the lecture weekday/time/room, MOSES deadline and Zoom link, propose a new schedule (lectures, course application deadline, project proposal deadline & group selection, project pitches, project presentations, final submission), apply it everywhere in the slides, then rewrite the ISIS course description and give the ISIS project-group opening/closing times. Use this whenever Fares mentions the next/new semester, Wintersemester/Sommersemester, WS/SS/WiSe/SoSe, updating dates or deadlines in the slides, the "Dates and Deadlines" slide, the ISIS description, or preparing the course for a new term, even if he only asks about one of these dates.
---

# Update semester dates

This course (Deep Learning for Audio Data, TU Berlin) runs every semester. Each rollover touches the same handful of dates across the slides and the ISIS course page. The work is mostly careful calendar arithmetic plus a few facts only Fares knows. So the workflow is: research → ask → propose → (approval) → apply → ISIS → group window.

Work through the phases in order. The schedule proposal in phase 4 is a checkpoint: do not edit any dates before Fares has agreed to it.

Two bundled scripts do the error-prone parts (run them from the repo root):
- `scripts/semester_calendar.py`: lecture slots, skipped days, Sundays, the MOSES rule, the ISIS group window, and past dates relative to today.
- `scripts/find_dates.py`: finds a list of dates in every format across the tracked files.

## Phase 1: Inventory the current state

Run `git status` first, so that pre-existing uncommitted changes don't get mixed into your summary later.

Read the "Dates and Deadlines" slide in `slides/00-introduction.md`. Extract the ordered list of rows (date, topic, whether it is a highlighted deadline row). That row order is the course's schedule template. The new semester should keep it unless Fares says otherwise. The slide is two `<table>`s side by side; note where the split currently is.

Then find every other place a schedule date lives:

```bash
python3 -I .claude/skills/update-semester-dates/scripts/find_dates.py <every old date as YYYY-MM-DD>
git grep -n "Project Proposal Deadline Reminder"   # which deck currently holds the reminder slide
git grep -nE "14:00 - 18:00|14:15"                  # old time strings
```

Typical locations, all in `slides/00-introduction.md`: the Dates and Deadlines table; Course Application (application deadline, MOSES enrollment deadline); Graded Deliverables (presentation date, git repository deadline and project paper deadline, both = final submission); Projects (proposal deadline). The "Project Proposal Deadline Reminder" slide has no date, but its *position* matters (phase 4). Avoid broad grep patterns of your own on `slides/assets/`, since it contains huge single-line SVGs that flood the output.

Work out which semester the slides currently describe and which one comes next.

## Phase 2: Research

Use web search, restricted to `tu.berlin` (`allowed_domains`). Unrestricted searches mostly return other Berlin universities' calendars.

1. **Vorlesungszeit and lecture-free periods.** The primary source is the *Amtliches Mitteilungsblatt* notice of the Akademischer Senat, "Festsetzung der Vorlesungszeiten für das akademische Jahr YYYY/YY". It covers WS + SS and usually appears around October of the preceding year (2026/27 was in Nr. 38 of 23.10.2025). Download the PDF with `curl` and read it with `pdftotext -layout file.pdf -`. Secondary sources: https://www.tu.berlin/en/studying/applying-and-enrolling/dates-deadlines and https://www.tu.berlin/en/international/getting-started-at-tu-berlin/living-in-berlin/semester-dates-public-holidays-and-school-holidays (the latter tends to lag a semester behind).
2. **Semester start and end.** At TU Berlin the WS runs 01.10.–31.03. and the SS 01.04.–30.09. The **final project submission is always the last day of the semester**, not the end of the Vorlesungszeit.
3. **Berlin public holidays** for every calendar year the semester touches, so both years for a WS. Only gesetzliche Feiertage *in Berlin* count: Berlin has Frauentag (08.03.) but not Reformationstag (31.10.). The recurring list is on berlin.de (Senatsverwaltung für Inneres, "Sonn- und Feiertagsrecht": https://www.berlin.de/sen/inneres/buerger-und-staat/verfassungs-und-verwaltungsrecht/artikel.1435639.php). The calculator computes these anyway. The point of the research is to catch **one-off holidays** the Abgeordnetenhaus occasionally adds (e.g. 08.05.2025); search for "einmaliger Feiertag Berlin <year>". Pass any you find via `--extra-holiday`.

Note the source URLs; you'll cite them in the proposal.

## Phase 3: Ask Fares

Ask everything in **one plain message**, because most answers are free text (a room, a URL, a date). For any value you can guess from the current slides, show the old value as the default so he can just confirm.

- Lecture weekday and time (start–end, e.g. 14:15–15:45)
- Room (e.g. E-N 152)
- Date of the first lecture, if it isn't the first lecture weekday in the Vorlesungszeit
- MOSES registration (enrollment) deadline
- New Zoom link
- Time window for Project Pitches and Project Presentations. These are extended sessions, currently "14:00 - 18:00". Ask whether it stays the same, especially if the lecture time changed.

## Phase 4: Propose the schedule

Run the calculator with the researched and collected values:

```bash
python3 -I .claude/skills/update-semester-dates/scripts/semester_calendar.py \
  --lecture-start YYYY-MM-DD --lecture-end YYYY-MM-DD --weekday fri \
  --break YYYY-MM-DD:YYYY-MM-DD [--first-lecture YYYY-MM-DD] [--extra-holiday YYYY-MM-DD:Name] \
  --lecture-end-time 15:45 --moses-deadline YYYY-MM-DD [--proposal-deadline YYYY-MM-DD]
```

Then map the schedule template from phase 1 onto the usable slots, following these rules:

- **Lectures on holidays or in lecture-free periods are skipped.** The topic moves to the next usable slot.
- **Keep the topic order** of the current table. Project Pitches and Project Presentations each take a lecture slot.
- **Project Presentations** go in the last usable slot of the Vorlesungszeit.
- **Final Project Submission** = last day of the semester (31.03. or 30.09.).
- **Course Application Deadline** is a Sunday, and it must be **at least 2 days before the MOSES registration deadline**. That means a calendar-day difference ≥ 2: Sun→Tue is fine, Sun→Mon is not. The calculator prints the latest valid Sunday. Start from its current relative position (currently the Sunday after the 2nd lecture). If that violates the MOSES rule, use the latest valid Sunday and **say explicitly that it moved and why**. Students then get less time between the introduction and the deadline, and Fares may prefer to move the MOSES date if that is his to set.
- **Project Proposal Deadline & Group Selection** (a Sunday) and **Project Pitches** should sit at the same relative position in the schedule as now, i.e. after the same topic. That keeps the project timeline consistent with the content students have seen. Read the positions from the current table.
- **Slot surplus or deficit.** WS usually has more usable slots than SS. Never silently drop or invent content. Look at the most recent semester *of the same type* in git history (`git log -p -- slides/00-introduction.md`, e.g. the previous WS table) to see how the extra or missing slots were handled then. Default for a surplus: extra "TBD" slots directly before the Project Presentations. For a deficit, propose options: the repo has condensed decks `slides/short-08-09-*` and `slides/short-09-10-*` that merge lectures 08–10. Let Fares choose.
- **Reminder slide.** The "Project Proposal Deadline Reminder" slide belongs in the last lecture held *before* the proposal deadline. If the deck found in phase 1 isn't that lecture, propose moving the slide there (same position, right after the title slide).
- **Late rollovers.** Compare every date with today (the calculator marks `(PAST)` and `(TODAY)`). If the update happens after the semester has started, mark which slots have already happened. Flag any deadline that is in the past or only a few days away, and offer alternatives rather than picking one silently.

Present the proposal as **one table**: new date (with weekday) | item | old date. Deadline rows go in chronological position, just like topics. Below it, list the skipped dates with their reasons, the MOSES-rule check (application date, MOSES date, gap in days), any surplus/deficit decision, the reminder-slide move, warnings about past or near dates, and the source links. Then wait for approval or corrections.

## Phase 5: Apply

After Fares approves:

1. Update every location found in phase 1. Keep each location's existing date format (the slides use `DD.MM.YYYY`). For the Dates and Deadlines table, keep the existing HTML structure. Deadline rows keep `class="fragment custom appear-table highlight"`. When an item changes position (topic *or* deadline row), move the whole `<tr>` rather than just rewriting dates. Keep the two columns roughly balanced; if rows were added or removed, re-split at a natural point (e.g. right after the pitches) so the left column has the same number of rows as the right, or one more. Update the pitch/presentation time window if it changed.
2. Move the reminder slide if that was agreed.
3. The room and Zoom link are **not** in the slides by design (the repo is public via GitHub Pages). Only update them in the repo if an old room string actually turned up in phase 1.
4. Verify: rerun `find_dates.py` with all old dates. It should report 0 hits; explain any hit you kept on purpose. Then run `npm run build` to make sure the decks still build.
5. Show `git diff --stat` and a short summary. Don't commit unless asked. If asked, follow the existing history style (e.g. "Updated dates for winter semester 2026/27"), and add no co-author trailer.

## Phase 6: ISIS description

Ask Fares to paste the current ISIS course description **as HTML source** (the code view of the ISIS editor). Then:

- Replace every date, weekday, time, room and Zoom link with the new values. Keep the formatting style each passage already uses: ISIS tends to use long dates like "Friday, 16 October 2026, at 14:15" and en dashes in time ranges. Make sure every weekday matches its date.
- If there is a preliminary note at the top (something like *"The dates in this description will be updated before the first lecture on …, it will be held in hybrid format, in room … and on Zoom (link to follow)."*), remove it entirely, including its wrapping element. It has done its job once the real dates are in.
- Leave all other content and markup untouched. Don't reformat or "improve" it.
- Return the **complete** new HTML in a single code block so it can be pasted straight back. Above it, give a short list of what changed. Mention anything in the ISIS text that didn't map cleanly onto the new schedule rather than guessing.

## Phase 7: ISIS project-group selection window

Give the opening and closing time for the ISIS project-group choice activity:

- **Opens:** after the lecture in the week *before* the week of the Project Proposal Deadline, at the lecture end time. Example: deadline Sun 22.11.2026 with Friday lectures 14:15–15:45 → opens Fri 13.11.2026 15:45.
- **Closes:** the proposal deadline Sunday, 23:59.

`--proposal-deadline` in the calculator prints exactly this. If the opening day is a holiday, it warns, and then you should ask Fares which day to use. Present both as date + time, ready to type into ISIS.
