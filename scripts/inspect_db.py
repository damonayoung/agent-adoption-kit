#!/usr/bin/env python3
"""One-off inspection tool for aak-generated SQLite event databases.

Not part of the aak package — a dev tool for eyeballing simulated .db files.
Per cohort, prints: users with zero events, mean turns per session, week-1
vs week-12 active-user rate, and the Gini coefficient of invocations per user.

Reads the provisioned-user roster the simulator writes alongside events, so
"users with zero events" and Gini (which needs to count unobserved users as
zero) come straight from the db — no guessing at population size required.

Usage:
    python scripts/inspect_db.py demo.db demo_aware.db
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aak.simulate.pathologies import gini_coefficient  # noqa: E402
from aak.store import read_events, read_provisioned_users  # noqa: E402

WEEK_DAYS = 7
WEEK12_START_DAY = 11 * WEEK_DAYS  # week 12 is the 12th 7-day block since rollout


def _turns_per_session(events) -> list[int]:
    by_session: dict[str, int] = defaultdict(int)
    for e in events:
        by_session[e.session_id] = max(by_session[e.session_id], e.turns)
    return list(by_session.values())


def _inspect_cohort(roster: list, cohort_events: list) -> dict:
    roster_ids = {u.user_id for u in roster}
    observed_users = {e.user_id for e in cohort_events}
    zero_users = len(roster_ids - observed_users)

    turns = _turns_per_session(cohort_events)
    mean_turns = statistics.fmean(turns) if turns else 0.0

    week1_active: set[str] = set()
    week12_active: set[str] = set()
    if roster:
        rollout = min(u.provisioned_date for u in roster)
        week1_end = rollout + timedelta(days=WEEK_DAYS)
        week12_start = rollout + timedelta(days=WEEK12_START_DAY)
        week12_end = week12_start + timedelta(days=WEEK_DAYS)
        week1_active = {e.user_id for e in cohort_events if rollout <= e.ts.date() < week1_end}
        week12_active = {e.user_id for e in cohort_events if week12_start <= e.ts.date() < week12_end}

    denominator = len(roster_ids)
    week1_rate = len(week1_active) / denominator if denominator else 0.0
    week12_rate = len(week12_active) / denominator if denominator else 0.0

    invocation_counts: dict[str, int] = {user_id: 0 for user_id in roster_ids}
    for e in cohort_events:
        if e.event_type == "invocation":
            invocation_counts[e.user_id] += 1
    gini = gini_coefficient(list(invocation_counts.values()))

    return {
        "zero_event_users": zero_users,
        "roster_size": denominator,
        "mean_turns_per_session": mean_turns,
        "week1_active_rate": week1_rate,
        "week12_active_rate": week12_rate,
        "gini_invocations": gini,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("db_paths", nargs="+", type=Path)
    args = parser.parse_args()

    for db_path in args.db_paths:
        print(f"=== {db_path} ===")
        roster = read_provisioned_users(db_path)
        events = read_events(db_path)

        roster_by_cohort: dict[str, list] = defaultdict(list)
        for u in roster:
            roster_by_cohort[u.cohort].append(u)
        events_by_cohort: dict[str, list] = defaultdict(list)
        for e in events:
            events_by_cohort[e.cohort].append(e)

        for cohort_name in sorted(roster_by_cohort):
            stats = _inspect_cohort(roster_by_cohort[cohort_name], events_by_cohort.get(cohort_name, []))
            print(
                f"  {cohort_name}: "
                f"zero_event_users={stats['zero_event_users']}/{stats['roster_size']}  "
                f"mean_turns/session={stats['mean_turns_per_session']:.2f}  "
                f"week1_active_rate={stats['week1_active_rate']:.1%}  "
                f"week12_active_rate={stats['week12_active_rate']:.1%}  "
                f"gini(invocations)={stats['gini_invocations']:.3f}"
            )

        orphaned_cohorts = set(events_by_cohort) - set(roster_by_cohort)
        if orphaned_cohorts:
            print(f"  NOTE: cohorts with events but no roster entries: {sorted(orphaned_cohorts)}")
        print()


if __name__ == "__main__":
    main()
