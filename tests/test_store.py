from datetime import date, datetime

from aak.models import Event, ProvisionedUser
from aak.store import (
    init_db,
    read_events,
    read_provisioned_users,
    truncate_db,
    write_events,
    write_provisioned_users,
)


def _sample_events() -> list[Event]:
    return [
        Event(
            user_id="user-0001",
            cohort="cohort-1",
            ts=datetime(2025, 1, 1, 9, 0, 0),
            event_type="invocation",
            session_id="user-0001:0",
            turns=1,
            latency_ms=300,
        ),
        Event(
            user_id="user-0001",
            cohort="cohort-1",
            ts=datetime(2025, 1, 1, 9, 0, 5),
            event_type="task_outcome",
            outcome="success",
            session_id="user-0001:0",
            turns=4,
            latency_ms=None,
        ),
        Event(
            user_id="user-0002",
            cohort="cohort-2",
            ts=datetime(2025, 1, 2, 10, 0, 0),
            event_type="escalation",
            session_id="user-0002:1",
            turns=1,
            latency_ms=500,
        ),
    ]


def _sample_roster() -> list[ProvisionedUser]:
    return [
        ProvisionedUser(user_id="user-0001", cohort="cohort-1", provisioned_date=date(2025, 1, 1)),
        ProvisionedUser(user_id="user-0002", cohort="cohort-2", provisioned_date=date(2025, 1, 15)),
        # a user with no matching event at all — the whole point of the roster
        ProvisionedUser(user_id="user-0003", cohort="cohort-2", provisioned_date=date(2025, 1, 15)),
    ]


def _sort_key(event: Event):
    return (event.user_id, event.ts, event.event_type)


def _roster_sort_key(user: ProvisionedUser):
    return (user.user_id, user.cohort)


def test_write_and_read_events_roundtrip(tmp_path):
    db_path = tmp_path / "events.db"
    events = _sample_events()

    init_db(db_path)
    write_events(db_path, events)
    read_back = read_events(db_path)

    assert len(read_back) == len(events)
    for original, restored in zip(sorted(events, key=_sort_key), sorted(read_back, key=_sort_key)):
        assert original.model_dump() == restored.model_dump()


def test_read_events_cohort_filter(tmp_path):
    db_path = tmp_path / "events.db"
    events = _sample_events()

    init_db(db_path)
    write_events(db_path, events)

    filtered = read_events(db_path, cohort="cohort-1")
    assert len(filtered) == 2
    assert all(event.cohort == "cohort-1" for event in filtered)


def test_read_events_empty_db(tmp_path):
    db_path = tmp_path / "events.db"
    init_db(db_path)
    assert read_events(db_path) == []


def test_write_and_read_provisioned_users_roundtrip(tmp_path):
    db_path = tmp_path / "roster.db"
    roster = _sample_roster()

    init_db(db_path)
    write_provisioned_users(db_path, roster)
    read_back = read_provisioned_users(db_path)

    assert len(read_back) == len(roster)
    for original, restored in zip(
        sorted(roster, key=_roster_sort_key), sorted(read_back, key=_roster_sort_key)
    ):
        assert original.model_dump() == restored.model_dump()


def test_read_provisioned_users_cohort_filter(tmp_path):
    db_path = tmp_path / "roster.db"
    roster = _sample_roster()

    init_db(db_path)
    write_provisioned_users(db_path, roster)

    filtered = read_provisioned_users(db_path, cohort="cohort-2")
    assert len(filtered) == 2
    assert all(user.cohort == "cohort-2" for user in filtered)


def test_provisioned_users_reveal_zero_event_users(tmp_path):
    db_path = tmp_path / "combined.db"
    init_db(db_path)
    write_events(db_path, _sample_events())
    write_provisioned_users(db_path, _sample_roster())

    roster_ids = {u.user_id for u in read_provisioned_users(db_path)}
    event_ids = {e.user_id for e in read_events(db_path)}
    zero_event_users = roster_ids - event_ids

    assert zero_event_users == {"user-0003"}


def test_truncate_db_empties_existing_data(tmp_path):
    db_path = tmp_path / "combined.db"
    init_db(db_path)
    write_events(db_path, _sample_events())
    write_provisioned_users(db_path, _sample_roster())

    truncate_db(db_path)

    assert read_events(db_path) == []
    assert read_provisioned_users(db_path) == []


def test_truncate_db_on_nonexistent_path_creates_empty_tables(tmp_path):
    db_path = tmp_path / "never-initialized.db"

    truncate_db(db_path)

    assert read_events(db_path) == []
    assert read_provisioned_users(db_path) == []
