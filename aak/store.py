"""SQLite persistence layer for adoption data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from sqlalchemy import Column, Date, DateTime, Integer, MetaData, String, Table, create_engine, insert, select

from aak.models import Event, ProvisionedUser

PathLike = Union[str, Path]

metadata = MetaData()

events_table = Table(
    "events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String, nullable=False),
    Column("cohort", String, nullable=False),
    Column("ts", DateTime, nullable=False),
    Column("event_type", String, nullable=False),
    Column("outcome", String, nullable=True),
    Column("session_id", String, nullable=False),
    Column("turns", Integer, nullable=False),
    Column("latency_ms", Integer, nullable=True),
)

provisioned_users_table = Table(
    "provisioned_users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String, nullable=False),
    Column("cohort", String, nullable=False),
    Column("provisioned_date", Date, nullable=False),
)


def _engine(path: PathLike):
    return create_engine(f"sqlite:///{path}")


def init_db(path: PathLike) -> None:
    """Create the events and provisioned_users tables at ``path`` if they don't already exist."""
    engine = _engine(path)
    metadata.create_all(engine)
    engine.dispose()


def truncate_db(path: PathLike) -> None:
    """Ensure the events/provisioned_users tables exist at ``path``, then empty them.

    Used by ``aak simulate``'s truncate-by-default behavior: safe to call whether ``path``
    already exists (with old data) or not.
    """
    init_db(path)
    engine = _engine(path)
    with engine.begin() as conn:
        conn.execute(events_table.delete())
        conn.execute(provisioned_users_table.delete())
    engine.dispose()


def write_events(path: PathLike, events: list[Event]) -> None:
    """Bulk-insert ``events`` into the database at ``path``."""
    if not events:
        return
    engine = _engine(path)
    rows = [event.model_dump() for event in events]
    with engine.begin() as conn:
        conn.execute(insert(events_table), rows)
    engine.dispose()


def read_events(path: PathLike, cohort: Optional[str] = None) -> list[Event]:
    """Read events from the database at ``path``, optionally filtered by ``cohort``."""
    engine = _engine(path)
    stmt = select(events_table)
    if cohort is not None:
        stmt = stmt.where(events_table.c.cohort == cohort)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    engine.dispose()
    return [
        Event(**{key: value for key, value in row.items() if key != "id"})
        for row in rows
    ]


def write_provisioned_users(path: PathLike, users: list[ProvisionedUser]) -> None:
    """Bulk-insert the provisioned-user roster into the database at ``path``."""
    if not users:
        return
    engine = _engine(path)
    rows = [user.model_dump() for user in users]
    with engine.begin() as conn:
        conn.execute(insert(provisioned_users_table), rows)
    engine.dispose()


def read_provisioned_users(path: PathLike, cohort: Optional[str] = None) -> list[ProvisionedUser]:
    """Read the provisioned-user roster from the database at ``path``, optionally filtered by ``cohort``."""
    engine = _engine(path)
    stmt = select(provisioned_users_table)
    if cohort is not None:
        stmt = stmt.where(provisioned_users_table.c.cohort == cohort)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    engine.dispose()
    return [
        ProvisionedUser(**{key: value for key, value in row.items() if key != "id"})
        for row in rows
    ]
