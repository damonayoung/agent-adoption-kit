# agent-adoption-kit — project conventions

- Python 3.11+. Pydantic models in aak/models.py are the single source of schema truth.
- Every analytics function gets a known-answer pytest before the phase it ships in is considered done.
- Every threshold default is marked PROPOSED in code comments and docs. No exceptions.
- No client data, client names, or engagement artifacts in this repo, ever.
- Native staging vocabulary is this project's own model; ADKAR may appear only in one attributed optional-mapping doc.
- No new dependencies without asking. SQLite only — no external databases, no Docker.
- Damon reviews all diffs in Cursor and writes his own commit messages. Do not commit or push unless explicitly asked.
