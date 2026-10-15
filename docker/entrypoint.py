"""Wait for PostgreSQL, import the knowledge base, then start the API."""

from __future__ import annotations

import os
import sys
import time


def wait_for_postgres(url: str, attempts: int = 30) -> None:
    """Retry until PostgreSQL accepts a connection."""
    import psycopg

    last_error = "unknown"
    for attempt in range(1, attempts + 1):
        try:
            with psycopg.connect(url, connect_timeout=3) as connection:
                connection.execute("SELECT 1")
            print("PostgreSQL is ready.", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - startup should report any connection failure
            last_error = str(exc)
            print(f"Waiting for PostgreSQL ({attempt}/{attempts}): {last_error}", flush=True)
            time.sleep(2)
    raise SystemExit(f"PostgreSQL did not become ready: {last_error}")


def main() -> None:
    """Import documents and replace this process with the API server."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise SystemExit("DATABASE_URL is required")
    wait_for_postgres(url)
    from apex_rag.cli import ingest

    ingest()
    os.execv(
        sys.executable,
        [sys.executable, "-m", "uvicorn", "apex_rag.api:app", "--host", "0.0.0.0", "--port", "18000"],
    )


if __name__ == "__main__":
    main()
