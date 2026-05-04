import os
import sqlite3


def resolve_db_path(database_path: str = "foguth_etf_models.db") -> str:
    """Resolve DB path relative to this repository, not process working directory."""
    if os.path.isabs(database_path):
        return database_path
    return os.path.abspath(os.path.join(os.path.dirname(__file__), database_path))


def get_db_cache_buster(database_path: str = "foguth_etf_models.db") -> str:
    """Return a lightweight fingerprint used to invalidate Streamlit caches.

    The fingerprint combines filesystem modification time and the max web log timestamp.
    If either changes, cached query functions that include this value as an argument are invalidated.
    """
    resolved_path = resolve_db_path(database_path)

    mtime_ns = 0
    try:
        mtime_ns = os.stat(resolved_path).st_mtime_ns
    except OSError:
        pass

    last_updated = ""
    conn = None
    try:
        conn = sqlite3.connect(resolved_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ffgwebUpdateLog'"
        )
        if cursor.fetchone() is not None:
            cursor.execute("SELECT MAX(updateDateTime) FROM ffgwebUpdateLog")
            value = cursor.fetchone()[0]
            if value:
                last_updated = str(value)
    except sqlite3.Error:
        pass
    finally:
        if conn is not None:
            conn.close()

    return f"{mtime_ns}|{last_updated}"
