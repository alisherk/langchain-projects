import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "db.sqlite"

def list_tables() -> str:
    """List all tables in the SQLite database."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return "\n".join([row[0] for row in cursor.fetchall()])
    finally:
        conn.close()


def run_sqlite_query(query: str) -> list:
    """Run a SQL query against a local SQLite database and return the results."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except sqlite3.OperationalError as err:
        return f"The following error occurred: {str(err)}"
    finally:
        conn.close()
