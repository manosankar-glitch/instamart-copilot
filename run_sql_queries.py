
import sqlite3
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DB = BASE / "data" / "instamart.db"
SQL_FILE = BASE / "sql" / "queries.sql"
OUTPUT_DIR = BASE / "reports" / "sql"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sql_text = SQL_FILE.read_text(encoding="utf-8")
    queries = [
        part.strip()
        for part in sql_text.split(";")
        if part.strip()
    ]

    if len(queries) != 7:
        raise ValueError(
            f"Expected 7 SQL queries, found {len(queries)}"
        )

    with sqlite3.connect(DB) as con:
        for i, query in enumerate(queries, start=1):
            # Remove leading SQL comment lines before execution.
            lines = [
                line for line in query.splitlines()
                if not line.lstrip().startswith("--")
            ]
            query_to_run = "\n".join(lines).strip()

            result = pd.read_sql_query(query_to_run, con)
            output_file = OUTPUT_DIR / f"q{i}_results.csv"
            result.to_csv(output_file, index=False)

            print(f"\n--- Q{i} ---")
            print(result.to_string(index=False))
            print(f"Saved: {output_file.relative_to(BASE)}")

    print("\nALL 7 SQL QUERIES EXECUTED SUCCESSFULLY")

if __name__ == "__main__":
    main()
