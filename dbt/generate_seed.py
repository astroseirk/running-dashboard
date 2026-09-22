"""Regenerate dbt/seeds/activities.csv from the canonical data/activities.csv.

The intervals.icu export wraps each data row in one extra layer of CSV quoting
(see src/data.py's _unwrap_double_quoted_rows for the same fix applied at
pandas-read time). dbt seed just loads a CSV as-is with no preprocessing hook,
so this script does that unwrapping once and writes a plain, dbt-seedable CSV.

Run this whenever data/activities.csv is updated, then `dbt seed && dbt run`.
"""
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "activities.csv"
SEED_PATH = Path(__file__).resolve().parent / "seeds" / "activities.csv"


def unwrap_double_quoted_rows(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    header, *rows = lines
    fixed_rows = []
    for row in rows:
        if row.startswith('"') and row.endswith('"'):
            row = row[1:-1].replace('""', '"')
        fixed_rows.append(row)
    return "\n".join([header, *fixed_rows])


if __name__ == "__main__":
    raw = RAW_PATH.read_text(encoding="utf-8-sig")
    SEED_PATH.write_text(unwrap_double_quoted_rows(raw), encoding="utf-8")
    n_rows = len(raw.splitlines()) - 1
    print(f"Wrote {SEED_PATH} ({n_rows} rows)")
