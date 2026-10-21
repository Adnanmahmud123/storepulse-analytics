from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
OUTPUTS = ROOT / "outputs"
EXPECTED_STORES = {f"STORE{i}" for i in range(1, 6)}

def main() -> None:
    summary_path = OUTPUTS / "all_stores_summary.json"
    html_path = DOCS / "index.html"
    if not summary_path.exists():
        raise SystemExit("FAIL: outputs/all_stores_summary.json is missing. Run python run_analysis.py first.")
    if not html_path.exists():
        raise SystemExit("FAIL: docs/index.html is missing. Run python run_analysis.py first.")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    missing = EXPECTED_STORES - set(data)
    if missing:
        raise SystemExit(f"FAIL: summaries missing stores: {sorted(missing)}")
    for store, result in data.items():
        required = ("total_revenue", "unique_users", "unique_author_sets", "books_sold", "top_5_days", "daily_revenue", "daily_books_sold", "best_buyer")
        absent = [key for key in required if key not in result]
        if absent:
            raise SystemExit(f"FAIL: {store} summary missing fields: {absent}")
        if not isinstance(result["top_5_days"], list) or len(result["top_5_days"]) > 5:
            raise SystemExit(f"FAIL: {store} top_5_days is invalid")
        for day in result["top_5_days"]:
            if len(day.get("date", "")) != 10 or day["date"][4] != "-" or day["date"][7] != "-":
                raise SystemExit(f"FAIL: {store} has a date not in YYYY-MM-dd format: {day}")
    html = html_path.read_text(encoding="utf-8")
    if "__DASHBOARD_DATA_JSON__" in html or "const DATA = {};" in html or "const DATA = {};" in html:
        raise SystemExit("FAIL: docs/index.html does not contain embedded analysis data. Regenerate it with python run_analysis.py.")
    if "How many books were sold?" not in html:
        raise SystemExit("FAIL: chat examples are missing from dashboard HTML")
    print("PASS: five store summaries exist; required metrics and top-day date formats are present.")
    print("PASS: dashboard HTML has embedded data and chat prompts. Review the live page after git push.")

if __name__ == "__main__":
    main()
