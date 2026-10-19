from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
DASHBOARD_TEMPLATE = ROOT / "dashboard" / "index_template.html"
DOCS_DIR = ROOT / "docs"
CENT = Decimal("0.01")
EURO_TO_USD = Decimal("1.2")
USER_ATTRIBUTES = ("name_norm", "address_norm", "phone_norm", "email_norm")


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().casefold() in {"", "nan", "none", "null", "nat"}


def clean_text(value: Any) -> str:
    return "" if is_blank(value) else str(value).strip()


def identifier(value: Any) -> str:
    """Convert numeric-looking IDs to stable strings without a trailing .0."""
    if is_blank(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    value = str(value).strip()
    if re.fullmatch(r"\d+\.0+", value):
        return value.split(".", 1)[0]
    return value


def id_sort_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value.casefold())


def normalize_words(value: Any) -> str:
    """Normalize a human-readable field for safe equality matching."""
    if is_blank(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def normalize_phone(value: Any) -> str:
    if is_blank(value):
        return ""
    return "".join(ch for ch in str(value) if ch.isdigit())


def normalize_email(value: Any) -> str:
    if is_blank(value):
        return ""
    return re.sub(r"\s+", "", str(value)).casefold()


def parse_price_to_usd(value: Any) -> float | None:
    """Parse a price string and return a USD amount rounded to cents.

    Explicit € / EUR values are converted at €1 = $1.20. Explicit dollar values,
    and values with no currency marker, are interpreted as USD. Invalid values
    return None and are removed from cleaned orders.
    """
    if is_blank(value):
        return None
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            return None
        currency_euro = False
    else:
        raw = unicodedata.normalize("NFKC", str(value)).strip()
        low = raw.casefold()
        currency_euro = ("€" in raw) or bool(re.search(r"\beur\b|\beuros?\b", low))
        # Keep numeric characters and separators only, then interpret decimal separators.
        matches = re.findall(r"[-+]?\d[\d.,]*", raw.replace("'", ""))
        if not matches:
            return None
        numeric = matches[0]
        if "," in numeric and "." in numeric:
            # The last punctuation is treated as the decimal separator.
            if numeric.rfind(",") > numeric.rfind("."):
                numeric = numeric.replace(".", "").replace(",", ".")
            else:
                numeric = numeric.replace(",", "")
        elif "," in numeric:
            left, right = numeric.rsplit(",", 1)
            if len(right) in (1, 2):
                numeric = left.replace(",", "") + "." + right
            else:
                numeric = numeric.replace(",", "")
        try:
            number = Decimal(numeric)
        except InvalidOperation:
            return None
    if not number.is_finite() or number < 0:
        return None
    if currency_euro:
        number *= EURO_TO_USD
    return float(number.quantize(CENT, rounding=ROUND_HALF_UP))



def parse_timestamps(values):
    import pandas as pd

    # Parse timestamps with mixed timezones consistently.
    # Naive timestamps are treated as UTC.
    parsed = pd.to_datetime(
        values,
        errors="coerce",
        format="mixed",
        utc=True
    )

    return parsed



class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def reconcile_users(users_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], dict[str, list[str]]]:
    """Merge user records when any three of name/address/phone/email match.

    This implements the assignment assumption that only one identifying field
    changes between records for the same real customer. Matching uses normalized
    values, and blank values never count as matches.
    """
    expected = ["id", "name", "address", "phone", "email"]
    for col in expected:
        if col not in users_raw.columns:
            users_raw[col] = ""
    users = users_raw[expected].copy()
    for col in expected:
        users[col] = users[col].map(clean_text)
    users["id"] = users["id"].map(identifier)
    users = users[users["id"].ne("")].reset_index(drop=True)

    users["name_norm"] = users["name"].map(normalize_words)
    users["address_norm"] = users["address"].map(normalize_words)
    users["phone_norm"] = users["phone"].map(normalize_phone)
    users["email_norm"] = users["email"].map(normalize_email)

    dsu = DisjointSet(len(users))

    # Identical IDs represent the same account even if duplicate rows differ.
    first_by_id: dict[str, int] = {}
    for idx, user_id in enumerate(users["id"]):
        if user_id in first_by_id:
            dsu.union(idx, first_by_id[user_id])
        else:
            first_by_id[user_id] = idx

    # If one field changed, the other three fields identify the same real person.
    bucket_first: dict[tuple[str, tuple[str, ...]], int] = {}
    for fields in combinations(USER_ATTRIBUTES, 3):
        for idx, row in users.iterrows():
            values = tuple(row[field] for field in fields)
            if not all(values):
                continue
            key = ("|".join(fields), values)
            if key in bucket_first:
                dsu.union(idx, bucket_first[key])
            else:
                bucket_first[key] = idx

    root_to_ids: dict[int, set[str]] = defaultdict(set)
    for idx, user_id in enumerate(users["id"]):
        root_to_ids[dsu.find(idx)].add(user_id)

    groups: dict[str, list[str]] = {}
    id_to_identity: dict[str, str] = {}
    for ids in root_to_ids.values():
        sorted_ids = sorted(ids, key=id_sort_key)
        canonical = sorted_ids[0]
        groups[canonical] = sorted_ids
        for user_id in sorted_ids:
            id_to_identity[user_id] = canonical

    # Human-readable reconciled export: one row per detected real customer.
    reconciled_rows = []
    for canonical, ids in sorted(groups.items(), key=lambda pair: id_sort_key(pair[0])):
        group_rows = users[users["id"].isin(ids)]
        def first_nonempty(col: str) -> str:
            vals = [v for v in group_rows[col].tolist() if v]
            return vals[0] if vals else ""
        reconciled_rows.append({
            "real_customer_id": canonical,
            "user_ids": json.dumps(ids, ensure_ascii=False),
            "name_examples": " | ".join(dict.fromkeys(v for v in group_rows["name"] if v)),
            "address_examples": " | ".join(dict.fromkeys(v for v in group_rows["address"] if v)),
            "phone_examples": " | ".join(dict.fromkeys(v for v in group_rows["phone"] if v)),
            "email_examples": " | ".join(dict.fromkeys(v for v in group_rows["email"] if v)),
            "representative_name": first_nonempty("name"),
        })
    return pd.DataFrame(reconciled_rows), id_to_identity, groups


def load_books(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or []
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rows.append({
            "book_id": identifier(item.get(":id", item.get("id", ""))),
            "title": clean_text(item.get(":title", item.get("title", ""))),
            "author": clean_text(item.get(":author", item.get("author", ""))),
            "genre": clean_text(item.get(":genre", item.get("genre", ""))),
            "publisher": clean_text(item.get(":publisher", item.get("publisher", ""))),
            "year": clean_text(item.get(":year", item.get("year", ""))),
        })
    books = pd.DataFrame(rows, columns=["book_id", "title", "author", "genre", "publisher", "year"])
    rows_before_dedupe = len(books)
    books = books[books["book_id"].ne("")].drop_duplicates(subset=["book_id"], keep="first").reset_index(drop=True)

    book_author_map: dict[str, dict[str, Any]] = {}
    cleaned_author_sets = []
    for _, book in books.iterrows():
        author_parts = [part.strip() for part in re.split(r"[,;]", book["author"]) if part.strip()]
        pairs = [(normalize_words(author), author) for author in author_parts]
        # Avoid duplicate entries within one author set, while retaining display labels.
        unique_authors: dict[str, str] = {}
        for norm, display in pairs:
            if norm:
                unique_authors.setdefault(norm, display)
        norm_authors = tuple(sorted(unique_authors.keys()))
        display_authors = [unique_authors[k] for k in norm_authors]
        if not norm_authors:
            continue
        label = ", ".join(display_authors)
        book_author_map[book["book_id"]] = {
            "author_keys": norm_authors,
            "authors": display_authors,
            "author_label": label,
            "title": book["title"],
        }
        cleaned_author_sets.append({
            **book.to_dict(),
            "authors_normalized": json.dumps(norm_authors, ensure_ascii=False),
            "authors_display": json.dumps(display_authors, ensure_ascii=False),
            "author_set_label": label,
        })

    books_clean = pd.DataFrame(cleaned_author_sets)
    if books_clean.empty:
        books_clean = pd.DataFrame(columns=["book_id", "title", "author", "genre", "publisher", "year", "authors_normalized", "authors_display", "author_set_label"])
    books_clean.attrs["duplicates_removed"] = rows_before_dedupe - len(books)
    return books_clean, book_author_map


def load_and_clean_orders(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    orders_raw = pd.read_parquet(path)
    # Normalize column labels in case the source varies in letter casing/spaces.
    orders_raw.columns = [str(col).strip().casefold() for col in orders_raw.columns]
    required = ["user_id", "book_id", "quantity", "unit_price", "timestamp"]
    missing_cols = [col for col in required if col not in orders_raw.columns]
    if missing_cols:
        raise ValueError(f"{path}: required order columns not found: {missing_cols}. Found: {list(orders_raw.columns)}")

    rows_read = len(orders_raw)
    exact_duplicates = int(orders_raw.duplicated().sum())
    orders = orders_raw.drop_duplicates().copy()
    orders["user_id"] = orders["user_id"].map(identifier)
    orders["book_id"] = orders["book_id"].map(identifier)
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    orders["unit_price_usd"] = orders["unit_price"].map(parse_price_to_usd)
    orders["timestamp"] = parse_timestamps(orders["timestamp"])
    orders["date"] = orders["timestamp"].dt.strftime("%Y-%m-%d")

    valid_id = orders["user_id"].ne("") & orders["book_id"].ne("")
    valid_quantity = orders["quantity"].notna() & orders["quantity"].gt(0) & (orders["quantity"] % 1 == 0)
    valid_price = orders["unit_price_usd"].notna() & orders["unit_price_usd"].ge(0)
    valid_timestamp = orders["timestamp"].notna()
    keep = valid_id & valid_quantity & valid_price & valid_timestamp
    removed_invalid = int((~keep).sum())
    orders = orders.loc[keep].copy()
    orders["quantity"] = orders["quantity"].astype("int64")
    orders["unit_price_usd"] = orders["unit_price_usd"].astype(float).round(2)
    orders["paid_price"] = (orders["quantity"] * orders["unit_price_usd"]).round(2)
    orders["date"] = orders["timestamp"].dt.strftime("%Y-%m-%d")
    orders["timestamp"] = orders["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_cols = ["user_id", "book_id", "quantity", "unit_price_usd", "paid_price", "timestamp", "date"]
    # Include shipping when present, since it can be useful for auditing the source.
    if "shipping" in orders.columns:
        orders["shipping"] = orders["shipping"].map(clean_text)
        output_cols.append("shipping")
    orders = orders[output_cols].reset_index(drop=True)
    quality = {
        "rows_read": rows_read,
        "exact_duplicate_rows_removed": exact_duplicates,
        "invalid_rows_removed": removed_invalid,
        "clean_rows": len(orders),
    }
    return orders, quality


def money(value: Any) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0


def build_store_result(store_dir: Path) -> dict[str, Any]:
    store_name = store_dir.name
    output_store = OUTPUT_DIR / store_name
    output_store.mkdir(parents=True, exist_ok=True)

    users_raw = pd.read_csv(store_dir / "users.csv", dtype=str, keep_default_na=False)
    users_reconciled, user_id_map, user_groups = reconcile_users(users_raw)
    users_reconciled.to_csv(output_store / "users_reconciled.csv", index=False, encoding="utf-8-sig")

    books_clean, book_author_map = load_books(store_dir / "books.yaml")
    books_clean.to_csv(output_store / "books_cleaned.csv", index=False, encoding="utf-8-sig")

    orders, order_quality = load_and_clean_orders(store_dir / "orders.parquet")
    orders["customer_key"] = orders["user_id"].map(lambda uid: user_id_map.get(uid, uid))
    orders.to_csv(output_store / "orders_cleaned.csv", index=False, encoding="utf-8-sig")

    # Revenue by calendar date; only validated, deduplicated rows participate.
    daily = (orders.groupby("date", as_index=False)["paid_price"].sum()
             .sort_values("date", ascending=True))
    daily["revenue"] = daily["paid_price"].round(2)
    daily = daily[["date", "revenue"]]
    daily.to_csv(output_store / "daily_revenue.csv", index=False, encoding="utf-8-sig")

    top_days = daily.sort_values(["revenue", "date"], ascending=[False, True]).head(5)
    top_5_days = [{"date": str(row.date), "revenue": money(row.revenue)} for row in top_days.itertuples(index=False)]

    # Distinct author sets are counted across the book catalog.
    unique_author_set_keys = {tuple(json.loads(v)) for v in books_clean["authors_normalized"].tolist()} if not books_clean.empty else set()

    # Sum quantity (copies sold) by each author set, so repeated purchases count.
    sales_by_author: dict[tuple[str, ...], dict[str, Any]] = {}
    books_sold_rows = orders[["book_id", "quantity"]]
    sold_total = int(orders["quantity"].sum()) if not orders.empty else 0
    for row in books_sold_rows.itertuples(index=False):
        meta = book_author_map.get(row.book_id)
        if meta is None:
            continue
        key = tuple(meta["author_keys"])
        entry = sales_by_author.setdefault(key, {"authors": meta["authors"], "author_label": meta["author_label"], "books_sold": 0, "titles": set()})
        entry["books_sold"] += int(row.quantity)
        if meta.get("title"):
            entry["titles"].add(meta["title"])

    if sales_by_author:
        max_sold = max(entry["books_sold"] for entry in sales_by_author.values())
        popular_sets = [
            {"authors": entry["authors"], "label": entry["author_label"], "books_sold": int(entry["books_sold"]), "distinct_titles": len(entry["titles"])}
            for entry in sales_by_author.values() if entry["books_sold"] == max_sold
        ]
        popular_sets.sort(key=lambda item: item["label"].casefold())
    else:
        max_sold = 0
        popular_sets = []

    # Buyer spend is combined across every user_id reconciled into the same identity.
    spent = orders.groupby("customer_key", as_index=False)["paid_price"].sum() if not orders.empty else pd.DataFrame(columns=["customer_key", "paid_price"])
    known_id_set = set(users_raw.get("id", pd.Series(dtype=str)).map(identifier))
    identity_to_ids = {key: ids.copy() for key, ids in user_groups.items()}
    for uid in orders["user_id"].dropna().unique().tolist():
        uid = identifier(uid)
        if uid and uid not in known_id_set and uid not in identity_to_ids:
            identity_to_ids[uid] = [uid]
    best_buyer = {"total_spending": 0.0, "user_ids": [], "real_customer_id": ""}
    if not spent.empty:
        max_spend = spent["paid_price"].max()
        winners = spent[spent["paid_price"].round(2).eq(round(float(max_spend), 2))]
        winner_rows = []
        for row in winners.itertuples(index=False):
            key = str(row.customer_key)
            ids = identity_to_ids.get(key, [key])
            winner_rows.append({"total_spending": money(row.paid_price), "user_ids": sorted(set(ids), key=id_sort_key), "real_customer_id": key})
        winner_rows.sort(key=lambda item: (-item["total_spending"], [id_sort_key(v) for v in item["user_ids"]]))
        best_buyer = winner_rows[0]

    # A simple Matplotlib line chart, saved separately for each store.
    chart_path = output_store / "daily_revenue.png"
    if not daily.empty:
        chart_dates = pd.to_datetime(daily["date"])
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(chart_dates, daily["revenue"], marker="o", markersize=2.5, linewidth=1.8)
        ax.set_title(f"{store_name} — Daily Revenue", fontsize=15, weight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue (USD)")
        ax.grid(True, linestyle="--", alpha=0.28)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        fig.tight_layout()
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.text(0.5, 0.5, "No valid orders to plot", ha="center", va="center")
        ax.axis("off")
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    result = {
        "store": store_name,
        "total_revenue": money(orders["paid_price"].sum()) if not orders.empty else 0.0,
        "orders_count": int(len(orders)),
        "unique_buyers_in_orders": int(orders["customer_key"].nunique()) if not orders.empty else 0,
        "books_sold": sold_total,
        "unique_users": int(len(user_groups)),
        "users_rows": int(len(users_raw)),
        "unique_author_sets": int(len(unique_author_set_keys)),
        "books_in_catalog": int(len(books_clean)),
        "top_5_days": top_5_days,
        "most_popular_author_sets": popular_sets,
        "best_buyer": best_buyer,
        "daily_revenue": [{"date": str(row.date), "revenue": money(row.revenue)} for row in daily.itertuples(index=False)],
        "quality": {
            **order_quality,
            "user_rows_read": int(len(users_raw)),
            "real_users_after_reconciliation": int(len(user_groups)),
            "user_id_aliases_merged": max(0, int(users_raw["id"].map(identifier).replace("", pd.NA).dropna().nunique()) - int(len(user_groups))),
            "book_rows_after_cleaning": int(len(books_clean)),
            "books_with_valid_author_sets": int(len(book_author_map)),
        },
    }
    (output_store / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    store_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"STORE\d+", p.name)], key=lambda p: int(re.search(r"\d+", p.name).group()))
    if not store_dirs:
        raise FileNotFoundError("No STORE folders found. Expected STORE1/users.csv, orders.parquet and books.yaml etc.")

    all_results = {}
    for store_dir in store_dirs:
        print(f"Processing {store_dir.name} ...")
        result = build_store_result(store_dir)
        all_results[store_dir.name] = result
        print(f"  Clean orders: {result['orders_count']:,}; unique users: {result['unique_users']:,}; author sets: {result['unique_author_sets']:,}; revenue: ${result['total_revenue']:,.2f}")

    payload = json.dumps(all_results, ensure_ascii=False, separators=(",", ":"))
    template = DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    dashboard_html = template.replace("__DASHBOARD_DATA_JSON__", payload)
    dashboard_dir = OUTPUT_DIR / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    (dashboard_dir / "index.html").write_text(dashboard_html, encoding="utf-8")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "index.html").write_text(dashboard_html, encoding="utf-8")

    (OUTPUT_DIR / "all_stores_summary.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDone!")
    print(f"Local dashboard: {dashboard_dir / 'index.html'}")
    print(f"GitHub Pages file: {DOCS_DIR / 'index.html'}")
    print(f"Detailed outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
