import unittest
from pathlib import Path

import pandas as pd

from run_analysis import load_books, parse_price_to_usd, parse_timestamps, reconcile_users


class ProcessingTests(unittest.TestCase):
    def test_mixed_timezone_and_bad_timestamps(self):
        values = pd.Series([
            "2025-01-02T12:00:00Z",
            "2025-01-02 12:00:00",
            "2025-01-03T12:00:00+06:00",
            "not-a-date",
            None,
        ])
        result = parse_timestamps(values)
        self.assertEqual(str(result.iloc[0]), "2025-01-02 12:00:00")
        self.assertEqual(str(result.iloc[1]), "2025-01-02 12:00:00")
        self.assertEqual(str(result.iloc[2]), "2025-01-03 12:00:00")
        self.assertTrue(pd.isna(result.iloc[3]))
        self.assertTrue(pd.isna(result.iloc[4]))

    def test_currency_conversion(self):
        self.assertEqual(parse_price_to_usd("€10,50"), 12.60)
        self.assertEqual(parse_price_to_usd("EUR 10.50"), 12.60)
        self.assertEqual(parse_price_to_usd("$1,234.56"), 1234.56)
        self.assertIsNone(parse_price_to_usd("not a price"))

    def test_reconcile_one_changed_user_field(self):
        users = pd.DataFrame([
            {"id": "1", "name": "Alice Smith", "address": "1 Main St", "phone": "123-456", "email": "a@example.com"},
            {"id": "2", "name": "Alice Smith", "address": "2 Main St", "phone": "123456", "email": "a@example.com"},
            {"id": "3", "name": "Bob Smith", "address": "9 Side St", "phone": "999", "email": "b@example.com"},
        ])
        _, mapping, groups = reconcile_users(users)
        self.assertEqual(len(groups), 2)
        self.assertEqual(mapping["1"], mapping["2"])
        self.assertNotEqual(mapping["1"], mapping["3"])

    def test_yaml_books_load(self):
        books, author_map = load_books(Path(__file__).resolve().parents[1] / "STORE1" / "books.yaml")
        self.assertGreater(len(books), 0)
        self.assertGreater(len(author_map), 0)


if __name__ == "__main__":
    unittest.main()
