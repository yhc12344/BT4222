"""Flatten audited JSON case files into a CSV for modelling.

Run from the project root:
    python src/json_to_df.py
"""
import json
from itertools import product
from pathlib import Path

import pandas as pd

from config import AUDIT_OUTPUT, CSV_OUTPUT


def build_dataframe(data_dir: Path) -> pd.DataFrame:
    rows = []
    for filepath in sorted(data_dir.glob("*.json")):
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Skipping {filepath.name}: invalid JSON.")
            continue

        plaintiffs = [r for r in data if r.get("Party_Details", {}).get("Role") == "Plaintiff"]
        defendants = [r for r in data if r.get("Party_Details", {}).get("Role") == "Defendant"]

        for p, d in product(plaintiffs, defendants):
            meta  = p.get("Metadata", {})
            p_det = p.get("Party_Details", {})
            d_det = d.get("Party_Details", {})
            rows.append({
                "Case_Number":          meta.get("Case_Number"),
                "Coram":                meta.get("Coram"),
                "Judge":                meta.get("Judge"),
                "Date":                 meta.get("Date"),
                "Tribunal_Court":       meta.get("Tribunal_Court"),
                "Plaintiff_Name":       p_det.get("Name"),
                "Defendant_Name":       d_det.get("Name"),
                "Combined_Facts":       [p_det.get("Facts", []), d_det.get("Facts", [])],
                "Combined_Issue":       [p_det.get("Issue", ""), d_det.get("Issue", "")],
                "Combined_Rule":        [p_det.get("Rule", ""), d_det.get("Rule", "")],
                "Combined_Application": [p_det.get("Application", ""), d_det.get("Application", "")],
                "plaintiff_label":      p.get("Label"),
                "defendant_label":      d.get("Label"),
            })

    return pd.DataFrame(rows)


def main() -> None:
    df = build_dataframe(AUDIT_OUTPUT)
    print(f"Total combinations: {len(df)}")
    print(df.head())
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"Saved -> {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
