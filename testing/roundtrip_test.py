"""
roundtrip_test.py

Round-trip test for the mdjsondictio Python port:

    e.g.dictionary.csv --[build_mdjson]--> mdJSON dict --[build_table]--> table

Compares the reconstructed table back against the original CSV to surface
any bugs in the (complex, loop-heavy) build_mdjson / build_table logic.

This does NOT expect a byte-perfect match -- some divergence is expected
and reported separately as "known coercion" rather than a bug:
  - JSON round-tripping stringifies everything, so int/float columns in the
    original CSV will come back as strings (e.g. 1.0 -> "1.0" or "1").
  - NaN/blank cells may come back as None, "NA", or empty string depending
    on the path they took through the dictionary structure.
  - Row order may differ; both tables are sorted identically before
    comparison to avoid spurious diffs.

Usage:
    python roundtrip_test.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from mdjsondictio import build_mdjson, build_table


DICTIONARY_CSV = "e.g.dictionary.csv"
DICTIONARY_TITLE = "Example Dictionary"

# Columns that are expected to only ever appear on "dataField" rows -- used
# to align comparisons since build_table's column set may not perfectly
# match the source CSV column order.
SORT_KEYS = ["codeName", "dataType", "domainItem_value"]


def normalize_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a table for comparison purposes:
      - stringify all values (so 1 == 1.0 == "1" == "1.0" after further
        cleanup, and NaN/None/'' collapse to a single sentinel)
      - strip trailing '.0' from strings that are whole-number floats
      - sort rows deterministically
    """
    out = df.copy()

    # Make sure the sort keys exist even if a table is missing some columns
    for k in SORT_KEYS:
        if k not in out.columns:
            out[k] = None

    def clean_cell(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        s = str(v).strip()
        if s.upper() in ("NA", "NAN", "NONE"):
            return ""
        # collapse "1.0" -> "1" so int/float/string round-trip forms match
        if s.endswith(".0"):
            try:
                float(s)
                s = s[:-2]
            except ValueError:
                pass
        return s

    for col in out.columns:
        out[col] = out[col].apply(clean_cell)

    out = out.sort_values(by=SORT_KEYS, kind="stable").reset_index(drop=True)
    return out


def compare_tables(original: pd.DataFrame, rebuilt: pd.DataFrame) -> dict:
    """
    Compares two normalized tables and returns a dict summarizing differences.
    """
    report = {}

    orig_cols = set(original.columns)
    rebuilt_cols = set(rebuilt.columns)

    report["columns_only_in_original"] = sorted(orig_cols - rebuilt_cols)
    report["columns_only_in_rebuilt"] = sorted(rebuilt_cols - orig_cols)
    report["row_count_original"] = len(original)
    report["row_count_rebuilt"] = len(rebuilt)

    shared_cols = sorted(orig_cols & rebuilt_cols)
    mismatches = []

    if len(original) != len(rebuilt):
        report["mismatches"] = (
            "Row counts differ -- skipping cell-level comparison. "
            "Check row_count_original vs row_count_rebuilt above."
        )
        return report

    for col in shared_cols:
        o = original[col].reset_index(drop=True)
        r = rebuilt[col].reset_index(drop=True)
        diff_mask = o != r
        n_diff = int(diff_mask.sum())
        if n_diff > 0:
            examples = []
            for idx in o[diff_mask].index[:5]:
                examples.append({
                    "row": int(idx),
                    "original": o.loc[idx],
                    "rebuilt": r.loc[idx],
                    "codeName": original.loc[idx, "codeName"] if "codeName" in original.columns else None,
                })
            mismatches.append({"column": col, "n_diff": n_diff, "examples": examples})

    report["column_mismatches"] = mismatches
    return report


def print_report(report: dict) -> None:
    print("=" * 70)
    print("ROUND-TRIP TEST REPORT")
    print("=" * 70)

    print(f"\nRow count -- original: {report['row_count_original']}, "
          f"rebuilt: {report['row_count_rebuilt']}")

    if report["columns_only_in_original"]:
        print(f"\nColumns only in ORIGINAL (missing from rebuilt table):")
        for c in report["columns_only_in_original"]:
            print(f"  - {c}")
    else:
        print("\nNo columns missing from rebuilt table. ✓")

    if report["columns_only_in_rebuilt"]:
        print(f"\nColumns only in REBUILT (unexpected new columns):")
        for c in report["columns_only_in_rebuilt"]:
            print(f"  - {c}")
    else:
        print("No unexpected extra columns in rebuilt table. ✓")

    if isinstance(report.get("mismatches"), str):
        print(f"\n{report['mismatches']}")
        return

    col_mismatches = report.get("column_mismatches", [])
    if not col_mismatches:
        print("\nNo cell-level differences found in shared columns. ✓")
        return

    print(f"\n{len(col_mismatches)} column(s) with differences:\n")
    for m in col_mismatches:
        print(f"  Column: {m['column']}  ({m['n_diff']} differing row(s))")
        for ex in m["examples"]:
            print(f"    row {ex['row']} (codeName={ex['codeName']!r}): "
                  f"original={ex['original']!r}  rebuilt={ex['rebuilt']!r}")
        print()


def main():
    print(f"Loading original dictionary table from {DICTIONARY_CSV} ...")
    try:
        original = pd.read_csv(DICTIONARY_CSV)
    except FileNotFoundError:
        print(f"ERROR: could not find {DICTIONARY_CSV} in the current directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(original)} rows, {len(original.columns)} columns.")

    print("\nRunning build_mdjson() ...")
    try:
        mdjson = build_mdjson(original, title=DICTIONARY_TITLE)
    except Exception as e:
        print(f"ERROR in build_mdjson(): {e}", file=sys.stderr)
        raise

    # Sanity check the produced JSON is well-formed and has the expected shape
    inner_json_str = mdjson["data"][0]["attributes"]["json"]
    try:
        inner = json.loads(inner_json_str)
    except json.JSONDecodeError as e:
        print(f"ERROR: build_mdjson() produced invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    n_attrs = len(inner["dataDictionary"]["entity"][0]["attribute"])
    n_domains = len(inner["dataDictionary"].get("domain", []))
    print(f"build_mdjson() succeeded. Produced {n_attrs} attribute(s) and "
          f"{n_domains} domain(s).")

    # Optionally save the intermediate JSON for manual inspection
    with open("e.g.dictionary_roundtrip.json", "w") as f:
        json.dump(mdjson, f, indent=2)
    print("Wrote intermediate mdJSON to e.g.dictionary_roundtrip.json for inspection.")

    print("\nRunning build_table() on the freshly-built mdJSON ...")
    try:
        rebuilt = build_table(mdjson, entity_num=1)
    except Exception as e:
        print(f"ERROR in build_table(): {e}", file=sys.stderr)
        raise

    print(f"build_table() succeeded. Produced {len(rebuilt)} rows, "
          f"{len(rebuilt.columns)} columns.")

    # Save rebuilt table too, for manual side-by-side inspection
    rebuilt.to_csv("e.g.dictionary_roundtrip.csv", index=False)
    print("Wrote rebuilt table to e.g.dictionary_roundtrip.csv for inspection.")

    print("\nNormalizing both tables for comparison ...")
    original_norm = normalize_for_compare(original.drop(columns=["notes"], errors="ignore"))
    rebuilt_norm = normalize_for_compare(rebuilt)

    report = compare_tables(original_norm, rebuilt_norm)
    print()
    print_report(report)


if __name__ == "__main__":
    main()