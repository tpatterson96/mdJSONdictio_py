"""
test_mdjsondictio.py

Pytest suite for the mdjsondictio Python port.

Uses the three real example files as fixtures:
  - e.g.dictionary.csv    (tabular data dictionary, 460 rows x 14 cols)
  - e.g.dictionary2.json  (real mdJSON dictionary, 100 attributes, 38 domains)
  - e.g.dataset.csv       (real dataset, 2901 rows x 92 cols)

Expected directory layout (adjust FIXTURES_DIR below if different):

    project/
      mdjsondictio.py
      test_mdjsondictio.py
      fixtures/
        e.g.dictionary.csv
        e.g.dictionary2.json
        e.g.dataset.csv

Run with:
    pytest test_mdjsondictio.py -v

If the fixture files aren't found, the file-dependent tests are skipped
(not failed) so the suite still runs in environments without the sample
data -- pure logic/regression tests below still execute normally.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mdjsondictio import (
    build_mdjson,
    build_table,
    validate_table,
    validate_mdjson,
    extract_mdjson,
    _coerce_bool_column,
    _is_integer_like,
    _dtype_matches,
    DATATYPE_RULES,
)


# ---------------------------------------------------------------------------
# Fixture file locations
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# Fall back to the current directory if there's no fixtures/ subfolder,
# so this also works if you just drop the CSV/JSON next to the test file.
if not FIXTURES_DIR.exists():
    FIXTURES_DIR = Path(__file__).parent

DICTIONARY_CSV = FIXTURES_DIR / "e.g.dictionary.csv"
DICTIONARY2_JSON = FIXTURES_DIR / "e.g.dictionary2.json"
DATASET_CSV = FIXTURES_DIR / "e.g.dataset.csv"


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"Fixture file not found: {path}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dictionary_table() -> pd.DataFrame:
    _require(DICTIONARY_CSV)
    return pd.read_csv(DICTIONARY_CSV)


@pytest.fixture(scope="module")
def dataset_table() -> pd.DataFrame:
    _require(DATASET_CSV)
    return pd.read_csv(DATASET_CSV)


@pytest.fixture(scope="module")
def dictionary2_mdjson() -> dict:
    _require(DICTIONARY2_JSON)
    with open(DICTIONARY2_JSON, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def built_mdjson(dictionary_table) -> dict:
    """build_mdjson() output from the real dictionary CSV, reused across tests."""
    return build_mdjson(dictionary_table, title="Example Dictionary")


@pytest.fixture(scope="module")
def rebuilt_table(built_mdjson) -> pd.DataFrame:
    """build_table() applied to the output of build_mdjson(), for round-trip tests."""
    return build_table(built_mdjson, entity_num=1)


@pytest.fixture(scope="module")
def validation_warnings(dictionary_table, dataset_table) -> pd.DataFrame:
    return validate_table(dictionary_table, dataset_table)


# ---------------------------------------------------------------------------
# build_table -- against a real, previously-existing mdJSON file
# ---------------------------------------------------------------------------

class TestBuildTableFromRealJSON:
    def test_returns_dataframe_with_expected_columns(self, dictionary2_mdjson):
        table = build_table(dictionary2_mdjson, entity_num=1)
        expected_cols = {
            "codeName", "domainItem_name", "domainItem_value", "definition",
            "dataType", "allowNull", "units", "unitsResolution",
            "isCaseSensitive", "fieldWidth", "missingValue", "minValue",
            "maxValue", "domainId",
        }
        assert expected_cols.issubset(set(table.columns))

    def test_nonempty_and_reasonable_size(self, dictionary2_mdjson):
        table = build_table(dictionary2_mdjson, entity_num=1)
        # The real file has 100 attributes plus many domain items across
        # 38 domains -- expect a substantial row count, not just the
        # attribute rows.
        assert len(table) > 100

    def test_dataField_rows_present(self, dictionary2_mdjson):
        table = build_table(dictionary2_mdjson, entity_num=1)
        datafield_rows = table[table["domainItem_value"] == "dataField"]
        assert len(datafield_rows) > 0
        # Known attribute from the fixture file
        assert "SpeciesCode" in datafield_rows["codeName"].values

    def test_raises_on_non_dictionary_record(self):
        bad_input = {"data": [{"type": "records", "attributes": {"json": "{}"}}]}
        with pytest.raises(ValueError, match="Dictionary not detected"):
            build_table(bad_input, entity_num=1)


# ---------------------------------------------------------------------------
# Round trip: build_mdjson -> build_table should reproduce the source table
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_row_count_preserved(self, dictionary_table, rebuilt_table):
        assert len(rebuilt_table) == len(dictionary_table)

    def test_no_columns_missing_from_rebuilt(self, dictionary_table, rebuilt_table):
        original_cols = set(dictionary_table.columns) - {"notes"}
        rebuilt_cols = set(rebuilt_table.columns)
        missing = original_cols - rebuilt_cols
        assert not missing, f"Columns dropped during round-trip: {missing}"

    def test_domainId_is_new_but_expected(self, dictionary_table, rebuilt_table):
        # domainId isn't in the source table but IS generated internally by
        # build_mdjson and correctly surfaced by build_table -- expected,
        # not a bug.
        assert "domainId" not in dictionary_table.columns
        assert "domainId" in rebuilt_table.columns

    def test_cell_values_match_after_normalization(self, dictionary_table, rebuilt_table):
        """
        Full cell-by-cell comparison, tolerant of expected JSON-roundtrip
        stringification (e.g. 1.0 -> "1") and blank-value representation
        differences (NaN vs None vs "" vs "NA").
        """
        sort_keys = ["codeName", "dataType", "domainItem_value"]

        def normalize(df):
            out = df.drop(columns=["notes"], errors="ignore").copy()
            for k in sort_keys:
                if k not in out.columns:
                    out[k] = None

            def clean(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return ""
                s = str(v).strip()
                if s.upper() in ("NA", "NAN", "NONE"):
                    return ""
                if s.endswith(".0"):
                    try:
                        float(s)
                        s = s[:-2]
                    except ValueError:
                        pass
                return s

            for col in out.columns:
                out[col] = out[col].apply(clean)
            return out.sort_values(by=sort_keys, kind="stable").reset_index(drop=True)

        orig_norm = normalize(dictionary_table)
        rebuilt_norm = normalize(rebuilt_table)

        shared_cols = sorted(set(orig_norm.columns) & set(rebuilt_norm.columns))
        mismatches = {}
        for col in shared_cols:
            diff = orig_norm[col] != rebuilt_norm[col]
            if diff.any():
                mismatches[col] = int(diff.sum())

        assert not mismatches, f"Cell-level mismatches by column: {mismatches}"

    def test_built_json_is_valid_and_well_formed(self, built_mdjson):
        inner_str = built_mdjson["data"][0]["attributes"]["json"]
        inner = json.loads(inner_str)  # should not raise
        assert "dataDictionary" in inner
        assert "entity" in inner["dataDictionary"]
        attrs = inner["dataDictionary"]["entity"][0]["attribute"]
        assert len(attrs) > 0

    def test_domain_items_share_domainId_within_codeName(self, rebuilt_table):
        """
        Sanity check: all domain items belonging to the same codeName
        (e.g. all Sex values M/F/U) should share one domainId.
        """
        domain_rows = rebuilt_table[rebuilt_table["domainItem_value"] != "dataField"]
        for code, group in domain_rows.groupby("codeName"):
            ids = group["domainId"].dropna().unique()
            assert len(ids) <= 1, f"codeName={code!r} has inconsistent domainId values: {ids}"


# ---------------------------------------------------------------------------
# validate_table -- against the real dataset
# ---------------------------------------------------------------------------

class TestValidateTableReal:
    def test_returns_expected_columns(self, validation_warnings):
        assert list(validation_warnings.columns) == ["Num", "Variable", "Category", "Message"]

    def test_finds_missing_codeName_columns(self, validation_warnings):
        flagged = set(validation_warnings.loc[validation_warnings["Category"] == "codeName", "Variable"])
        # These dataset columns are known not to appear in the dictionary's codeName list
        for expected in ["LocCountry", "LocStateProvince", "LocCity", "LocArea"]:
            assert expected in flagged

    def test_finds_domain_value_mismatches(self, validation_warnings):
        flagged = set(validation_warnings.loc[validation_warnings["Category"] == "domainItem_value", "Variable"])
        assert "SpeciesCode" in flagged

    def test_integer_nan_upcast_not_falsely_flagged(self, validation_warnings):
        """
        Regression test: columns that are conceptually integer in the
        dictionary but stored as float64 in pandas purely because of NaN
        values (e.g. Month, Day, FatScore) should NOT trigger a
        dataType_RdataType warning.
        """
        rdatatype_flagged = set(
            validation_warnings.loc[validation_warnings["Category"] == "dataType_RdataType", "Variable"]
        )
        false_positive_candidates = {"Month", "Day", "FatScore", "BodyMolt", "RectWear",
                                      "BandNumberIn", "BandNumberOut"}
        offenders = rdatatype_flagged & false_positive_candidates
        assert not offenders, f"Integer/NaN upcast false positives reappeared: {offenders}"

    def test_genuine_dtype_mismatches_still_flagged(self, validation_warnings):
        """
        Columns that are genuinely the wrong type (not just NaN-upcast
        integers) should still be flagged.
        """
        rdatatype_flagged = set(
            validation_warnings.loc[validation_warnings["Category"] == "dataType_RdataType", "Variable"]
        )
        for expected in ["BanderID", "ScribeID", "BBLStatus"]:
            assert expected in rdatatype_flagged

    def test_allowNull_warnings_fire_with_string_values(self, validation_warnings):
        """
        Regression test: allowNull is stored as the string "TRUE"/"FALSE" in
        CSV-sourced dictionaries. The check must still fire correctly rather
        than silently no-op due to a strict `is False` comparison against a
        string.
        """
        allownull_flagged = set(
            validation_warnings.loc[validation_warnings["Category"] == "allowNull", "Variable"]
        )
        assert len(allownull_flagged) > 0

    def test_no_exceptions_on_full_real_dataset(self, dictionary_table, dataset_table):
        # Simply must not raise, on the full 2901x92 real dataset.
        result = validate_table(dictionary_table, dataset_table)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# validate_mdjson -- thin wrapper around build_table + validate_table
# ---------------------------------------------------------------------------

class TestValidateMdjson:
    def test_wrapper_matches_manual_pipeline(self, dictionary2_mdjson, dataset_table):
        """
        validate_mdjson(x, y) should produce identical results to manually
        calling build_table(x) then validate_table(table, y).
        """
        # Only run this against dictionary2 (json) since that's what
        # validate_mdjson expects as input -- reuse dataset_table just to
        # confirm the wrapper doesn't error, not for semantic correctness
        # (the dictionary2 domains don't correspond to dataset_table's cols).
        manual_table = build_table(dictionary2_mdjson, entity_num=1)
        manual_result = validate_table(manual_table, dataset_table)
        wrapper_result = validate_mdjson(dictionary2_mdjson, dataset_table, entity_num=1)

        pd.testing.assert_frame_equal(
            manual_result.reset_index(drop=True),
            wrapper_result.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# extract_mdjson
# ---------------------------------------------------------------------------

class TestExtractMdjson:
    def test_single_matching_record_returned_directly(self, dictionary2_mdjson):
        result = extract_mdjson(dictionary2_mdjson, record_type="dictionaries")
        assert len(result["data"]) == 1
        assert result["data"][0]["type"] == "dictionaries"

    def test_no_matching_records_returns_empty(self, dictionary2_mdjson):
        result = extract_mdjson(dictionary2_mdjson, record_type="contacts")
        assert result["data"] == []

    def test_multiple_records_with_explicit_selection(self):
        fake_input = {
            "data": [
                {"type": "dictionaries", "attributes": {"json": json.dumps({"citation": {"title": "Dict A"}})}},
                {"type": "dictionaries", "attributes": {"json": json.dumps({"citation": {"title": "Dict B"}})}},
            ]
        }
        result = extract_mdjson(fake_input, record_type="dictionaries", multiple=False, selection="Dict B")
        assert len(result["data"]) == 1
        picked = json.loads(result["data"][0]["attributes"]["json"])
        assert picked["citation"]["title"] == "Dict B"

    def test_all_flag_selects_everything(self):
        fake_input = {
            "data": [
                {"type": "dictionaries", "attributes": {"json": json.dumps({"citation": {"title": "Dict A"}})}},
                {"type": "dictionaries", "attributes": {"json": json.dumps({"citation": {"title": "Dict B"}})}},
            ]
        }
        result = extract_mdjson(fake_input, record_type="dictionaries", all=True)
        assert len(result["data"]) == 2


# ---------------------------------------------------------------------------
# build_mdjson -- input validation / error paths (synthetic data, no fixtures needed)
# ---------------------------------------------------------------------------

class TestBuildMdjsonValidation:
    def _minimal_valid_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "codeName": ["Foo"],
            "domainItem_name": ["dataField"],
            "domainItem_value": ["dataField"],
            "definition": ["A test field"],
            "dataType": ["character varying"],
            "allowNull": [True],
        })

    def test_missing_required_column_raises(self):
        df = self._minimal_valid_df().drop(columns=["dataType"])
        with pytest.raises(ValueError, match="missing required column"):
            build_mdjson(df, title="Test")

    def test_v1_template_yes_no_allowNull_raises(self):
        df = self._minimal_valid_df()
        df["allowNull"] = ["yes"]
        with pytest.raises(ValueError, match="only accept logical values"):
            build_mdjson(df, title="Test")

    def test_conflicting_domainItem_name_value_raises(self):
        df = self._minimal_valid_df()
        df["domainItem_value"] = ["somethingElse"]
        with pytest.raises(ValueError, match="[Cc]onflicting entries"):
            build_mdjson(df, title="Test")

    def test_missing_dataType_on_dataField_raises(self):
        df = self._minimal_valid_df()
        df["dataType"] = [None]
        with pytest.raises(ValueError, match="dataType"):
            build_mdjson(df, title="Test")

    def test_valid_minimal_input_succeeds(self):
        df = self._minimal_valid_df()
        result = build_mdjson(df, title="Minimal Test Dictionary")
        inner = json.loads(result["data"][0]["attributes"]["json"])
        assert inner["dataDictionary"]["citation"]["title"] == "Minimal Test Dictionary"
        attrs = inner["dataDictionary"]["entity"][0]["attribute"]
        assert len(attrs) == 1
        assert attrs[0]["codeName"] == "Foo"


# ---------------------------------------------------------------------------
# Regression tests for specific bugs found during manual testing
# ---------------------------------------------------------------------------

class TestRegressionFixes:
    def test_coerce_bool_column_handles_string_true_false(self):
        s = pd.Series(["TRUE", "FALSE", "True", "false", None, True, False])
        result = _coerce_bool_column(s)
        assert result.tolist()[:4] == [True, False, True, False]
        assert result.iloc[4] is None or pd.isna(result.iloc[4])
        assert result.iloc[5] is True
        assert result.iloc[6] is False

    def test_coerce_bool_column_leaves_other_values_untouched(self):
        s = pd.Series(["yes", "no", "maybe"])
        result = _coerce_bool_column(s)
        # Non-TRUE/FALSE strings pass through unchanged (the "yes"/"no"
        # v1-template check upstream is responsible for catching these).
        assert result.tolist() == ["yes", "no", "maybe"]

    def test_is_integer_like_true_for_native_int(self):
        s = pd.Series([1, 2, 3], dtype="int64")
        assert _is_integer_like(s) is True

    def test_is_integer_like_true_for_float_whole_numbers_with_nan(self):
        # This is the exact scenario from the real dataset: a column of
        # whole numbers (e.g. Month: 1-12) with some missing values, which
        # pandas silently upcasts to float64.
        s = pd.Series([1.0, 2.0, np.nan, 12.0])
        assert s.dtype == np.float64
        assert _is_integer_like(s) is True

    def test_is_integer_like_false_for_true_fractional_values(self):
        s = pd.Series([1.5, 2.0, 3.25])
        assert _is_integer_like(s) is False

    def test_is_integer_like_true_for_all_nan_column(self):
        s = pd.Series([np.nan, np.nan, np.nan])
        assert _is_integer_like(s) is True  # vacuously true, matches R's na.rm behavior intent

    def test_dtype_matches_none_rtype_returns_none(self):
        s = pd.Series([1, 2, 3])
        assert _dtype_matches(s, None) is None

    def test_dtype_matches_integer_rule_uses_integer_like(self):
        s = pd.Series([1.0, 2.0, np.nan])
        assert _dtype_matches(s, "is.integer") is True


# ---------------------------------------------------------------------------
# DATATYPE_RULES sanity checks (catches transcription errors in the
# hardcoded ported table)
# ---------------------------------------------------------------------------

class TestDatatypeRulesIntegrity:
    def test_forty_rows(self):
        assert len(DATATYPE_RULES) == 40

    def test_no_duplicate_datatype_values(self):
        assert DATATYPE_RULES["value"].is_unique

    def test_known_rows_present(self):
        values = set(DATATYPE_RULES["value"])
        for expected in ["character", "integer", "boolean", "date", "datetime",
                          "numeric", "decimal", "enum"]:
            assert expected in values

    def test_boolean_row_has_distinctValue_two(self):
        row = DATATYPE_RULES[DATATYPE_RULES["value"] == "boolean"].iloc[0]
        assert row["distinctValue"] == 2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))