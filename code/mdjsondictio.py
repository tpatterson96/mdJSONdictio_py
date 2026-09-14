"""
mdjsondictio
------------
Python port of the R package `mdJSONdictio`:
Tools to Extract, Build, Modify, and Validate mdEditor Dictionary records.

Original R package: https://github.com/hdvincelette/mdJSONdictio

Public functions
----------------
- extract_mdjson(x, record_type="dictionaries", multiple=True, all=False, selection=None)
- build_table(x, entity_num=1)
- build_mdjson(x, title=None)
- validate_table(dict_df, data_df)
- validate_mdjson(x, y, entity_num=1)

Notes on deviations from the original R source (see chat for full explanation):
1. Fixed a bug in build.table.R: entities/domains are read from
   newlist["dataDictionary"]["entity"/"domain"], not newlist["entity"/"domain"].
2. Interactive GUI record/entity selection (utils::select.list) is replaced with
   either an explicit argument or a plain-text console prompt.
3. build_mdjson stores real bool/int/float values in the JSON output instead of
   R's approach of stringifying everything and then regex-patching booleans back.
4. "is.decimal" (referenced in datatype.rules but not a base R function) is treated
   as a generic numeric check.
5. Multibyte string validation is a best-effort analog of R's validEnc(), not a
   byte-identical port.
"""

from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reference data: datatype.rules (ported from sysdata.rda via dput())
# ---------------------------------------------------------------------------

_DATATYPE_ROWS = [
    # value, definition, RdataType, maxLength, maxPrecision, minValue_unsigned, maxValue_unsigned, distinctValue, distinctLength
    ("character", "Fixed length character strings", "is.character", 255, None, None, None, None, 1),
    ("character varying", "Variable length character strings", "is.character", 65535, None, None, None, None, None),
    ("character large object", "Character large object; large objects are designed to hold extremely large column values", "is.character", 2147483647, None, None, None, None, None),
    ("national character", "Fixed length national character strings (2 byte unicode); UTF-8 or UTF-16 encoded characters as defined by the Unicode Standard", "is.character", 255, None, None, None, None, 1),
    ("national character varying", "Variable length national character strings (2 byte unicode); UTF-8 or UTF-16 encoded characters as defined by the Unicode Standard", "is.character", 65535, None, None, None, None, None),
    ("national character large object", "National character (2 byte unicode) large object; UTF-8 or UTF-16 encoded characters as defined by the Unicode Standard", "is.character", 2147483647, None, None, None, None, None),
    ("text", "Variable string to maximum of 65k characters", "is.character", 65535, None, None, None, None, None),
    ("tinytext", "Variable string to maximum of 255 characters", "is.character", 255, None, None, None, None, None),
    ("mediumtext", "Variable string to maximum of 16m characters", "is.character", 16777215, None, None, None, None, None),
    ("longtext", "Variable string to maximum of 4g characters", "is.character", 4294967295, None, None, None, None, None),
    ("binary", "Fixed length binary", "is.integer", 255, None, None, None, None, 1),
    ("binary varying", "Variable length binary", "is.integer", 32704, None, None, None, None, None),
    ("binary large object", "Binary large object to maximum of 65k bytes; large objects are designed to hold extremely large column values", "is.integer", 2147483647, None, None, None, None, None),
    ("mediumblob", "Binary large object to maximum of 16m bytes; large objects are designed to hold extremely large column values", "is.integer", 16777215, None, None, None, None, None),
    ("longblob", "Binary large object to maximum of 4g bytes; large objects are designed to hold extremely large column values", "is.integer", 4294967295, None, None, None, None, None),
    ("integer", "Integers number (+-2b)", "is.integer", None, None, None, None, None, None),
    ("tinyint", "Integer numbers (+-128)", "is.integer", None, None, -2147483648, 2147483648, None, None),
    ("mediumint", "Integer numbers (+-16k)", "is.integer", None, None, -128, 127, None, None),
    ("smallint", "Integer numbers (+-32k)", "is.integer", None, None, -8388608, 8388608, None, None),
    ("bigint", "Integer numbers (+-1e27)", "is.integer", None, None, -32767, 32767, None, None),
    ("float", "Floating point numbers", "is.numeric", None, 24, -9.22e18, -9.22e18, None, None),
    ("real", "Low precision floating point numbers", "is.numeric", None, None, -1.79e308, 1.79e308, None, None),
    ("double precision", "High precision floating point numbers", "is.double", None, 53, -3.4e38, 3.4e38, None, None),
    ("numeric", "Fixed precision and scale decimal numbers", "is.numeric", None, 38, -1.79e308, 1.79e308, None, None),
    ("decimal", "Fixed precision and scale decimal numbers (numeric alternate)", "is.decimal", None, 38, -1e38, 1e38, None, None),
    ("bit", "Fixed length bit strings", "is.integer", None, None, -1e38, 1e38, None, 1),
    ("bit varying", "Variable length bit strings", "is.integer", None, None, -9.22e18, -9.22e18, None, None),
    ("date", "Calendar date", None, None, None, -9.22e18, -9.22e18, None, None),
    ("time", "Clock time", None, None, None, None, None, None, None),
    ("datetime", "Date and time", None, None, None, None, None, None, None),
    ("timestamp", "Number of seconds since the unix epoch (1970-01-01t00:00:00 utc)", "is.numeric", None, None, None, None, None, None),
    ("year", "Year", "is.integer", None, None, 1, 9999, None, None),
    ("interval", "Time intervals (i.e., length of time in year, month, day, hour, minute, or second)", "is.numeric", None, None, None, None, None, None),
    ("interval day", "Day intervals", "is.numeric", None, None, None, None, None, None),
    ("interval year", "Year intervals", "is.numeric", None, None, None, None, None, None),
    ("currency", "Monetary value", "is.numeric", 19, None, -3.96e28, 3.96e28, None, None),
    ("money", "Monetary value", "is.numeric", None, 4, -9.22e14, 9.22e14, None, None),
    ("boolean", "Boolean value (yes/no)", None, None, None, None, None, 2, None),
    ("xml", "Extensible Markup Language (XML) formatted data", None, None, None, None, None, None, None),
    ("enum", "List of possible values: enum('a','b','c')", None, None, None, None, None, 65535, None),
]

DATATYPE_RULES = pd.DataFrame(
    _DATATYPE_ROWS,
    columns=[
        "value", "definition", "RdataType", "maxLength", "maxPrecision",
        "minValue_unsigned", "maxValue_unsigned", "distinctValue", "distinctLength",
    ],
)


# ---------------------------------------------------------------------------
# Reference data: blankjson template (ported from sysdata.rda via dput())
# ---------------------------------------------------------------------------

BLANK_DICTIONARY_TEMPLATE = {
    "dictionaryId": None,
    "dataDictionary": {
        "citation": {
            "title": None,
            "date": [{"date": None, "dateType": "creation"}],
        },
        "domain": [
            {
                "domainId": None,
                "domainItem": [{"name": None, "value": None, "definition": None}],
                "codeName": None,
                "description": None,
            }
        ],
        "entity": [
            {
                "entityId": None,
                "attribute": [
                    {
                        "allowNull": None,
                        "codeName": None,
                        "dataType": None,
                        "definition": None,
                        "domainId": None,
                        "units": None,
                        "unitsResolution": None,
                        "isCaseSensitive": None,
                        "missingValue": None,
                        "minValue": None,
                        "maxValue": None,
                        "fieldWidth": None,
                    }
                ],
            }
        ],
    },
}


def _make_blank_outer() -> dict:
    """Equivalent of R's `blankjson` object (the outer mdJSON envelope)."""
    return {
        "data": [
            {
                "id": "NA",
                "attributes": {
                    "profile": "org.adiwg.profile.full",
                    "json": json.dumps(BLANK_DICTIONARY_TEMPLATE),
                    "date-updated": {},
                },
                "type": "dictionaries",
            }
        ]
    }


# ---------------------------------------------------------------------------
# extract.mdJSON
# ---------------------------------------------------------------------------

def extract_mdjson(x: dict, record_type="dictionaries", multiple: bool = True,
                    all: bool = False, selection=None) -> dict:
    """
    Plucks one or more mdJSON records from an mdJSON file comprised of other
    types of records.

    Parameters
    ----------
    x : dict
        Parsed mdJSON object (e.g. from json.load).
    record_type : str or list[str]
        Type(s) of records to extract, e.g. "dictionaries", "records", "contacts".
    multiple : bool
        Whether multiple records may be selected in the interactive fallback.
    all : bool
        If True, automatically select all matching records (skips selection).
    selection : list[str] | list[int] | str | int | None
        Programmatic selection by name or 0-based index. Use this to avoid the
        interactive console prompt in scripts/tests.

    Returns
    -------
    dict with a "data" key containing the selected record(s).
    """
    if isinstance(record_type, str):
        record_type = [record_type]

    filtered_records = [
        rec for rec in x.get("data", []) if rec.get("type") in record_type
    ]

    if len(filtered_records) == 0:
        joined = " or ".join(record_type)
        print(f"Operation canceled. No {joined} were found")
        return {"data": []}

    if len(filtered_records) == 1:
        return {"data": filtered_records}

    # Multiple candidates -- need a selection
    names = []
    for rec in filtered_records:
        record_list = json.loads(rec["attributes"]["json"])
        if rec["type"] == "dictionaries":
            names.append(record_list.get("citation", {}).get("title"))
        elif rec["type"] == "records":
            names.append(
                record_list.get("metadata", {})
                .get("resourceInfo", {})
                .get("citation", {})
                .get("title")
            )
        elif rec["type"] == "contacts":
            names.append(record_list.get("name"))
        else:
            names.append(None)

    if all:
        chosen_indices = list(range(len(filtered_records)))
    elif selection is not None:
        sel = selection if isinstance(selection, (list, tuple)) else [selection]
        chosen_indices = [s if isinstance(s, int) else names.index(s) for s in sel]
        if not multiple:
            chosen_indices = chosen_indices[:1]
    else:
        print("The mdJSON list object contains more than one record.")
        for i, n in enumerate(names):
            print(f"  [{i}] {n}")
        raw = input("Select record index(es), comma-separated: ")
        chosen_indices = [int(v.strip()) for v in raw.split(",") if v.strip()]
        if not multiple:
            chosen_indices = chosen_indices[:1]

    return {"data": [filtered_records[i] for i in chosen_indices]}


# ---------------------------------------------------------------------------
# build.table
# ---------------------------------------------------------------------------

_TABLE_COLUMNS = [
    "codeName", "domainItem_name", "domainItem_value", "definition", "dataType",
    "allowNull", "units", "unitsResolution", "isCaseSensitive", "fieldWidth",
    "missingValue", "minValue", "maxValue", "domainId",
]


def build_table(x: dict, entity_num: int = 1) -> pd.DataFrame:
    """
    Translates a parsed mdJSON data dictionary (dict) into a pandas DataFrame.

    Parameters
    ----------
    x : dict
        Parsed mdJSON object.
    entity_num : int
        1-based index of the entity to use, if the file contains more than one.

    Returns
    -------
    pandas.DataFrame
    """
    if len(x.get("data", [])) > 1:
        json_dictionary = extract_mdjson(x, record_type="dictionaries", multiple=False)
    else:
        json_dictionary = x

    if json_dictionary["data"][0]["type"] != "dictionaries":
        raise ValueError("Dictionary not detected.")

    dictionary_string = json_dictionary["data"][0]["attributes"]["json"]

    if "entity" not in dictionary_string:
        raise ValueError('No Entity detected.')
    if "attribute" not in dictionary_string:
        raise ValueError('Entity requires at least one attribute.')
    if "dataType" not in dictionary_string:
        raise ValueError('Entity requires at least one attribute.')

    new_list = json.loads(dictionary_string)

    # NOTE: fixed from the R original, which read newlist["entity"]/["domain"]
    # (top level) instead of the correct nested path.
    entity_list = new_list["dataDictionary"]["entity"]
    domain_list = new_list["dataDictionary"].get("domain")

    if entity_num is None:
        entity_num = 1
    idx = entity_num - 1

    rows = []
    domaincount = 0
    attributes = entity_list[idx].get("attribute", []) or []

    for attr in attributes:
        row = {c: None for c in _TABLE_COLUMNS}
        for col, entry in attr.items():
            if entry not in (None, [], ""):
                row[col] = entry
        row["_domainNum"] = 0
        if row.get("domainId") is not None:
            domaincount += 1
            row["_domainNum"] = domaincount
        row["domainItem_name"] = "dataField"
        row["domainItem_value"] = "dataField"
        rows.append(row)

    if domain_list:
        for domain in domain_list:
            for item in (domain.get("domainItem") or []):
                row = {c: None for c in _TABLE_COLUMNS}
                row["domainId"] = domain.get("domainId")
                row["codeName"] = domain.get("codeName")
                for col, entry in item.items():
                    key = {"name": "domainItem_name", "value": "domainItem_value"}.get(col, col)
                    if entry not in (None, [], ""):
                        row[key] = entry
                row["_domainNum"] = 0
                rows.append(row)

    df = pd.DataFrame(rows, columns=_TABLE_COLUMNS + ["_domainNum"])
    # NB: R's dplyr::arrange() sorts NA values *last* by default -- match that here.
    df = df.sort_values(
        by=["codeName", "dataType", "_domainNum"], na_position="last", kind="stable"
    ).drop(columns=["_domainNum"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# build.mdJSON
# ---------------------------------------------------------------------------

def build_mdjson(x: pd.DataFrame, title: str | None = None) -> dict:
    """
    Translates a tabular data dictionary (DataFrame) into a dict that can be
    serialized to mdJSON and imported to mdEditor as a Dictionary record.

    Parameters
    ----------
    x : pandas.DataFrame
        Tabular data dictionary, formatted to the mdJSONdictio template
        (columns: codeName, domainItem_name, domainItem_value, definition,
        dataType, allowNull, units, unitsResolution, minValue, maxValue,
        missingValue, fieldWidth, isCaseSensitive, notes[optional]).
    title : str, optional
        Title of the Dictionary record in mdEditor.

    Returns
    -------
    dict -- the full mdJSON envelope (equivalent of R's `newjson`).
    """
    df = x.copy()

    if title is None:
        title = "Untitled Dictionary"

    for col in ["codeName", "domainItem_name", "domainItem_value", "definition"]:
        if col in df.columns:
            df[col] = df[col].fillna("NA")

    if "allowNull" in df.columns:
        vals = set(df["allowNull"].dropna().astype(str).str.lower().unique())
        if "yes" in vals or "no" in vals:
            raise ValueError(
                "'allowNull' and 'isCaseSensitive' only accept logical values "
                "(True/False). Correct these fields before continuing. Use the "
                "v2 tabular data dictionary template."
            )
        df["allowNull"] = _coerce_bool_column(df["allowNull"])
    if "isCaseSensitive" in df.columns:
        df["isCaseSensitive"] = _coerce_bool_column(df["isCaseSensitive"])

    required_cols = ["codeName", "domainItem_name", "domainItem_value",
                      "definition", "dataType", "allowNull"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Data frame missing required column(s): {', '.join(missing_cols)}.")

    for i, row in df.iterrows():
        if row["domainItem_name"] == "dataField" and row["domainItem_value"] != "dataField":
            raise ValueError(
                f"Conflicting entries at row {i + 1}: domainItem_name='dataField' "
                f"but domainItem_value='{row['domainItem_value']}'."
            )
        if row["domainItem_name"] != "dataField" and row["domainItem_value"] == "dataField":
            raise ValueError(
                f"Conflicting entries at row {i + 1}: domainItem_value='dataField' "
                f"but domainItem_name='{row['domainItem_name']}'."
            )
        if row["domainItem_name"] == "dataField" and pd.isna(row.get("dataType")):
            raise ValueError(f"Required field incomplete: dataType is missing at row {i + 1}.")
        if row["domainItem_name"] == "dataField" and pd.isna(row.get("allowNull")):
            raise ValueError(f"Required field incomplete: allowNull is missing at row {i + 1}.")
        if isinstance(row.get("fieldWidth"), str):
            raise ValueError(f"fieldWidth has an incompatible data type at row {i + 1}.")
        if isinstance(row.get("unitsResolution"), str):
            raise ValueError(f"unitsResolution has an incompatible data type at row {i + 1}.")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda v: v.replace('"', "'") if isinstance(v, str) else v)
    df = df.drop(columns=[c for c in ["notes"] if c in df.columns])

    df["domainId"] = None
    code_counts = df["codeName"].value_counts()
    for i, row in df.iterrows():
        if row["domainItem_name"] == "dataField" and code_counts.get(row["codeName"], 0) > 1:
            df.at[i, "domainId"] = "true"

    rec_id = str(uuid.uuid4()).split("-")[0]
    dictionary_id = str(uuid.uuid4())
    entity_id = str(uuid.uuid4())

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%dT%H:%M") + ":00.000Z"

    domain_rows = df[(df["domainItem_value"] == "dataField") & (df["domainId"] == "true")]
    code_to_uuid = {code: str(uuid.uuid4()) for code in domain_rows["codeName"].tolist()}
    for i, row in df.iterrows():
        if row["domainItem_value"] == "dataField" and row["codeName"] in code_to_uuid:
            df.at[i, "domainId"] = code_to_uuid[row["codeName"]]

    df["entityNum"] = 0
    df["domainNum"] = 0
    entitycount = 0
    for i, row in df.iterrows():
        if row["domainItem_name"] == "dataField":
            entitycount += 1
            df.at[i, "entityNum"] = entitycount

    domainref_a = df[(df["domainItem_name"] == "dataField") & (df["domainId"].notna())].copy()

    domaincount = 0
    for i, row in df.iterrows():
        if row["domainItem_name"] != "dataField":
            match = domainref_a[domainref_a["codeName"] == row["codeName"]]
            if not match.empty:
                domaincount += 1
                df.at[i, "entityNum"] = match.iloc[0]["entityNum"]
                df.at[i, "domainNum"] = domaincount

    domainref_i = df[df["domainNum"] != 0].copy()
    entityref = df[df["domainItem_name"] == "dataField"].reset_index(drop=True)

    for _, erow in entityref.iterrows():
        domainref_i.loc[domainref_i["codeName"] == erow["codeName"], "domainId"] = erow["domainId"]

    dictionary = copy.deepcopy(BLANK_DICTIONARY_TEMPLATE)
    dictionary["dataDictionary"]["entity"][0]["entityId"] = entity_id

    attr_fields = list(dictionary["dataDictionary"]["entity"][0]["attribute"][0].keys())

    attributes = []
    for _, erow in entityref.iterrows():
        attr = {}
        for field in attr_fields:
            if field in erow.index:
                val = erow[field]
                if pd.isna(val):
                    continue
                attr[field] = str(val)
        attributes.append(attr)
    dictionary["dataDictionary"]["entity"][0]["attribute"] = attributes

    for attr in dictionary["dataDictionary"]["entity"][0]["attribute"]:
        for numfield in ("unitsResolution", "fieldWidth"):
            if numfield in attr:
                try:
                    s = attr[numfield]
                    attr[numfield] = float(s) if "." in s else int(s)
                except (ValueError, TypeError):
                    del attr[numfield]
        for boolfield in ("allowNull", "isCaseSensitive"):
            if boolfield in attr:
                v = str(attr[boolfield]).strip().lower()
                if v in ("true", "false"):
                    attr[boolfield] = (v == "true")

    if not domainref_a.empty:
        domains = []
        for _, drow in domainref_a.iterrows():
            items_df = domainref_i[domainref_i["entityNum"] == drow["entityNum"]]
            domain_items = []
            for _, irow in items_df.iterrows():
                domain_items.append({
                    "name": str(irow["domainItem_name"]),
                    "value": str(irow["domainItem_value"]),
                    "definition": (str(irow["definition"]) if pd.notna(irow["definition"]) else None),
                })
            domains.append({
                "codeName": str(drow["codeName"]),
                "domainId": str(drow["domainId"]),
                "description": (str(drow["definition"]) if pd.notna(drow["definition"]) else None),
                "domainItem": domain_items,
            })
        dictionary["dataDictionary"]["domain"] = domains
    else:
        dictionary["dataDictionary"].pop("domain", None)

    dictionary["dictionaryId"] = dictionary_id
    dictionary["dataDictionary"]["citation"]["title"] = title
    dictionary["dataDictionary"]["citation"]["date"][0]["date"] = date_str

    outer = _make_blank_outer()
    outer["data"][0]["id"] = rec_id
    outer["data"][0]["attributes"]["date-updated"] = date_str
    outer["data"][0]["attributes"]["json"] = json.dumps(dictionary)

    return outer


# ---------------------------------------------------------------------------
# validate.table
# ---------------------------------------------------------------------------

def _is_integer_like(series: pd.Series) -> bool:
    """
    True if the column is an integer dtype, OR a float dtype whose non-null
    values all happen to be whole numbers.

    This exists because R integer vectors can natively hold NA and still
    report is.integer()==TRUE, whereas pandas silently upcasts an
    integer-valued column to float64 the moment it contains any missing
    values (no native int+NaN support pre-nullable-Int64 dtype). Without this
    accommodation, any dictionary column marked "integer" that has *any*
    blank entries in the dataset (extremely common) would incorrectly fail
    the datatype check every time.
    """
    if pd.api.types.is_integer_dtype(series):
        return True
    if pd.api.types.is_float_dtype(series):
        non_null = series.dropna()
        if len(non_null) == 0:
            return True
        return bool(np.all(np.mod(non_null, 1) == 0))
    return False


def _dtype_matches(series: pd.Series, rtype: str | None):
    """Analog of calling an R RdataType predicate (e.g. is.character) on a column."""
    if rtype is None or (isinstance(rtype, float) and pd.isna(rtype)):
        return None  # no check defined -- caller should skip
    if rtype == "is.character":
        return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    if rtype == "is.integer":
        return _is_integer_like(series)
    if rtype == "is.numeric":
        return pd.api.types.is_numeric_dtype(series)
    if rtype == "is.double":
        return pd.api.types.is_float_dtype(series)
    if rtype == "is.decimal":  # not a base R function; treated as numeric (assumption)
        return pd.api.types.is_numeric_dtype(series)
    return True


def _is_iso_date(v) -> bool:
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%j"):
        try:
            datetime.strptime(str(v), fmt)
            return True
        except ValueError:
            continue
    return False


def _coerce_bool_column(series: pd.Series) -> pd.Series:
    """
    Normalizes a column that should hold logical TRUE/FALSE values but may have
    arrived as native bool, or as strings/mixed values (common when reading from
    CSV, which has no boolean dtype). Non-boolean-looking values are left as-is
    so the v1-template ('yes'/'no') check upstream can still catch them.
    """
    def coerce(v):
        if isinstance(v, bool) or pd.isna(v):
            return v
        s = str(v).strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
        return v
    return series.apply(coerce)


def validate_table(dict_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares a tabular data dictionary to a tabular dataset and summarizes
    discrepancies in a DataFrame.

    Parameters
    ----------
    dict_df : pandas.DataFrame
        Tabular data dictionary (mdJSONdictio template format).
    data_df : pandas.DataFrame
        The dataset to validate.

    Returns
    -------
    pandas.DataFrame with columns: Num, Variable, Category, Message.
    """
    dict_df = dict_df.copy()
    data_df = data_df.copy()

    if "allowNull" in dict_df.columns:
        vals = set(dict_df["allowNull"].dropna().astype(str).str.lower().unique())
        if "yes" in vals or "no" in vals:
            raise ValueError(
                "'allowNull' and 'isCaseSensitive' only accept logical values "
                "(True/False). Use the v2 tabular data dictionary template."
            )
        # CSV/Excel often round-trips booleans as strings ("TRUE"/"FALSE") rather
        # than a native boolean dtype -- normalize so downstream `is False`/`is True`
        # comparisons behave correctly regardless of source format.
        dict_df["allowNull"] = _coerce_bool_column(dict_df["allowNull"])

    warnings_rows: list[dict] = []

    def add_warning(variable, category, message):
        warnings_rows.append({
            "Num": len(warnings_rows) + 1,
            "Variable": variable,
            "Category": category,
            "Message": message,
        })

    data_df = data_df.replace("", np.nan)
    dict_df = dict_df.replace("", np.nan)

    dict_vars = dict_df["codeName"].unique().tolist()
    dict_domain = dict_df[dict_df["domainItem_value"] != "dataField"]
    dict_datafield = dict_df[dict_df["domainItem_value"] == "dataField"]
    domain_groups = {code: grp for code, grp in dict_domain.groupby("codeName")}

    # missingValue -> NaN
    dict_miss = dict_datafield[dict_datafield["missingValue"].notna()]
    data_na = data_df.copy()
    for col in data_na.columns:
        for _, r in dict_miss.iterrows():
            if col == r["codeName"] and r["missingValue"] in data_na[col].unique():
                data_na[col] = data_na[col].replace(r["missingValue"], np.nan)
                data_na[col] = pd.to_numeric(data_na[col], errors="ignore")

    # --- Required fields: codeName, domainItem_value, allowNull ---

    for col in data_na.columns:
        if col not in dict_vars:
            add_warning(col, "codeName", 'Dataset variable not listed under "codeName" in dictionary')

    if dict_df["domainItem_value"].nunique() != 1:
        for col in data_na.columns:
            if col in domain_groups:
                allowed = set(domain_groups[col]["domainItem_value"].dropna().unique())
                present = set(data_na[col].dropna().unique())
                diff = present - allowed
                if diff:
                    quoted = ", ".join(f"'{d}'" for d in diff)
                    add_warning(col, "domainItem_value",
                                f'Dataset variable contains entry value(s) not listed under '
                                f'"domainItem_Value" in dictionary: {quoted}')

    for col in data_na.columns:
        for _, r in dict_datafield[dict_datafield["codeName"] == col].iterrows():
            if r["allowNull"] is False and data_na[col].isna().any():
                add_warning(col, "allowNull",
                            'Dataset variable contains blank values, which is inconsistent '
                            'with "allowNull" in dictionary ')

    # --- Required fields: dataType ---

    for col in data_na.columns:
        for _, r in dict_datafield[dict_datafield["codeName"] == col].iterrows():
            rule = DATATYPE_RULES[DATATYPE_RULES["value"] == r["dataType"]]
            if rule.empty:
                continue
            rtype = rule.iloc[0]["RdataType"]
            result = _dtype_matches(data_na[col], rtype)
            if result is False and data_na[col].dtype != bool:
                add_warning(col, "dataType_RdataType",
                            f"Dataset variable is detected as a different datatype "
                            f"({data_na[col].dtype}) than indicated in dictionary "
                            f"({r['dataType']})")

    # date / datetime
    for col in data_na.columns:
        matches = dict_datafield[
            (dict_datafield["codeName"] == col) & (dict_datafield["dataType"].isin(["date", "datetime"]))
        ]
        if not matches.empty:
            vals = data_na[col].dropna()
            if len(vals) > 0:
                bad = (
                    any(not _is_iso_date(v) for v in vals)
                    or any("/" in str(v) or "AM" in str(v) or "PM" in str(v) for v in vals)
                )
                if bad:
                    add_warning(col, "dataType_datetime",
                                'Dataset variable has entry value(s) not in standard ISO 1806 datetime format')

    # time
    for col in data_na.columns:
        matches = dict_datafield[(dict_datafield["codeName"] == col) & (dict_datafield["dataType"] == "time")]
        if not matches.empty:
            vals = data_na[col].dropna()
            if any("AM" in str(v) or "PM" in str(v) for v in vals):
                add_warning(col, "dataType_datetime",
                            'Dataset variable has entry value(s) not in standard ISO 1806 time format')

    # maxLength
    for col in data_na.columns:
        for _, r in dict_datafield[dict_datafield["codeName"] == col].iterrows():
            rule = DATATYPE_RULES[DATATYPE_RULES["value"] == r["dataType"]]
            if rule.empty or pd.isna(rule.iloc[0]["maxLength"]):
                continue
            max_allowed = rule.iloc[0]["maxLength"]
            lengths = data_na[col].dropna().astype(str).map(len)
            if len(lengths) and lengths.max() > max_allowed:
                add_warning(col, "dataType_maxLength",
                            f"Dataset variable has entry value(s) with a greater length "
                            f"({lengths.max()}) than allowed for the datatype ({max_allowed})")

    # maxPrecision
    for col in data_na.columns:
        if data_na[col].dtype == object:
            continue
        for _, r in dict_datafield[dict_datafield["codeName"] == col].iterrows():
            rule = DATATYPE_RULES[DATATYPE_RULES["value"] == r["dataType"]]
            if rule.empty or pd.isna(rule.iloc[0]["maxPrecision"]):
                continue
            max_prec = rule.iloc[0]["maxPrecision"]
            vals = data_na[col].dropna().astype(str).map(lambda s: len(s.lstrip("0")))
            if len(vals) and vals.max() > max_prec:
                add_warning(col, "dataType_maxPrecision",
                            f"Dataset variable has entry value(s) with greater precision "
                            f"({vals.max()}) than allowed for the datatype ({max_prec})")

    # minValue / maxValue (datatype-rule based)
    for col in data_na.columns:
        if data_na[col].dtype == object:
            continue
        for _, r in dict_datafield[dict_datafield["codeName"] == col].iterrows():
            rule = DATATYPE_RULES[DATATYPE_RULES["value"] == r["dataType"]]
            if rule.empty:
                continue
            min_allowed = rule.iloc[0]["minValue_unsigned"]
            max_allowed = rule.iloc[0]["maxValue_unsigned"]
            vals = data_na[col].dropna()
            if len(vals) == 0:
                continue
            if pd.notna(min_allowed) and vals.min() < min_allowed:
                add_warning(col, "dataType_minValue",
                            f"Dataset variable has entry value(s) with a smaller value "
                            f"({vals.min()}) than allowed for the datatype ({min_allowed})")
            if pd.notna(max_allowed) and vals.max() > max_allowed:
                add_warning(col, "dataType_maxValue",
                            f"Dataset variable has entry value(s) with a greater value "
                            f"({vals.max()}) than allowed for the datatype ({max_allowed})")

    # distinctValue / distinctLength
    for col in data_na.columns:
        for _, r in dict_datafield[dict_datafield["codeName"] == col].iterrows():
            rule = DATATYPE_RULES[DATATYPE_RULES["value"] == r["dataType"]]
            if rule.empty:
                continue
            dv, dl = rule.iloc[0]["distinctValue"], rule.iloc[0]["distinctLength"]
            vals = data_na[col].dropna()
            if len(vals) == 0:
                continue
            if pd.notna(dv):
                n = vals.nunique()
                if n > dv:
                    add_warning(col, "dataType_distinctValue",
                                f"Dataset variable has a greater number of distinct values "
                                f"({n}) than allowed for the datatype ({dv})")
            if pd.notna(dl):
                n = vals.astype(str).map(len).nunique()
                if n > dl:
                    add_warning(col, "dataType_distinctLength",
                                f'Dataset variable has entry values with more than one length '
                                f'({n}) while the datatype indicates values should be "fixed length"')

    # --- Optional fields ---

    # unitsResolution
    for col in data_na.columns:
        for _, r in dict_df[(dict_df["codeName"] == col) & (dict_df["unitsResolution"].notna())].iterrows():
            def ndecimal(v):
                s = str(v)
                return len(s.split(".")[1]) if "." in s else 0
            data_ndec = data_na[col].dropna().map(ndecimal)
            dict_ndec = ndecimal(r["unitsResolution"])
            if len(data_ndec) == 0:
                continue
            if data_ndec.min() < dict_ndec:
                add_warning(col, "unitsResolution",
                            'Dataset variable contains entry value(s) with lower resolution '
                            'than "unitsResolution" in dictionary')
            if data_ndec.max() > dict_ndec:
                add_warning(col, "unitsResolution",
                            'Dataset variable contains entry value(s) with higher resolution '
                            'than "unitsResolution" in dictionary')

    # fieldWidth
    for col in data_na.columns:
        for _, r in dict_df[(dict_df["codeName"] == col) & (dict_df["fieldWidth"].notna())].iterrows():
            vals = data_na[col].dropna().astype(str).map(len)
            if len(vals) and vals.max() > r["fieldWidth"]:
                add_warning(col, "fieldWidth",
                            'Dataset variable contains entry value(s) that exceeds "fieldWidth" in dictionary')

    # missingValue
    for col in data_df.columns:
        for _, r in dict_df[(dict_df["codeName"] == col) & (dict_df["missingValue"].notna())].iterrows():
            if data_df[col].isna().any():
                add_warning(col, "missingValue",
                            f'Dataset variable contains blank values rather than "missingValue" '
                            f'in dictionary: {r["missingValue"]}')

    # minValue / maxValue (dictionary-level)
    for col in data_na.columns:
        for _, r in dict_df[(dict_df["codeName"] == col) & (dict_df["minValue"].notna())].iterrows():
            vals = data_na[col].dropna()
            if len(vals) and vals.min() < float(r["minValue"]):
                add_warning(col, "minValue", 'Dataset variable contains entry value(s) less than "minValue" in dictionary')

    for col in data_na.columns:
        for _, r in dict_df[(dict_df["codeName"] == col) & (dict_df["maxValue"].notna())].iterrows():
            vals = data_na[col].dropna()
            if len(vals) and vals.max() > float(r["maxValue"]):
                add_warning(col, "maxValue", 'Dataset variable contains entry value(s) greater than "maxValue" in dictionary')

    return pd.DataFrame(warnings_rows, columns=["Num", "Variable", "Category", "Message"])


# ---------------------------------------------------------------------------
# validate.mdJSON
# ---------------------------------------------------------------------------

def validate_mdjson(x: dict, y: pd.DataFrame, entity_num: int = 1) -> pd.DataFrame:
    """
    Compares an mdJSON data dictionary to a tabular dataset and summarizes
    discrepancies in a DataFrame.
    """
    dict_table = build_table(x, entity_num=entity_num)
    return validate_table(dict_table, y)


# ---------------------------------------------------------------------------
# Example usage (mirrors the R package's help-page examples)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # import pandas as pd
    #
    # input_table = pd.read_excel("e.g.dictionary.xlsx")
    # new_dict = build_mdjson(input_table, title="Example Dictionary")
    # with open("e.g.dictionary.json", "w") as f:
    #     json.dump(new_dict, f, indent=2)
    #
    # with open("e.g.dictionary2.json") as f:
    #     input_list = json.load(f)
    # new_table = build_table(input_list, entity_num=1)
    # new_table.to_csv("e.g.dictionary2.csv", index=False)
    #
    # input_dxnry = pd.read_excel("e.g.dictionary.xlsx")
    # input_data = pd.read_csv("e.g.dataset.csv")
    # all_warnings = validate_table(input_dxnry, input_data)
    # all_warnings.to_csv("e.g.warnings2.csv", index=False)
    pass
