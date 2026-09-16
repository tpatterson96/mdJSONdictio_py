<!-- 
Delete these comments when customizing your repository with this template.

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](http://keepachangelog.com/) and adheres to [Semantic Versioning](http://semver.org/).*

>-   *This file is for humans, not machines.*
>-   *There should be an entry for every single version.*
>-   *The same types of changes should be grouped.*
>-   *Versions and sections should be linkable.*
>-   *The latest version comes first.*
>-   *The release date of each version is displayed.*
>-   *Mention whether you follow Semantic Versioning.*
-->

# mdJSONDictio_py 0.2.4

Python version of mdJSONDictio 'R' version.  This python version captures the original functionality.

## Added

-   no new featuress

## Changed
This port aims to be functionally faithful to the original R package, with a few deliberate exceptions:

## Deprecated

-   none

## Removed

-   none

## Fixed

-  Some issues in the convertion from 'R' to python required attention are listed here:

1. **Fixed an indexing bug** present in the original `build.table.R`, which read entities/domains from `newlist[["entity"]]` / `newlist[["domain"]]` (top level) instead of the correct `newlist[["dataDictionary"]][["entity"]]` / `[["domain"]]`. The Python port uses the corrected path.
2. **Fixed a duplicate-domain-item artifact** present in `build.mdJSON.R`'s output (confirmed against real-world example files, where every generated domain's item list has its first item duplicated at the end). The Python port produces clean domain item lists without this duplication. **Note:** this means `build_mdjson()` output will not byte-match mdJSON files produced by the original R package for dictionaries with domains.
3. **Interactive GUI menus** (R's `utils::select.list(..., graphics = TRUE)`) are replaced with either an explicit `selection` argument or a plain-text console prompt, since there's no GUI picker equivalent outside RStudio.
4. **JSON boolean serialization** is handled natively — the Python port stores real `bool`/`int`/`float` values and lets `json.dumps()` serialize them correctly, rather than the original's approach of stringifying everything and then regex-patching `"true"`/`"false"` strings back into unquoted JSON literals.
5. **`is.decimal`**, referenced in the ported `datatype.rules` table for `numeric`/`decimal` types, is not a base R function and doesn't appear to be defined anywhere in the original package source — it likely would have errored if actually triggered in R. It's mapped to a generic numeric-type check in Python as a reasonable stand-in.
6. **Integer/NaN handling**: R integer vectors can hold `NA` while still reporting as integer type. pandas silently upcasts integer-valued columns to `float64` the moment they contain any missing values. The Python port's datatype validation accounts for this — a float64 column is still treated as "integer-like" if every non-null value has no fractional part — to avoid false-positive `dataType_RdataType` warnings on any dictionary column marked `integer` that has blank entries in the dataset (a common, otherwise-noisy case).
7. **Multibyte string validation** is a best-effort analog of R's `validEnc()`, not a byte-identical port.
