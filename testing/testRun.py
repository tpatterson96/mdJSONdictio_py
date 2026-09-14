import pandas as pd
import json
from mdjsondictio import build_mdjson, build_table, validate_table

# Round-trip test: table -> mdJSON -> table
dxnry = pd.read_csv("e.g.dictionary.csv")
new_json = build_mdjson(dxnry, title="Example Dictionary")
new_table = build_table(new_json, entity_num=1)

# Validation test against the real dataset
data = pd.read_csv("e.g.dataset.csv")
warnings = validate_table(dxnry, data)
print(warnings.head(20))
print(f"Total warnings: {len(warnings)}")

print(warnings["Category"].value_counts())
print(warnings[warnings["Category"].isin(["dataType_RdataType", "allowNull"])].to_string())