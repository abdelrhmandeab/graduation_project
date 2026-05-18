import sys
import re

filepath = r"core/command_parser.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Pattern for conjunction_pattern block
con_search = r"conjunction_pattern = re\.compile\(.*?re\.IGNORECASE\s+\)"
new_con = r"""conjunction_pattern = re.compile(
        r'(?:\s+(?:and|or|then)\s+|\s+(?:\u0648|\u0623\u0648|\u062b\u0645)\s+)',
        re.IGNORECASE
    )"""

# Pattern for delete_ar_pattern block
del_search = r"delete_ar_pattern = re\.compile\(.*?re\.IGNORECASE,\s+\)"
new_del = r"""delete_ar_pattern = re.compile(
        r"^(?:\u0627\u062d\u0630\u0641|\u0623\u0632\u0644|\u0627\u0645\u0633\u062d)\s+(?:\u0645\u0644\u0641\u0627\u062a)\s+(.+?)(?:\s+(?:\u0641\u064i|\u0645\u0646)\s+(.+))?$",
        re.IGNORECASE,
    )"""

# Use lambda or string escape for replacement to avoid re.sub character escape issues
content = re.sub(con_search, lambda m: new_con, content, flags=re.DOTALL)
content = re.sub(del_search, lambda m: new_del, content, flags=re.DOTALL)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated patterns successfully")
