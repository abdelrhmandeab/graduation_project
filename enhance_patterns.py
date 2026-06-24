
# Read the file
with open("core/command_parser.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: Ensure batch patterns come early
batch_pattern_fix = '''    # Phase 3: Batch file delete patterns (highest priority for batch)
    (
        re.compile(
            r"^(?:delete|remove|rm)\\s+(?:files?|items?)\\s+(.+?)(?:\\s+from\\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION_BATCH",
        "delete_multiple",
        lambda m: {"files": m.group(1).strip(), "location": (m.group(2) or "").strip()},
        0.92,
    ),
    (
        re.compile(
            r"^(?:احذف|امسح)\\s+(?:ملفات|مستندات)\\s+(.+?)(?:\\s+(?:من|في)\\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION_BATCH",
        "delete_multiple",
        lambda m: {"files": m.group(1).strip(), "location": (m.group(2) or "").strip()},
        0.92,
    ),
'''

# Find where to insert batch patterns (right after index/search patterns in _PRIORITY_REGEX_TABLE)
insert_pos = content.find('    # Phase 3: Advanced search patterns')
if insert_pos != -1:
    # Check if already inserted to avoid duplicates
    if "OS_FILE_NAVIGATION_BATCH" not in content[:insert_pos + 500]:
        content = content[:insert_pos] + batch_pattern_fix + content[insert_pos:]
        print("Inserted batch patterns")
    else:
        print("Batch patterns already present")

# Fix 2: Improve search pattern regex
old_search = r"^(?:find|search|locate|look\\s+for)\\s+(?:files?|documents?)\\s+(?:about|containing|with|named)?\\s*(.+?)(?:\\s+in\\s+(.+?))?$"
new_search = r"^(?:find|search|locate|look\\s+for)\\s+(?:files?|documents?|docs)\\s+(?:about|containing|with|named|for)?\\s*(.+?)(?:\\s+in\\s+(.+?))?$"

if old_search in content:
    content = content.replace(old_search, new_search)
    print("Updated English search pattern")
else:
    # Try a slightly different version in case of escape character differences
    old_search_alt = old_search.replace("\\", "\\\\").replace("\\\\", "\\")
    if old_search_alt in content:
         content = content.replace(old_search_alt, new_search)
         print("Updated English search pattern (alt match)")

# Write back
with open("core/command_parser.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Patterns enhanced successfully")
