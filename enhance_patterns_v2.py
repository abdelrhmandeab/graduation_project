
with open("core/command_parser.py", "r", encoding="utf-8") as f:
    content = f.read()

batch_pattern_fix = """    # Phase 3: Batch file delete patterns (highest priority for batch)
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
"""

if 're.compile(r"^(?:index find|' in content:
    # Insert after the Search Index entry in _PRIORITY_REGEX_TABLE
    search_index_end = content.find('0.95,', content.find('"SEARCH_INDEX_COMMAND"'))
    if search_index_end != -1:
        # Find the next closing parenthesis and comma
        insert_pt = content.find('),', search_index_end) + 2
        if "OS_FILE_NAVIGATION_BATCH" not in content[insert_pt:insert_pt+500]:
            content = content[:insert_pt] + batch_pattern_fix + content[insert_pt:]
            print("Successfully inserted Batch patterns into _PRIORITY_REGEX_TABLE")

# Search fix
old_search = r"^(?:find|search|locate|look\\s+for)\\s+(?:files?|documents?)\\s+(?:about|containing|with|named)?\\s*(.+?)(?:\\s+in\\s+(.+?))?$"
new_search = r"^(?:find|search|locate|look\\s+for)\\s+(?:files?|documents?|docs)\\s+(?:about|containing|with|named|for)?\\s*(.+?)(?:\\s+in\\s+(.+?))?$"

if old_search in content:
    content = content.replace(old_search, new_search)
    print("Updated English search pattern")

with open("core/command_parser.py", "w", encoding="utf-8") as f:
    f.write(content)
