import re

with open("core/command_parser.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix search regex
old_search = r"re\.compile\(\s*r\"\^\(\?:find\|search\|locate\|look\\s\+for\)\\s\+\(\?:files\?\|documents\?\)\\s\+\(\?:about\|containing\|with\|named\)\?\\s\*\(\.\+\?\)\(\?:\\s\+in\\s\+\(\.\+\?\)\)\?\$\",\s*re\.IGNORECASE,\s*\)"
new_search_regex = 're.compile(\n            r"^(?:find|search|locate|look\\s+for)\\s+(?:files?|documents?|docs)\\s+(?:about|containing|with|named|for)?\\s*(.+?)(?:\\s+in\\s+(.+?))?$",\n            re.IGNORECASE,\n        )'

# Attempt replacement using a more flexible approach if direct string fail
matches = list(re.finditer(r"re\.compile\(\s*r\"\^\(\?:find\|search\|locate\|look\\s\+for\).+?re\.IGNORECASE,\s*\)", content, re.DOTALL))
if matches:
    content = content[:matches[0].start()] + new_search_regex + content[matches[0].end():]
    print("Updated English search pattern regex")

with open("core/command_parser.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Finished updates")
