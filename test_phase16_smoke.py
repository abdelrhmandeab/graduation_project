"""Phase 1.6 Consolidation Smoke Tests"""
from core.command_parser import parse_command

# Test cases for newly consolidated patterns
tests = [
    # Audio profile consolidation
    ("set audio ux profile responsive", "VOICE_COMMAND", "audio_ux_profile_set"),
    ("خلي تجربة الصوت سريع", "VOICE_COMMAND", "audio_ux_profile_set"),
    ("latency status", "VOICE_COMMAND", "latency_status"),
    ("الكمون عامل ايه", "VOICE_COMMAND", "latency_status"),
    
    # Browser consolidation
    ("new tab", "OS_SYSTEM_COMMAND", ""),
    ("search google for machine learning", "OS_SYSTEM_COMMAND", ""),
    ("دور على الويب python", "OS_SYSTEM_COMMAND", ""),
    ("open google.com", "OS_SYSTEM_COMMAND", ""),
    
    # Window consolidation
    ("maximize window", "OS_SYSTEM_COMMAND", ""),
    ("snap window left", "OS_SYSTEM_COMMAND", ""),
    ("focus chrome", "OS_SYSTEM_COMMAND", ""),
    
    # File operations consolidation
    ("create folder my documents", "OS_FILE_NAVIGATION", "create_directory"),
    ("move file.txt to desktop", "OS_FILE_NAVIGATION", "move_item"),
    ("rename old_name.txt to new_name.txt", "OS_FILE_NAVIGATION", "rename_item"),
    ("delete temp.txt permanently", "OS_FILE_NAVIGATION", "delete_item_permanent"),
]

passed = 0
failed = 0

for raw_cmd, exp_intent, exp_action in tests:
    try:
        result = parse_command(raw_cmd)
        if result.intent == exp_intent and (not exp_action or result.action == exp_action):
            passed += 1
            print(f"✓ {raw_cmd[:45]:<45} → {result.intent} {result.action}".rstrip())
        else:
            failed += 1
            print(f"✗ {raw_cmd[:45]:<45} → Got {result.intent} {result.action}, expected {exp_intent} {exp_action}".rstrip())
    except Exception as e:
        failed += 1
        print(f"✗ {raw_cmd[:45]:<45} → ERROR: {str(e)[:40]}")

print(f"\n{passed}/{passed+failed} tests passed")
