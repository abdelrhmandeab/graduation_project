import os
from core.logger import logger
from core.config import MAX_FILE_RESULTS

def find_files(filename, search_path="C:\\"):
    try:
        matches = []
        for root, _, files in os.walk(search_path):
            for name in files:
                if filename.lower() in name.lower():
                    matches.append(os.path.join(root, name))
            if len(matches) >= MAX_FILE_RESULTS:
                break
        return matches
    except Exception as e:
        logger.error(f"File search failed: {e}")
        return []
