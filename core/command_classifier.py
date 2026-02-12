def classify(text: str):
    text = text.lower()

    if text.startswith("find file"):
        return "OS_FILE_SEARCH"

    if text.startswith("open"):
        return "OS_APP_OPEN"

    return "LLM_QUERY"
