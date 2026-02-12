from core.logger import logger
from core.command_classifier import classify
from os_control.file_ops import find_files
from llm.ollama_client import ask_llm

def route_command(text):
    command_type = classify(text)
    logger.info(f"Command classified as: {command_type}")

    if command_type == "OS_FILE_SEARCH":
        filename = text.replace("find file", "").strip()
        logger.info(f"Searching for file: {filename}")
        results = find_files(filename)
        return "\n".join(results) if results else "File not found."

    if command_type == "LLM_QUERY":
        logger.info("Forwarding query to LLM")
        return ask_llm(text)

    logger.warning("Unsupported command")
    return "Command not supported."
