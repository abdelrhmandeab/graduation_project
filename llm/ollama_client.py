import subprocess
import time
from core.logger import logger
from core.config import LLM_MODEL

def ask_llm(prompt):
    try:
        start = time.time()
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        latency = time.time() - start
        logger.info(f"LLM latency: {latency:.2f}s")
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return "Sorry, I had an internal error."
