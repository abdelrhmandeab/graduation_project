from core.shutdown import setup_shutdown
from core.logger import logger
from audio.wake_word import listen_for_wake_word
from audio.mic import record_until_silence
from audio.stt import transcribe
from core.command_router import route_command
from core.metrics import metrics

def run():
    setup_shutdown()
    logger.info("Jarvis started")

    while True:
        listen_for_wake_word()
        logger.info("Wake word detected")

        metrics.start("pipeline")

        record_until_silence()
        text = transcribe("input.wav")

        if not text:
            logger.warning("No valid speech detected")
            continue

        response = route_command(text)
        latency = metrics.end("pipeline")

        logger.info(f"Pipeline latency: {latency:.2f}s")
        print("🤖 Jarvis:", response)

if __name__ == "__main__":
    run()
