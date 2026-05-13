import argparse
import os

from core.orchestrator import run


def _parse_args():
    parser = argparse.ArgumentParser(description="Jarvis voice AI assistant")
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        default=False,
        help="Show intent/confidence overlay in console for demo presentations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.demo_mode:
        os.environ["JARVIS_DEMO_MODE"] = "1"
    run()
