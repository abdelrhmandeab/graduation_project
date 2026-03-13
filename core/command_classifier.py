from core.command_parser import parse_command


def classify(text: str):
    return parse_command(text).intent
