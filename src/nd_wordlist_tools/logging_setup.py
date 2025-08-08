import logging
import os
import sys

def setup_logging() -> None:
    level = os.getenv("ND_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
