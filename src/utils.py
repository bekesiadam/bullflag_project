import logging
import sys
from pathlib import Path

def get_logger(name: str = "dl-project") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    h = logging.StreamHandler(sys.stdout)  # Docker ezt kapja el
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    return logger

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
