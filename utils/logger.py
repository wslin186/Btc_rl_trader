# btc_rl_trader/utils/logger.py
import logging
from pathlib import Path


def get_logger(name: str = "btc_rl", level: int = logging.INFO) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_dir / f"{name}.log", "a", "utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
