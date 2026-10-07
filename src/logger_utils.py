import sys
from datetime import datetime
from pathlib import Path

from config import (
    PROJECT_ROOT,
    EXPERIMENT_NAME,
)

# pasta base dos outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "experiments" / EXPERIMENT_NAME
LOGS_DIR = OUTPUTS_DIR / "logs"


def ensure_dirs():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


class TeeLogger:
    def __init__(self, filepath: Path):
        self.file = open(filepath, "a", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def setup_logging(script_name: str):
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{script_name}_{ts}.log"

    logger = TeeLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    print(f"[LOG] Experiência: {EXPERIMENT_NAME}")
    print(f"[LOG] A gravar logs em: {log_file}")
