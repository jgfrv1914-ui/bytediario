"""Utilidades compartidas por todo el pipeline: carga de config, logging y cola de trabajo."""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
LOGS_DIR = ROOT / "logs"
QUEUE_FILE = DATA_DIR / "queue.json"

load_dotenv(ROOT / ".env")


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _load_queue() -> list:
    if not QUEUE_FILE.exists():
        return []
    with open(QUEUE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_queue(queue: list) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, ensure_ascii=False, indent=2)


def add_to_queue(item: dict, status: str = "news_ready") -> str:
    queue = _load_queue()
    item_id = str(uuid.uuid4())[:8]
    item["id"] = item_id
    item["status"] = status
    item["created_at"] = datetime.now(timezone.utc).isoformat()
    queue.append(item)
    _save_queue(queue)
    return item_id


def get_item(item_id: str) -> dict:
    queue = _load_queue()
    for item in queue:
        if item["id"] == item_id:
            return item
    raise ValueError(f"No se encontró el item {item_id} en la cola")


def update_item(item_id: str, **fields) -> dict:
    queue = _load_queue()
    for item in queue:
        if item["id"] == item_id:
            item.update(fields)
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save_queue(queue)
            return item
    raise ValueError(f"No se encontró el item {item_id} en la cola")


def next_item_with_status(status: str) -> dict | None:
    queue = _load_queue()
    for item in queue:
        if item["status"] == status:
            return item
    return None


def ensure_dirs() -> None:
    for d in (DATA_DIR, OUTPUT_DIR, LOGS_DIR):
        d.mkdir(exist_ok=True, parents=True)
