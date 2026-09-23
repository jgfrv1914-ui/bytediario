"""
Paso 1: trae noticias tech recientes de las fuentes RSS configuradas,
filtra por antigüedad, prioriza por palabras clave, y elige las top N
para combinarlas en un solo video del día.

Uso:
    python scripts/01_fetch_news.py
"""
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser

from utils import add_to_queue, ensure_dirs, get_logger, load_config

logger = get_logger("01_fetch_news")


def _entry_age_hours(entry) -> float:
    published = entry.get("published") or entry.get("updated")
    if not published:
        return 0.0
    try:
        dt = parsedate_to_datetime(published)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except Exception:
        return 0.0
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600


def _score(entry, keywords: list[str]) -> int:
    text = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
    return sum(1 for kw in keywords if kw.lower() in text)


def fetch_top_stories(cfg: dict) -> list[dict]:
    news_cfg = cfg["news"]
    all_entries = []

    for source in cfg["news_sources"]:
        logger.info(f"Leyendo feed: {source['name']}")
        try:
            feed = feedparser.parse(source["url"])
        except Exception as e:
            logger.warning(f"No se pudo leer {source['name']}: {e}")
            continue

        for entry in feed.entries:
            age = _entry_age_hours(entry)
            if age > news_cfg["max_age_hours"]:
                continue
            all_entries.append(
                {
                    "source": source["name"],
                    "title": entry.get("title", "").strip(),
                    "summary": (entry.get("summary", "") or "")[:600].strip(),
                    "link": entry.get("link", ""),
                    "age_hours": round(age, 1),
                    "score": _score(entry, news_cfg["keywords_boost"]),
                }
            )

    if not all_entries:
        logger.error("No se encontró ninguna noticia reciente en las fuentes configuradas")
        sys.exit(1)

    # ordena por score (relevancia) y luego por recencia
    all_entries.sort(key=lambda e: (e["score"], -e["age_hours"]), reverse=True)

    # evita elegir dos noticias casi idénticas del mismo tema (dedupe simple por título)
    chosen, seen_titles = [], set()
    for e in all_entries:
        key = e["title"][:40].lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        chosen.append(e)
        if len(chosen) >= news_cfg["stories_per_video"]:
            break

    return chosen


def main():
    ensure_dirs()
    cfg = load_config()

    stories = fetch_top_stories(cfg)
    logger.info(f"Noticias seleccionadas ({len(stories)}):")
    for s in stories:
        logger.info(f"  - [{s['source']}] {s['title']} (score={s['score']}, {s['age_hours']}h)")

    item_id = add_to_queue({"stories": stories}, status="news_ready")
    logger.info(f"Noticias guardadas en la cola con id={item_id}")
    print(item_id)


if __name__ == "__main__":
    main()
