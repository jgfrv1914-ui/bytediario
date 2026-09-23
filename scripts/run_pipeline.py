"""
Orquestador: corre el pipeline completo
(noticias -> guion -> audio -> subtítulos -> miniatura -> video -> subida)
para la cantidad de videos definida en config.yaml (channel.videos_per_run).

Este es el script que conectarías a un cron / tarea programada para que el canal
funcione solo, por ejemplo una vez al día.

Uso:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --sin-subir     # genera todo pero no publica en YouTube
"""
import argparse
import importlib
import sys

from utils import add_to_queue, get_logger, load_config

fetch_news = importlib.import_module("01_fetch_news")
gen_script = importlib.import_module("02_generate_script")
gen_audio = importlib.import_module("03_generate_audio")
gen_captions = importlib.import_module("04_generate_captions")
gen_thumbnail = importlib.import_module("05_generate_thumbnail")
assemble = importlib.import_module("06_assemble_video")
upload = importlib.import_module("07_upload_youtube")

logger = get_logger("run_pipeline")


def run_once(cfg: dict, skip_upload: bool) -> str:
    logger.info("=== Iniciando pipeline de Tech Daily ===")

    stories = fetch_news.fetch_top_stories(cfg)
    item_id = add_to_queue({"stories": stories}, status="news_ready")
    logger.info(f"[1/6] Noticias listas (id={item_id}): {len(stories)} historias")

    data = gen_script.generate_script(item_id, cfg)
    logger.info(f"[2/6] Guion listo: {data['titulo_video']}")

    gen_audio.generate_audio(item_id)
    logger.info("[3/6] Audio listo")

    gen_captions.generate_captions(item_id)
    logger.info("[4/6] Subtítulos listos")

    gen_thumbnail.generate_thumbnail(item_id)
    logger.info("[5/6] Miniatura lista")

    assemble.assemble_video(item_id)
    logger.info("[5/6] Video ensamblado")

    if skip_upload:
        logger.info("[6/6] Subida omitida (--sin-subir). Revisa el video y la miniatura en output/ antes de publicar.")
    else:
        url = upload.upload_video(item_id)
        logger.info(f"[6/6] Publicado: {url}")

    logger.info(f"=== Pipeline completo para id={item_id} ===")
    return item_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sin-subir", action="store_true", help="Genera el video pero no lo publica en YouTube")
    args = parser.parse_args()

    cfg = load_config()
    n = cfg["channel"]["videos_per_run"]

    for i in range(n):
        logger.info(f"--- Video {i + 1}/{n} ---")
        try:
            run_once(cfg, skip_upload=args.__dict__["sin_subir"])
        except Exception as e:
            logger.error(f"Falló la corrida {i + 1}: {e}")
            continue


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
