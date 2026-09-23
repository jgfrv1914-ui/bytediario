"""
Paso 4: transcribe el audio con timestamps por palabra (para subtítulos) usando
faster-whisper (corre localmente, no necesita API key).

Uso:
    python scripts/04_generate_captions.py --id abc12345
"""
import argparse
import json
import sys

from faster_whisper import WhisperModel

from utils import OUTPUT_DIR, get_item, get_logger, next_item_with_status, update_item

logger = get_logger("04_generate_captions")

_MODEL = None


def _get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        logger.info("Cargando modelo Whisper (small)... la primera vez descarga los pesos.")
        _MODEL = WhisperModel("small", device="cpu", compute_type="int8")
    return _MODEL


def generate_captions(item_id: str) -> str:
    item = get_item(item_id)
    audio_path = item.get("audio_path")
    if not audio_path:
        logger.error(f"El item {item_id} no tiene audio_path; corre primero 03_generate_audio.py")
        sys.exit(1)

    model = _get_model()
    logger.info(f"Transcribiendo {audio_path}... (puede tardar varios minutos en un video largo)")
    segments, _info = model.transcribe(audio_path, word_timestamps=True, language="es")

    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)})

    captions_path = OUTPUT_DIR / f"{item_id}_captions.json"
    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    update_item(item_id, status="captions_ready", captions_path=str(captions_path))
    logger.info(f"Subtítulos ({len(words)} palabras) guardados en {captions_path}")
    return str(captions_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None)
    args = parser.parse_args()

    item_id = args.id
    if not item_id:
        item = next_item_with_status("audio_ready")
        if not item:
            logger.error("No hay ningún audio pendiente de subtítulos (status=audio_ready) en la cola")
            sys.exit(1)
        item_id = item["id"]

    generate_captions(item_id)
    print(item_id)


if __name__ == "__main__":
    main()
