"""
Paso 3: convierte el guion en audio narrado (TTS) con un modelo local de Hugging Face.

Uso:
    python scripts/03_generate_audio.py --id abc12345
"""
import argparse
import os
import re
import sys
import traceback

import numpy as np
import soundfile as sf
from transformers import pipeline

from utils import OUTPUT_DIR, get_item, get_logger, load_config, next_item_with_status, update_item

logger = get_logger("03_generate_audio")

_PIPELINES = {}

MAX_CHUNK_CHARS = 300  # VITS/MMS-TTS no soporta texto largo en una sola pasada


def _split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Corta el guion en oraciones y las agrupa en trozos cortos para el TTS.

    Guiones de 10-15 min (~1500-1900 palabras) son demasiado largos para una
    sola pasada de VITS/MMS-TTS (se queda sin memoria o falla en silencio),
    así que se sintetiza por partes y se concatena el audio resultante.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def generate_audio(item_id: str) -> str:
    item = get_item(item_id)
    if item["status"] != "script_ready":
        logger.warning(f"El item {item_id} tiene status '{item['status']}', se esperaba 'script_ready'")

    logger.info(f"Generando audio para item {item_id}...")
    cfg = load_config()
    model_name = os.environ.get("HF_TTS_MODEL") or cfg.get("tts", {}).get("model", "facebook/mms-tts-spa")
    try:
        if model_name not in _PIPELINES:
            logger.info(f"Cargando modelo HF TTS '{model_name}' (primera ejecución descarga los pesos)...")
            _PIPELINES[model_name] = pipeline("text-to-speech", model=model_name)
        tts = _PIPELINES[model_name]

        chunks = _split_into_chunks(item["guion_completo"])
        logger.info(f"Sintetizando audio en {len(chunks)} fragmentos...")
        sample_rate = None
        audio_parts = []
        silence_gap = None
        for i, chunk in enumerate(chunks):
            result = tts(chunk)
            part = np.asarray(result["audio"], dtype=np.float32)
            if part.ndim > 1:
                part = np.squeeze(part)
            if sample_rate is None:
                sample_rate = int(result["sampling_rate"])
                silence_gap = np.zeros(int(sample_rate * 0.35), dtype=np.float32)
            if audio_parts:
                audio_parts.append(silence_gap)
            audio_parts.append(part)
            if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                logger.info(f"  fragmento {i + 1}/{len(chunks)} listo")

        audio = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
    except Exception as exc:
        logger.error("Traceback completo del fallo de TTS:\n" + traceback.format_exc())
        raise RuntimeError(
            f"No se pudo cargar o ejecutar el modelo HF TTS '{model_name}'. "
            "Revisa HF_TTS_MODEL, memoria y acceso al modelo."
        ) from exc

    audio_path = OUTPUT_DIR / f"{item_id}_audio.wav"
    sf.write(str(audio_path), audio, sample_rate)

    duration_seconds = round(len(audio) / sample_rate, 1)
    update_item(
        item_id,
        status="audio_ready",
        audio_path=str(audio_path),
        duracion_audio_final_seg=duration_seconds,
    )
    logger.info(f"Audio guardado en {audio_path} ({duration_seconds}s)")
    if not (600 <= duration_seconds <= 900):
        logger.warning(
            f"La duración real del audio ({duration_seconds}s) queda fuera del rango "
            "objetivo de 600-900s (10-15 min). Revisa min_words/max_words en config.yaml "
            "si esto se repite seguido."
        )
    return str(audio_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None)
    args = parser.parse_args()

    item_id = args.id
    if not item_id:
        item = next_item_with_status("script_ready")
        if not item:
            logger.error("No hay ningún guion pendiente de audio (status=script_ready) en la cola")
            sys.exit(1)
        item_id = item["id"]

    generate_audio(item_id)
    print(item_id)


if __name__ == "__main__":
    main()
