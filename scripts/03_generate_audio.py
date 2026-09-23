"""
Paso 3: convierte el guion en audio narrado (TTS) con un modelo local de Hugging Face.

Uso:
    python scripts/03_generate_audio.py --id abc12345
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
from transformers import pipeline

from utils import OUTPUT_DIR, get_item, get_logger, load_config, next_item_with_status, update_item

logger = get_logger("03_generate_audio")

_PIPELINES = {}


def generate_audio(item_id: str) -> str:
    item = get_item(item_id)
    if item["status"] != "script_ready":
        logger.warning(f"El item {item_id} tiene status '{item['status']}', se esperaba 'script_ready'")

    logger.info(f"Generando audio para item {item_id}...")
    cfg = load_config()
    model_name = os.environ.get("HF_TTS_MODEL", cfg.get("tts", {}).get("model", "facebook/mms-tts-spa"))
    try:
        if model_name not in _PIPELINES:
            logger.info(f"Cargando modelo HF TTS '{model_name}' (primera ejecución descarga los pesos)...")
            _PIPELINES[model_name] = pipeline("text-to-speech", model=model_name)
        result = _PIPELINES[model_name](item["guion_completo"])
        audio = np.asarray(result["audio"], dtype=np.float32)
        if audio.ndim > 1:
            audio = np.squeeze(audio)
        sample_rate = int(result["sampling_rate"])
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo cargar o ejecutar el modelo HF TTS '{model_name}'. "
            "Revisa HF_TTS_MODEL, memoria y acceso al modelo."
        ) from exc

    audio_path = OUTPUT_DIR / f"{item_id}_audio.wav"
    sf.write(str(audio_path), audio, sample_rate)

    update_item(item_id, status="audio_ready", audio_path=str(audio_path))
    logger.info(f"Audio guardado en {audio_path}")
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
