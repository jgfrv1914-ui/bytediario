"""
Paso 2: convierte las noticias seleccionadas en un guion narrado, siguiendo
la plantilla y el tono del canal "FerTechZone" (presentador cercano, hook fuerte,
explicación en 3 partes: qué pasó / por qué importa / qué significa para nosotros).

Uso:
    python scripts/02_generate_script.py --id abc12345
"""
import argparse
import json
import sys

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_item, get_logger, load_config, next_item_with_status, update_item

logger = get_logger("02_generate_script")

_MODEL_CACHE = {}

SYSTEM_PROMPT = """Actúa como productor de contenido tecnológico y guionista para un canal \
de YouTube en español llamado "{channel_name}", presentado por {presenter}. \
El video explica de manera clara, dinámica y atractiva las noticias tecnológicas más \
importantes del día, para una audiencia interesada en tecnología pero no experta.

Sigue esta estructura obligatoria:
1. Saludo inicial ("¡Hola a todos! Bienvenidos a {channel_name}...")
2. Hook fuerte que genere curiosidad sobre la noticia principal
3. Para CADA noticia que te den (usualmente 2-3): explica qué pasó, por qué es \
   importante y qué significa para la audiencia / la industria
4. Resumen final que conecte las noticias con una idea más grande (tendencia, impacto)
5. Cierre con llamado a la acción: comentar, compartir, dar like, suscribirse y activar la campanita

Tono: {tone}. Lenguaje sencillo, sin tecnicismos exagerados, frases cortas y dinámicas.
Longitud objetivo: entre {min_w} y {max_w} palabras (para un video de 10 a 15 minutos hablado).

Responde SIEMPRE en JSON válido, sin texto adicional, con este formato exacto:
{{
  "titulo_video": "titulo llamativo para YouTube, max 90 caracteres, sin comillas",
  "guion_completo": "el guion completo narrado, listo para leer en voz alta, del saludo al cierre",
  "descripcion_youtube": "descripcion de 3-4 lineas resumiendo las noticias del video, con 5-8 hashtags al final",
  "capitulos": [
    {{"tiempo_aprox_seg": 0, "titulo": "Introducción"}},
    {{"tiempo_aprox_seg": 20, "titulo": "titulo corto de la noticia 1"}}
  ],
  "palabras_clave_visuales": ["4 a 8 palabras clave en ingles para buscar video stock relacionado con las noticias"]
}}"""


def generate_script(item_id: str, cfg: dict) -> dict:
    item = get_item(item_id)
    stories = item["stories"]

    system = SYSTEM_PROMPT.format(
        channel_name=cfg["channel"]["name"],
        presenter=cfg["channel"]["presenter_name"],
        tone=cfg["script"]["tone"],
        min_w=cfg["script"]["min_words"],
        max_w=cfg["script"]["max_words"],
    )

    stories_block = "\n\n".join(
        f"NOTICIA {i+1} (fuente: {s['source']}):\nTítulo: {s['title']}\nResumen: {s['summary']}\nLink: {s['link']}"
        for i, s in enumerate(stories)
    )
    user_prompt = f"Estas son las noticias de hoy:\n\n{stories_block}\n\nGenera el guion completo del video."

    model_name = os.environ.get("HF_SCRIPT_MODEL", cfg["script"].get("model", "Qwen/Qwen2.5-3B-Instruct"))
    try:
        tokenizer, model = _load_model(model_name)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=int(cfg["script"].get("max_new_tokens", 2048)),
                do_sample=True,
                temperature=float(cfg["script"].get("temperature", 0.7)),
                top_p=float(cfg["script"].get("top_p", 0.9)),
            )
        raw_text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo cargar o ejecutar el modelo HF de guion '{model_name}'. "
            "Revisa HF_SCRIPT_MODEL, memoria y acceso al modelo."
        ) from exc

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text
        raw_text = raw_text.rsplit("```", 1)[0]

    # Algunos modelos envuelven el JSON en texto o bloques Markdown.
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    data = json.loads(match.group(0) if match else raw_text)
    update_item(item_id, status="script_ready", **data)
    return data


def _load_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        logger.info(f"Cargando modelo HF de guion '{model_name}' (primera ejecución descarga los pesos)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)
    return _MODEL_CACHE[model_name]


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None)
    args = parser.parse_args()

    item_id = args.id
    if not item_id:
        item = next_item_with_status("news_ready")
        if not item:
            logger.error("No hay noticias pendientes de guion (status=news_ready); corre primero 01_fetch_news.py")
            sys.exit(1)
        item_id = item["id"]

    try:
        data = generate_script(item_id, cfg)
    except json.JSONDecodeError as e:
        logger.error(f"El modelo no devolvió JSON válido: {e}")
        sys.exit(1)

    logger.info(f"Guion generado para id={item_id}: {data['titulo_video']}")
    print(item_id)


if __name__ == "__main__":
    main()
