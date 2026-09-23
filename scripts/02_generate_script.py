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
from datetime import datetime, timedelta, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_item, get_logger, load_config, next_item_with_status, update_item

logger = get_logger("02_generate_script")

_MODEL_CACHE = {}

SYSTEM_PROMPT = """Actúa como guionista y editor de un canal de tecnología en español llamado \
"{channel_name}". El presentador se llama {presenter}.

OBJETIVO
Escribes el guion de un único video diario sobre noticias y datos curiosos de tecnología. \
Apunta a una narración de entre {min_w} y {max_w} palabras, ajustada al ritmo de {presenter}, \
para un audio final de entre 10 y 15 minutos (idealmente 12-13 minutos).

IDENTIDAD Y TONO
Tono: {tone}.
- Español natural, frases claras y ejemplos cotidianos.
- Combina frases cortas con explicaciones más pausadas.
- Transmite curiosidad sin exagerar. Explica los términos técnicos cuando aparezcan.
- No inventes experiencias personales de {presenter}.
- Evita repetir ideas o añadir relleno para completar palabras.
- No incluyas indicaciones visuales ni acotaciones dentro del texto que leerá el narrador \
(eso va aparte, en "plan_visual").

FUENTES: SOLO LAS NOTICIAS PROPORCIONADAS
No tienes acceso a internet. Las únicas fuentes válidas son las noticias que te paso abajo, \
cada una con su medio, fecha real y enlace. No inventes datos, citas, cifras, productos ni \
supuestas respuestas entre empresas que no estén en esas noticias. Si un dato no aparece en \
las noticias dadas, no lo afirmes como hecho: acláralo como algo que sigue sin confirmarse o \
simplemente no lo menciones. Distingue rumores/anuncios de funciones ya disponibles cuando la \
noticia lo indique. Usa exactamente el nombre del medio, la fecha y la URL de cada noticia tal \
como te las doy, en el campo "fuentes" del JSON de salida (no las cambies ni las inventes).

ESTRUCTURA DEL GUION (referencia para ~12-13 minutos; ajusta proporciones si tienes menos \
minutos objetivo)
1. Saludo y gancho (0:00-0:45): abre con una pregunta o un hecho de la noticia principal. \
Integra de forma natural un saludo equivalente a "¡Hola! Soy {presenter} y esto es {channel_name}". \
Adelanta brevemente qué va a ver el espectador. Varía la apertura respecto a otros días.
2. Tema principal (0:45-5:15): qué ocurrió, qué contexto necesita el espectador, cómo funciona \
la tecnología, qué cambia en una situación cotidiana, qué límites tiene y qué sigue sin \
confirmarse. Incluye un ejemplo concreto y separa hechos de tu análisis.
3. Segundo tema (5:15-8:30): transición natural (sin forzar conexión con el tema anterior). \
Qué aporta, a quién le interesa, qué limitaciones tiene. Usa una comparación sencilla si ayuda.
4. Tercer tema o dato curioso (8:30-11:30): deja un aprendizaje útil. Si es de seguridad, nombra \
producto/versión afectada solo si la noticia lo confirma. Si es un dato curioso, explica por qué \
sucede, no solo que sucede.
5. Cierre y despedida (11:30-12:30): resume las ideas clave sin repetir el guion, haz una sola \
pregunta concreta para invitar a comentar, y termina con algo equivalente a "Si quieres recibir \
más datos curiosos y novedades sobre tecnología, suscríbete a {channel_name}. Soy {presenter}. \
¡Gracias por acompañarme y hasta mañana!".

REGLAS DE PRODUCCIÓN
- Genera un único video (no dividas el contenido en varias publicaciones).
- Si el conteo de palabras de tu narración queda corto para 10 minutos, añade contexto, \
ejemplos o explicaciones útiles (nunca relleno vacío). Si se pasa de 15 minutos, elimina \
repeticiones y detalles secundarios. No indiques que verificaste la duración del audio: eso se \
mide después, fuera de tu control.

Responde SIEMPRE en JSON válido, sin comentarios ni comas finales ni texto fuera del JSON, con \
este formato exacto (sustituye los valores de ejemplo por los resultados reales; deja \
"duracion_audio_final_seg" en null, se completa después de generar el audio):
{{
  "canal": "{channel_name}",
  "presentador": "{presenter}",
  "fecha_video": "AAAA-MM-DD",
  "titulo_video": "titulo llamativo y fiel al contenido, max 90 caracteres, sin comillas",
  "guion_completo": "texto integro de la narracion: saludo, desarrollo de los 3 temas y despedida",
  "descripcion_youtube": "resumen de 3-4 lineas del contenido, con las URLs de las fuentes y 5-8 hashtags al final",
  "capitulos": [
    {{"tiempo_aprox_seg": 0, "titulo": "Introducción"}},
    {{"tiempo_aprox_seg": 45, "titulo": "titulo corto del tema principal"}}
  ],
  "plan_visual": [
    {{"seccion": "Introducción", "indicaciones": "imagenes o graficos sugeridos para esta seccion", "palabras_clave_busqueda": ["termino en ingles"]}}
  ],
  "fuentes": [
    {{"tema": "tema que respalda", "nombre": "nombre exacto del medio dado", "url": "URL exacta dada", "fecha_publicacion": "AAAA-MM-DD"}}
  ],
  "conteo_palabras_narracion": 0,
  "duracion_estimada_seg": 0,
  "duracion_audio_final_seg": null
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

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stories_block = "\n\n".join(
        f"NOTICIA {i+1} (medio: {s['source']}, "
        f"fecha aprox. de publicación: {(datetime.now(timezone.utc) - timedelta(hours=s['age_hours'])).strftime('%Y-%m-%d')}):\n"
        f"Título: {s['title']}\nResumen: {s['summary']}\nURL: {s['link']}"
        for i, s in enumerate(stories)
    )
    user_prompt = (
        f"Fecha del video: {today}\n\nEstas son las únicas noticias disponibles para hoy:\n\n"
        f"{stories_block}\n\nGenera el guion completo del video siguiendo todas las reglas del "
        "sistema, citando estas mismas noticias en \"fuentes\"."
    )

    model_name = os.environ.get("HF_SCRIPT_MODEL") or cfg["script"].get("model", "Qwen/Qwen2.5-3B-Instruct")
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
