---
title: FerTechZone — pipeline local de noticias
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
short_description: Pipeline local para videos de noticias en español
python_version: "3.12"
startup_duration_timeout: 1h
---

# FerTechZone: pipeline local

Space y CLI para crear videos de noticias tecnológicas en español:

1. RSS público (`01_fetch_news.py`)
2. Guion con `transformers` y un modelo local de Hugging Face
3. TTS español local (`facebook/mms-tts-spa` por defecto)
4. Whisper local para subtítulos
5. Miniatura con Pillow
6. Video con MoviePy/FFmpeg, fondo degradado generado localmente y música local
7. Subida opcional con YouTube Data API v3

No se usan Anthropic, ElevenLabs, Pexels ni otras APIs de IA externas. El fondo
del video se genera localmente con un degradado; no requiere claves ni descargas
de stock.

El canal de destino es [youtube.com/@fertech-m8e](https://www.youtube.com/@fertech-m8e).

La publicación diaria automática corre en **GitHub Actions** (cron gratuito,
ver `.github/workflows/daily-video.yml`), no en un Space activo: crear Spaces
con SDK Gradio/Docker en Hugging Face requiere PRO en algunas cuentas, mientras
que Actions es gratis e ilimitado en repos públicos. `app.py` queda como panel
manual opcional (para ejecutar/depurar el pipeline a mano, en local o en un
Space estático si se quiere).

## Configuración

En local:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Instala FFmpeg y déjalo en `PATH`.

Nunca pongas tokens o credenciales en `config.yaml` o en el repositorio.

### Publicación automática con GitHub Actions (para siempre, gratis)

`.github/workflows/daily-video.yml` corre el pipeline completo todos los días
a las 14:00 UTC en un runner gratuito de GitHub (repos públicos: minutos
ilimitados). Configúralo así:

1. En tu máquina local, obtén `youtube_client_secret.json` (Google Cloud
   Console -> credenciales OAuth de escritorio), autorizando la cuenta de
   Google dueña del canal
   [youtube.com/@fertech-m8e](https://www.youtube.com/@fertech-m8e).
2. Corre `python scripts/get_refresh_token.py`. Se abrirá el navegador para
   autorizar y al final imprime 3 valores.
3. En el repo de GitHub, ve a **Settings → Secrets and variables → Actions**
   y agrega como **Secrets**:
   - `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_REFRESH_TOKEN`
     (los 3 valores del paso 2)
   - `HF_TOKEN` (solo si usas modelos privados/gated de Hugging Face)
4. Opcionalmente, en la pestaña **Variables** del mismo lugar, agrega
   `HF_SCRIPT_MODEL` / `HF_TTS_MODEL` si quieres otros modelos distintos a los
   valores por defecto de `config.yaml`.
5. Listo: cada día el workflow instala dependencias y FFmpeg, descarga los
   modelos de Hugging Face, genera el video y lo publica en el canal. Puedes
   disparar una corrida manual desde la pestaña **Actions → Publicar video
   diario → Run workflow**, y revisar los logs como artefacto descargable de
   cada ejecución.

Los pesos de Hugging Face se descargan en la primera ejecución y requieren
memoria suficiente. `config.yaml` contiene los valores por defecto del canal,
fuentes RSS, guion, TTS, video y YouTube; las variables de modelo tienen prioridad.

## Uso CLI

```bash
cd scripts
python run_pipeline.py --sin-subir
python run_pipeline.py
```

Los pasos individuales aceptan `--id`. La cola se guarda en `data/queue.json` y
los resultados en `output/`. `run_pipeline.py` conserva la compatibilidad con el
flujo existente y solo publica cuando no se pasa `--sin-subir`.

## Uso en Gradio

Ejecuta `app.py` o abre el Space, revisa los logs y pulsa **Ejecutar pipeline**.
Marca **Publicar en YouTube al terminar** únicamente cuando quieras llamar a la
YouTube Data API.
