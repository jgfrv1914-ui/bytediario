"""
Paso 7: sube el video final a YouTube (con miniatura personalizada) usando la Data API v3.

Primera vez (autenticación):
  1. Ve a https://console.cloud.google.com/ -> crea un proyecto
  2. Habilita "YouTube Data API v3"
  3. Crea credenciales OAuth 2.0 (tipo "Aplicación de escritorio")
  4. Descarga el JSON y guárdalo como youtube_client_secret.json en la raíz del proyecto
  5. Corre este script una vez: se abrirá el navegador para que autorices tu cuenta de YouTube.

Uso:
    python scripts/07_upload_youtube.py --id abc12345
"""
import argparse
import json
import os
import sys
from pathlib import Path

import google.auth.transport.requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from utils import ROOT, get_item, get_logger, load_config, next_item_with_status, update_item

logger = get_logger("07_upload_youtube")

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # necesario para subir la pista de subtítulos (CC)
]
TOKEN_FILE = ROOT / "token.json"


def _credentials_from_env() -> Credentials | None:
    """En el Space no hay disco persistente: la sesión OAuth vive en secrets.

    Requiere YOUTUBE_REFRESH_TOKEN, YOUTUBE_CLIENT_ID y YOUTUBE_CLIENT_SECRET
    como secrets del Space (generados una vez en local con get_refresh_token.py).
    """
    refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN")
    client_id = os.environ.get("YOUTUBE_CLIENT_ID")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET")
    if not (refresh_token and client_id and client_secret):
        return None

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds


def _get_credentials() -> Credentials:
    creds = _credentials_from_env()
    if creds:
        return creds

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            secrets_file = ROOT / os.environ.get("YOUTUBE_CLIENT_SECRETS_FILE", "youtube_client_secret.json")
            if not secrets_file.exists():
                logger.error(
                    f"No encuentro {secrets_file}. Descárgalo desde Google Cloud Console "
                    "(credenciales OAuth de escritorio) y ponlo en la raíz del proyecto."
                )
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets_file), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return creds


def upload_video(item_id: str) -> str:
    cfg = load_config()
    item = get_item(item_id)

    video_path = item.get("video_path")
    if not video_path or not Path(video_path).exists():
        logger.error(f"El item {item_id} no tiene un video_path válido; corre primero 06_assemble_video.py")
        sys.exit(1)

    creds = _get_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": item["titulo_video"][:100],
            "description": item.get("descripcion_youtube", ""),
            "tags": cfg["upload"]["tags"],
            "categoryId": cfg["upload"]["category_id"],
        },
        "status": {
            "privacyStatus": cfg["channel"]["privacy_status"],
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
    logger.info(f"Subiendo '{item['titulo_video']}' a YouTube (privacidad: {cfg['channel']['privacy_status']})...")

    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            logger.info(f"Progreso de subida: {int(status.progress() * 100)}%")

    video_id = response["id"]
    video_url = f"https://youtube.com/watch?v={video_id}"

    thumbnail_path = item.get("thumbnail_path")
    if thumbnail_path and Path(thumbnail_path).exists():
        logger.info("Subiendo miniatura personalizada...")
        try:
            youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_path)).execute()
        except Exception as e:
            logger.warning(f"No se pudo subir la miniatura (¿cuenta sin verificar en YouTube?): {e}")

    captions_srt_path = item.get("captions_srt_path")
    if captions_srt_path and Path(captions_srt_path).exists():
        logger.info("Subiendo subtítulos como pista opcional (CC)...")
        try:
            youtube.captions().insert(
                part="snippet",
                body={
                    "snippet": {
                        "videoId": video_id,
                        "language": "es",
                        "name": "Español",
                        "isDraft": False,
                    }
                },
                media_body=MediaFileUpload(captions_srt_path, mimetype="application/octet-stream"),
            ).execute()
        except Exception as e:
            logger.warning(f"No se pudo subir la pista de subtítulos: {e}")

    update_item(item_id, status="uploaded", youtube_video_id=video_id, youtube_url=video_url)
    logger.info(f"¡Video publicado! {video_url}")
    return video_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None)
    args = parser.parse_args()

    item_id = args.id
    if not item_id:
        item = next_item_with_status("video_ready")
        if not item:
            logger.error("No hay ningún video listo para subir (status=video_ready) en la cola")
            sys.exit(1)
        item_id = item["id"]

    upload_video(item_id)


if __name__ == "__main__":
    main()
