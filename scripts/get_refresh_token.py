"""Genera un refresh token de YouTube para usar en el Space (sin disco persistente).

Corre esto UNA VEZ en tu máquina local, autorizando la cuenta de Google dueña
del canal https://www.youtube.com/@fertech-m8e. Copia los 3 valores que imprime
al final como Secrets del Space en HF: Settings -> Variables and secrets.

Requisitos previos:
  1. https://console.cloud.google.com/ -> crea un proyecto
  2. Habilita "YouTube Data API v3"
  3. Crea credenciales OAuth 2.0 (tipo "Aplicación de escritorio")
  4. Descarga el JSON y guárdalo como youtube_client_secret.json en la raíz del proyecto

Uso:
    python scripts/get_refresh_token.py
"""
import json
import os
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

ROOT = Path(__file__).resolve().parent.parent
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # necesario para subir la pista de subtítulos (CC)
]


def main():
    secrets_file = ROOT / os.environ.get("YOUTUBE_CLIENT_SECRETS_FILE", "youtube_client_secret.json")
    if not secrets_file.exists():
        print(f"No encuentro {secrets_file}. Descárgalo desde Google Cloud Console primero.")
        sys.exit(1)

    with open(secrets_file, "r", encoding="utf-8") as f:
        client_config = json.load(f)
    client_id = client_config["installed"]["client_id"]
    client_secret = client_config["installed"]["client_secret"]

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_file), SCOPES)
    creds = flow.run_local_server(port=0)

    print("\n=== Copia estos valores como Secrets del Space en Hugging Face ===")
    print(f"YOUTUBE_CLIENT_ID={client_id}")
    print(f"YOUTUBE_CLIENT_SECRET={client_secret}")
    print(f"YOUTUBE_REFRESH_TOKEN={creds.refresh_token}")
    print("====================================================================\n")


if __name__ == "__main__":
    main()
