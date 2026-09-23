"""Interfaz Gradio para ejecutar el pipeline manualmente (dashboard/monitor).

La publicación automática diaria corre en GitHub Actions (ver
.github/workflows/daily-video.yml), no en este proceso: un Space Gradio no
tiene cómputo gratuito garantizado para mantener un hilo en background vivo.
Este archivo sirve para pruebas manuales y para revisar el estado de la cola.
"""
import importlib
import io
import logging
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

run_pipeline = importlib.import_module("run_pipeline")
utils = importlib.import_module("utils")


def execute_pipeline(publish: bool):
    """Ejecuta un video manualmente y devuelve logs, estado y ruta del resultado."""
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        cfg = utils.load_config()
        item_id = run_pipeline.run_once(cfg, skip_upload=not publish)
        item = utils.get_item(item_id)
        result = item.get("youtube_url") or item.get("video_path", "Pipeline terminado")
        return buffer.getvalue(), f"Completado: {result}"
    except Exception as exc:
        safe_error = str(exc).replace("\r", " ").replace("\n", " ")[:500]
        logging.getLogger(__name__).error("Falló la ejecución manual: %s", safe_error)
        return buffer.getvalue(), f"ERROR: {safe_error}"
    finally:
        root_logger.removeHandler(handler)


with gr.Blocks(title="ByteDiario — pipeline local") as demo:
    gr.Markdown(
        "# ByteDiario\n"
        "La publicación diaria automática de "
        "[youtube.com/@fertech-m8e](https://www.youtube.com/@fertech-m8e) corre "
        "en GitHub Actions. Este panel sirve para ejecutar el pipeline "
        "manualmente y revisar logs."
    )
    publish = gr.Checkbox(label="Publicar en YouTube al terminar", value=False)
    run = gr.Button("Ejecutar pipeline", variant="primary")
    status = gr.Textbox(label="Resultado", interactive=False)
    logs = gr.Textbox(label="Logs", lines=18, interactive=False)
    run.click(execute_pipeline, inputs=[publish], outputs=[logs, status])


if __name__ == "__main__":
    demo.launch()
