"""
Paso 6: arma el video final en formato horizontal (16:9):
  - genera fondos visuales acordes a cada tema con un modelo de difusión de Hugging Face
    (o un degradado simple si video.background.mode = "gradient")
  - pone el audio narrado + música de fondo de bajo volumen
  - opcionalmente quema subtítulos palabra-por-palabra (video.burn_subtitles)
  - agrega una barra inferior con el nombre del canal

Uso:
    python scripts/06_assemble_video.py --id abc12345
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    afx,
)

from utils import OUTPUT_DIR, ROOT, get_item, get_logger, load_config, next_item_with_status, update_item

logger = get_logger("06_assemble_video")

_SD_PIPELINE = {}


def _build_gradient(duration: float, cfg: dict) -> ImageClip:
    """Fondo degradado en memoria, sin descargar ni generar material externo."""
    w, h = cfg["video"]["width"], cfg["video"]["height"]
    top = np.array([11, 15, 26], dtype=np.float32)
    bottom = np.array([24, 63, 105], dtype=np.float32)
    gradient = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    pixels = np.broadcast_to(top * (1 - gradient) + bottom * gradient, (h, w, 3))
    return ImageClip(np.asarray(pixels, dtype=np.uint8)).set_duration(duration)


def _get_sd_pipeline(model_name: str):
    if model_name not in _SD_PIPELINE:
        from diffusers import StableDiffusionPipeline

        logger.info(f"Cargando modelo HF de imágenes '{model_name}' (primera ejecución descarga los pesos)...")
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32, safety_checker=None)
        pipe.set_progress_bar_config(disable=True)
        _SD_PIPELINE[model_name] = pipe
    return _SD_PIPELINE[model_name]


def _generate_topic_image(prompt: str, cfg: dict, out_path: Path) -> Path:
    bg_cfg = cfg["video"]["background"]
    model_name = os.environ.get("HF_IMAGE_MODEL") or bg_cfg.get("model", "runwayml/stable-diffusion-v1-5")
    pipe = _get_sd_pipeline(model_name)
    full_prompt = (
        f"{prompt}, digital illustration, tech news graphic, cinematic lighting, high detail, no text, no watermark"
    )
    image = pipe(
        full_prompt,
        num_inference_steps=int(bg_cfg.get("steps", 15)),
        width=int(bg_cfg.get("width", 768)),
        height=int(bg_cfg.get("height", 432)),
    ).images[0]
    image.save(out_path)
    return out_path


def _fit_to_canvas(image_path: Path, w: int, h: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    src_ratio = img.width / img.height
    dst_ratio = w / h
    if src_ratio > dst_ratio:
        new_h = h
        new_w = int(h * src_ratio)
    else:
        new_w = w
        new_h = int(w / src_ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    img = img.crop((left, top, left + w, top + h))
    return np.asarray(img, dtype=np.uint8)


def _build_ai_background(item: dict, item_id: str, duration: float, cfg: dict) -> list:
    plan_visual = item.get("plan_visual") or []
    if not plan_visual:
        logger.info("El guion no trajo plan_visual; uso fondo degradado como respaldo")
        return [_build_gradient(duration, cfg)]

    w, h = cfg["video"]["width"], cfg["video"]["height"]
    segment_duration = duration / len(plan_visual)
    clips = []
    for i, section in enumerate(plan_visual):
        keywords = section.get("palabras_clave_busqueda") or [section.get("seccion", "technology")]
        prompt = ", ".join(keywords)
        img_path = OUTPUT_DIR / f"{item_id}_bg_{i}.png"
        try:
            _generate_topic_image(prompt, cfg, img_path)
            frame = _fit_to_canvas(img_path, w, h)
        except Exception as exc:
            logger.warning(f"No se pudo generar la imagen para la sección '{prompt}': {exc}. Uso degradado.")
            clips.append(
                _build_gradient(segment_duration, cfg).set_start(i * segment_duration)
            )
            continue
        clip = (
            ImageClip(frame)
            .set_start(i * segment_duration)
            .set_duration(segment_duration)
            .fadein(min(0.5, segment_duration / 4))
        )
        clips.append(clip)
    return clips


def _build_subtitles(words: list[dict], cfg: dict, video_w: int, video_h: int):
    sub_cfg = cfg["video"]["subtitle"]
    chunks = []
    chunk_size = 6  # frases un poco más largas para video horizontal
    for i in range(0, len(words), chunk_size):
        group = words[i : i + chunk_size]
        if not group:
            continue
        text = " ".join(w["word"] for w in group).strip()
        chunks.append((text, group[0]["start"], group[-1]["end"]))

    clips = []
    y_pos = video_h - sub_cfg["position_from_bottom"]
    for text, start, end in chunks:
        if not text:
            continue
        txt_clip = (
            TextClip(
                text,
                fontsize=sub_cfg["fontsize"],
                font=sub_cfg["font"],
                color=sub_cfg["color"],
                stroke_color=sub_cfg["stroke_color"],
                stroke_width=sub_cfg["stroke_width"],
                method="caption",
                size=(video_w - 200, None),
                align="center",
            )
            .set_start(start)
            .set_duration(max(end - start, 0.05))
            .set_position(("center", y_pos))
        )
        clips.append(txt_clip)
    return clips


def _build_channel_bar(cfg: dict, duration: float):
    bar = TextClip(
        f"  {cfg['channel']['name']}  ",
        fontsize=30,
        font=cfg["video"]["subtitle"]["font"],
        color="white",
        bg_color="#3ea6ff",
    ).set_position((30, 30)).set_duration(duration)
    return bar


def _write_srt(words: list[dict], srt_path: Path, chunk_size: int = 8) -> Path:
    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    idx = 1
    for i in range(0, len(words), chunk_size):
        group = words[i : i + chunk_size]
        if not group:
            continue
        text = " ".join(w["word"] for w in group).strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{fmt(group[0]['start'])} --> {fmt(group[-1]['end'])}")
        lines.append(text)
        lines.append("")
        idx += 1

    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def assemble_video(item_id: str) -> str:
    cfg = load_config()
    item = get_item(item_id)

    audio_path = item.get("audio_path")
    captions_path = item.get("captions_path")
    if not audio_path or not captions_path:
        logger.error(f"Falta audio_path o captions_path en el item {item_id}; corre los pasos 3 y 4 primero")
        sys.exit(1)

    narration = AudioFileClip(audio_path)
    duration = narration.duration

    with open(captions_path, "r", encoding="utf-8") as f:
        words = json.load(f)

    srt_path = OUTPUT_DIR / f"{item_id}_captions.srt"
    _write_srt(words, srt_path)

    bg_mode = cfg["video"]["background"].get("mode", "gradient")
    if bg_mode == "ai_generated":
        bg_clips = _build_ai_background(item, item_id, duration, cfg)
    else:
        bg_clips = [_build_gradient(duration, cfg)]

    overlay_clips = []
    if cfg["video"].get("burn_subtitles", False):
        overlay_clips = _build_subtitles(words, cfg, cfg["video"]["width"], cfg["video"]["height"])

    channel_bar = _build_channel_bar(cfg, duration)

    final_video = CompositeVideoClip(
        [*bg_clips, *overlay_clips, channel_bar], size=(cfg["video"]["width"], cfg["video"]["height"])
    )

    music_dir = ROOT / "assets" / "music"
    music_files = list(music_dir.glob("*.mp3")) if music_dir.exists() else []
    if music_files:
        music = AudioFileClip(str(random.choice(music_files))).fx(afx.audio_loop, duration=duration)
        music = music.volumex(cfg["video"]["background_music_volume"])
        narration_v = narration.volumex(cfg["video"]["voice_volume"])
        final_audio = CompositeAudioClip([music, narration_v])
    else:
        logger.info("No hay música en assets/music/, el video llevará solo la narración")
        final_audio = narration

    final_video = final_video.set_audio(final_audio).set_duration(duration)

    out_path = OUTPUT_DIR / f"{item_id}_final.mp4"
    logger.info(f"Renderizando video final en {out_path} (puede tardar varios minutos, es un video largo)...")
    final_video.write_videofile(
        str(out_path),
        fps=cfg["video"]["fps"],
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
        logger=None,
    )

    update_item(item_id, status="video_ready", video_path=str(out_path), captions_srt_path=str(srt_path))
    logger.info(f"Video final listo: {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None)
    args = parser.parse_args()

    item_id = args.id
    if not item_id:
        item = next_item_with_status("captions_ready")
        if not item:
            logger.error("No hay ningún item listo para ensamblar (status=captions_ready) en la cola")
            sys.exit(1)
        item_id = item["id"]

    assemble_video(item_id)
    print(item_id)


if __name__ == "__main__":
    main()
