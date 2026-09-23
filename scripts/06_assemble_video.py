"""
Paso 6: arma el video final en formato horizontal (16:9):
  - genera un fondo visual local con gradiente (sin servicios de stock)
  - pone el audio narrado + música de fondo de bajo volumen
  - quema los subtítulos palabra-por-palabra encima
  - agrega una barra inferior con el nombre del canal

Uso:
    python scripts/06_assemble_video.py --id abc12345
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
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

def _build_background(duration: float, cfg: dict) -> ImageClip:
    """Crea un fondo degradado en memoria, sin descargar material externo."""
    w, h = cfg["video"]["width"], cfg["video"]["height"]
    top = np.array([11, 15, 26], dtype=np.float32)
    bottom = np.array([24, 63, 105], dtype=np.float32)
    gradient = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    pixels = np.broadcast_to(top * (1 - gradient) + bottom * gradient, (h, w, 3))
    return ImageClip(np.asarray(pixels, dtype=np.uint8)).set_duration(duration)


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
    w, h = cfg["video"]["width"], cfg["video"]["height"]
    bar = TextClip(
        f"  {cfg['channel']['name']}  ",
        fontsize=30,
        font=cfg["video"]["subtitle"]["font"],
        color="white",
        bg_color="#3ea6ff",
    ).set_position((30, 30)).set_duration(duration)
    return bar


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

    bg = _build_background(duration, cfg)

    with open(captions_path, "r", encoding="utf-8") as f:
        words = json.load(f)
    sub_clips = _build_subtitles(words, cfg, cfg["video"]["width"], cfg["video"]["height"])
    channel_bar = _build_channel_bar(cfg, duration)

    final_video = CompositeVideoClip(
        [bg, *sub_clips, channel_bar], size=(cfg["video"]["width"], cfg["video"]["height"])
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

    update_item(item_id, status="video_ready", video_path=str(out_path))
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
