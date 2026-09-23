"""
Paso 5: genera una miniatura (thumbnail) 1280x720 automáticamente a partir del
título del video, con un diseño simple tipo "tech news" (fondo oscuro + acento + texto grande).

Si pones una imagen de fondo en assets/thumbnail/background.jpg, se usa como base;
si no, genera un fondo con gradiente simple.

Uso:
    python scripts/05_generate_thumbnail.py --id abc12345
"""
import argparse
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from utils import OUTPUT_DIR, ROOT, get_item, get_logger, load_config, next_item_with_status, update_item

logger = get_logger("05_generate_thumbnail")


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def _hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def generate_thumbnail(item_id: str) -> str:
    cfg = load_config()
    item = get_item(item_id)
    title = item.get("titulo_video", "Tech Daily")

    t_cfg = cfg["thumbnail"]
    w, h = t_cfg["width"], t_cfg["height"]
    bg_color = _hex_to_rgb(t_cfg["background_color"])
    accent = _hex_to_rgb(t_cfg["accent_color"])

    custom_bg = ROOT / "assets" / "thumbnail" / "background.jpg"
    if custom_bg.exists():
        img = Image.open(custom_bg).convert("RGB").resize((w, h))
        overlay = Image.new("RGB", (w, h), (0, 0, 0))
        img = Image.blend(img, overlay, alpha=0.45)
    else:
        img = Image.new("RGB", (w, h), bg_color)

    draw = ImageDraw.Draw(img)

    # franja de acento abajo a la izquierda
    draw.rectangle([(0, h - 18), (w, h)], fill=accent)

    # título envuelto en varias líneas, centrado verticalmente
    font = _load_font(t_cfg["font_size_title"])
    wrapped = textwrap.fill(title.upper(), width=16)
    lines = wrapped.split("\n")[:4]

    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
    total_h = sum(line_heights) + (len(lines) - 1) * 14
    y = (h - total_h) / 2 - 30

    for line, lh in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x = (w - line_w) / 2
        # contorno negro para legibilidad
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3), (-3, -3), (3, 3), (-3, 3), (3, -3)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += lh + 14

    # etiqueta del canal arriba a la izquierda
    small_font = _load_font(36)
    draw.text((40, 30), cfg["channel"]["name"].upper(), font=small_font, fill=accent)

    out_path = OUTPUT_DIR / f"{item_id}_thumbnail.jpg"
    img.save(out_path, quality=92)

    update_item(item_id, thumbnail_path=str(out_path))
    logger.info(f"Miniatura guardada en {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None)
    args = parser.parse_args()

    item_id = args.id
    if not item_id:
        item = next_item_with_status("captions_ready")
        if not item:
            logger.error("No hay ningún item con guion listo (status=captions_ready) en la cola")
            sys.exit(1)
        item_id = item["id"]

    generate_thumbnail(item_id)
    print(item_id)


if __name__ == "__main__":
    main()
