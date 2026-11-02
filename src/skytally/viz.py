"""Aerial instrument overlay for tracked frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from skytally import CLASS_NAMES

PHOSPHOR = (120, 255, 170)
AMBER = (255, 186, 72)
PANEL = (6, 16, 12, 178)
DIM = (120, 255, 170, 120)
BRACKET_FRACTION = 0.32


def _font(size: int) -> ImageFont.ImageFont:
    """DejaVu Sans Mono from matplotlib when installed, otherwise Pillow's default."""
    try:
        import matplotlib

        path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf"
        return ImageFont.truetype(str(path), size)
    except (ImportError, OSError):
        return ImageFont.load_default(size)


@dataclass
class Overlay:
    line: tuple[tuple[float, float], tuple[float, float]]
    caption: str = ""
    trail: int = 14
    _history: dict[int, deque] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._small = _font(10)
        self._label = _font(12)
        self._big = _font(20)

    def draw(self, pixels: np.ndarray, frame: int, fps: float, tracks, counter) -> Image.Image:
        image = Image.fromarray(pixels).convert("RGBA")
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        self._edge_scales(draw, image.size)
        self._counting_line(draw)
        for track in tracks:
            self._track(draw, track)
        self._panels(draw, image.size, frame, fps, len(tracks), counter)
        return Image.alpha_composite(image, layer).convert("RGB")

    def _edge_scales(self, draw, size) -> None:
        """Tick scales along the frame edges, spaced every 40 px, as in a gimbal display."""
        width, height = size
        for x in range(0, width, 40):
            length = 9 if x % 200 == 0 else 4
            draw.line((x, height - 1, x, height - 1 - length), fill=DIM)
        for y in range(0, height, 40):
            length = 9 if y % 200 == 0 else 4
            draw.line((width - 1, y, width - 1 - length, y), fill=DIM)
        cx, cy = width // 2, height // 2
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            draw.line((cx + dx * 8, cy + dy * 8, cx + dx * 20, cy + dy * 20), fill=DIM)

    def _counting_line(self, draw) -> None:
        start, end = self.line
        draw.line((*start, *end), fill=(*AMBER, 235), width=2)
        for x, y in (start, end):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(*AMBER, 255), width=2)

    def _track(self, draw, track) -> None:
        trail = self._history.setdefault(track.track_id, deque(maxlen=self.trail))
        trail.append(tuple(track.center))
        if len(trail) > 1:
            draw.line(list(trail), fill=(*PHOSPHOR, 120), width=1)
        x0, y0, x1, y1 = track.box
        arm = max(4.0, min(x1 - x0, y1 - y0) * BRACKET_FRACTION)
        colour = (*PHOSPHOR, 255)
        for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            draw.line((cx, cy, cx + sx * arm, cy), fill=colour, width=1)
            draw.line((cx, cy, cx, cy + sy * arm), fill=colour, width=1)
        draw.text((x0, y0 - 12), f"{track.track_id:03d}", fill=colour, font=self._small)

    def _panels(self, draw, size, frame, fps, active, counter) -> None:
        width, _ = size
        draw.rectangle((10, 10, 296, 74), fill=PANEL)
        draw.text((18, 15), "SKYTALLY", fill=PHOSPHOR, font=self._big)
        draw.text(
            (18, 40), f"FRAME {frame:05d}   T+{frame / fps:6.1f} s", fill=PHOSPHOR, font=self._label
        )
        draw.text((18, 56), f"TRACKS {active:3d}", fill=PHOSPHOR, font=self._label)

        totals = counter.direction_totals()
        draw.rectangle((width - 226, 10, width - 10, 96), fill=PANEL)
        draw.text((width - 218, 15), "LINE COUNT", fill=AMBER, font=self._label)
        draw.text((width - 218, 32), f"FWD {totals['forward']:3d}", fill=AMBER, font=self._big)
        draw.text((width - 118, 32), f"BWD {totals['backward']:3d}", fill=AMBER, font=self._big)
        by_label = {}
        for (_, label), value in counter.totals().items():
            by_label[label] = by_label.get(label, 0) + value
        summary = "  ".join(
            f"{CLASS_NAMES[k][:5].upper()} {v}" for k, v in sorted(by_label.items())
        )
        draw.text((width - 218, 66), summary[:30], fill=AMBER, font=self._small)
        if self.caption:
            draw.text((width - 218, 80), self.caption, fill=(*AMBER, 200), font=self._small)
