"""Paint a simulated sequence as aerial frames.

The frames exist so the whole pipeline, detector included, can run on a sequence with
known truth and be shown publicly. They are simple on purpose: nobody should mistake them
for real footage, and a detector trained on VisDrone meets them as a new domain.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from skytally.sim.traffic import BUS, TRUCK, ObjectState, SceneConfig, road_axes

BODY_COLOURS = [
    (232, 232, 228),
    (205, 208, 212),
    (150, 154, 160),
    (58, 60, 66),
    (24, 26, 30),
    (156, 36, 34),
    (34, 66, 128),
]
SUPERSAMPLE = 4


def _value_noise(rng: np.random.Generator, height: int, width: int, octaves: int = 5) -> np.ndarray:
    """Layered random fields, upsampled, in [0, 1]. Reads as ground texture from altitude."""
    field = np.zeros((height, width))
    for octave in range(octaves):
        cells = 4 * 2**octave
        coarse = rng.random((cells * height // width + 2, cells + 2))
        layer = Image.fromarray((coarse * 255).astype(np.uint8)).resize(
            (width, height), Image.Resampling.BICUBIC
        )
        field += np.asarray(layer, dtype=np.float64) / 255 / 2**octave
    field -= field.min()
    return field / field.max()


def _quad(origin, along, normal, start, stop, low, high) -> list[tuple[float, float]]:
    corners = [
        origin + along * start + normal * low,
        origin + along * stop + normal * low,
        origin + along * stop + normal * high,
        origin + along * start + normal * high,
    ]
    return [tuple(c) for c in corners]


def paint_background(config: SceneConfig, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    width, height = config.width, config.height
    noise = _value_noise(rng, height, width)
    fine = rng.normal(0, 6, (height, width, 1))
    dark = np.array([64, 84, 52], dtype=np.float64)
    light = np.array([138, 140, 96], dtype=np.float64)
    ground = dark + (light - dark) * noise[..., None] + fine
    image = Image.fromarray(np.clip(ground, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")

    origin, along, normal, length = road_axes(config)
    half_road = config.median_width / 2 + config.lanes_per_direction * config.lane_width
    _paint_buildings(draw, rng, origin, along, normal, length, half_road)

    draw.polygon(
        _quad(origin, along, normal, 0, length, -half_road - 12, half_road + 12),
        fill=(150, 146, 136, 255),
    )
    draw.polygon(
        _quad(origin, along, normal, 0, length, -half_road, half_road),
        fill=(70, 72, 76, 255),
    )
    draw.polygon(
        _quad(origin, along, normal, 0, length, -config.median_width / 2, config.median_width / 2),
        fill=(92, 104, 74, 255),
    )
    for sign in (-1, 1):
        for lane in range(1, config.lanes_per_direction):
            offset = sign * (config.median_width / 2 + lane * config.lane_width)
            for start in np.arange(0, length, 36.0):
                draw.polygon(
                    _quad(origin, along, normal, start, start + 16, offset - 0.8, offset + 0.8),
                    fill=(224, 224, 216, 200),
                )
        edge = sign * (half_road - 2)
        draw.polygon(
            _quad(origin, along, normal, 0, length, edge - 0.7, edge + 0.7),
            fill=(224, 224, 216, 200),
        )

    asphalt = rng.normal(0, 4, (height, width, 1))
    pixels = np.asarray(image, dtype=np.float64) + asphalt
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def _paint_buildings(draw, rng, origin, along, normal, length, half_road) -> None:
    roofs = [(176, 168, 158), (140, 86, 70), (196, 198, 200), (112, 112, 118)]
    for sign in (-1, 1):
        position = float(rng.uniform(0, 60))
        while position < length:
            footprint = rng.uniform(50, 120)
            depth = rng.uniform(60, 110)
            setback = rng.uniform(34, 70)
            near = sign * (half_road + setback)
            far = near + sign * depth
            start, stop = position, position + footprint
            shadow = [
                (x + 7, y + 9) for x, y in _quad(origin, along, normal, start, stop, near, far)
            ]
            draw.polygon(shadow, fill=(20, 22, 20, 120))
            colour = roofs[int(rng.integers(len(roofs)))]
            draw.polygon(_quad(origin, along, normal, start, stop, near, far), fill=(*colour, 255))
            position = stop + rng.uniform(16, 70)


def _vehicle_sprite(state: ObjectState, colour: tuple[int, int, int]) -> Image.Image:
    """Top-down vehicle facing +x, drawn large and reduced so edges anti-alias."""
    s = SUPERSAMPLE
    length, width = state.length * s, state.width * s
    pad = 2 * s
    size = (round(length + 2 * pad), round(width + 2 * pad))
    sprite = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    x0, y0, x1, y1 = pad, pad, pad + length, pad + width
    draw.rounded_rectangle((x0, y0, x1, y1), radius=width * 0.22, fill=(*colour, 255))
    glass = (34, 40, 48, 255)
    if state.label in (TRUCK, BUS):
        draw.rectangle(
            (x1 - length * 0.16, y0 + width * 0.12, x1 - length * 0.05, y1 - width * 0.12),
            fill=glass,
        )
    else:
        draw.rectangle(
            (x0 + length * 0.60, y0 + width * 0.12, x0 + length * 0.76, y1 - width * 0.12),
            fill=glass,
        )
        draw.rectangle(
            (x0 + length * 0.14, y0 + width * 0.14, x0 + length * 0.24, y1 - width * 0.14),
            fill=glass,
        )
    return sprite.resize((size[0] // s, size[1] // s), Image.Resampling.LANCZOS)


class SceneRenderer:
    def __init__(self, config: SceneConfig, seed: int = 0) -> None:
        self.config = config
        self.background = paint_background(config, seed)
        self.seed = seed
        self._colours: dict[int, tuple[int, int, int]] = {}
        self._sprites: dict[int, Image.Image] = {}
        self._shadows: dict[int, Image.Image] = {}
        self._rng = np.random.default_rng(seed + 1)

    def _sprite(self, state: ObjectState) -> tuple[Image.Image, Image.Image]:
        key = state.vehicle_id
        if key not in self._sprites:
            colour = BODY_COLOURS[int(self._rng.integers(len(BODY_COLOURS)))]
            sprite = _vehicle_sprite(state, colour).rotate(
                -np.rad2deg(state.heading), expand=True, resample=Image.Resampling.BICUBIC
            )
            alpha = sprite.getchannel("A").point(lambda v: int(v * 0.42))
            shadow = Image.merge("RGBA", (*[Image.new("L", sprite.size, 8)] * 3, alpha))
            self._sprites[key] = sprite
            self._shadows[key] = shadow.filter(ImageFilter.GaussianBlur(1.0))
        return self._sprites[key], self._shadows[key]

    def render(self, states: list[ObjectState]) -> np.ndarray:
        """Compose one frame and pass it through a sensor model: blur, noise, JPEG."""
        frame = self.background.copy()
        for state in states:
            if not state.visible:
                continue
            sprite, shadow = self._sprite(state)
            x = round(state.center[0] - sprite.width / 2)
            y = round(state.center[1] - sprite.height / 2)
            frame.paste(shadow, (x + 3, y + 3), shadow)
            frame.paste(sprite, (x, y), sprite)
        if self.config.occluder is not None:
            frame = self._cover_occluder(frame)
        frame = frame.filter(ImageFilter.GaussianBlur(0.6))
        pixels = np.asarray(frame, dtype=np.float32)
        pixels = pixels + self._rng.normal(0, 2.0, pixels.shape)
        noisy = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
        buffer = io.BytesIO()
        noisy.save(buffer, "JPEG", quality=85)
        buffer.seek(0)
        return np.asarray(Image.open(buffer).convert("RGB"))

    def _cover_occluder(self, frame: Image.Image) -> Image.Image:
        _origin, along, normal, _ = road_axes(self.config)
        center = np.array([self.config.width / 2, self.config.height / 2])
        low, high = self.config.occluder
        start = center + along * (low - self.config.width / 2)
        stop = center + along * (high - self.config.width / 2)
        reach = 200.0
        draw = ImageDraw.Draw(frame)
        draw.polygon(
            [
                tuple(start - normal * reach),
                tuple(stop - normal * reach),
                tuple(stop + normal * reach),
                tuple(start + normal * reach),
            ],
            fill=(96, 94, 90),
        )
        return frame
