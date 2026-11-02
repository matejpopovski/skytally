import numpy as np
import pytest
import torch
from PIL import Image

from skytally.assign import pyramid_levels
from skytally.data import (
    Annotations,
    CropConfig,
    EpochSampler,
    TrainingSamples,
    letterbox_size,
    load_split,
    parse_annotation,
    random_crop,
    to_truth,
)


def test_parse_keeps_objects_and_routes_ignored_rows():
    ann = parse_annotation(
        "684,8,273,116,0,0,0,0\n"  # ignored region
        "406,119,265,70,1,4,0,1\n"  # car
        "10,10,5,5,1,11,0,0\n"  # others
        "20,20,5,5,0,1,0,0\n"  # excluded from evaluation
        "30,30,0,5,1,1,0,0\n"  # zero width
    )
    assert ann.labels.tolist() == [3]
    assert ann.boxes.tolist() == [[406, 119, 671, 189]]
    assert len(ann.ignore) == 3


def test_parse_handles_trailing_comma_and_blank_lines():
    ann = parse_annotation("1,2,3,4,1,1,0,0,\n\n")
    assert ann.boxes.tolist() == [[1, 2, 4, 6]]


def test_to_truth_repeats_ignore_regions_for_each_class():
    ann = Annotations(
        np.array([[1, 2, 4, 6]], np.float32),
        np.array([2]),
        np.array([[0, 0, 10, 10]], np.float32),
    )
    truth = to_truth(ann, num_classes=3)
    assert truth.boxes[0].tolist() == [1, 2, 3, 4]
    assert truth.labels.tolist() == [2, 0, 1, 2]
    assert truth.crowd.tolist() == [False, True, True, True]


def make_image(width=400, height=300):
    pixels = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def test_crop_keeps_boxes_in_bounds_and_on_the_object():
    # A bright square on a black image: the transformed box must still enclose it.
    pixels = np.zeros((300, 400, 3), dtype=np.uint8)
    pixels[100:140, 200:260] = 255
    ann = Annotations(
        np.array([[200, 100, 260, 140]], np.float32), np.array([0]), np.empty((0, 4), np.float32)
    )
    config = CropConfig(size=256, flip=0.0, color_jitter=0.0, object_centered=1.0)
    for seed in range(20):
        crop, out = random_crop(Image.fromarray(pixels), ann, config, np.random.default_rng(seed))
        assert crop.shape == (256, 256, 3)
        if len(out.boxes):
            x0, y0, x1, y1 = out.boxes[0].round().astype(int)
            assert 0 <= x0 < x1 <= 256 and 0 <= y0 < y1 <= 256
            inside = crop[y0 + 2 : y1 - 2, x0 + 2 : x1 - 2]
            assert inside.mean() > 200


def test_flip_mirrors_boxes():
    ann = Annotations(
        np.array([[50, 60, 70, 90]], np.float32), np.array([1]), np.empty((0, 4), np.float32)
    )
    pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    pixels[60:90, 50:70] = 255
    config = CropConfig(
        size=256, scale_min=1.0, scale_max=1.0, flip=1.0, color_jitter=0.0, object_centered=0.0
    )
    crop, out = random_crop(Image.fromarray(pixels), ann, config, np.random.default_rng(0))
    assert out.boxes[0].tolist() == pytest.approx([186, 60, 206, 90], abs=1.0)
    assert crop[75, 196].mean() > 200


def test_sliced_box_becomes_ignore_region_not_background():
    ann = Annotations(
        np.array([[0, 100, 100, 110]], np.float32), np.array([0]), np.empty((0, 4), np.float32)
    )
    config = CropConfig(
        size=256, scale_min=1.0, scale_max=1.0, flip=0.0, color_jitter=0.0, object_centered=0.0
    )
    image = make_image(1000, 300)
    rng = next(
        np.random.default_rng(seed)
        for seed in range(1000)
        if 80 < _crop_left(seed, config, image) < 90
    )
    _, out = random_crop(image, ann, config, rng)
    assert len(out.boxes) == 0
    assert len(out.ignore) == 1


def _crop_left(seed, config, image):
    rng = np.random.default_rng(seed)
    rng.uniform(0, 0)
    rng.random()
    cx = rng.uniform(0, image.size[0])
    return np.clip(cx - config.size / 2, 0, image.size[0] - config.size)


def test_small_image_is_padded_rather_than_rejected():
    ann = Annotations(
        np.array([[10, 10, 30, 30]], np.float32), np.array([0]), np.empty((0, 4), np.float32)
    )
    config = CropConfig(size=256, scale_min=1.5, scale_max=1.5, flip=0.0, color_jitter=0.0)
    crop, _ = random_crop(make_image(100, 80), ann, config, np.random.default_rng(1))
    assert crop.shape == (256, 256, 3)


def write_dataset(root, count=4):
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    rng = np.random.default_rng(0)
    for i in range(count):
        pixels = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / f"{i:03d}.jpg")
        (root / "annotations" / f"{i:03d}.txt").write_text("20,30,16,12,1,4,0,0\n")


def test_samples_are_a_pure_function_of_the_key(tmp_path):
    write_dataset(tmp_path)
    images, annotations = load_split(tmp_path)
    crop = CropConfig(size=128)
    dataset = TrainingSamples(images, annotations, pyramid_levels(2), crop)
    first = dataset[(0, 3, 1)]
    second = dataset[(0, 3, 1)]
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)
    assert not torch.equal(first[0], dataset[(0, 4, 1)][0])


def test_load_split_requires_images(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_visdrone"):
        load_split(tmp_path)


def test_epoch_sampler_resumes_mid_epoch():
    full = list(EpochSampler(20, seed=1, epoch=2))
    resumed = list(EpochSampler(20, seed=1, epoch=2, start=8))
    assert resumed == full[8:]
    assert sorted(k[2] for k in full) == list(range(20))
    assert full != list(EpochSampler(20, seed=1, epoch=3))


def test_letterbox_rounds_up_to_divisor():
    assert letterbox_size(2000, 1500, 1024) == (1024, 768, 0.512)
    width, height, _ = letterbox_size(1360, 765, 1024)
    assert width % 32 == 0 and height % 32 == 0
    assert height >= round(765 * 1024 / 1360)
