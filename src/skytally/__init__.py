"""skytally: tiny-object detection, tracking and line counting for drone footage."""

CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
NUM_CLASSES = len(CLASS_NAMES)
# Classes a road traffic counter cares about. Matches results/operating_point.json.
VEHICLE_CLASSES = tuple(CLASS_NAMES.index(n) for n in ("car", "van", "truck", "bus"))
