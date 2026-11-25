"""Distracted driver detection (State Farm) — training, evaluation and inference."""

CLASS_IDS = [f"c{i}" for i in range(10)]
CLASS_NAMES = [
    "safe driving",
    "texting - right",
    "talking on the phone - right",
    "texting - left",
    "talking on the phone - left",
    "operating the radio",
    "drinking",
    "reaching behind",
    "hair and makeup",
    "talking to passenger",
]
NUM_CLASSES = len(CLASS_IDS)

__version__ = "2.0.0"
