"""Gradio demo: upload a dashboard-camera frame, get the predicted behaviour + Grad-CAM.

    pip install gradio
    python app.py --checkpoint runs/resnet34_168x224/best.pt
"""

import argparse
from pathlib import Path

import gradio as gr

from ddd import CLASS_IDS, CLASS_NAMES
from ddd.data import get_split, load_metadata, resolve_data_dir, subset
from ddd.gradcam import gradcam, overlay
from ddd.predict import Predictor

ROOT = Path(__file__).resolve().parent


def example_images(per_class: int = 1) -> list[str]:
    """One image per class from the held-out *test* drivers, if the dataset is downloaded."""
    try:
        df = load_metadata(resolve_data_dir())
    except FileNotFoundError:
        return []
    test = subset(df, get_split(df), "test")
    return test.groupby("classname").head(per_class).sort_values("classname").path.tolist()


def build(checkpoint: Path):
    predictor = Predictor(checkpoint)

    def run(image):
        if image is None:
            return None, None
        cam, probs = gradcam(predictor, image)
        labels = {f"{c}: {n}": float(p) for c, n, p in zip(CLASS_IDS, CLASS_NAMES, probs)}
        return labels, overlay(image, cam)

    return gr.Interface(
        fn=run,
        inputs=gr.Image(type="pil", label="Dashboard camera frame"),
        outputs=[gr.Label(num_top_classes=4, label="Prediction"), gr.Image(label="Grad-CAM: where the model looked")],
        examples=[[e] for e in example_images()] or None,
        title="Distracted Driver Detection",
        description=f"Model: `{predictor.ckpt['arch']}`, evaluated on drivers never seen during training.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "resnet34_168x224" / "best.pt")
    p.add_argument("--share", action="store_true")
    a = p.parse_args()
    build(a.checkpoint).launch(share=a.share)
