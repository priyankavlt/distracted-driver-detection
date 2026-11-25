import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from ddd import NUM_CLASSES
from ddd.data import DriverImages, build_transforms, make_driver_split
from ddd.evaluate import compute_metrics, fit_temperature, softmax
from ddd.models import ResNet50Scratch, build_model, gradcam_target_layer


def _fake_df(n_drivers=26, per_driver=20):
    rows = [{"subject": f"p{d:03d}", "classname": f"c{i % 10}", "img": f"{d}_{i}.jpg", "label": i % 10}
            for d in range(n_drivers) for i in range(per_driver)]
    return pd.DataFrame(rows)


def test_driver_split_is_disjoint_and_complete():
    df = _fake_df()
    s = make_driver_split(df, n_val=4, n_test=4, seed=0)
    tr, va, te = map(set, (s["train"], s["val"], s["test"]))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(df.subject)
    assert len(va) == 4 and len(te) == 4
    assert make_driver_split(df, seed=0) == s  # deterministic


def test_committed_split_has_no_leakage():
    from ddd.data import DEFAULT_SPLIT_PATH
    from ddd.utils import load_json
    s = load_json(DEFAULT_SPLIT_PATH)
    groups = [set(s[k]) for k in ("train", "val", "test")]
    assert sum(map(len, groups)) == len(set().union(*groups)) == 26


def test_scratch_resnet50_shapes():
    m = ResNet50Scratch().eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, 64, 64)).shape == (2, NUM_CLASSES)
        assert m(torch.randn(1, 3, 168, 224)).shape == (1, NUM_CLASSES)
    # Depth as in the paper: 1 stem conv + 16 bottlenecks x 3 convs + 1 fc = 50 weighted layers,
    # plus 4 projection-shortcut convs (one per stage).
    n_conv = sum(isinstance(x, torch.nn.Conv2d) for x in m.modules())
    n_fc = sum(isinstance(x, torch.nn.Linear) for x in m.modules())
    assert (n_conv - 4) + n_fc == 50
    assert gradcam_target_layer(m) is m.stage5[-1]


def test_timm_model_builds_without_download():
    m = build_model("resnet18", pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 3, 168, 224)).shape == (1, NUM_CLASSES)


def test_train_transforms_do_not_flip(tmp_path):
    # Left half white, right half black: any horizontal flip would invert the mean of the halves.
    arr = np.zeros((240, 320, 3), np.uint8)
    arr[:, :160] = 255
    tf = build_transforms((168, 224), train=True, mean=(0, 0, 0), std=(1, 1, 1))
    torch.manual_seed(0)
    for _ in range(30):
        x = tf(Image.fromarray(arr))
        assert x.shape == (3, 168, 224)
        assert x[:, :, :60].mean() > x[:, :, -60:].mean()


def test_dataset_reads_images(tmp_path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (320, 240), (10, 20, 30)).save(p)
    ds = DriverImages([p], [3], build_transforms((64, 64), train=False))
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and int(y) == 3


def test_metrics_and_temperature():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 500)
    logits = rng.normal(size=(500, 10)).astype(np.float32)
    logits[np.arange(500), labels] += 2.0
    logits *= 4  # overconfident
    t = fit_temperature(logits, labels)
    assert t > 1.5
    m_raw = compute_metrics(softmax(logits), labels)
    m_cal = compute_metrics(softmax(logits, t), labels)
    assert m_cal["log_loss"] < m_raw["log_loss"]
    assert m_cal["accuracy"] == pytest.approx(m_raw["accuracy"])
