"""Prediction and Grad-CAM for single images (used by the Streamlit app)."""
import time

import numpy as np
import torch
from PIL import Image

from config import CLASSES, MODELS_DIR
from model import build_model, eval_transform

# "Inferno"-like colour ramp for the heatmap: dark purple -> red -> yellow.
_RAMP = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], dtype=np.float32)


def load_model(path=MODELS_DIR / "model.pt"):
    m = build_model(len(CLASSES), pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    # Frozen weights: autograd then only tracks the graph after the Grad-CAM layer (see predict).
    m.requires_grad_(False)
    return m.eval()


def predict(model, img: Image.Image):
    """Return (class probabilities, Grad-CAM map in [0, 1] at the image's size, seconds taken).

    Grad-CAM weights the last convolutional block's activations by the gradient
    of the predicted class score, showing which regions drove the decision.
    """
    t0 = time.perf_counter()
    x = eval_transform()(img.convert("RGB"))[None]
    acts = {}

    def keep(module, inp, out):
        acts["a"] = out.requires_grad_()   # gradients start here, saving memory upstream

    hook = model.features[-1].register_forward_hook(keep)
    try:
        out = model(x)
    finally:
        hook.remove()
    probs = out.softmax(1)[0]
    grads = torch.autograd.grad(out[0, probs.argmax()], acts["a"])[0]
    cam = torch.relu((grads.mean((2, 3), keepdim=True) * acts["a"]).sum(1))[0]
    cam = (cam / (cam.max() + 1e-8)).detach().numpy()
    cam = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BILINEAR)) / 255.0
    return probs.detach().numpy(), cam, time.perf_counter() - t0


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.6) -> Image.Image:
    """Blend the heatmap onto the grayscale CT slice; weak regions stay gray."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)[..., None].repeat(3, 2)
    pos = np.linspace(0, 1, len(_RAMP))
    color = np.stack([np.interp(cam, pos, _RAMP[:, c]) for c in range(3)], -1)
    w = alpha * cam[..., None]
    return Image.fromarray(np.uint8((1 - w) * gray + w * color))
