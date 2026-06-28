"""Image transforms: multi-stage degradation (train + robust eval) and the
shortcut-control normalization (canonical resize + optional JPEG re-encode).

PIL.Image -> normalized CHW float tensor.
"""
import random
import numpy as np
import cv2
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_bgr(pil):
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def canonical_resize(bgr, size):
    """Resize short side to `size` (area), center-crop size x size. size must be /16."""
    h, w = bgr.shape[:2]
    s = size / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    y, x = (nh - size) // 2, (nw - size) // 2
    return bgr[y:y + size, x:x + size]


def jpeg_reencode(bgr, q):
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else bgr


# --- degradation ops (mirror the NTIRE robust regime) ---
def _deg_jpeg(b):
    return jpeg_reencode(b, random.randint(30, 95))


def _deg_resize(b):
    h, w = b.shape[:2]
    f = random.uniform(0.5, 1.0)
    s = cv2.resize(b, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)


def _deg_blur(b):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(b, (k, k), 0)


def _deg_noise(b):
    sigma = random.uniform(2, 12)
    out = b.astype(np.float32) + np.random.randn(*b.shape) * sigma
    return np.clip(out, 0, 255).astype(np.uint8)


DEG_OPS = [_deg_jpeg, _deg_resize, _deg_blur, _deg_noise]
SEVERITY = {"clean": 0, "mild": 1, "moderate": 2, "heavy": 3}


def apply_degradation(bgr, n_ops):
    for op in random.sample(DEG_OPS, k=min(n_ops, len(DEG_OPS))):
        bgr = op(bgr)
    return bgr


class DetectionTransform:
    """train=True -> random-severity aug; fixed_severity set -> deterministic robust eval."""

    def __init__(self, size=256, train=False, reencode_jpeg=True, jpeg_quality=90,
                 severity_weights=(0.4, 0.3, 0.2, 0.1), fixed_severity=None,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.size = size
        self.train = train
        self.reencode = reencode_jpeg
        self.q = jpeg_quality
        self.sw = severity_weights
        self.fixed = fixed_severity
        self.mean = np.array(mean, np.float32)
        self.std = np.array(std, np.float32)

    def _n_ops(self):
        if self.fixed is not None:
            return SEVERITY[self.fixed]
        return random.choices([0, 1, 2, 3], weights=self.sw)[0]

    def __call__(self, pil):
        b = _to_bgr(pil)
        if self.train or self.fixed is not None:
            n = self._n_ops()
            if n > 0:
                b = apply_degradation(b, n)
        b = canonical_resize(b, self.size)
        if self.reencode:                       # shortcut-control normalization
            b = jpeg_reencode(b, self.q)
        rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        return torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous().float()
