"""Torch-free image descriptions for the demo (Prof. Lu's second feedback round): numpy + PIL only.

  palette()             6 dominant colours: k-means in CIELAB on a downscaled copy, fixed seed, deterministic
  aesthetic_measures()  descriptive aesthetic properties (colour harmony, visual balance, visual complexity,
                        contrast); descriptions of the pixels, NOT a quality score
  map_similarity()      Pearson correlation of two 7 x 7 attention maps (49 weights) + a plain label
  difference_overlay()  A - B attention difference over a grey copy of the centre square, diverging colours
  with_grid()           the 7 x 7 grid CLIP's regions lie on, drawn over a square image

The measures are computed exactly as artfae.features.aesthetic.extract prepares its input (RGB, resized to
256 x 256 bilinear, Rec. 709 luminance, edges = gradient magnitude > 0.08), so edge density, RMS contrast and the
luminance range agree with the 28 attributes. Percentiles: app/assets/aesthetic_percentiles.json, built over the
training paintings by app/build_aesthetic_percentiles.py.
"""
import numpy as np
from PIL import Image, ImageDraw

PALETTE_K = 6
PALETTE_SIDE = 128   # the downscaled copy: longest side in pixels
PALETTE_SEED = 0
KMEANS_ITERS = 100

# (key, group) in display order; groups: harmony, balance, complexity, contrast
MEASURES = [("hue_spread", "harmony"), ("complementary", "harmony"),
            ("balance_lum_lr", "balance"), ("balance_lum_tb", "balance"),
            ("balance_edge_lr", "balance"), ("balance_edge_tb", "balance"),
            ("edge_density", "complexity"), ("colour_entropy", "complexity"),
            ("rms_contrast", "contrast"), ("luminance_range", "contrast")]
MEASURE_NAMES = [k for k, _ in MEASURES]
GROUPS = ["harmony", "balance", "complexity", "contrast"]


# ---------------------------------------------------------------- colour spaces (sRGB D65 <-> CIELAB)
_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_M_INV = np.linalg.inv(_M)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb):
    """rgb: (..., 3) in 0-255 -> CIELAB (D65)."""
    c = np.asarray(rgb, np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab):
    """CIELAB (D65) -> sRGB 0-255 floats (unclipped)."""
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    lin = xyz @ _M_INV.T
    lin = np.clip(lin, 0, None)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return c * 255.0


def hex_code(rgb) -> str:
    """(r, g, b) 0-255 -> '#RRGGBB' (uppercase)."""
    r, g, b = (int(v) for v in np.clip(np.rint(rgb), 0, 255))
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(code: str):
    return tuple(int(code[i:i + 2], 16) for i in (1, 3, 5))


# ---------------------------------------------------------------- palette
def downscaled(img: Image.Image, side: int = PALETTE_SIDE) -> np.ndarray:
    im = img.convert("RGB")
    w, h = im.size
    s = side / max(w, h)
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BILINEAR)
    return np.asarray(im, np.uint8).reshape(-1, 3)


def kmeans(x, k, seed=PALETTE_SEED, iters=KMEANS_ITERS):
    """Lloyd's k-means with k-means++ initialisation; deterministic for a given seed. -> (centres, labels)."""
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(len(x))]]
    d2 = ((x - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        total = d2.sum()
        i = int(rng.choice(len(x), p=d2 / total)) if total > 0 else int(d2.argmax())
        centres.append(x[i])
        d2 = np.minimum(d2, ((x - x[i]) ** 2).sum(1))
    c = np.array(centres, np.float64)
    labels = None
    for _ in range(iters):
        dist = ((x[:, None, :] - c[None, :, :]) ** 2).sum(2)
        new = dist.argmin(1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                c[j] = x[m].mean(0)
            else:  # empty cluster: take the point farthest from its centre
                far = int(dist[np.arange(len(x)), labels].argmax())
                c[j] = x[far]; labels[far] = j
    return c, labels


def round_shares(shares, total: int = 1000) -> list:
    """Largest-remainder rounding to `total` units (1000 = tenths of a percent), so the shown values sum exactly."""
    q = np.asarray(shares, np.float64) * total / max(np.sum(shares), 1e-12)
    out = np.floor(q).astype(int)
    for i in sorted(range(len(q)), key=lambda i: (-(q[i] - out[i]), i))[: total - int(out.sum())]:
        out[i] += 1
    return out.tolist()


def palette(img: Image.Image, k: int = PALETTE_K) -> list:
    """-> [{"hex": "#RRGGBB", "rgb": [r, g, b], "lab": [L, a, b], "share": float, "percent": float}, ...]
    sorted by share (largest first). share sums to 1; percent (one decimal) sums to exactly 100."""
    px = downscaled(img)
    lab = srgb_to_lab(px)
    k = min(k, len(np.unique(px, axis=0)))
    centres, labels = kmeans(lab, k)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    order = sorted(range(k), key=lambda j: (-counts[j], -centres[j][0]))
    shares = counts[order] / counts.sum()
    tenths = round_shares(shares)
    out = []
    for j, s, t in zip(order, shares, tenths):
        rgb = np.clip(np.rint(lab_to_srgb(centres[j])), 0, 255).astype(int)
        out.append({"hex": hex_code(rgb), "rgb": rgb.tolist(), "lab": [round(float(v), 4) for v in centres[j]],
                    "share": float(s), "percent": t / 10})
    return out


# ---------------------------------------------------------------- descriptive aesthetic properties
def _balance(a: float, b: float) -> float:
    """1 = both halves equal; 0 = everything in one half."""
    return 1.0 if a + b <= 1e-12 else float(1 - abs(a - b) / (a + b))


def aesthetic_measures(img: Image.Image) -> dict:
    """Descriptive aesthetic properties of an image (each in [0, 1]); none of them is a quality score.

      hue_spread        1 - mean resultant length of the saturation-weighted hues of coloured pixels
                        (0 = one hue family, 1 = hues spread around the colour wheel)
      complementary     share of the saturation-weighted hue mass (12 bins of 30 deg, centred on 0, 30, ... deg)
                        matched by mass in the opposite bin (180 deg away): 0 = no complementary pairs, 1 = all paired
      balance_lum_lr/tb luminance balance of the left/right and top/bottom halves (1 = equal)
      balance_edge_lr/tb  edge balance of the same halves (1 = equal)
      edge_density      share of edge pixels (as attribute edge_density)
      colour_entropy    Shannon entropy of a 4 x 4 x 4 RGB histogram / log(64) (0 = one colour, 1 = all 64 equal)
      rms_contrast      standard deviation of luminance (as attribute luminance_std)
      luminance_range   P95 - P5 of luminance (as attribute contrast_p95_p05)
    plus the sides ("left"/"right", "top"/"bottom") that are brighter or busier, for the plain-language text.
    """
    im = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    rgb8 = np.asarray(im, np.uint8)
    rgb = rgb8.astype(np.float32) / 255
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255
    h, s, _ = hsv.transpose(2, 0, 1)
    y = rgb @ np.array([.2126, .7152, .0722])
    dy, dx = np.gradient(y)
    edge = np.hypot(dx, dy) > .08

    w = np.where(s > .15, s, 0.0).astype(np.float64)
    if w.sum() > 1e-8:
        ang = h.astype(np.float64) * 2 * np.pi
        r_len = np.hypot((w * np.cos(ang)).sum(), (w * np.sin(ang)).sum()) / w.sum()
        hue_spread = float(1 - r_len)
        hist = np.histogram((h + 1 / 24) % 1, bins=12, range=(0, 1), weights=w)[0]  # bins centred on 0, 30, ... deg
        hist = hist / hist.sum()
        complementary = float(sum(2 * min(hist[i], hist[i + 6]) for i in range(6)))
    else:  # no coloured pixels (greyscale): no hue spread, no complementary pairs
        hue_spread = complementary = 0.0

    yl, yr, yt, yb = y[:, :128].mean(), y[:, 128:].mean(), y[:128].mean(), y[128:].mean()
    el, er, et, eb = edge[:, :128].mean(), edge[:, 128:].mean(), edge[:128].mean(), edge[128:].mean()
    q = (rgb8 // 64).reshape(-1, 3).astype(np.int64)
    counts = np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(64))
    values = {"hue_spread": hue_spread, "complementary": complementary,
              "balance_lum_lr": _balance(yl, yr), "balance_lum_tb": _balance(yt, yb),
              "balance_edge_lr": _balance(el, er), "balance_edge_tb": _balance(et, eb),
              "edge_density": float(edge.mean()), "colour_entropy": entropy,
              "rms_contrast": float(y.std()), "luminance_range": float(np.percentile(y, 95) - np.percentile(y, 5))}
    sides = {"lum_lr": "left" if yl >= yr else "right", "lum_tb": "top" if yt >= yb else "bottom",
             "edge_lr": "left" if el >= er else "right", "edge_tb": "top" if et >= eb else "bottom"}
    return {"values": values, "sides": sides}


def measure_vector(m: dict) -> np.ndarray:
    return np.array([m["values"][k] for k in MEASURE_NAMES], np.float64)


# ---------------------------------------------------------------- attention maps
SIMILAR_VERY, SIMILAR = 0.9, 0.7


def map_similarity(a, b):
    """Pearson correlation of the 49 weights of two maps -> (r, label) with label very_similar (r >= 0.9),
    similar (r >= 0.7) or different. Identical maps give r = 1; a constant map has no correlation (nan, different)."""
    x, y = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if np.array_equal(x, y):
        r = 1.0
    elif x.std() < 1e-15 or y.std() < 1e-15:
        r = float("nan")
    else:
        r = float(np.corrcoef(x, y)[0, 1])
    label = "very_similar" if r >= SIMILAR_VERY else "similar" if r >= SIMILAR else "different"
    return r, label


# diverging, colour-blind-safe (blue - light grey - orange); A higher = orange, B higher = blue
DIVERGING = np.array([[33, 102, 172], [146, 197, 222], [247, 247, 247], [244, 165, 130], [178, 88, 6]], float)


def diverging(x):
    """x in [-1, 1] -> RGB; -1 = B higher (blue), 0 = equal (light grey), +1 = A higher (orange)."""
    pos = np.linspace(-1, 1, len(DIVERGING))
    return np.stack([np.interp(x, pos, DIVERGING[:, i]) for i in range(3)], -1)


def difference_overlay(square: Image.Image, map_a, map_b, size: int = 640, alpha: float = 0.85):
    """A - B (both 7 x 7) upsampled (bicubic) over a grey copy of the centre square.
    -> (image, max |A - B| in percentage points); the colour scale is symmetric around 0."""
    d = np.asarray(map_a, np.float64) - np.asarray(map_b, np.float64)
    vmax = float(np.abs(d).max())
    scaled = d / vmax if vmax > 1e-15 else np.zeros_like(d)
    u8 = Image.fromarray(np.uint8(np.clip(np.rint((scaled + 1) * 127.5), 0, 255)))
    up = np.asarray(u8.resize((size, size), Image.Resampling.BICUBIC), np.float64) / 127.5 - 1
    grey = np.asarray(square.convert("L").resize((size, size), Image.Resampling.LANCZOS), np.float64)[..., None]
    a = (0.2 + (alpha - 0.2) * np.abs(up))[..., None]  # colour opacity grows with |A - B|: unchanged areas stay grey
    out = (1 - a) * np.repeat(grey, 3, -1) + a * diverging(up)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), vmax * 100


def with_grid(img: Image.Image, n: int = 7, colour=(255, 255, 255)) -> Image.Image:
    """The n x n region grid over a square image (lines at every 1/n; an outer frame too)."""
    out = img.convert("RGB").copy()
    w, h = out.size
    d = ImageDraw.Draw(out)
    width = max(1, round(w / 320))
    for i in range(n + 1):
        x = min(round(i * w / n), w - 1); yy = min(round(i * h / n), h - 1)
        d.line([(x, 0), (x, h - 1)], fill=colour, width=width)
        d.line([(0, yy), (w - 1, yy)], fill=colour, width=width)
    return out
