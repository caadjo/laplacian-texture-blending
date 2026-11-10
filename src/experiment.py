"""Evaluate LOD skipping on the albedo textures bundled with the WebGL demo."""

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import PIL
import scipy
from PIL import Image
from scipy.ndimage import gaussian_filter

from laplacian_blend import (
    laplacian_blend, load_inputs, local_variance, mip_schedule, naive_linear_blend,
)

ROOT = Path(__file__).resolve().parents[1]
DEPTH = 6
STRIDE = 2
LUMA = (0.2989, 0.5870, 0.1140)
PAIRS = {"materiais": ("1.jpg", "2.jpg"), "rotacao": ("1.jpg", "1rot.jpg")}
MASKS = {"perlin": ROOT / "demo_webgl/perlin.jpg", "rampa": "linear"}


def similarity(reference, image, region):
    """PSNR in the ROI and mean RGB SSIM at ROI centers with full 11x11 support."""
    a, b = reference.astype(np.float64), image.astype(np.float64)

    def smooth(x):
        return gaussian_filter(x, sigma=1.5, radius=5, axes=(0, 1))

    mu_a, mu_b = smooth(a), smooth(b)
    var_a = smooth(a * a) - mu_a * mu_a
    var_b = smooth(b * b) - mu_b * mu_b
    cov = smooth(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)
            / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))
    mse = np.mean((a[region] - b[region]) ** 2)
    psnr = float(-10 * np.log10(mse)) if mse else float("inf")
    return float(ssim[5:-5, 5:-5][region[5:-5, 5:-5]].mean()), psnr


def save_panel(path, images):
    """Three columns, with inputs above the three blending outputs."""
    size = 384
    panel = Image.new("RGB", (3 * size, 2 * size))
    for i, array in enumerate(images):
        encoded = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
        thumbnail = Image.fromarray(encoded).resize((size, size), Image.Resampling.LANCZOS)
        panel.paste(thumbnail, (i % 3 * size, i // 3 * size))
    panel.save(path)


def run_case(outdir, pair_name, files, mask_name, mask_source):
    t0, t1, mask = load_inputs(*(ROOT / "demo_webgl" / f for f in files), mask_source)
    region = (mask[..., 0] > 0.05) & (mask[..., 0] < 0.95)
    linear = naive_linear_blend(t0, t1, mask)
    full = laplacian_blend(t0, t1, mask, DEPTH)
    skipped = laplacian_blend(t0, t1, mask, DEPTH, STRIDE)
    save_panel(outdir / f"{pair_name}_{mask_name}.png", (
        t0, np.repeat(mask, 3, axis=2), t1, linear, full, skipped,
    ))
    error = np.rint(np.clip(5 * np.abs(full - skipped), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(error).save(outdir / f"{pair_name}_{mask_name}_erro5x.png")

    rows = []
    for method, image, lods in (
        ("lerp", linear, (0,)),
        ("lap_completo", full, mip_schedule(DEPTH)),
        ("lap_salto2", skipped, mip_schedule(DEPTH, STRIDE)),
    ):
        ssim, psnr = similarity(full, image, region)
        rows.append({
            "par": pair_name,
            "mascara": mask_name,
            "metodo": method,
            "pixels_roi": int(region.sum()),
            "percentual_roi": f"{100 * region.mean():.3f}",
            "lods": " ".join(map(str, lods)),
            "amostras_logicas": 3 * len(lods),
            "ssim_roi": f"{ssim:.6f}",
            "psnr_roi_db": f"{psnr:.4f}",
            "variancia_local_roi": f"{local_variance(image @ LUMA)[region].mean():.9f}",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair_name, files in PAIRS.items():
        for mask_name, source in MASKS.items():
            rows.extend(run_case(args.outdir, pair_name, files, mask_name, source))

    with (args.outdir / "metrics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with (args.outdir / "environment.json").open("w", encoding="utf-8") as output:
        json.dump({
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "Pillow": PIL.__version__,
            "scipy": scipy.__version__,
            "depth": DEPTH,
            "stride": STRIDE,
            "pairs": PAIRS,
            "masks": {"perlin": "demo_webgl/perlin.jpg", "rampa": "linear"},
        }, output, indent=2)
        output.write("\n")
    print(f"{len(rows)} medições salvas em {args.outdir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
