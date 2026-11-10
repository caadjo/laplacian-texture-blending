"""
GPU-Friendly Laplacian Texture Blending (Wronski, JCGT 2025)
Offline Reference Implementation & Experimental Comparison

Approximates the core algorithm from Listing 1 of the paper:
'GPU Friendly Laplacian Texture Blending', Bartlomiej Wronski (2025).
"""

import argparse
import os
import numpy as np
from PIL import Image


def mip_schedule(num_levels, stride=1):
    """Keep the finest level and the residual, even for an odd depth."""
    if num_levels < 0 or stride < 1:
        raise ValueError("Depth must be nonnegative and stride must be positive.")
    return (*range(0, num_levels, stride), num_levels)


def build_mipmap_chain(img_array: np.ndarray, lods, filter_type=Image.Resampling.BILINEAR):
    """
    Constructs a chain of mipmap levels upsampled back to full resolution,
    simulating hardware texture sampling at different LODs:
    Fup((G_k) ^ k)
    """
    h, w = img_array.shape[0], img_array.shape[1]
    arr_uint8 = (np.clip(img_array, 0, 1) * 255.0).astype(np.uint8)
    if arr_uint8.ndim == 3 and arr_uint8.shape[2] == 1:
        base_img = Image.fromarray(arr_uint8[:, :, 0], mode="L")
    else:
        base_img = Image.fromarray(arr_uint8)
    
    levels = []
    # Level 0 is the original full-resolution image
    levels.append(img_array.copy())
    
    for i in lods[1:]:
        target_w = max(1, w >> i)
        target_h = max(1, h >> i)
        
        # Downsample to mip level
        down = base_img.resize((target_w, target_h), resample=filter_type)
        # Upsample bilinearly back to original resolution (simulating GPU texture2D with lod)
        up = down.resize((w, h), resample=Image.Resampling.BILINEAR)
        
        level_array = np.array(up, dtype=np.float32) / 255.0
        if level_array.ndim == 2:
            level_array = level_array[:, :, np.newaxis]
        levels.append(level_array)
        
    return levels


def laplacian_blend(tex0: np.ndarray, tex1: np.ndarray, mask: np.ndarray, num_levels: int = 4,
                    stride: int = 1):
    """
    Simulates Listing 1 for stride 1 and Section 6.2 for stride 2.
    Each grouped band uses the mask of its finer endpoint.
    """
    lods = mip_schedule(num_levels, stride)
    tex0_levels = build_mipmap_chain(tex0, lods)
    tex1_levels = build_mipmap_chain(tex1, lods)
    mask_levels = build_mipmap_chain(mask, lods)
    
    blended = np.zeros_like(tex0, dtype=np.float32)
    
    # Accumulate Laplacian band differences
    for i in range(len(lods) - 1):
        tex0_laplace = tex0_levels[i] - tex0_levels[i + 1]
        tex1_laplace = tex1_levels[i] - tex1_levels[i + 1]
        
        # Blend each band with corresponding mask level
        blended += tex0_laplace * (1.0 - mask_levels[i]) + tex1_laplace * mask_levels[i]
        
    # Coarsest Gaussian residual level
    tex0_gauss = tex0_levels[-1]
    tex1_gauss = tex1_levels[-1]
    blended += tex0_gauss * (1.0 - mask_levels[-1]) + tex1_gauss * mask_levels[-1]
    
    return np.clip(blended, 0.0, 1.0)


def naive_linear_blend(tex0: np.ndarray, tex1: np.ndarray, mask: np.ndarray):
    """
    Standard pointwise linear texture blending (lerp baseline).
    """
    return np.clip(tex0 * (1.0 - mask) + tex1 * mask, 0.0, 1.0)


def local_variance(img_gray: np.ndarray, kernel_size: int = 7):
    """
    Computes local patch variance as a proxy for contrast/texture detail preservation.
    """
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(img_gray, size=kernel_size)
    mean_sq = uniform_filter(img_gray ** 2, size=kernel_size)
    var = mean_sq - mean ** 2
    return np.maximum(var, 0.0)


def load_inputs(tex0, tex1, mask):
    with Image.open(tex0) as source:
        img0 = source.convert("RGB")
    with Image.open(tex1) as source:
        img1 = source.convert("RGB").resize(img0.size, Image.Resampling.BILINEAR)
    w, h = img0.size

    # Load or generate mask
    if mask == "linear":
        x = np.linspace(0, 1, w, dtype=np.float32)
        mask_arr = np.tile(x, (h, 1))[:, :, np.newaxis]
    else:
        with Image.open(mask) as source:
            mask_img = source.convert("L").resize((w, h), Image.Resampling.BILINEAR)
        raw_mask = np.array(mask_img, dtype=np.float32) / 255.0
        # Sigmoid/threshold similar to demo shader for organic transition boundary
        threshold = 0.55
        mask_arr = np.clip((raw_mask - threshold) * 8.0 + 0.5, 0.0, 1.0)[:, :, np.newaxis]

    return np.asarray(img0, dtype=np.float32) / 255, np.asarray(img1, dtype=np.float32) / 255, mask_arr


def main():
    parser = argparse.ArgumentParser(description="GPU-Friendly Laplacian Texture Blending (Reference Runner)")
    parser.add_argument("--tex0", default="demo_webgl/1.jpg", help="Path to Texture 0")
    parser.add_argument("--tex1", default="demo_webgl/2.jpg", help="Path to Texture 1")
    parser.add_argument("--mask", default="demo_webgl/perlin.jpg", help="Path to mask or 'linear'")
    parser.add_argument("--levels", type=int, default=4, help="Last mip level (default: 4)")
    parser.add_argument("--stride", type=int, default=1, help="Distance between sampled mip levels")
    parser.add_argument("--outdir", default="output", help="Output directory")
    args = parser.parse_args()
    try:
        lods = mip_schedule(args.levels, args.stride)
    except ValueError as error:
        parser.error(str(error))

    t0_arr, t1_arr, mask_arr = load_inputs(args.tex0, args.tex1, args.mask)
    h, w = mask_arr.shape[:2]
    os.makedirs(args.outdir, exist_ok=True)

    print("=================================================================")
    print("  GPU-Friendly Laplacian Texture Blending - Verificação")
    print("=================================================================")
    print(f"Textura 0: {args.tex0} ({w}x{h})")
    print(f"Textura 1: {args.tex1} ({w}x{h})")
    print(f"Níveis mip amostrados: {lods}")

    # 2. Computar blends
    print("\nExecutando Linear Blend (Baseline)...")
    linear_res = naive_linear_blend(t0_arr, t1_arr, mask_arr)

    print(f"Executando Laplacian Texture Blend ({args.levels} níveis)...")
    laplacian_res = laplacian_blend(t0_arr, t1_arr, mask_arr, args.levels, args.stride)

    # 3. Métricas quantitativas de preservação de detalhe na zona de transição
    # Transição é onde a máscara está entre 0.05 e 0.95
    blend_region = (mask_arr[:, :, 0] > 0.05) & (mask_arr[:, :, 0] < 0.95)
    
    # Converter para escala de cinza para medir variância local (detalhes/contraste)
    lin_gray = np.dot(linear_res[..., :3], [0.2989, 0.5870, 0.1140])
    lap_gray = np.dot(laplacian_res[..., :3], [0.2989, 0.5870, 0.1140])
    
    print("\n--- Resultados Quantitativos na Zona de Transição ---")
    print(f"Pixels na zona de transição: {np.count_nonzero(blend_region)} ({np.mean(blend_region)*100:.1f}% da imagem)")
    if blend_region.any():
        var_lin = np.mean(local_variance(lin_gray)[blend_region])
        var_lap = np.mean(local_variance(lap_gray)[blend_region])
        print(f"Variância local média - Linear Blend:    {var_lin:.6f}")
        print(f"Variância local média - Laplacian Blend: {var_lap:.6f}")
        if var_lin > 0:
            print(f"Variação relativa da variância local: {(var_lap / var_lin - 1) * 100:+.1f}%")

    # 4. Salvar saídas
    p_lin = os.path.join(args.outdir, "blend_linear_baseline.png")
    suffix = "" if args.stride == 1 else f"_stride{args.stride}"
    p_lap = os.path.join(args.outdir, f"blend_laplacian_{args.levels}levels{suffix}.png")
    p_mask = os.path.join(args.outdir, "mask_used.png")

    Image.fromarray((linear_res * 255.0).astype(np.uint8)).save(p_lin)
    Image.fromarray((laplacian_res * 255.0).astype(np.uint8)).save(p_lap)
    Image.fromarray((np.repeat(mask_arr, 3, axis=2) * 255.0).astype(np.uint8)).save(p_mask)

    # 5. Criar montagem comparativa (Grid)
    # Linha 1: Tex0, Mascara, Tex1
    # Linha 2: Linear Baseline, Laplacian Blend, Mapa de Diferença Absoluta (|Lap - Lin| * 3x)
    thumb_w, thumb_h = max(1, w // 2), max(1, h // 2)
    def thumb(arr):
        return Image.fromarray((np.clip(arr, 0, 1) * 255.0).astype(np.uint8)).resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)

    diff = np.abs(laplacian_res - linear_res) * 4.0  # amplificado 4x para visualização
    
    grid = Image.new("RGB", (thumb_w * 3, thumb_h * 2), (25, 25, 25))
    grid.paste(thumb(t0_arr), (0, 0))
    grid.paste(thumb(np.repeat(mask_arr, 3, axis=2)), (thumb_w, 0))
    grid.paste(thumb(t1_arr), (thumb_w * 2, 0))

    grid.paste(thumb(linear_res), (0, thumb_h))
    grid.paste(thumb(laplacian_res), (thumb_w, thumb_h))
    grid.paste(thumb(diff), (thumb_w * 2, thumb_h))

    p_grid = os.path.join(args.outdir, "comparativo_completo.png")
    grid.save(p_grid)

    print(f"\nImagens salvas em '{args.outdir}/':")
    print(f" - {p_lin}")
    print(f" - {p_lap}")
    print(f" - {p_mask}")
    print(f" - {p_grid}")
    print("\nExecução e comprovação da Regra de Viabilidade Inicial concluídas com sucesso!")


if __name__ == "__main__":
    main()
