import os
import time

# --- HOTFIX WINDOWS ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from config import REAL_TEST_DIR, FAKE_TEST_DIR, IMG_EXTS
from logger_utils import setup_logging


def list_images(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def make_random_mask(w: int, h: int, min_frac=0.18, max_frac=0.35) -> Image.Image: # tinha 0.25 0.55
    """
    Máscara branca (255) = região a inpaint
    Máscara preta (0) = manter
    Faz 1-3 retângulos aleatórios.
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # n = random.randint(1, 3)
    n = 1
    for _ in range(n):
        rw = int(random.uniform(min_frac, max_frac) * w)
        rh = int(random.uniform(min_frac, max_frac) * h)


        # x0 = random.randint(0, max(0, w - rw))
        # y0 = random.randint(0, max(0, h - rh))
        x_min = int(0.15 * w)
        x_max = int(0.85 * w) - rw
        y_min = int(0.15 * h)
        y_max = int(0.85 * h) - rh

        x0 = random.randint(x_min, max(x_min, x_max))
        y0 = random.randint(y_min, max(y_min, y_max))

        x1 = x0 + rw
        y1 = y0 + rh

        draw.rectangle([x0, y0, x1, y1], fill=255)

    return mask


def main():

    start_time = time.time()


    setup_logging("generate_fakes_inpainting")

    # --- 1) Modelo de inpainting (Diffusers) ---
    # Tens 2 opções comuns:
    #  - "runwayml/stable-diffusion-inpainting"
    #  - "stabilityai/stable-diffusion-2-inpainting"
    model_id = os.environ.get("INPAINT_MODEL", "runwayml/stable-diffusion-inpainting")

    prompt = os.environ.get("INPAINT_PROMPT", "a photo of a person, realistic, natural skin")
    negative_prompt = os.environ.get("INPAINT_NEG_PROMPT", "cartoon, anime, painting, deformed, distorted, low quality, blurry, extra fingers")

    # num_inference_steps = int(os.environ.get("INPAINT_STEPS", "8")) #quantos steps por imagem
    # guidance_scale = float(os.environ.get("INPAINT_GUIDANCE", "6.5")) #Forca que o modelo segue o prompt textual, valores baixos o modelo ignora o texto, mais altos forca demaisado e pode criar artefactos
    # strength = float(os.environ.get("INPAINT_STRENGTH", "0.75")) # 0 = imagem quase igual a original, 1  = imagem quase totalemnte regenerada
    seed = int(os.environ.get("INPAINT_SEED", "123"))

    NUM_INFERENCE_STEPS = 12
    GUIDANCE_SCALE = 5.5
    STRENGTH = 0.65
    MAX_IMAGES = 12000  # número de imagens a processar tinha 12000


    real_files = list_images(REAL_TEST_DIR)

    # Limitar número de imagens para não demorar tanto
    real_files = real_files[:MAX_IMAGES]

    print(f"[INFO] A gerar fakes para {len(real_files)} imagens")

    if not real_files:
        raise ValueError(f"Não encontrei imagens em {REAL_TEST_DIR}")

    FAKE_TEST_DIR.mkdir(parents=True, exist_ok=True)

    MASKS_DIR = FAKE_TEST_DIR.parent / "masks"
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    # Import aqui para falhar cedo se não estiver instalado
    from diffusers import StableDiffusionInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    random.seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)

    pipe.safety_checker = None

    # Pequena otimização quando há GPU
    # Pequenas otimizações quando há GPU (xformers é opcional)
    # if device == "cuda":
    #     try:
    #         if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
    #             pipe.enable_xformers_memory_efficient_attention()
    #             print("[OK] xformers ativado.")
    #     except ModuleNotFoundError:
    #         print("[AVISO] xformers não instalado. A correr sem xformers.")
    # Não usar xformers (evita conflitos no Windows)

    # xformers é opcional
    # if device == "cuda" and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
    #     try:
    #         pipe.enable_xformers_memory_efficient_attention()
    #         print("[OK] xformers ativado.")
    #     except ModuleNotFoundError:
    #         print("[AVISO] xformers não instalado. A correr sem xformers.")
    #     except Exception as e:
    #         print(f"[AVISO] Falhou ativação do xformers ({e}). A correr sem xformers.")

    print(f"[INFO] Real test: {REAL_TEST_DIR}")
    print(f"[INFO] Fake test: {FAKE_TEST_DIR}")
    print(f"[INFO] Model: {model_id}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Prompt: {prompt}")

    print(
        f"[INFO] Steps: {NUM_INFERENCE_STEPS} | "
        f"Guidance: {GUIDANCE_SCALE} | "
        f"Strength: {STRENGTH} | "
        f"MAX_IMAGES: {MAX_IMAGES}"
    )

    for i, real_path in enumerate(real_files, start=1):
        out_path = FAKE_TEST_DIR / real_path.name

        # Se já existe, salta (útil se interromperes e voltares a correr)
        if out_path.exists():
            continue

        img = Image.open(real_path).convert("RGB")
        w, h = img.size
        mask = make_random_mask(w, h)

        mask_out_path = MASKS_DIR / f"{real_path.stem}_mask.png"
        mask.save(mask_out_path)

        if device == "cuda":
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                result = pipe(
                    prompt=prompt,
                    image=img,
                    mask_image=mask,
                    negative_prompt=negative_prompt,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    strength=STRENGTH,
                ).images[0]
        else:
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    image=img,
                    mask_image=mask,
                    negative_prompt=negative_prompt,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    strength=STRENGTH,
                ).images[0]

        result.save(out_path)
        if i % 100 == 0:
            print(f"#############################[OK] {i}/{len(real_files)} gerados#############################")

    print(f"[DONE] Gerados fakes para {len(real_files)} reais.")

    end_time = time.time()
    total_seconds = end_time - start_time

    print(f"[TIME] Tempo total de execução: {total_seconds / 60:.2f} minutos")


if __name__ == "__main__":
    main()
