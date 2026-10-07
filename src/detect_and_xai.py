import os

# --- HOTFIX WINDOWS (OpenMP) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"


import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import PAIRS_CSV, PROJECT_ROOT
from logger_utils import setup_logging
from train_colorizer import UNetSmall, IMG_SIZE, read_rgb, rgb_to_lab


# --- Paths/outputs ---
MODEL_PATH = PROJECT_ROOT / "outputs" / "colorizer" / "best_colorizer.pt"
OUT_DIR = PROJECT_ROOT / "outputs" / "detection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MASKS_DIR = PROJECT_ROOT / "data" / "masks"


def load_mask(mask_path: Path, target_shape_hw):
    # Lê máscara (0..255) e garante que tem o mesmo tamanho de E (H,W)
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Falha a ler máscara: {mask_path}")

    H, W = target_shape_hw
    if m.shape[0] != H or m.shape[1] != W:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    return m


def read_pairs(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((Path(row["real_path"]), Path(row["fake_path"])))
    return rows


def load_lab(path: Path):
    rgb = read_rgb(path, IMG_SIZE) # mesma leitura + resize do treino
    if rgb is None:
        raise ValueError(f"Falha ao ler imagem: {path}")

    L, ab = rgb_to_lab(rgb)  # mesma normalização do treino
    return L, ab



@torch.no_grad()
def predict_ab(model, L01: np.ndarray):
    x = torch.from_numpy(L01.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    if DEVICE == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(x)
    else:
        pred = model(x)

    pred = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return pred



def error_map(pred_ab: np.ndarray, orig_ab: np.ndarray):
    # E = ||ab_hat - ab_orig||_2 por pixel
    diff = pred_ab - orig_ab
    E = np.sqrt(np.sum(diff * diff, axis=2))  # (H,W)
    return E


def save_heatmap(E: np.ndarray,
                 out_path: Path,
                 rgb_uint8: np.ndarray = None,
                 top_percentile: float = 92.0,
                 dilate_iters: int = 2,
                 keep_largest: bool = True,
                 draw_bbox: bool = True):
    hm = E.astype(np.float32)
    hm = np.maximum(hm, 0.0)

    # Contraste robusto
    p_low, p_high = np.percentile(hm, [2.0, 99.5])
    hm = np.clip(hm, p_low, p_high)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

    # Suavização (menos "pontinhos")
    hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=2.2, sigmaY=2.2)

    # Threshold menos agressivo para não ficar esparso
    thr = np.percentile(hm, top_percentile)
    mask = (hm >= thr).astype(np.uint8)  # 0/1

    # Dilatar para formar "mancha"
    if dilate_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iters)

    # Opcional: ficar só com a maior componente (uma zona principal)
    bbox = None
    if keep_largest:
        num, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num > 1:
            best_j = None
            best_score = -1e18

            # ignora fundo (idx=0)
            for j in range(1, num):
                x, y, w, h, area = stats[j]
                if area < 60:
                    continue  # ignora manchas minúsculas

                comp = (labels_cc == j)

                # energia = soma das intensidades do heatmap dentro da componente
                energy = float(hm[comp].sum())

                # bónus leve por área (estabilidade, mas não domina)
                score = energy + 0.001 * float(area)

                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j is not None:
                mask = (labels_cc == best_j).astype(np.uint8)
                x, y, w, h, area = stats[best_j]
                bbox = (x, y, w, h)

    # Aplicar máscara no heatmap para ficar limpo
    hm = hm * mask.astype(np.float32)

    # Color map
    hm_u8 = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    if rgb_uint8 is None:
        cv2.imwrite(str(out_path), hm_color)
        return

    img_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.60, hm_color, 0.40, 0)

    # Desenhar bounding box da principal região
    if draw_bbox and bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(str(out_path), overlay)


def diff_ab_map(ab_real: np.ndarray, ab_fake: np.ndarray):
    # delta = ||ab_fake - ab_real||_2 por pixel
    diff = ab_fake - ab_real
    D = np.sqrt(np.sum(diff * diff, axis=2))  # (H,W)
    return D


def roc_auc(scores, labels):
    # Implementação simples (sem sklearn) para não depender de nada.
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    order = np.argsort(-scores)  # desc
    labels = labels[order]

    P = np.sum(labels == 1)
    N = np.sum(labels == 0)
    if P == 0 or N == 0:
        return None, None, None

    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)
    tpr = tps / P
    fpr = fps / N

    # AUC trapezoidal
    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc


def main():

    # Para decidir automaticamente se o score tem de ser invertido
    comp_real_list = []
    comp_fake_list = []

    setup_logging("detect_and_xai")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

    pairs = read_pairs(PAIRS_CSV)
    print(f"[INFO] PARES: {len(pairs)} -> {PAIRS_CSV}")

    model = UNetSmall().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[INFO] Modelo carregado: {MODEL_PATH} | DEVICE={DEVICE}")

    rows_out = []
    scores = []
    labels = []

    # Para não explodir disco, só guardo heatmaps de alguns exemplos
    heatmaps_dir = OUT_DIR / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    for i, (real_p, fake_p) in enumerate(pairs, start=1):
        # --- REAL ---
        Lr, abr = load_lab(real_p)
        pred_r = predict_ab(model, Lr)
        Er = error_map(pred_r, abr)

        #METER EM COMENTARIO PARA JA DEPOIS VE-SE SE DESCOMENTO OU NAO
        mask_path_real = MASKS_DIR / f"{real_p.stem}_mask.png"
        mask_path_fake = MASKS_DIR / f"{fake_p.stem}_mask.png"

        mask_path = None
        if mask_path_fake.exists():
            mask_path = mask_path_fake
        elif mask_path_real.exists():
            mask_path = mask_path_real

        if mask_path is not None:
            m = load_mask(mask_path, Er.shape)
            masked_pixels = (m > 0)
        else:
            masked_pixels = None

        # --- FAKE ---
        Lf, abf = load_lab(fake_p)
        pred_f = predict_ab(model, Lf)
        Ef = error_map(pred_f, abf)

        # =========================
        # NOVOS SCORES (melhor recall)
        # =========================
        eps = 1e-8

        # -------------------------
        # Vetores sem máscara (global)
        # -------------------------
        Er_all = Er.reshape(-1)
        Ef_all = Ef.reshape(-1)

        # -------------------------
        # Vetores com máscara (ROI)
        # Se a máscara for pequena, cai para global para não instabilizar.
        # -------------------------
        min_pixels = 100

        if masked_pixels is None:
            Er_roi = Er_all
            Ef_roi = Ef_all
        else:
            Er_roi = Er[masked_pixels]
            Ef_roi = Ef[masked_pixels]
            if Er_roi.size < min_pixels or Ef_roi.size < min_pixels:
                Er_roi = Er_all
                Ef_roi = Ef_all

        # ---------- GLOBAL ----------
        P_HIGH = 85 #melhor -> 85

        p99_r = float(np.percentile(Er_all, P_HIGH))
        p99_f = float(np.percentile(Ef_all, P_HIGH))

        p95_r = float(np.percentile(Er_all, 95))
        p95_f = float(np.percentile(Ef_all, 95))

        k_all = max(1, int(0.01 * Er_all.size))
        top1_r = float(np.mean(np.partition(Er_all, -k_all)[-k_all:]))
        top1_f = float(np.mean(np.partition(Ef_all, -k_all)[-k_all:]))

        # ---------- MASK/ROI ----------
        p99_r_m = float(np.percentile(Er_roi, P_HIGH))
        p99_f_m = float(np.percentile(Ef_roi, P_HIGH))

        p95_r_m = float(np.percentile(Er_roi, 95))
        p95_f_m = float(np.percentile(Ef_roi, 95))

        k_roi = max(1, int(0.01 * Er_roi.size))
        top1_r_m = float(np.mean(np.partition(Er_roi, -k_roi)[-k_roi:]))
        top1_f_m = float(np.mean(np.partition(Ef_roi, -k_roi)[-k_roi:]))


        # -------------------------
        # SCORES POR IMAGEM (não constantes)
        # -------------------------
        score_p99_real = p99_r
        score_p99_fake = p99_f

        score_top1_real = top1_r
        score_top1_fake = top1_f

        # Score composto simples (melhora ranking/AUC)
        score_comp_real = score_p99_real + score_top1_real
        score_comp_fake = score_p99_fake + score_top1_fake

        comp_real_list.append(score_comp_real)
        comp_fake_list.append(score_comp_fake)


        # -------------------------
        # SCORES POR PAR (mantemos para análise)
        # -------------------------
        score_ratio_p99 = (p99_f + eps) / (p99_r + eps)
        score_delta_top1 = top1_f - top1_r


        score_ratio_p99_mask = (p99_f_m + eps) / (p99_r_m + eps)
        score_delta_top1_mask = top1_f_m - top1_r_m

        score_log_ratio_p99 = float(np.log(score_ratio_p99 + eps))
        score_log_ratio_p99_mask = float(np.log(score_ratio_p99_mask + eps))

        # --- NOVO 1: PEAK RATIO (mais estável que p99.99 sozinho) ---
        peak_f = (p99_f - p95_f)
        peak_r = (p99_r - p95_r)
        score_peak_ratio = float((peak_f + eps) / (peak_r + eps))

        peak_f_m = (p99_f_m - p95_f_m)
        peak_r_m = (p99_r_m - p95_r_m)
        score_peak_ratio_mask = float((peak_f_m + eps) / (peak_r_m + eps))

        # --- NOVO 2: SCORE MIX (combina log_ratio + delta_top1) ---
        LAMBDA_DELTA = 0.9 # melhor com 0.9
        score_mix = float(score_log_ratio_p99 + LAMBDA_DELTA * score_delta_top1)
        score_mix_mask = float(score_log_ratio_p99_mask + LAMBDA_DELTA * score_delta_top1_mask)


        # -------------------------
        # FUSÃO DE SCORES (por par) depois posso tirar
        # -------------------------
        # Normalização simples por par:
        # - ratio_p99 já é "normalizado" (divide pela dificuldade do real)
        # - delta_top1 pode ser positivo/negativo, damos-lhe um peso menor
        alpha = 1.0 #melhor com 1.5
        fused_score = score_ratio_p99 + alpha * score_delta_top1

        # Guardar tudo no CSV (uma linha por imagem)
        # rows_out.append([str(real_p), 0, score_p99_real, score_top1_real, score_comp_real, 1.0, 0.0])
        # rows_out.append([str(fake_p), 1, score_p99_fake, score_top1_fake, score_comp_fake, score_ratio_p99, score_delta_top1])

        # rows_out.append([str(real_p), 0, score_p99_real, score_top1_real, score_comp_real, 1.0, 0.0, 1.0])
        # rows_out.append(
        #     [str(fake_p), 1, score_p99_fake, score_top1_fake, score_comp_fake, score_ratio_p99, score_delta_top1,
        #     fused_score]
        # )

        # log-ratio por par (no real guardamos 0.0 só para manter formato de "uma linha por imagem") TESTAR
        # score_log_ratio_p99 = float(np.log((p99_f + eps) / (p99_r + eps)))
        #
        # rows_out.append([str(real_p), 0, score_p99_real, score_top1_real, score_comp_real, 1.0, 0.0, fused_score, 0.0])
        # rows_out.append(
        #     [str(fake_p), 1, score_p99_fake, score_top1_fake, score_comp_fake, score_ratio_p99, score_delta_top1,
        #      fused_score, score_log_ratio_p99])

        # rows_out.append([str(real_p), 0,
        #                  p99_r, top1_r, (p99_r + top1_r),
        #                  1.0, 0.0,
        #                  1.0, 0.0,
        #                  fused_score,
        #                  0.0, 0.0])
        #
        # rows_out.append([str(fake_p), 1,
        #                  p99_f, top1_f, (p99_f + top1_f),
        #                  score_ratio_p99, score_delta_top1,
        #                  score_ratio_p99_mask, score_delta_top1_mask,
        #                  fused_score,
        #                  score_log_ratio_p99, score_log_ratio_p99_mask])

        rows_out.append([str(real_p), 0,
                         p99_r, top1_r, (p99_r + top1_r),
                         1.0, 0.0,
                         1.0, 0.0,
                         fused_score,
                         0.0, 0.0,
                         1.0, 1.0,  # peak_ratio (real = 1.0 para manter formato)
                         0.0, 0.0])  # score_mix (real = 0.0)

        rows_out.append([str(fake_p), 1,
                         p99_f, top1_f, (p99_f + top1_f),
                         score_ratio_p99, score_delta_top1,
                         score_ratio_p99_mask, score_delta_top1_mask,
                         fused_score,
                         score_log_ratio_p99, score_log_ratio_p99_mask,
                         score_peak_ratio, score_peak_ratio_mask,
                         score_mix, score_mix_mask])




        # Para ROC/AUC rápida neste script, usamos o score_comp (melhor separação global)
        # scores.append(score_comp_real); labels.append(0)
        # scores.append(score_comp_fake); labels.append(1) #########Ao fazer com este dava 0.52

        # scores.append(1.0); labels.append(0)
        # scores.append(score_ratio_p99); labels.append(1) ############Ao fazer com este dava 0.65

        # ESTE FOI O QUE OBTIVE SEGUNDO MELHOR RESULTADO
        # scores.append(1.0); labels.append(0)
        # scores.append(fused_score); labels.append(1)

        # scores.append(1.0); labels.append(0)
        # scores.append(score_ratio_tail); labels.append(1)

        #Sem mascara 0.7574
        # scores.append(0.0); labels.append(0)
        # scores.append(score_log_ratio_p99); labels.append(1)

        # Sem mascara (usar score_mix)Melhor resultado!!!!!
        scores.append(0.0); labels.append(0)
        scores.append(score_mix); labels.append(1)

        # Guardar heatmaps só para alguns exemplos
        if i in {1, 10, 100, 500, 1000}:
            # heatmap que localiza a alteração (diferença direta real vs fake em ab)
            D = diff_ab_map(abr, abf)

            rgb_real = read_rgb(real_p, IMG_SIZE)
            rgb_fake = read_rgb(fake_p, IMG_SIZE)

            # # overlay em cima do fake para veres “onde mexeu”
            # save_heatmap(D, heatmaps_dir / f"diffab_fake_{i:05d}.png", rgb_fake, top_percentile=95.0)
            #
            # # (opcional) também podes ver em cima do real
            # save_heatmap(D, heatmaps_dir / f"diffab_real_{i:05d}.png", rgb_real, top_percentile=95.0)
            # save_heatmap(Er, heatmaps_dir / f"realerr_{i:05d}.png", rgb_real, top_percentile=95.0)
            # save_heatmap(Ef, heatmaps_dir / f"fakeerr_{i:05d}.png", rgb_fake, top_percentile=95.0)


            save_heatmap(D, heatmaps_dir / f"diffab_fake_{i:05d}.png", rgb_fake, top_percentile=90.0, dilate_iters=2,keep_largest=True, draw_bbox=True)
            save_heatmap(D, heatmaps_dir / f"diffab_real_{i:05d}.png", rgb_real, top_percentile=90.0, dilate_iters=2,keep_largest=True, draw_bbox=True)

            # Os erros do modelo (Er/Ef) podes deixar mais suaves também
            save_heatmap(Er, heatmaps_dir / f"realerr_{i:05d}.png", rgb_real, top_percentile=90.0, dilate_iters=1,keep_largest=True, draw_bbox=False)
            save_heatmap(Ef, heatmaps_dir / f"fakeerr_{i:05d}.png", rgb_fake, top_percentile=90.0, dilate_iters=1,keep_largest=True, draw_bbox=False)

        if i % 200 == 0:
            print(f"[OK] {i}/{len(pairs)} pares processados")

    # =========================
    # Ajustar orientação do score_comp (muito importante!)
    # Queremos: fakes com score MAIOR que reais.
    # =========================
    mean_real = float(np.mean(comp_real_list)) if len(comp_real_list) > 0 else 0.0
    mean_fake = float(np.mean(comp_fake_list)) if len(comp_fake_list) > 0 else 0.0
    print(f"[DEBUG] mean(score_comp) real={mean_real:.6f} | fake={mean_fake:.6f}")

    flip = (mean_fake < mean_real)
    # if flip:
    #     print("[DEBUG] A inverter score_comp (porque fake < real em média).")
    #     # rows_out tem: [path, label, score_p99, score_top1_mean, score_comp, score_ratio_p99, score_delta_top1_mean]
    #     # índice do score_comp = 4
    #     for row in rows_out:
    #         row[4] = -float(row[4])

        # inverter também o array usado na ROC/AUC (scores)
        #scores = [-float(v) for v in scores]


    # Guardar scores
    scores_csv = OUT_DIR / "scores.csv"
    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["path", "label",
             "score_p99", "score_top1_mean", "score_comp",
             "score_ratio_p99", "score_delta_top1_mean",
             "score_ratio_p99_mask", "score_delta_top1_mean_mask",
             "fused_score",
             "score_log_ratio_p99", "score_log_ratio_p99_mask",
             "score_peak_ratio", "score_peak_ratio_mask",
             "score_mix", "score_mix_mask"]
        )

        w.writerows(rows_out)

    # ROC/AUC
    fpr, tpr, auc = roc_auc(scores, labels)
    if auc is None:
        print("[WARN] Não consegui calcular ROC/AUC (labels insuficientes).")
    else:
        print(f"[RESULT] AUC = {auc:.4f}")

        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC (AUC={auc:.4f})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "roc_curve.png", dpi=200)
        plt.close()

    print(f"[DONE] Guardado: {scores_csv}")
    print(f"[DONE] Guardado: {OUT_DIR / 'roc_curve.png'} (se AUC calculada)")
    print(f"[DONE] Heatmaps: {heatmaps_dir}")


if __name__ == "__main__":
    main()
