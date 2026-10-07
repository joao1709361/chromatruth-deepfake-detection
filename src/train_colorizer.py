import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from logger_utils import setup_logging
from torch.cuda.amp import autocast, GradScaler

# -------------------------
# Configs rápidas (podes ajustar depois)
# -------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
REAL_TRAIN_DIR = DATA_DIR / "real_train"
REAL_VAL_DIR = DATA_DIR / "real_test"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "colorizer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ACC_THRESHOLD = 0.1

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
NUM_WORKERS = 4
PERSISTENT_WORKERS = True
PIN_MEMORY = True
PREFETCH_FACTOR = 2
MAX_TRAIN_IMAGES = 17000  # para testar rápido agora (depois sobes)
MAX_VAL_IMAGES = 9000


# -------------------------
# Utils LAB
# -------------------------
def read_rgb(path: Path, size: int = 256):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def rgb_to_lab(rgb_uint8: np.ndarray):
    # OpenCV espera RGB em uint8 OK
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    # L em [0,255], a/b em [0,255] (OpenCV shift)
    # Normalizar para treino estável:
    L = lab[..., 0:1] / 255.0                     # [0,1]
    ab = (lab[..., 1:3] - 128.0) / 128.0          # ~[-1,1]
    return L, ab


# def list_images(folder: Path):
#     exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
#     files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
#     files.sort()
#     return files

def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files




# -------------------------
# Dataset
# -------------------------
class RealFacesLAB(Dataset):
    def __init__(self, folder: Path, max_images: int | None = None, img_size: int = 256):
        self.files = list_images(folder)
        if max_images is not None:
            self.files = self.files[:max_images]
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        rgb = read_rgb(path, self.img_size)
        if rgb is None:
            # fallback: devolve outro
            return self.__getitem__((idx + 1) % len(self.files))

        L, ab = rgb_to_lab(rgb)

        # (H,W,C) -> (C,H,W)
        L = torch.from_numpy(L).permute(2, 0, 1).contiguous()     # (1,H,W)
        ab = torch.from_numpy(ab).permute(2, 0, 1).contiguous()   # (2,H,W)
        return L, ab


# -------------------------
# Modelo (U-Net simples e leve)
# Input: 1 canal (L)
# Output: 2 canais (ab)
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = ConvBlock(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.out = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        m = self.mid(self.pool3(e3))

        d3 = self.up3(m)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.tanh(self.out(d1))  # ab ~ [-1,1]


# -------------------------
# Train / Val
# -------------------------
# def run_epoch(model, loader, optimizer, criterion, device, train: bool, scaler=None, use_amp: bool = False):
#     n = 0
#     model.train(train)
#     total_loss = 0.0
#     total_pixels = 0
#     correct_pixels = 0
#
#     for i, (L, ab) in enumerate(loader):
#         if i == 0:
#             print("[DEBUG] primeira batch chegou")
#
#         L = L.to(device, non_blocking=True)
#         ab = ab.to(device, non_blocking=True)
#
#         if train:
#             optimizer.zero_grad(set_to_none=True)
#
#         with autocast(enabled=use_amp):
#             pred = model(L)
#             loss = criterion(pred, ab)
#
#         if train:
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#         bs = L.size(0)
#         total_loss += loss.item() * bs
#         n += bs
#
#         error = torch.abs(pred - ab)          # (B,2,H,W)
#         error = torch.mean(error, dim=1)      # (B,H,W)
#         correct_pixels += (error < ACC_THRESHOLD).sum().item()
#         total_pixels += error.numel()
#
#     avg_loss = total_loss / max(n, 1)
#     acc = correct_pixels / max(total_pixels, 1)
#     return avg_loss, acc


def run_epoch(model, loader, optimizer, criterion, device, train: bool, scaler=None, use_amp: bool = False):
    n = 0
    model.train(train)
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    for i, (L, ab) in enumerate(loader):
        if i == 0:
            print("[DEBUG] primeira batch chegou")

        L = L.to(device, non_blocking=True)
        ab = ab.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            pred = model(L)
            loss = criterion(pred, ab)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs = L.size(0)
        total_loss += loss.item() * bs
        n += bs

        error = torch.abs(pred - ab)     # (B,2,H,W)
        error = torch.mean(error, dim=1) # (B,H,W)
        correct_pixels += (error < ACC_THRESHOLD).sum().item()
        total_pixels += error.numel()

    avg_loss = total_loss / max(n, 1)
    acc = correct_pixels / max(total_pixels, 1)
    return avg_loss, acc




def main():
    print("[DEBUG] A iniciar treino (primeira batch pode ser mais lenta)...")

    start = time.time()
    print("[DEBUG] antes do setup_logging")
    setup_logging("train_colorizer")
    print("[DEBUG] 1) a criar datasets...")

    if not REAL_TRAIN_DIR.exists():
        raise FileNotFoundError(f"Não existe: {REAL_TRAIN_DIR}")
    if not REAL_VAL_DIR.exists():
        raise FileNotFoundError(f"Não existe: {REAL_VAL_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True

    use_amp = (device == "cuda")
    scaler = GradScaler(enabled=use_amp)

    model = UNetSmall().to(device)

    criterion = nn.SmoothL1Loss(beta=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-5,
        verbose=True,
    )

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Train dir: {REAL_TRAIN_DIR}")
    print(f"[INFO] Val dir:   {REAL_VAL_DIR}")

    train_ds = RealFacesLAB(REAL_TRAIN_DIR, max_images=MAX_TRAIN_IMAGES, img_size=IMG_SIZE)
    val_ds = RealFacesLAB(REAL_VAL_DIR, max_images=MAX_VAL_IMAGES, img_size=IMG_SIZE)

    print(f"[DEBUG] 2 train_ds size = {len(train_ds)}")
    print(f"[DEBUG] 3 val_ds size   = {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )

    print("[DEBUG] 4) dataloaders criados")

    # model = UNetSmall().to(device)
    # criterion = nn.L1Loss()  # no PDF é permitido L1 ou MSE :contentReference[oaicite:1]{index=1}
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_path = OUTPUT_DIR / "best_colorizer.pt"



    # -------------------------
    # Early Stopping
    # -------------------------
    PATIENCE = 10
    MIN_DELTA = 1e-4
    epochs_no_improve = 0

    print("[DEBUG] 5) a iniciar loop de treino")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, optimizer, criterion, device,
            train=True, scaler=scaler, use_amp=use_amp
        )

        va_loss, va_acc = run_epoch(
            model, val_loader, optimizer, criterion, device,
            train=False, scaler=scaler, use_amp=use_amp
        )



        print(
            f"[EPOCH {epoch}/{EPOCHS}] "
            f"train_loss={tr_loss:.5f} | train_acc={tr_acc * 100:.2f}% || "
            f"val_loss={va_loss:.5f} | val_acc={va_acc * 100:.2f}%"
        )

        scheduler.step(va_loss)

        # Melhorou?
        if best_val - va_loss > MIN_DELTA:
            best_val = va_loss

            epochs_no_improve = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "img_size": IMG_SIZE,
                },
                best_path,
            )
            print(f"[OK] Novo melhor modelo guardado: {best_path} (val_loss={best_val:.5f})")

        else:
            epochs_no_improve += 1
            print(f"[INFO] EarlyStopping: {epochs_no_improve}/{PATIENCE} sem melhoria")

            if epochs_no_improve >= PATIENCE:
                print("[STOP] Early stopping ativado")
                break

    elapsed = (time.time() - start) / 60.0
    print(f"[DONE] Treino concluído.")
    print(f"[TIME] Tempo total: {elapsed:.2f} minutos")


if __name__ == "__main__":
    main()
