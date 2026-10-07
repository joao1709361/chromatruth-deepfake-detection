import json
import random
import shutil
from pathlib import Path

from config import REAL_ALL_DIR, REAL_TRAIN_DIR, REAL_TEST_DIR, IMG_EXTS, SPLITS_JSON
from logger_utils import setup_logging



def list_images(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    setup_logging("make_splits")

    seed = 42

    train_ratio = 0.8
    test_ratio = 0.2

    assert abs(train_ratio + test_ratio - 1.0) < 1e-6, "As percentagens têm de somar 1.0"

    safe_mkdir(REAL_TRAIN_DIR)
    safe_mkdir(REAL_TEST_DIR)

    images = list_images(REAL_ALL_DIR)
    n_total = len(images)

    if n_total == 0:
        raise ValueError(f"Não há imagens em {REAL_ALL_DIR}")

    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train

    random.seed(seed)
    random.shuffle(images)

    train_imgs = images[:n_train]
    test_imgs = images[n_train:]

    for p in train_imgs:
        shutil.copy2(p, REAL_TRAIN_DIR / p.name)

    for p in test_imgs:
        shutil.copy2(p, REAL_TEST_DIR / p.name)

    splits = {
        "seed": seed,
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "n_total": n_total,
        "n_train": n_train,
        "n_test": n_test,
        "train_files": [p.name for p in train_imgs],
        "test_files": [p.name for p in test_imgs],
    }

    with open(SPLITS_JSON, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"[OK] Total: {n_total} imagens")
    print(f"[OK] Train: {n_train} imagens ({train_ratio*100:.0f}%) -> {REAL_TRAIN_DIR}")
    print(f"[OK] Test:  {n_test} imagens ({test_ratio*100:.0f}%) -> {REAL_TEST_DIR}")



if __name__ == "__main__":
    main()
