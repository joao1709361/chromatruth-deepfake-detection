import csv
from pathlib import Path

from config import REAL_TEST_DIR, FAKE_TEST_DIR, PAIRS_CSV, IMG_EXTS
from logger_utils import setup_logging


def list_images(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def main():
    setup_logging("build_pairs")


    real_files = list_images(REAL_TEST_DIR)

    # Limitar aos mesmos fakes gerados
    MAX_IMAGES = 6000
    real_files = real_files[:MAX_IMAGES]

    print(f"[INFO] A criar pares para {len(real_files)} imagens")

    if not real_files:
        raise ValueError(f"Não encontrei imagens em {REAL_TEST_DIR}.")

    missing = []
    rows = []

    for r in real_files:
        f = FAKE_TEST_DIR / r.name
        if not f.exists():
            missing.append(r.name)
            continue
        rows.append((str(r), str(f)))

    if missing:
        print("[ERRO] Estão a faltar fakes para estes ficheiros (mesmo nome!):")
        for name in missing[:30]:
            print(" -", name)
        if len(missing) > 30:
            print(f" ... e mais {len(missing) - 30}")
        raise ValueError("Gera os fakes em data/fake_test com o MESMO nome dos reais em data/real_test.")

    with open(PAIRS_CSV, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["real_path", "fake_path"])
        w.writerows(rows)

    print(f"[OK] pairs.csv criado com {len(rows)} pares -> {PAIRS_CSV}")


if __name__ == "__main__":
    main()
