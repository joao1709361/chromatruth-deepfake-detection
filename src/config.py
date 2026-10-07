from pathlib import Path


# Nome da experiência (muda isto sempre que quiseres guardar uma nova execução)
EXPERIMENT_NAME = "exp_inpainting_v1"


# Raiz do projeto (ajusta se precisares)
PROJECT_ROOT = Path(__file__).resolve().parent

# Pasta de dados
DATA_DIR = PROJECT_ROOT / "data"

# Pastas (Fase 1)
REAL_ALL_DIR = DATA_DIR / "real_all"         # aqui metes todas as imagens reais (ex.: FFHQ)
REAL_TRAIN_DIR = DATA_DIR / "real_train"     # treino do colorizador (apenas reais)
REAL_TEST_DIR = DATA_DIR / "real_test"       # reais para teste (pares)
FAKE_TEST_DIR = DATA_DIR / "fake_test"       # fakes correspondentes a real_test

# Ficheiros
PAIRS_CSV = DATA_DIR / "pairs.csv"
SPLITS_JSON = DATA_DIR / "splits.json"

# Extensões permitidas
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Tamanhos (vamos usar mais tarde)
IMG_SIZE = 256  # mantemos 256 por enquanto (bom equilíbrio)
