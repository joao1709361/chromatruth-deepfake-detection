import csv
from pathlib import Path
import numpy as np

SCORES_CSV = Path("outputs/detection/scores.csv")
SCORE_COL = "score_mix"

def load_scores(path: Path):
    y = []
    s = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            y.append(int(row["label"]))
            s.append(float(row[SCORE_COL]))
    return np.array(y, dtype=np.int32), np.array(s, dtype=np.float64)

def confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def roc_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    order = np.argsort(-scores)  # desc
    labels = labels[order]

    P = np.sum(labels == 1)
    N = np.sum(labels == 0)
    if P == 0 or N == 0:
        return None

    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)
    tpr = tps / P
    fpr = fps / N

    auc = np.trapezoid(tpr, fpr)
    return float(auc)


def metrics(tp, tn, fp, fn):
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    beta = 2.0
    f2 = (1 + beta**2) * prec * rec / max((beta**2) * prec + rec, 1e-12)
    return acc, prec, rec, f1, f2


def best_threshold(y, s):
    # thresholds candidatos: valores únicos (ordenados)
    thresh = np.unique(s)
    best = None

    for t in thresh:
        y_pred = (s >= t).astype(np.int32)
        tp, tn, fp, fn = confusion(y, y_pred)
        acc, prec, rec, f1, f2 = metrics(tp, tn, fp, fn)
        score = f1

        cand = {"t": float(t), "score": float(score), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "acc": acc, "prec": prec, "rec": rec, "f1": f1, "f2": f2}

        if best is None:
            best = cand
        else:
            # 1) melhor score (F1 ou F2)
            if cand["score"] > best["score"] + 1e-12:
                best = cand
            # 2) empate no score: preferir MENOS FP
            elif abs(cand["score"] - best["score"]) <= 1e-12 and cand["fp"] < best["fp"]:
                best = cand
            # 3) novo empate: preferir threshold mais alto (mais conservador)
            elif abs(cand["score"] - best["score"]) <= 1e-12 and cand["fp"] == best["fp"] and cand["t"] > best["t"]:
                best = cand

    return best

def main():
    y, s = load_scores(SCORES_CSV)

    auc_all = roc_auc(s, y)
    if auc_all is not None:
        print(f"[AUC] Global ({SCORE_COL}): {auc_all:.4f}")

    # -------------------------------------------------
    # Split por "grupo/par": mantém real+fake juntos
    # Grupo = stem do filename do campo "path" no CSV
    # -------------------------------------------------
    paths = []
    with open(SCORES_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            paths.append(row["path"])

    groups = np.array([Path(p).stem for p in paths], dtype=object)

    uniq = np.unique(groups)

    rng = np.random.default_rng(123)
    rng.shuffle(uniq)

    cut = int(0.8 * len(uniq))
    g_cal = set(uniq[:cut])
    g_eval = set(uniq[cut:])

    idx_cal = np.array([i for i, g in enumerate(groups) if g in g_cal], dtype=np.int64)
    idx_eval = np.array([i for i, g in enumerate(groups) if g in g_eval], dtype=np.int64)

    print(f"[SPLIT] cal: {len(idx_cal)} | eval: {len(idx_eval)}")
    print(f"[SPLIT] cal positives={int(np.sum(y[idx_cal] == 1))} negatives={int(np.sum(y[idx_cal] == 0))}")
    print(f"[SPLIT] eval positives={int(np.sum(y[idx_eval] == 1))} negatives={int(np.sum(y[idx_eval] == 0))}")

    auc_eval = roc_auc(s[idx_eval], y[idx_eval])
    if auc_eval is not None:
        print(f"[AUC] Eval  ({SCORE_COL}): {auc_eval:.4f}")


    best = best_threshold(y[idx_cal], s[idx_cal])
    t = best["t"]



    y_pred_eval = (s[idx_eval] >= t).astype(np.int32)
    tp, tn, fp, fn = confusion(y[idx_eval], y_pred_eval)
    acc, prec, rec, f1, f2 = metrics(tp, tn, fp, fn)

    print(f"[CAL] best_threshold (F1): {t:.6f} | F1_cal={best['f1']:.4f} | F2_cal={best['f2']:.4f}")
    print("[EVAL] Confusion: TP={} TN={} FP={} FN={}".format(tp, tn, fp, fn))
    print("[EVAL] Acc={:.4f} Prec={:.4f} Rec={:.4f} F1={:.4f} F2={:.4f}".format(acc, prec, rec, f1, f2))


if __name__ == "__main__":
    main()
