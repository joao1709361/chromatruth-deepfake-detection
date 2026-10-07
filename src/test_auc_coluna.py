import csv
import numpy as np
from pathlib import Path

SCORES_CSV = Path("outputs/detection/scores.csv")

def roc_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    order = np.argsort(-scores)
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

def load_col(col):
    y, s = [], []
    with open(SCORES_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            y.append(int(row["label"]))
            s.append(float(row[col]))
    return np.array(y), np.array(s)

def main():
    cols = ["score_p99", "score_top1_mean", "score_comp", "score_ratio_p99", "score_delta_top1_mean"]
    for c in cols:
        y, s = load_col(c)
        auc = roc_auc(s, y)
        print(f"[AUC] {c}: {auc:.4f}")

if __name__ == "__main__":
    main()