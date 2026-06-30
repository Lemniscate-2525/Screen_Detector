import glob
import json
import os

import numpy as np
import cv2

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

from predictor import get_embedding

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model_weights.json")

def load_dataset():
    real_paths = glob.glob(os.path.join(DATASET_DIR, "real", "*"))
    screen_paths = glob.glob(os.path.join(DATASET_DIR, "screen", "*"))
    X, y = [], []

    for p in real_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"Skipping unreadable file: {p}")
            continue
        X.append(get_embedding(img))
        y.append(0)

    for p in screen_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"Skipping unreadable file: {p}")
            continue
        X.append(get_embedding(img))
        y.append(1)

    return np.array(X), np.array(y)


def main():
    print("Loading Dataset and Extracting Embeddings")

    X, y = load_dataset()

    print(f"Loaded {len(y)} images ({(y == 0).sum()} real, {(y == 1).sum()} screen)")
    print(f"Embedding shape: {X.shape}")

    if len(y) < 10:
        print("WARNING : Tiny Dataset, Accuracy will be Noisy.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C = 0.1, max_iter = 2000)

    n_splits = min(5, np.bincount(y).min())
    n_splits = max(n_splits, 2)

    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=skf)

    print(f"\nCross-validated accuracy ({n_splits}-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Per-fold scores: {np.round(cv_scores, 4)}")

    clf.fit(X_scaled, y)

    weights = clf.coef_[0].tolist()
    bias = float(clf.intercept_[0])

    model = {
        "weights": weights,
        "bias": bias,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(model, f)

    print(f"\nSaved final model weights to {WEIGHTS_PATH}")

if __name__ == "__main__":
    main()
