import glob
import os
import time

from predictor import predict_proba_from_path

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

def main() :

    real_paths = glob.glob(os.path.join(DATASET_DIR, "real", "*"))
    screen_paths = glob.glob(os.path.join(DATASET_DIR, "screen", "*"))

    tp = fp = tn = fn = 0
    latencies = []

    for p in real_paths :

        start = time.time()
        proba, _ = predict_proba_from_path(p)
        latencies.append((time.time() - start) * 1000)
        pred = 1 if proba >= 0.5 else 0

        if pred == 0:
            tn += 1
        else:
            fp += 1

    for p in screen_paths:

        start = time.time()
        proba, _ = predict_proba_from_path(p)
        latencies.append((time.time() - start) * 1000)
        pred = 1 if proba >= 0.5 else 0

        if pred == 1:
            tp += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0

    print(f"Total images : {total}")
    print(f"Accuracy (on training data, NOT honest estimate) : {accuracy:.4f}")
    print()

    print("Confusion Matrix :")
    print(f"  True Real,   Pred Real:   {tn}")
    print(f"  True Real,   Pred Screen: {fp}  <- False Positives")
    print(f"  True Screen, Pred Screen: {tp}")
    print(f"  True Screen, Pred Real:   {fn}  <- False Negatives")

    print()
    print(f"Avg latency: {sum(latencies)/len(latencies):.2f} ms")
    print(f"Max latency: {max(latencies):.2f} ms")

if __name__ == "__main__":
    main()
