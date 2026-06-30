import argparse
import sys

import time

from predictor import predict_proba_from_path

def main() :
    parser = argparse.ArgumentParser(
        description = "Predict whether an image is a real photo or a photo of a screen."
    )
    parser.add_argument("image_path", type = str, help = "Path to the input image")

    parser.add_argument(
        "--verbose", "-v",
        action = "store_true",
        help = "Print per-feature scores and timing info in addition to the probability",
    )
    args = parser.parse_args()

    start = time.time()

    try:
        proba, feature_scores = predict_proba_from_path(args.image_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    elapsed_ms = (time.time() - start) * 1000

    if args.verbose:
        label = "SCREEN" if proba >= 0.5 else "REAL"

        print(f"Prediction: {label}")
        print(f"Confidence: {proba:.4f}")
        print()
        print("Feature Scores:")

        for name, score in feature_scores.items():
            print(f"  {name.capitalize()}: {score:.4f}")
        print()
        print(f"Latency: {elapsed_ms:.1f} ms")
    else:
        print(f"{proba:.4f}")

if __name__ == "__main__":
    main()
