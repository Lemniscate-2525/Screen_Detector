import json
import os

import numpy as np
import cv2

import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model_weights.json")

_weights_enum = MobileNet_V2_Weights.IMAGENET1K_V2
_model = mobilenet_v2(weights = _weights_enum)
_model.classifier = torch.nn.Identity()
_model.eval()
_preprocess = _weights_enum.transforms()

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def _load_model() -> dict:
    with open(WEIGHTS_PATH, "r") as f:
        return json.load(f)

def get_embedding(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = _preprocess(torch.from_numpy(image_rgb).permute(2, 0, 1))
    with torch.no_grad():
        emb = _model(tensor.unsqueeze(0))
    return emb.squeeze(0).numpy()

def predict_proba(image: np.ndarray) -> tuple[float, dict]:
    emb = get_embedding(image)
    model = _load_model()

    mean = np.array(model["scaler_mean"])
    scale = np.array(model["scaler_scale"])
    x = (emb - mean) / scale

    w = np.array(model["weights"])
    b = model["bias"]

    z = float(np.dot(w, x) + b)
    proba = _sigmoid(z)

    return proba, {"embedding_logit": z}

def predict_proba_from_path(image_path: str) -> tuple[float, dict]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    return predict_proba(image)
