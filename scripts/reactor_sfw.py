from transformers import pipeline
from PIL import Image
import logging
import os
from scripts.reactor_logger import logger

logging.getLogger("transformers").setLevel(logging.ERROR)

AGE_SCORE_THRESHOLD = 0.85  # Flag if model is confident subject is underage

def ensure_model_exists(model_path: str):
    """Checks if local age model exists. Does NOT attempt download."""
    required_files = ["config.json", "model.safetensors", "preprocessor_config.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            raise FileNotFoundError(f"Missing required age model file: {file} in {model_path}")

def nsfw_image(img_path: str, model_path: str) -> bool:
    """
    Compatibility function name. Now does age-only filtering using local model.
    Raises PermissionError(403) if subject is underage.
    """
    ensure_model_exists(model_path)
    classifier = pipeline("image-classification", model=model_path)

    with Image.open(img_path) as img:
        results = classifier(img)
        top_result = results[0]
        label = top_result["label"].lower()
        score = top_result["score"]

        if label in ["child", "teen"] and score > AGE_SCORE_THRESHOLD:
            logger.status(f"🚫 Underage content detected: {label} ({score:.2f}) — blocking.")
            raise PermissionError("403 Forbidden: Underage content detected")

    return False

def ensure_nsfw_model(nsfwdet_model_path):
    """
    Deprecated function kept for compatibility.
    """
    return
