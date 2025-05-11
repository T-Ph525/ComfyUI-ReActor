from transformers import pipeline
from PIL import Image
import logging
import os

# Optional: custom logger setup
logger = logging.getLogger("AgeFilter")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Reduce noisy transformer logs
logging.getLogger("transformers").setLevel(logging.ERROR)

# Path to the local age classification model
MODEL_PATH = "/root/comfy/ComfyUI/models/age_classifier"

# Minimum confidence to block underage content
AGE_SCORE_THRESHOLD = 0.85


def ensure_model_exists(model_path: str):
    """Check that the required model files are present."""
    required_files = ["config.json", "model.safetensors", "preprocessor_config.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            raise FileNotFoundError(f"Missing required model file: {file} in {model_path}")


def nsfw_image(img_path: str, model_path: str) -> bool:
    """
    Detect underage subjects in an image using a local classification model.
    Raises PermissionError if confident the subject is a child/teen.
    """
    ensure_model_exists(model_path)
    classifier = pipeline("image-classification", model=model_path)

    with Image.open(img_path) as img:
        results = classifier(img)
        top_result = results[0]
        label = top_result["label"].lower()
        score = top_result["score"]

        if label in ["child", "teen"] and score > AGE_SCORE_THRESHOLD:
            logger.info(f"🚫 Underage content detected: {label} ({score:.2f}) — blocking.")
            raise PermissionError("403 Forbidden: Underage content detected")

    logger.info("✅ Image passed age check.")
    return False


# Example usage
if __name__ == "__main__":
    test_image_path = "/path/to/your/image.jpg"  # Replace with actual image path
    try:
        nsfw_image(test_image_path, MODEL_PATH)
    except PermissionError as e:
        print(str(e))
