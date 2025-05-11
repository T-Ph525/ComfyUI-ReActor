from transformers import pipeline
from PIL import Image
import logging
import os

# Optional: custom logger setup
logger = logging.getLogger("AgeFilter")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Silence transformer logs
logging.getLogger("transformers").setLevel(logging.ERROR)

# Constants
AGE_SCORE_THRESHOLD = 0.85
MODEL_PATH = "/root/comfy/ComfyUI/models/age_classifier"


def ensure_model_exists(model_path: str):
    """
    Checks if the required age classification model files exist in the local path.
    """
    required_files = ["config.json", "model.safetensors", "preprocessor_config.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            raise FileNotFoundError(f"Missing required model file: {file} in {model_path}")


def ensure_nsfw_model(model_path: str):
    """
    Deprecated: Retained for backward compatibility. No action taken.
    """
    logger.info("⚠️ ensure_nsfw_model() is deprecated and no longer needed.")
    return None


def nsfw_image(img_path: str, model_path: str = MODEL_PATH) -> bool:
    """
    Uses a local age classification model to block images with children/teens.
    Raises PermissionError if subject is underage.
    
    Parameters:
        img_path (str): Path to the image to check.
        model_path (str): Local path to the age classification model.
        
    Returns:
        bool: False if image passes, otherwise raises PermissionError.
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
    test_image_path = "/path/to/your/image.jpg"  # Replace with real image path
    try:
        nsfw_image(test_image_path)
    except PermissionError as e:
        print(str(e))
