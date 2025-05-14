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
MODEL_PATH = "/comfy/models/age-classifier"

def ensure_nsfw_model(model_path: str):
    if not os.path.exists(nsfwdet_model_path):
    os.makedirs(nsfwdet_model_path)
    nd_urls = [
        "https://huggingface.co/nateraw/vit-age-classifier/resolve/main/model.safetensors",
        "https://huggingface.co/nateraw/vit-age-classifier/resolve/main/preprocessor_config.json",
        "https://huggingface.co/nateraw/vit-age-classifier/resolve/main/config.json",
    ]
    for model_url in nd_urls:
        model_name = os.path.basename(model_url)
        model_path = os.path.join(nsfwdet_model_path, model_name)
        download(model_url, model_path, model_name)


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
