from PIL import Image
import io
import logging
from scripts.reactor_logger import logger

# Completely disable model check and classification
def nsfw_image(img_data, model_path: str):
    logger.status("NSFW check bypassed (always returns False).")
    return False
