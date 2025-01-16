# File: src/utils/preprocessing.py

import numpy as np
from typing import Tuple
import cv2

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image values, matching the network's preprocessing.
    
    Args:
        image: Input image array
        
    Returns:
        Normalized image array
    """
    # Match the exact normalization from ModernPoseNetwork
    image = image.astype(np.float32)
    image = ((image / 255.0) - 0.5) * 2.0
    return image

def prepare_image_for_network(image: np.ndarray, target_size: int = 299) -> np.ndarray:
    """Prepare an image for input to the pose network.
    
    Args:
        image: Input image
        target_size: Target size (default 299 for InceptionResNet)
        
    Returns:
        Prepared image
    """
    # Resize to network input size
    image = cv2.resize(image, (target_size, target_size))
    
    # Convert to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3 and image.dtype == np.uint8:  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # Normalize
    image = normalize_image(image)
    
    return image

def prepare_batch_for_network(images: List[np.ndarray], target_size: int = 299) -> np.ndarray:
    """Prepare a batch of images for the network.
    
    Args:
        images: List of input images
        target_size: Target size for each image
        
    Returns:
        Batch of prepared images (N x H x W x 3)
    """
    prepared_images = []
    for image in images:
        prepared = prepare_image_for_network(image, target_size)
        prepared_images.append(prepared)
    
    return np.stack(prepared_images)