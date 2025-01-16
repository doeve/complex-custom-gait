# File: src/utils/visualization.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

JOINT_NAMES = [
    'right ankle',
    'right knee',
    'right hip',
    'left hip',
    'left knee',
    'left ankle',
    'pelvis',
    'thorax',
    'upper neck',
    'head top',
    'right wrist',
    'right elbow',
    'right shoulder',
    'left shoulder',
    'left elbow',
    'left wrist'
]

# Match the original color scheme
JOINT_COLORS = ['r', 'r', 'b', 'm', 'm', 'y', 'g', 'g', 'b', 'c', 'r', 'r', 'b', 'm', 'm', 'c']

def visualize_pose_estimation(frame: np.ndarray, 
                            y: np.ndarray, 
                            x: np.ndarray, 
                            confidence: np.ndarray,
                            save_path: Optional[str] = None) -> np.ndarray:
    """Visualize pose estimation on a frame, matching original implementation.
    
    Args:
        frame: Input frame (H x W x 3)
        y: Y coordinates of joints (16,)
        x: X coordinates of joints (16,)
        confidence: Confidence values for each joint (16,)
        save_path: Optional path to save visualization
        
    Returns:
        Visualized frame
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(frame)
    
    # Draw connections between joints, matching original implementation
    for i in range(16):
        if i < 15 and i not in {5, 9}:
            if confidence[i] > 0.5 and confidence[i + 1] > 0.5:
                plt.plot([x[i], x[i + 1]], 
                        [y[i], y[i + 1]], 
                        color=JOINT_COLORS[i], 
                        linewidth=5)
    
    # Draw joint points
    for i in range(16):
        if confidence[i] > 0.5:
            plt.plot(x[i], y[i], 'o', color=JOINT_COLORS[i])
            
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        # Load and return the saved image
        return cv2.imread(save_path)
    else:
        # Convert plot to image array
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return image

def plot_gait_signatures(signatures: List[np.ndarray], 
                        labels: List[str],
                        save_path: Optional[str] = None):
    """Plot multiple gait signatures for comparison.
    
    Args:
        signatures: List of gait signature vectors (each from GaitNetwork)
        labels: List of labels for each signature
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for signature, label in zip(signatures, labels):
        # Normalize signature for better visualization
        normalized_sig = (signature - signature.mean()) / signature.std()
        plt.plot(normalized_sig, label=label, alpha=0.7)
    
    plt.title('Normalized Gait Signature Comparison')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def print_joint_confidences(confidence: np.ndarray):
    """Print confidence values for each joint, matching original format.
    
    Args:
        confidence: Confidence values for each joint (16,)
    """
    for i, (name, conf) in enumerate(zip(JOINT_NAMES, confidence)):
        print(f'{name}: {conf*100:.2f}%')