# File: src/utils/metrics.py

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import confusion_matrix, classification_report

def compute_similarity_matrix(signatures: List[np.ndarray]) -> np.ndarray:
    """Compute similarity matrix between all pairs of signatures.
    
    Args:
        signatures: List of signature vectors
        
    Returns:
        Similarity matrix
    """
    n = len(signatures)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            similarity = cosine_similarity(signatures[i], signatures[j])
            similarity_matrix[i, j] = similarity
            
    return similarity_matrix

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Similarity score between 0 and 1
    """
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Distance value
    """
    return np.linalg.norm(v1 - v2)

def compute_identification_metrics(true_labels: List[int], 
                                predicted_labels: List[int],
                                label_names: Optional[Dict[int, str]] = None) -> Dict:
    """Compute identification metrics.
    
    Args:
        true_labels: List of true user IDs
        predicted_labels: List of predicted user IDs
        label_names: Optional dictionary mapping user IDs to names
        
    Returns:
        Dictionary containing various metrics
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Compute classification report
    if label_names:
        target_names = [label_names[id] for id in sorted(set(true_labels))]
    else:
        target_names = [str(id) for id in sorted(set(true_labels))]
        
    report = classification_report(true_labels, predicted_labels,
                                 target_names=target_names,
                                 output_dict=True)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': accuracy
    }

def compute_equal_error_rate(similarities: List[float], 
                           genuine: List[bool]) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) for verification.
    
    Args:
        similarities: List of similarity scores
        genuine: List of boolean flags (True for genuine pairs)
        
    Returns:
        Tuple of (EER, threshold at EER)
    """
    far = []  # False Accept Rate
    frr = []  # False Reject Rate
    thresholds = np.linspace(0, 1, 100)
    
    for threshold in thresholds:
        predictions = np.array(similarities) >= threshold
        
        # Compute FAR and FRR
        far_value = np.sum((~np.array(genuine)) & predictions) / np.sum(~np.array(genuine))
        frr_value = np.sum(np.array(genuine) & (~predictions)) / np.sum(np.array(genuine))
        
        far.append(far_value)
        frr.append(frr_value)
    
    # Find the intersection point
    far = np.array(far)
    frr = np.array(frr)
    eer_threshold_idx = np.argmin(np.abs(far - frr))
    eer = (far[eer_threshold_idx] + frr[eer_threshold_idx]) / 2
    
    return eer, thresholds[eer_threshold_idx]