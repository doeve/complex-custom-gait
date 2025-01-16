# File: src/utils/__init__.py

from .visualization import (
    visualize_pose_estimation,
    plot_gait_signatures,
    print_joint_confidences,
    JOINT_NAMES,
    JOINT_COLORS
)

from .preprocessing import (
    normalize_image,
    prepare_image_for_network,
    prepare_batch_for_network
)

from .metrics import (
    compute_similarity_matrix,
    cosine_similarity,
    compute_identification_metrics
)

__all__ = [
    # Visualization
    'visualize_pose_estimation',
    'plot_gait_signatures',
    'print_joint_confidences',
    'JOINT_NAMES',
    'JOINT_COLORS',
    
    # Preprocessing
    'normalize_image',
    'prepare_image_for_network',
    'prepare_batch_for_network',
    
    # Metrics
    'compute_similarity_matrix',
    'cosine_similarity',
    'compute_identification_metrics'
]