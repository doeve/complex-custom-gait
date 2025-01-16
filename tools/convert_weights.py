# File: tools/convert_weights.py

import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
from loguru import logger

# Force CPU usage
tf.config.set_visible_devices([], 'GPU')

# Import our network implementations
import sys
sys.path.append('.')  # Add project root to path
from src.networks.pose_network import ModernPoseNetwork
from src.networks.gait_network import ModernGaitNetwork

def load_tf1_checkpoint(ckpt_path):
    """Load weights from a TensorFlow 1.x checkpoint."""
    reader = tf.train.load_checkpoint(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    weights_dict = {}
    
    for key in var_to_shape_map:
        try:
            tensor = reader.get_tensor(key)
            weights_dict[key] = tensor
        except Exception as e:
            logger.warning(f"Could not load tensor {key}: {e}")
    
    return weights_dict

def convert_pose_network_weights(ckpt_path, output_path):
    """Convert pose network weights from TF1 to Keras format."""
    logger.info(f"Converting pose network weights from {ckpt_path}")
    
    # Create new model
    model = ModernPoseNetwork()
    # Create sample input to build the model
    sample_input = tf.random.normal([1, 299, 299, 3])
    _ = model(sample_input)
    
    # Load TF1 weights
    weights_dict = load_tf1_checkpoint(ckpt_path)
    
    # Map old names to new names
    name_mapping = {
        'InceptionResnetV2/Conv2d_1a_3x3/weights': 'conv1/kernel',
        'InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/beta': 'bn1/beta',
        'InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/gamma': 'bn1/gamma',
        'InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/moving_mean': 'bn1/moving_mean',
        'InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/moving_variance': 'bn1/moving_variance',
        # Add more mappings as needed
    }
    
    # Transfer weights
    for old_name, new_name in name_mapping.items():
        if old_name in weights_dict:
            logger.info(f"Transferring weights for {old_name} -> {new_name}")
            # Find corresponding layer
            for layer in model.layers:
                if new_name.split('/')[0] in layer.name:
                    weights = layer.get_weights()
                    if len(weights) > 0:  # Layer has weights
                        weights[0] = weights_dict[old_name]
                        layer.set_weights(weights)
                        break
    
    # Save in Keras format
    output_path = str(output_path)
    if not output_path.endswith('.weights.h5'):
        output_path += '.weights.h5'
    model.save_weights(output_path)
    logger.info(f"Saved converted weights to {output_path}")

def convert_gait_network_weights(ckpt_path, output_path):
    """Convert gait network weights from TF1 to Keras format."""
    logger.info(f"Converting gait network weights from {ckpt_path}")
    
    # Create new model
    model = ModernGaitNetwork()

    model.summary()

    # Create sample input to build the model
    sample_input = tf.random.normal([1, 64, 64, 512])
    _ = model(sample_input)
    
    # Load TF1 weights
    weights_dict = load_tf1_checkpoint(ckpt_path)
    
    # Map old names to new names
    name_mapping = {
        'GaitNN/Block17_0/Conv2d_1x1/weights': 'block_17_0/conv1/kernel',
        'GaitNN/Block17_1/Conv2d_1x1/weights': 'block_17_1/conv1/kernel',
        'GaitNN/Block17_2/Conv2d_1x1/weights': 'block_17_2/conv1/kernel',
        # Add more mappings based on checkpoint inspection
    }
    
    # Transfer weights
    for old_name, new_name in name_mapping.items():
        if old_name in weights_dict:
            logger.info(f"Transferring weights for {old_name} -> {new_name}")
            # Find corresponding layer
            for layer in model.layers:
                if new_name.split('/')[0] in layer.name:
                    weights = layer.get_weights()
                    if len(weights) > 0:  # Layer has weights
                        weights[0] = weights_dict[old_name]
                        layer.set_weights(weights)
                        break
    
    # Save in Keras format
    output_path = str(output_path)
    if not output_path.endswith('.keras'):
        output_path += '.keras'
    model.save_weights(output_path)
    logger.info(f"Saved converted weights to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert TF1 checkpoints to Keras format')
    parser.add_argument('--pose-ckpt', type=str, help='Path to pose network checkpoint')
    parser.add_argument('--gait-ckpt', type=str, help='Path to gait network checkpoint')
    parser.add_argument('--output-dir', type=str, default='models/converted', 
                       help='Output directory for converted weights')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert pose network weights if provided
    if args.pose_ckpt:
        pose_output = output_dir / 'pose_network'
        convert_pose_network_weights(args.pose_ckpt, str(pose_output))
    
    # Convert gait network weights if provided
    if args.gait_ckpt:
        gait_output = output_dir / 'gait_network'
        convert_gait_network_weights(args.gait_ckpt, str(gait_output))

if __name__ == '__main__':
    main()