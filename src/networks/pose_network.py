# File: src/networks/pose_network.py

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, List

class ModernPoseNetwork(tf.keras.Model):
    """Modern implementation of the Pose Estimation Network with TF2 compatibility."""
    
    def __init__(self, image_size: int = 299, heatmap_size: int = 289, features: int = 32):
        """Initialize the pose network.
        
        Args:
            image_size: Input image size
            heatmap_size: Output heatmap size
            features: Number of features
        """
        super(ModernPoseNetwork, self).__init__()
        
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.features = features
        self.smooth_size = 21
        self.sigma = 1.0
        
        # Build network components
        self._build_layers()
        self._build_gaussian_filter()

    def _build_layers(self):
        """Build all network layers."""
        # Initial convolutions
        self.conv1 = tf.keras.layers.Conv2D(
            32, kernel_size=3, strides=2, padding='valid', name='conv1'
        )
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        
        # Inception-ResNet blocks
        self.inception_blocks = []
        for i in range(3):
            block = self._build_inception_block(f'inception_block_{i}')
            self.inception_blocks.append(block)
        
        # Auxiliary tower
        self.aux_pool = tf.keras.layers.AveragePooling2D(5, strides=1, padding='same')
        self.aux_conv1 = tf.keras.layers.Conv2D(128, 1, name='aux_conv1')
        self.aux_conv2 = tf.keras.layers.Conv2D(768, 5, padding='same', name='aux_conv2')
        
        # Final layers
        self.final_conv = tf.keras.layers.Conv2D(self.features, 1, name='final_conv')
        
        # Upsampling
        self.upsample = tf.keras.layers.Conv2DTranspose(
            16,  # Number of joints
            kernel_size=17,
            strides=17,
            padding='valid',
            name='upsample'
        )

    def _build_inception_block(self, name: str) -> tf.keras.Model:
        """Build an Inception-ResNet block."""
        input_layer = tf.keras.layers.Input(shape=(None, None, self.features))
        
        # Branch 1
        branch1 = tf.keras.layers.Conv2D(32, 1, name=f'{name}_branch1_conv')(input_layer)
        
        # Branch 2
        branch2 = tf.keras.layers.Conv2D(32, 1, name=f'{name}_branch2_conv1')(input_layer)
        branch2 = tf.keras.layers.Conv2D(32, 3, padding='same',
                                       name=f'{name}_branch2_conv2')(branch2)
        
        # Branch 3
        branch3 = tf.keras.layers.Conv2D(32, 1, name=f'{name}_branch3_conv1')(input_layer)
        branch3 = tf.keras.layers.Conv2D(48, 3, padding='same',
                                       name=f'{name}_branch3_conv2')(branch3)
        branch3 = tf.keras.layers.Conv2D(64, 3, padding='same',
                                       name=f'{name}_branch3_conv3')(branch3)
        
        # Concatenate branches
        concat = tf.keras.layers.Concatenate(axis=-1)([branch1, branch2, branch3])
        
        # 1x1 convolution to match input dimensions
        output = tf.keras.layers.Conv2D(self.features, 1,
                                      name=f'{name}_project')(concat)
        
        # Residual connection
        scaled = tf.keras.layers.Lambda(
            lambda x: x[0] + 0.1 * x[1])([input_layer, output])
        
        return tf.keras.Model(inputs=input_layer, outputs=scaled, name=name)

    def _build_gaussian_filter(self):
        """Build Gaussian smoothing filter."""
        # Create Gaussian kernel
        size = self.smooth_size
        x = tf.range(-size//2 + 1, size//2 + 1, dtype=tf.float32)
        gaussian = tf.exp(-tf.square(x) / (2 * self.sigma**2))
        gaussian = gaussian / tf.reduce_sum(gaussian)
        
        # Create separable 2D filters for each channel
        gaussian_h = tf.reshape(gaussian, [size, 1, 1, 1])
        gaussian_v = tf.reshape(gaussian, [1, size, 1, 1])
        
        # Store filters
        self.gaussian_h = gaussian_h
        self.gaussian_v = gaussian_v

    def call(self, inputs, training=False):
        """Forward pass of the network."""
        # Initial processing
        x = self._preprocess(inputs)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        # Pass through Inception-ResNet blocks
        for block in self.inception_blocks:
            x = block(x)
        
        # Store feature tensor for gait recognition
        self.feature_tensor = x
        
        # Auxiliary tower
        x = self.aux_pool(x)
        x = self.aux_conv1(x)
        x = self.aux_conv2(x)
        
        # Final processing
        x = self.final_conv(x)
        x = self.upsample(x)
        
        # Apply Gaussian smoothing
        x = self._apply_gaussian_smoothing(x)
        
        return x

    def _preprocess(self, inputs: tf.Tensor) -> tf.Tensor:
        """Preprocess input images."""
        x = tf.cast(inputs, tf.float32)
        x = (x / 255.0 - 0.5) * 2.0
        return x

    def _apply_gaussian_smoothing(self, x: tf.Tensor) -> tf.Tensor:
        """Apply Gaussian smoothing to heatmaps."""
        # Get shape
        batch_size = tf.shape(x)[0]
        num_channels = tf.shape(x)[-1]
        
        # Separate channels and apply smoothing
        channels = tf.unstack(x, axis=-1)
        smoothed_channels = []
        
        for channel in channels:
            # Add channel dimension
            channel = tf.expand_dims(channel, -1)
            
            # Apply horizontal smoothing
            channel = tf.nn.conv2d(
                channel,
                self.gaussian_h,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            
            # Apply vertical smoothing
            channel = tf.nn.conv2d(
                channel,
                self.gaussian_v,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            
            smoothed_channels.append(tf.squeeze(channel, -1))
        
        # Stack channels back together
        return tf.stack(smoothed_channels, axis=-1)

    @tf.function
    def feed_forward_features(self, x: tf.Tensor) -> tf.Tensor:
        """Extract features from input images."""
        self(x)  # Forward pass
        return self.feature_tensor

    @tf.function
    def estimate_joints(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Estimate joint positions from input images."""
        heatmaps = self(x)
        
        # Process heatmaps to get joint positions
        batch_size = tf.shape(heatmaps)[0]
        height = tf.shape(heatmaps)[1]
        width = tf.shape(heatmaps)[2]
        num_joints = tf.shape(heatmaps)[3]
        
        # Find peak locations
        heatmaps_flat = tf.reshape(heatmaps, [batch_size, -1, num_joints])
        max_vals = tf.reduce_max(heatmaps_flat, axis=1)
        argmax = tf.argmax(heatmaps_flat, axis=1)
        
        # Convert to coordinates
        y_coords = tf.cast(argmax // width, tf.float32)
        x_coords = tf.cast(argmax % width, tf.float32)
        
        # Scale coordinates to original image size
        scale = tf.cast(self.image_size / self.heatmap_size, tf.float32)
        y_coords = y_coords * scale
        x_coords = x_coords * scale
        
        return y_coords, x_coords, max_vals

    def load_weights(self, filepath: str):
        """Load model weights.
        
        Args:
            filepath: Path to the weights file
        """
        try:
            super().load_weights(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load weights from {filepath}: {str(e)}")

    def save_weights(self, filepath: str):
        """Save model weights.
        
        Args:
            filepath: Path to save the weights
        """
        try:
            super().save_weights(filepath)
        except Exception as e:
            raise ValueError(f"Failed to save weights to {filepath}: {str(e)}")