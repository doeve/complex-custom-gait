# File: src/networks/gait_network.py

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, List

class ModernGaitNetwork(tf.keras.Model):
    """Modern implementation of the GaitNetwork with TF2 compatibility."""
    
    def __init__(self, features: int = 512, num_rnn_layers: int = 2, 
                 recurrent_unit: str = 'GRU', is_training: bool = False):
        """Initialize the gait network.
        
        Args:
            features: Number of features in the network
            num_rnn_layers: Number of recurrent layers
            recurrent_unit: Type of RNN unit ('GRU' or 'LSTM')
            is_training: Whether the network is in training mode
        """
        super(ModernGaitNetwork, self).__init__()
        
        self.features = features
        self.num_rnn_layers = num_rnn_layers
        self.recurrent_unit = recurrent_unit
        self.is_training = is_training
        
        # Build network components
        self._build_layers()

    def _build_layers(self):
        """Build all network layers."""
        # Downsample block
        self.initial_conv = tf.keras.layers.Conv2D(
            256, kernel_size=1, activation='relu', name='initial_conv'
        )
        
        # Build residual blocks
        self.residual_blocks = []
        
        # 17x17 blocks
        for i in range(3):
            block = self._build_residual_block(256, 64, f'block_17_{i}')
            self.residual_blocks.append(block)
        
        # 8x8 blocks
        self.residual_blocks.append(
            self._build_residual_block(512, 64, 'block_8_0', stride=2)
        )
        for i in range(2):
            block = self._build_residual_block(512, 128, f'block_8_{i+1}')
            self.residual_blocks.append(block)
        
        # 4x4 blocks
        self.residual_blocks.append(
            self._build_residual_block(512, 128, 'block_4_0', stride=2)
        )
        self.residual_blocks.append(
            self._build_residual_block(512, 256, 'block_4_1')
        )
        
        # Final convolutions
        self.final_conv1 = tf.keras.layers.Conv2D(256, 1, name='final_conv1')
        self.final_conv2 = tf.keras.layers.Conv2D(256, 3, name='final_conv2')
        
        # Dense layer
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(512, name='pre_rnn_dense')
        
        # Dropout for training
        self.dropout = tf.keras.layers.Dropout(0.7)
        
        # RNN layers
        rnn_class = tf.keras.layers.GRU if self.recurrent_unit == 'GRU' else tf.keras.layers.LSTM
        
        self.rnn_layers = []
        for i in range(self.num_rnn_layers):
            cell = rnn_class(
                self.features,
                return_sequences=True,
                name=f'rnn_layer_{i}'
            )
            self.rnn_layers.append(cell)

    def _build_residual_block(self, channels: int, bottleneck_channels: int,
                            name: str, stride: int = 1) -> tf.keras.Model:
        """Build a residual block."""
        input_layer = tf.keras.layers.Input(shape=(None, None, channels))
        
        # Skip connection
        if stride > 1:
            shortcut = tf.keras.layers.Conv2D(
                channels, 1, strides=stride,
                name=f'{name}_shortcut'
            )(input_layer)
            shortcut = tf.keras.layers.BatchNormalization(
                name=f'{name}_shortcut_bn'
            )(shortcut)
        else:
            shortcut = input_layer
        
        # Main path
        x = tf.keras.layers.Conv2D(
            bottleneck_channels, 1,
            name=f'{name}_conv1'
        )(input_layer)
        x = tf.keras.layers.BatchNormalization(
            name=f'{name}_bn1'
        )(x)
        x = tf.keras.layers.ReLU()(x)
        
        x = tf.keras.layers.Conv2D(
            bottleneck_channels, 3,
            strides=stride,
            padding='same',
            name=f'{name}_conv2'
        )(x)
        x = tf.keras.layers.BatchNormalization(
            name=f'{name}_bn2'
        )(x)
        x = tf.keras.layers.ReLU()(x)
        
        x = tf.keras.layers.Conv2D(
            channels, 1,
            name=f'{name}_conv3'
        )(x)
        x = tf.keras.layers.BatchNormalization(
            name=f'{name}_bn3'
        )(x)
        
        # Combine paths
        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.ReLU()(x)
        
        return tf.keras.Model(inputs=input_layer, outputs=x, name=name)

    def call(self, inputs, training=False):
        """Forward pass of the network."""
        # Initial processing
        x = self.initial_conv(inputs)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x, training=training)
        
        # Final convolutions
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        
        # Dense processing
        x = self.flatten(x)
        x = self.dense(x)
        
        if training:
            x = self.dropout(x)
        
        # Reshape for RNN
        x = tf.expand_dims(x, axis=0)
        
        # Pass through RNN layers
        rnn_states = []
        for rnn_layer in self.rnn_layers:
            x, state = rnn_layer(x, training=training)
            rnn_states.append(state)
        
        # Store gait signature (temporal average pooling)
        self.gait_signature = tf.reduce_mean(x, axis=1)
        
        return x, rnn_states

    @tf.function
    def feed_forward(self, x: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Generate gait signature from input features.
        
        Args:
            x: Input tensor of shape [batch_size, height, width, channels]
            
        Returns:
            Tuple of (gait_signature, rnn_states)
        """
        outputs, states = self(x, training=False)
        return self.gait_signature, states

    def compute_similarity(self, sig1: tf.Tensor, sig2: tf.Tensor) -> tf.Tensor:
        """Compute similarity between two gait signatures.
        
        Args:
            sig1: First gait signature
            sig2: Second gait signature
            
        Returns:
            Similarity score
        """
        # Normalize signatures
        sig1_norm = tf.nn.l2_normalize(sig1, axis=-1)
        sig2_norm = tf.nn.l2_normalize(sig2, axis=-1)
        
        # Compute cosine similarity
        similarity = tf.reduce_sum(sig1_norm * sig2_norm, axis=-1)
        
        return similarity

    def load_weights(self, filepath: str):
        """Load model weights."""
        super().load_weights(filepath)

    def save_weights(self, filepath: str):
        """Save model weights."""
        super().save_weights(filepath)

    def get_config(self) -> Dict:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'features': self.features,
            'num_rnn_layers': self.num_rnn_layers,
            'recurrent_unit': self.recurrent_unit,
            'is_training': self.is_training
        })
        return config

    @classmethod
    def from_config(cls, config: Dict):
        """Create model from configuration."""
        return cls(**config)