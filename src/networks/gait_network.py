import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, List

class ModernGaitNetwork:
    """Modern implementation of the Gait Recognition Network with pure TensorFlow."""
    
    def __init__(self, features: int = 256, num_rnn_layers: int = 2, 
                 recurrent_unit: str = 'lstm', is_training: bool = False):
        """Initialize the gait network.
        
        Args:
            features: Number of features in the output signature
            num_rnn_layers: Number of recurrent layers
            recurrent_unit: Type of recurrent unit ('lstm' or 'gru')
            is_training: Whether the network is in training mode
        """
        self.features = features
        self.num_rnn_layers = num_rnn_layers
        self.recurrent_unit = recurrent_unit.lower()
        self.is_training = is_training
        self.dropout_rate = 0.5 if is_training else 0.0
        
        # Validate recurrent unit type
        if self.recurrent_unit not in ['lstm', 'gru']:
            raise ValueError("recurrent_unit must be either 'lstm' or 'gru'")
        
        # Initialize weights
        self.weights = {}
        self.build_network()

    def build_network(self):
        """Build network and initialize weights."""
        # Feature reduction layers
        self.weights['feature_reduction'] = {
            'conv1': {
                'kernel': tf.Variable(tf.random.normal([1, 1, 32, 64])),
                'bias': tf.Variable(tf.zeros([64]))
            },
            'conv2': {
                'kernel': tf.Variable(tf.random.normal([1, 1, 64, 128])),
                'bias': tf.Variable(tf.zeros([128]))
            }
        }
        
        # Initialize RNN weights
        input_size = 128
        for i in range(self.num_rnn_layers):
            layer_name = f'rnn_layer_{i}'
            if self.recurrent_unit == 'lstm':
                self._init_lstm_weights(layer_name, input_size)
            else:  # GRU
                self._init_gru_weights(layer_name, input_size)
            input_size = self.features
        
        # Final dense layer
        self.weights['dense'] = {
            'kernel': tf.Variable(tf.random.normal([self.features, self.features])),
            'bias': tf.Variable(tf.zeros([self.features]))
        }

    def _init_lstm_weights(self, name: str, input_size: int):
        """Initialize LSTM weights for a layer."""
        # LSTM has 4 gates: input, forget, cell, output
        self.weights[name] = {
            'kernel': tf.Variable(
                tf.random.normal([input_size + self.features, 4 * self.features])),
            'bias': tf.Variable(tf.zeros([4 * self.features])),
            'initial_state': tf.Variable(tf.zeros([2, self.features]))  # [h, c]
        }

    def _init_gru_weights(self, name: str, input_size: int):
        """Initialize GRU weights for a layer."""
        # GRU has 3 gates: reset, update, new
        self.weights[name] = {
            'kernel': tf.Variable(
                tf.random.normal([input_size + self.features, 3 * self.features])),
            'bias': tf.Variable(tf.zeros([3 * self.features])),
            'initial_state': tf.Variable(tf.zeros([self.features]))
        }

    @tf.function
    def __call__(self, inputs, states=None):
        """Forward pass of the network."""
        return self.forward(inputs, states)

    def forward(self, inputs, states=None):
        """Forward pass implementation."""
        batch_size = tf.shape(inputs)[0]
        
        # Feature reduction
        x = tf.nn.conv2d(inputs, self.weights['feature_reduction']['conv1']['kernel'],
                        strides=1, padding='VALID')
        x = tf.nn.bias_add(x, self.weights['feature_reduction']['conv1']['bias'])
        x = tf.nn.relu(x)
        
        x = tf.nn.conv2d(x, self.weights['feature_reduction']['conv2']['kernel'],
                        strides=1, padding='VALID')
        x = tf.nn.bias_add(x, self.weights['feature_reduction']['conv2']['bias'])
        x = tf.nn.relu(x)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=[1, 2])  # Average over spatial dimensions
        
        # Initialize states if not provided
        if states is None:
            states = self._get_initial_states(batch_size)
        
        # Process through RNN layers
        rnn_outputs = []
        current_states = []
        
        for i in range(self.num_rnn_layers):
            layer_name = f'rnn_layer_{i}'
            layer_states = states[i] if states else None
            
            if self.recurrent_unit == 'lstm':
                x, new_states = self._lstm_forward(x, self.weights[layer_name], layer_states)
            else:  # GRU
                x, new_states = self._gru_forward(x, self.weights[layer_name], layer_states)
            
            current_states.append(new_states)
            
            if self.is_training:
                x = tf.nn.dropout(x, rate=self.dropout_rate)
        
        # Final dense layer
        x = tf.matmul(x, self.weights['dense']['kernel'])
        x = tf.nn.bias_add(x, self.weights['dense']['bias'])
        
        # L2 normalize the output to create gait signature
        signature = tf.nn.l2_normalize(x, axis=-1)
        
        return signature, current_states

    def _lstm_forward(self, inputs: tf.Tensor, weights: Dict[str, tf.Tensor],
                     states: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Forward pass through LSTM layer."""
        if states is None:
            h_state = tf.zeros([tf.shape(inputs)[0], self.features])
            c_state = tf.zeros([tf.shape(inputs)[0], self.features])
        else:
            h_state, c_state = states
        
        # Concatenate input and previous hidden state
        concat = tf.concat([inputs, h_state], axis=1)
        
        # Calculate gates
        gates = tf.matmul(concat, weights['kernel'])
        gates = tf.nn.bias_add(gates, weights['bias'])
        
        # Split into individual gates
        i, f, c, o = tf.split(gates, 4, axis=1)
        
        # Apply gate activations
        i = tf.sigmoid(i)  # input gate
        f = tf.sigmoid(f)  # forget gate
        c_tilde = tf.tanh(c)  # candidate cell state
        o = tf.sigmoid(o)  # output gate
        
        # Update states
        new_c = f * c_state + i * c_tilde
        new_h = o * tf.tanh(new_c)
        
        return new_h, (new_h, new_c)

    def _gru_forward(self, inputs: tf.Tensor, weights: Dict[str, tf.Tensor],
                    state: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass through GRU layer."""
        if state is None:
            state = tf.zeros([tf.shape(inputs)[0], self.features])
        
        # Concatenate input and previous state
        concat = tf.concat([inputs, state], axis=1)
        
        # Calculate gates
        gates = tf.matmul(concat, weights['kernel'])
        gates = tf.nn.bias_add(gates, weights['bias'])
        
        # Split into individual gates
        r, z, n = tf.split(gates, 3, axis=1)
        
        # Apply gate activations
        r = tf.sigmoid(r)  # reset gate
        z = tf.sigmoid(z)  # update gate
        n = tf.tanh(n)  # new gate
        
        # Update state
        new_state = z * state + (1 - z) * n
        
        return new_state, new_state

    def _get_initial_states(self, batch_size: int) -> List:
        """Get initial states for all RNN layers."""
        states = []
        for i in range(self.num_rnn_layers):
            layer_name = f'rnn_layer_{i}'
            if self.recurrent_unit == 'lstm':
                h = tf.tile(self.weights[layer_name]['initial_state'][0:1], [batch_size, 1])
                c = tf.tile(self.weights[layer_name]['initial_state'][1:2], [batch_size, 1])
                states.append((h, c))
            else:  # GRU
                h = tf.tile(self.weights[layer_name]['initial_state'][None], [batch_size, 1])
                states.append(h)
        return states

    @tf.function
    def feed_forward(self, x: tf.Tensor) -> Tuple[tf.Tensor, List]:
        """Feed forward pass for input sequence."""
        return self(x)

    def load_weights(self, ckpt_path: str):
        """Load weights from checkpoint file.
        
        Args:
            ckpt_path: Path to the checkpoint file
        """
        checkpoint = tf.train.Checkpoint(**self.weights)
        status = checkpoint.restore(ckpt_path)
        status.expect_partial()

    def save_weights(self, ckpt_path: str):
        """Save weights to checkpoint file.
        
        Args:
            ckpt_path: Path to save the checkpoint
        """
        checkpoint = tf.train.Checkpoint(**self.weights)
        checkpoint.save(ckpt_path)