import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, List
from loguru import logger

class ModernGaitNetwork:
    """Modern implementation of the Gait Recognition Network with extensive debugging."""
    
    def __init__(self, features: int = 256, num_rnn_layers: int = 2, 
                 recurrent_unit: str = 'lstm', is_training: bool = False):
        """Initialize the gait network."""
        self.features = features
        self.num_rnn_layers = num_rnn_layers
        self.recurrent_unit = recurrent_unit.lower()
        self.is_training = is_training
        self.dropout_rate = 0.5 if is_training else 0.0
        
        # Debug flags
        self.debug = True
        self.layer_debug = True
        self.weight_debug = True
        
        # Initialize weights
        self.weights = {}
        self.build_network()
        
        if self.weight_debug:
            self._debug_weights()

    def _debug_tensor(self, tensor, name: str, layer_name: str = ""):
        """Debug helper for tensors."""
        if not self.debug:
            return

        prefix = f"[{layer_name}] " if layer_name else ""
        
        # Use tf operations instead of numpy for tensor analysis
        shape = tf.shape(tensor)
        min_val = tf.reduce_min(tensor)
        max_val = tf.reduce_max(tensor)
        mean_val = tf.reduce_mean(tensor)
        std_val = tf.math.reduce_std(tensor)
        
        # Print stats using tf.print for compatibility with tf.function
        tf.print(f"\n{prefix}{name} stats:",
                "\nShape:", shape,
                "\nMin/Max:", min_val, max_val,
                "\nMean/Std:", mean_val, std_val)
        
        # Check for numerical issues
        nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(tensor), tf.int32))
        inf_count = tf.reduce_sum(tf.cast(tf.math.is_inf(tensor), tf.int32))
        
        if nan_count > 0 or inf_count > 0:
            tf.print(f"{prefix}{name} has", nan_count, "NaNs and", inf_count, "Infs")
        
        # Check for constant values
        first_val = tensor[0]
        is_constant = tf.reduce_all(tf.equal(tensor, first_val))
        if is_constant:
            tf.print(f"{prefix}{name} has all identical values:", first_val)
        
        # Sample random values for distribution check
        flat_tensor = tf.reshape(tensor, [-1])
        tensor_size = tf.size(flat_tensor)
        num_samples = tf.minimum(1000, tensor_size)  # Take at most 1000 samples
        
        if num_samples > 0:
            indices = tf.random.uniform([num_samples], 0, tensor_size, dtype=tf.int32)
            samples = tf.gather(flat_tensor, indices)
            samples = tf.sort(samples)
            
            # Get approximate percentiles from samples
            sample_indices = tf.cast(tf.linspace(0.0, tf.cast(num_samples-1, tf.float32), 5), tf.int32)
            percentiles = tf.gather(samples, sample_indices)
            
            tf.print(f"{prefix}{name} approximate percentiles [0,25,50,75,100] from {num_samples} samples:", 
                    percentiles)

    def _debug_weights(self):
        """Debug weight statistics and potential issues."""
        logger.info("Debugging network weights...")
        
        for layer_name, layer_weights in self.weights.items():
            logger.debug(f"\nAnalyzing weights for layer: {layer_name}")
            
            for weight_name, weight in layer_weights.items():
                if isinstance(weight, tf.Variable):
                    weight_data = weight.numpy()
                    
                    # Basic statistics
                    logger.debug(f"{layer_name}/{weight_name}:")
                    logger.debug(f"  Shape: {weight_data.shape}")
                    logger.debug(f"  Min/Max: {np.min(weight_data):.6f}/{np.max(weight_data):.6f}")
                    logger.debug(f"  Mean/Std: {np.mean(weight_data):.6f}/{np.std(weight_data):.6f}")
                    
                    # Check for issues
                    if np.any(np.isnan(weight_data)):
                        logger.error(f"  NaN values found in {layer_name}/{weight_name}")
                    if np.any(np.isinf(weight_data)):
                        logger.error(f"  Inf values found in {layer_name}/{weight_name}")
                    if np.allclose(weight_data, 0):
                        logger.warning(f"  All weights near zero in {layer_name}/{weight_name}")
                    if np.allclose(weight_data, weight_data.flatten()[0]):
                        logger.warning(f"  All weights identical in {layer_name}/{weight_name}")

    def build_network(self):
        """Build network and initialize weights."""
        # Feature reduction layers
        self.weights['feature_reduction'] = {
            'conv1': {
                'kernel': tf.Variable(tf.random.normal([1, 1, 32, 64], stddev=0.1),
                                   name='conv1_kernel'),
                'bias': tf.Variable(tf.zeros([64]), name='conv1_bias')
            },
            'conv2': {
                'kernel': tf.Variable(tf.random.normal([1, 1, 64, 128], stddev=0.1),
                                   name='conv2_kernel'),
                'bias': tf.Variable(tf.zeros([128]), name='conv2_bias')
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
            'kernel': tf.Variable(tf.random.normal([self.features, self.features], stddev=0.1),
                               name='dense_kernel'),
            'bias': tf.Variable(tf.zeros([self.features]), name='dense_bias')
        }

    def _init_lstm_weights(self, name: str, input_size: int):
        """Initialize LSTM weights for a layer."""
        stddev = 1.0 / np.sqrt(input_size)
        self.weights[name] = {
            'kernel': tf.Variable(
                tf.random.normal([input_size + self.features, 4 * self.features], 
                               stddev=stddev),
                name=f'{name}_kernel'
            ),
            'bias': tf.Variable(tf.zeros([4 * self.features]), name=f'{name}_bias'),
            'initial_state': tf.Variable(tf.zeros([2, self.features]), 
                                      name=f'{name}_initial_state')
        }

    def _init_gru_weights(self, name: str, input_size: int):
        """Initialize GRU weights for a layer."""
        stddev = 1.0 / np.sqrt(input_size)
        self.weights[name] = {
            'kernel': tf.Variable(
                tf.random.normal([input_size + self.features, 3 * self.features],
                               stddev=stddev),
                name=f'{name}_kernel'
            ),
            'bias': tf.Variable(tf.zeros([3 * self.features]), name=f'{name}_bias'),
            'initial_state': tf.Variable(tf.zeros([self.features]), 
                                      name=f'{name}_initial_state')
        }

    def _debug_layer_output(self, x: tf.Tensor, layer_name: str, operation: str):
        """Debug layer outputs."""
        if not self.layer_debug:
            return
        
        prefix = f"{layer_name}/{operation}"
        self._debug_tensor(x, "output", prefix)

    def forward(self, inputs, states=None):
        """Forward pass implementation with detailed debugging."""
        batch_size = tf.shape(inputs)[0]
        
        # Debug input
        self._debug_tensor(inputs, "network_input", "input")
        
        # Feature reduction
        x = tf.nn.conv2d(inputs, self.weights['feature_reduction']['conv1']['kernel'],
                        strides=1, padding='VALID')
        x = tf.nn.bias_add(x, self.weights['feature_reduction']['conv1']['bias'])
        x = tf.nn.relu(x)
        self._debug_layer_output(x, "feature_reduction", "conv1")
        
        x = tf.nn.conv2d(x, self.weights['feature_reduction']['conv2']['kernel'],
                        strides=1, padding='VALID')
        x = tf.nn.bias_add(x, self.weights['feature_reduction']['conv2']['bias'])
        x = tf.nn.relu(x)
        self._debug_layer_output(x, "feature_reduction", "conv2")
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=[1, 2])
        self._debug_layer_output(x, "feature_reduction", "global_pool")
        
        # Initialize states if not provided
        if states is None:
            states = self._get_initial_states(batch_size)
        
        # Process through RNN layers
        current_states = []
        for i in range(self.num_rnn_layers):
            layer_name = f'rnn_layer_{i}'
            layer_states = states[i] if states else None
            
            # Debug pre-RNN
            self._debug_tensor(x, "pre_rnn_input", layer_name)
            
            if self.recurrent_unit == 'lstm':
                x, new_states = self._lstm_forward(x, self.weights[layer_name], 
                                                 layer_states, layer_name)
            else:  # GRU
                x, new_states = self._gru_forward(x, self.weights[layer_name], 
                                                layer_states, layer_name)
            
            # Debug post-RNN
            self._debug_tensor(x, "post_rnn_output", layer_name)
            
            current_states.append(new_states)
            
            if self.is_training:
                x = tf.nn.dropout(x, rate=self.dropout_rate)
        
        # Final dense layer
        x = tf.matmul(x, self.weights['dense']['kernel'])
        x = tf.nn.bias_add(x, self.weights['dense']['bias'])
        self._debug_layer_output(x, "dense", "output")
        
        # L2 normalize
        signature = tf.nn.l2_normalize(x, axis=-1)
        self._debug_layer_output(signature, "final", "normalized")
        
        return signature, current_states

    def _lstm_forward(self, inputs: tf.Tensor, weights: Dict[str, tf.Tensor],
                     states: Optional[Tuple[tf.Tensor, tf.Tensor]], 
                     layer_name: str) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Forward pass through LSTM layer with debugging."""
        if states is None:
            h_state = tf.zeros([tf.shape(inputs)[0], self.features])
            c_state = tf.zeros([tf.shape(inputs)[0], self.features])
        else:
            h_state, c_state = states
        
        # Debug states
        self._debug_tensor(h_state, "h_state", f"{layer_name}/initial")
        self._debug_tensor(c_state, "c_state", f"{layer_name}/initial")
        
        # Concatenate input and previous hidden state
        concat = tf.concat([inputs, h_state], axis=1)
        self._debug_tensor(concat, "concat_input", layer_name)
        
        # Calculate gates
        gates = tf.matmul(concat, weights['kernel'])
        gates = tf.nn.bias_add(gates, weights['bias'])
        self._debug_tensor(gates, "gates", layer_name)
        
        # Split into individual gates
        i, f, c, o = tf.split(gates, 4, axis=1)
        
        # Debug individual gates
        self._debug_tensor(i, "input_gate", layer_name)
        self._debug_tensor(f, "forget_gate", layer_name)
        self._debug_tensor(c, "cell_gate", layer_name)
        self._debug_tensor(o, "output_gate", layer_name)
        
        # Apply gate activations
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        c_tilde = tf.tanh(c)
        o = tf.sigmoid(o)
        
        # Update states
        new_c = f * c_state + i * c_tilde
        new_h = o * tf.tanh(new_c)
        
        # Debug final states
        self._debug_tensor(new_c, "new_c_state", layer_name)
        self._debug_tensor(new_h, "new_h_state", layer_name)
        
        return new_h, (new_h, new_c)

    def _gru_forward(self, inputs: tf.Tensor, weights: Dict[str, tf.Tensor],
                    state: Optional[tf.Tensor], layer_name: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass through GRU layer with debugging."""
        if state is None:
            state = tf.zeros([tf.shape(inputs)[0], self.features])
        
        # Debug initial state
        self._debug_tensor(state, "initial_state", layer_name)
        
        # Concatenate input and previous state
        concat = tf.concat([inputs, state], axis=1)
        self._debug_tensor(concat, "concat_input", layer_name)
        
        # Calculate gates
        gates = tf.matmul(concat, weights['kernel'])
        gates = tf.nn.bias_add(gates, weights['bias'])
        self._debug_tensor(gates, "gates", layer_name)
        
        # Split into individual gates
        r, z, n = tf.split(gates, 3, axis=1)
        
        # Debug individual gates
        self._debug_tensor(r, "reset_gate", layer_name)
        self._debug_tensor(z, "update_gate", layer_name)
        self._debug_tensor(n, "new_gate", layer_name)
        
        # Apply gate activations
        r = tf.sigmoid(r)
        z = tf.sigmoid(z)
        n = tf.tanh(n)
        
        # Update state
        new_state = z * state + (1 - z) * n
        self._debug_tensor(new_state, "new_state", layer_name)
        
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

    def feed_forward(self, x: tf.Tensor) -> Tuple[tf.Tensor, List]:
        """Feed forward pass for input sequence."""
        # Run eager execution for better debugging
        if self.debug:
            return self(x)
        else:
            # Use tf.function only when not debugging
            @tf.function
            def _feed_forward(x):
                return self(x)
            return _feed_forward(x)

    def __call__(self, inputs, states=None):
        """Forward pass of the network."""
        return self.forward(inputs, states)

    def load_weights(self, ckpt_path: str):
        """Load weights from checkpoint file with extensive validation."""
        logger.info(f"Loading weights from {ckpt_path}...")
        
        # Store original weights for comparison
        if self.weight_debug:
            original_weights = {
                name: {k: v.numpy() for k, v in layer_weights.items() 
                      if isinstance(v, tf.Variable)}
                for name, layer_weights in self.weights.items()
            }
        
        try:
            # Create checkpoint
            checkpoint = tf.train.Checkpoint(**self.weights)
            status = checkpoint.restore(ckpt_path)
            status.expect_partial()
            
            if self.weight_debug:
                # Compare weights before and after loading
                logger.info("Analyzing weight changes after loading...")
                for layer_name, layer_weights in self.weights.items():
                    for weight_name, weight in layer_weights.items():
                        if not isinstance(weight, tf.Variable):
                            continue
                            
                        new_weight = weight.numpy()
                        old_weight = original_weights[layer_name][weight_name]
                        
                        if np.allclose(new_weight, old_weight):
                            logger.warning(f"Weights unchanged for {layer_name}/{weight_name}")
                        else:
                            diff_stats = {
                                'max_diff': np.max(np.abs(new_weight - old_weight)),
                                'mean_diff': np.mean(np.abs(new_weight - old_weight)),
                                'std_diff': np.std(new_weight - old_weight)
                            }
                            logger.info(f"Weight changes for {layer_name}/{weight_name}:")
                            logger.info(f"  Max diff: {diff_stats['max_diff']:.6f}")
                            logger.info(f"  Mean diff: {diff_stats['mean_diff']:.6f}")
                            logger.info(f"  Std diff: {diff_stats['std_diff']:.6f}")
                
                # Validate loaded weights
                self._validate_weights()
                
        except Exception as e:
            logger.error(f"Failed to load weights from {ckpt_path}: {str(e)}")
            raise

    def _validate_weights(self):
        """Validate loaded weights for potential issues."""
        logger.info("Validating loaded weights...")
        
        for layer_name, layer_weights in self.weights.items():
            for weight_name, weight in layer_weights.items():
                if not isinstance(weight, tf.Variable):
                    continue
                    
                weight_data = weight.numpy()
                
                # Check for numerical issues
                if np.any(np.isnan(weight_data)):
                    logger.error(f"NaN values found in {layer_name}/{weight_name}")
                if np.any(np.isinf(weight_data)):
                    logger.error(f"Inf values found in {layer_name}/{weight_name}")
                    
                # Check for potential initialization issues
                if np.allclose(weight_data, 0):
                    logger.warning(f"All weights near zero in {layer_name}/{weight_name}")
                if np.allclose(weight_data, weight_data.flatten()[0]):
                    logger.warning(f"All weights identical in {layer_name}/{weight_name}")
                    
                # Check for reasonable magnitudes
                magnitude = np.abs(weight_data).mean()
                if magnitude > 100 or magnitude < 1e-6:
                    logger.warning(f"Unusual weight magnitude in {layer_name}/{weight_name}: {magnitude:.6f}")
                    
                # For RNN weights, check gate distributions
                if 'rnn_layer' in layer_name and weight_name == 'kernel':
                    if self.recurrent_unit == 'lstm':
                        # Check LSTM gate distributions
                        num_gates = 4
                        gate_size = weight_data.shape[-1] // num_gates
                        gates = np.split(weight_data, num_gates, axis=-1)
                        gate_names = ['input', 'forget', 'cell', 'output']
                        
                        for gate, name in zip(gates, gate_names):
                            mean = np.mean(gate)
                            std = np.std(gate)
                            logger.debug(f"{layer_name} {name} gate stats - mean: {mean:.6f}, std: {std:.6f}")
                    else:  # GRU
                        # Check GRU gate distributions
                        num_gates = 3
                        gate_size = weight_data.shape[-1] // num_gates
                        gates = np.split(weight_data, num_gates, axis=-1)
                        gate_names = ['reset', 'update', 'new']
                        
                        for gate, name in zip(gates, gate_names):
                            mean = np.mean(gate)
                            std = np.std(gate)
                            logger.debug(f"{layer_name} {name} gate stats - mean: {mean:.6f}, std: {std:.6f}")

    def save_weights(self, ckpt_path: str):
        """Save weights to checkpoint file with validation."""
        logger.info(f"Saving weights to {ckpt_path}...")
        
        try:
            # Validate weights before saving
            self._validate_weights()
            
            # Create checkpoint and save
            checkpoint = tf.train.Checkpoint(**self.weights)
            checkpoint.save(ckpt_path)
            
            logger.info("Weights saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save weights to {ckpt_path}: {str(e)}")
            raise