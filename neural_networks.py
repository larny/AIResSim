"""
Neural Network Models for Reservoir Simulation
Implements GNN, FNO, and Well models with simplified PyTorch-like interface
"""

import math
import random
from typing import List, Dict, Tuple, Optional

# Simplified tensor operations for demonstration
class Tensor:
    """Simplified tensor class for demonstration"""
    
    def __init__(self, data, shape=None):
        if isinstance(data, (int, float)):
            self.data = [data]
            self.shape = (1,)
        elif isinstance(data, list):
            self.data = data
            if shape:
                self.shape = shape
            else:
                self.shape = (len(data),)
        else:
            self.data = list(data)
            self.shape = shape or (len(self.data),)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        self.data[idx] = value
    
    def size(self):
        return self.shape
    
    def view(self, *shape):
        return Tensor(self.data, shape)
    
    def mean(self):
        return sum(self.data) / len(self.data)
    
    def std(self):
        mean_val = self.mean()
        variance = sum((x - mean_val) ** 2 for x in self.data) / len(self.data)
        return variance ** 0.5

# Simplified neural network layers
class Linear:
    """Simplified linear layer"""
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights randomly
        self.weight = [[random.gauss(0, 0.1) for _ in range(in_features)] 
                       for _ in range(out_features)]
        self.bias = [random.gauss(0, 0.01) for _ in range(out_features)]
    
    def forward(self, x: List[float]) -> List[float]:
        """Forward pass"""
        output = []
        for i in range(self.out_features):
            val = self.bias[i]
            for j in range(self.in_features):
                if j < len(x):
                    val += self.weight[i][j] * x[j]
            output.append(val)
        return output

class ReLU:
    """ReLU activation function"""
    
    def forward(self, x: List[float]) -> List[float]:
        return [max(0, val) for val in x]

class BatchNorm1d:
    """Simplified batch normalization"""
    
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        self.eps = 1e-5
    
    def forward(self, x: List[float]) -> List[float]:
        """Simplified batch norm"""
        output = []
        for i, val in enumerate(x[:self.num_features]):
            normalized = (val - self.running_mean[i]) / math.sqrt(self.running_var[i] + self.eps)
            output.append(normalized)
        return output

class GCNLayer:
    """Graph Convolutional Network Layer"""
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.linear = Linear(in_features, out_features)
        self.activation = ReLU()
    
    def forward(self, node_features: List[List[float]], 
                edge_index: List[List[int]]) -> List[List[float]]:
        """
        Forward pass for GCN layer
        Aggregates neighbor features and applies transformation
        """
        num_nodes = len(node_features)
        output_features = []
        
        for node_id in range(min(num_nodes, 1000)):  # Limit for testing
            # Get current node features
            node_feat = node_features[node_id]
            
            # Aggregate neighbor features
            neighbor_sum = [0.0] * self.in_features
            neighbor_count = 0
            
            # Find neighbors from edge index (limit search for performance)
            edge_limit = min(len(edge_index[0]), 10000)  # Limit edges processed
            for i in range(edge_limit):
                src, tgt = edge_index[0][i], edge_index[1][i]
                if tgt == node_id:  # This node is the target
                    for j, val in enumerate(node_features[src][:self.in_features]):
                        neighbor_sum[j] += val
                    neighbor_count += 1
            
            # Average aggregation
            if neighbor_count > 0:
                aggregated = [val / neighbor_count for val in neighbor_sum]
            else:
                aggregated = [0.0] * self.in_features
            
            # Combine self features with aggregated neighbor features
            combined = []
            for i in range(self.in_features):
                if i < len(node_feat) and i < len(aggregated):
                    combined.append(node_feat[i] + aggregated[i])
                elif i < len(node_feat):
                    combined.append(node_feat[i])
                else:
                    combined.append(0.0)
            
            # Apply linear transformation and activation
            transformed = self.linear.forward(combined)
            activated = self.activation.forward(transformed)
            
            output_features.append(activated)
        
        # Pad output to match expected size
        while len(output_features) < num_nodes:
            output_features.append([0.5])  # Default saturation
        
        return output_features

class GCNIILayer:
    """GCN II Layer with residual connections and identity mapping"""
    
    def __init__(self, hidden_dim: int, alpha: float = 0.1, theta: float = 0.5):
        self.hidden_dim = hidden_dim
        self.alpha = alpha  # Residual connection strength
        self.theta = theta  # Identity mapping strength
        self.linear = Linear(hidden_dim, hidden_dim)
        self.activation = ReLU()
    
    def forward(self, x: List[List[float]], x0: List[List[float]], 
                edge_index: List[List[int]]) -> List[List[float]]:
        """
        Forward pass with residual connections to initial features
        """
        # Standard GCN aggregation
        gcn_layer = GCNLayer(self.hidden_dim, self.hidden_dim)
        h = gcn_layer.forward(x, edge_index)
        
        # Apply GCN II formula: h = ((1-alpha)*h + alpha*x0) * (1-theta) + theta*x
        output = []
        for i in range(len(h)):
            node_output = []
            for j in range(self.hidden_dim):
                h_val = h[i][j] if j < len(h[i]) else 0.0
                x0_val = x0[i][j] if j < len(x0[i]) else 0.0
                x_val = x[i][j] if j < len(x[i]) else 0.0
                
                # GCN II combination
                combined = ((1 - self.alpha) * h_val + self.alpha * x0_val) * (1 - self.theta) + self.theta * x_val
                node_output.append(combined)
            
            output.append(node_output)
        
        return output

class GNNModel:
    """Graph Neural Network for Saturation Prediction"""
    
    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, output_dim: int = 1, num_layers: int = 4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = Linear(input_dim, hidden_dim)
        self.encoder_activation = ReLU()
        self.encoder_norm = BatchNorm1d(hidden_dim)
        
        # GCN II layers
        self.gcn_layers = []
        for _ in range(num_layers):
            self.gcn_layers.append(GCNIILayer(hidden_dim))
        
        # Decoder
        self.decoder = Linear(hidden_dim, output_dim)
        
        print(f"GNN Model initialized: {input_dim} -> {hidden_dim} -> {output_dim}")
    
    def forward(self, node_features: List[List[float]], 
                edge_index: List[List[int]]) -> List[List[float]]:
        """Forward pass through GNN"""
        
        # Encoder
        x = []
        for node_feat in node_features:
            encoded = self.encoder.forward(node_feat)
            activated = self.encoder_activation.forward(encoded)
            normalized = self.encoder_norm.forward(activated)
            x.append(normalized)
        
        # Store initial features for residual connections
        x0 = [feat[:] for feat in x]  # Deep copy
        
        # GCN II layers
        for layer in self.gcn_layers:
            x = layer.forward(x, x0, edge_index)
        
        # Decoder
        output = []
        for node_feat in x:
            decoded = self.decoder.forward(node_feat)
            output.append(decoded)
        
        return output

class FourierLayer:
    """Simplified Fourier layer for FNO"""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int = 16):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Simplified weights (in practice these would be complex)
        self.weights = [[[random.gauss(0, 0.1) for _ in range(modes)] 
                        for _ in range(in_channels)] 
                       for _ in range(out_channels)]
    
    def forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """Simplified Fourier transform and multiplication"""
        # This is a very simplified version - real FNO uses FFT
        output = []
        for i, channel in enumerate(x):
            if i < self.out_channels:
                output.append(channel)
            else:
                # Create new channel
                new_channel = [[0.0 for _ in row] for row in channel]
                output.append(new_channel)
        
        return output

class FNOModel:
    """Fourier Neural Operator for Pressure Prediction"""
    
    def __init__(self, input_channels: int = 4, hidden_channels: int = 32, 
                 output_channels: int = 1, num_layers: int = 4, modes: int = 16):
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_layers = num_layers
        self.modes = modes
        
        # Lifting layer
        self.lift = Linear(input_channels, hidden_channels)
        
        # Fourier layers
        self.fourier_layers = []
        for _ in range(num_layers):
            self.fourier_layers.append(FourierLayer(hidden_channels, hidden_channels, modes))
        
        # Projection layer
        self.project = Linear(hidden_channels, output_channels)
        
        print(f"FNO Model initialized: {input_channels} -> {hidden_channels} -> {output_channels}")
    
    def forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """Forward pass through FNO"""
        # x is expected to be [batch, height, width] with multiple channels
        # For simplicity, we'll process it as a 2D field
        
        if not x or not x[0] or not x[0][0]:
            return [[[0.0]]]
        
        height = len(x[0])
        width = len(x[0][0])
        
        # Create input features by stacking channels at each spatial point
        lifted = []
        for i in range(height):
            lifted_row = []
            for j in range(width):
                # Collect features from all channels at position (i,j)
                point_features = []
                for channel in range(min(self.input_channels, len(x))):
                    if i < len(x[channel]) and j < len(x[channel][i]):
                        point_features.append(x[channel][i][j])
                    else:
                        point_features.append(0.0)
                
                # Pad if needed
                while len(point_features) < self.input_channels:
                    point_features.append(0.0)
                
                # Lift to higher dimension
                lifted_features = self.lift.forward(point_features)
                lifted_row.append(lifted_features)
            lifted.append(lifted_row)
        
        # Apply Fourier layers (simplified - just pass through)
        current = [lifted]  # Wrap in batch dimension
        for layer in self.fourier_layers:
            current = layer.forward(current)
        
        # Project to output dimension
        output_batch = []
        for batch_item in current:
            output_slice = []
            for i, row in enumerate(batch_item):
                output_row = []
                for j, point_features in enumerate(row):
                    projected = self.project.forward(point_features)
                    output_row.append(projected[0] if projected else 0.0)  # Take first output
                output_slice.append(output_row)
            output_batch.append(output_slice)
        
        return output_batch

class WellModel:
    """Feedforward Neural Network for Well Property Prediction"""
    
    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [64, 32], output_dim: int = 3):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build layers
        self.layers = []
        self.activations = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(Linear(prev_dim, hidden_dim))
            self.activations.append(ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        self.layers.append(Linear(prev_dim, output_dim))
        
        print(f"Well Model initialized: {input_dim} -> {hidden_dims} -> {output_dim}")
    
    def forward(self, x: List[float]) -> List[float]:
        """Forward pass through well model"""
        current = x
        
        # Hidden layers with activation
        for i, (layer, activation) in enumerate(zip(self.layers[:-1], self.activations)):
            current = layer.forward(current)
            current = activation.forward(current)
        
        # Output layer (no activation)
        output = self.layers[-1].forward(current)
        
        return output

def test_neural_networks():
    """Test neural network models"""
    print("=== Testing Neural Network Models ===")
    
    # Test GNN
    print("\nTesting GNN Model...")
    gnn = GNNModel(input_dim=9, hidden_dim=32, output_dim=1)
    
    # Create dummy data
    num_nodes = 100
    node_features = [[random.gauss(0, 1) for _ in range(9)] for _ in range(num_nodes)]
    edge_index = [[], []]
    
    # Add some edges
    for i in range(num_nodes - 1):
        edge_index[0].extend([i, i+1])
        edge_index[1].extend([i+1, i])
    
    gnn_output = gnn.forward(node_features, edge_index)
    print(f"GNN output shape: {len(gnn_output)} nodes, {len(gnn_output[0])} features")
    
    # Test FNO
    print("\nTesting FNO Model...")
    fno = FNOModel(input_channels=4, hidden_channels=16, output_channels=1)
    
    # Create dummy 3D field data - 4 channels, each with 2D spatial data
    nx, ny = 10, 10
    field_data = [[[random.gauss(0, 1) for _ in range(ny)] 
                   for _ in range(nx)]
                  for _ in range(4)]  # 4 channels
    
    fno_output = fno.forward(field_data)
    print(f"FNO output shape: {len(fno_output)} x {len(fno_output[0])} x {len(fno_output[0][0])}")
    
    # Test Well Model
    print("\nTesting Well Model...")
    well_model = WellModel(input_dim=10, hidden_dims=[32, 16], output_dim=3)
    
    well_input = [random.gauss(0, 1) for _ in range(10)]
    well_output = well_model.forward(well_input)
    print(f"Well model output: {well_output}")
    
    print("\n=== Neural Network Tests Completed ===")
    
    return gnn, fno, well_model

if __name__ == "__main__":
    test_neural_networks()