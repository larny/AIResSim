"""
GNN-FNO Coupling Workflow for Reservoir Simulation
Implements the iterative pressure-saturation prediction workflow
"""

import math
from typing import Dict, List, Tuple, Optional
from neural_networks import GNNModel, FNOModel, WellModel
from graph_constructor import GraphConstructor
from feature_extractor import FeatureExtractor

class ReservoirSimulationWorkflow:
    """
    Main workflow class that couples GNN and FNO models
    Following the SPE-223907-MS paper methodology
    """
    
    def __init__(self, case_name: str, data_dir: str, config: Dict = None):
        self.case_name = case_name
        self.data_dir = data_dir
        self.config = config or self.get_default_config()
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(case_name, data_dir)
        self.features = None
        self.graph = None
        
        # Initialize models
        self.gnn_model = None
        self.fno_model = None
        self.well_model = None
        
        # State variables
        self.current_pressure = None
        self.current_saturation = None
        self.timestep = 0
        
        print(f"Reservoir Simulation Workflow initialized for case: {case_name}")
    
    def get_default_config(self) -> Dict:
        """Get default configuration parameters"""
        return {
            # Model parameters
            'gnn_hidden_dim': 64,
            'gnn_num_layers': 4,
            'fno_hidden_channels': 32,
            'fno_modes': 16,
            'well_hidden_dims': [64, 32],
            
            # Training parameters
            'pressure_rollout_steps': 10,
            'saturation_rollout_steps': 5,
            'learning_rate': 0.001,
            'batch_size': 1,
            
            # Simulation parameters
            'max_timesteps': 100,
            'dt': 1.0,  # Time step size
            'pressure_tolerance': 1e-3,
            'saturation_tolerance': 1e-4
        }
    
    def initialize_models(self):
        """Initialize neural network models"""
        print("Initializing neural network models...")
        
        # GNN for saturation prediction
        self.gnn_model = GNNModel(
            input_dim=9,  # pressure, saturation, porosity, 3 perms, 3 coords
            hidden_dim=self.config['gnn_hidden_dim'],
            output_dim=1,  # saturation
            num_layers=self.config['gnn_num_layers']
        )
        
        # FNO for pressure prediction
        self.fno_model = FNOModel(
            input_channels=4,  # saturation, porosity, well controls
            hidden_channels=self.config['fno_hidden_channels'],
            output_channels=1,  # pressure
            modes=self.config['fno_modes']
        )
        
        # Well model for production prediction
        self.well_model = WellModel(
            input_dim=10,  # pressure, saturation, well properties
            hidden_dims=self.config['well_hidden_dims'],
            output_dim=3   # oil rate, water rate, BHP
        )
        
        print("Models initialized successfully")
    
    def load_data(self):
        """Load and process reservoir data"""
        print("Loading reservoir data...")
        
        # Extract features
        self.features = self.feature_extractor.extract_all_features()
        
        # Build graph
        graph_constructor = GraphConstructor(self.features)
        self.graph = graph_constructor.build_graph()
        
        # Initialize state variables
        self.initialize_state()
        
        print("Data loaded and processed successfully")
    
    def initialize_state(self):
        """Initialize pressure and saturation fields"""
        print("Initializing simulation state...")
        
        grid_dims = self.features['grid_dims']
        nx, ny, nz = grid_dims
        total_cells = nx * ny * nz
        
        # Initialize pressure field
        initial_pressure = self.features.get('initial_pressure', [])
        if initial_pressure and len(initial_pressure) > 0:
            # Use actual initial pressure
            self.current_pressure = self.reshape_to_3d(initial_pressure.data, grid_dims)
        else:
            # Default pressure field
            self.current_pressure = [[[2000.0 for _ in range(nz)] 
                                     for _ in range(ny)] 
                                    for _ in range(nx)]
        
        # Initialize saturation field
        initial_saturation = self.features.get('initial_saturation', [])
        if initial_saturation and len(initial_saturation) > 0:
            # Use actual initial saturation
            self.current_saturation = self.reshape_to_3d(initial_saturation.data, grid_dims)
        else:
            # Default saturation field
            self.current_saturation = [[[0.8 for _ in range(nz)] 
                                       for _ in range(ny)] 
                                      for _ in range(nx)]
        
        self.timestep = 0
        print(f"State initialized: pressure range [{self.get_field_min_max(self.current_pressure)}], "
              f"saturation range [{self.get_field_min_max(self.current_saturation)}]")
    
    def reshape_to_3d(self, data: List[float], grid_dims: Tuple[int, int, int]) -> List[List[List[float]]]:
        """Reshape 1D array to 3D grid"""
        nx, ny, nz = grid_dims
        field_3d = []
        
        idx = 0
        for i in range(nx):
            layer_x = []
            for j in range(ny):
                layer_y = []
                for k in range(nz):
                    if idx < len(data):
                        layer_y.append(data[idx])
                    else:
                        layer_y.append(0.0)
                    idx += 1
                layer_x.append(layer_y)
            field_3d.append(layer_x)
        
        return field_3d
    
    def get_field_min_max(self, field: List[List[List[float]]]) -> str:
        """Get min and max values of a 3D field"""
        values = []
        for i in range(len(field)):
            for j in range(len(field[i])):
                for k in range(len(field[i][j])):
                    values.append(field[i][j][k])
        
        if values:
            return f"{min(values):.2f}, {max(values):.2f}"
        return "0.0, 0.0"
    
    def predict_pressure(self, saturation: List[List[List[float]]], 
                        timestep: int) -> List[List[List[float]]]:
        """
        Use FNO to predict pressure field at next timestep
        Input: current saturation field
        Output: pressure field at t+1
        """
        # Prepare input for FNO
        # Input channels: saturation, porosity, well controls
        grid_dims = self.features['grid_dims']
        nx, ny, nz = grid_dims
        
        # Get porosity field
        porosity = self.features.get('porosity', [])
        porosity_3d = self.reshape_to_3d(porosity.data if porosity else [0.2] * (nx*ny*nz), grid_dims)
        
        # Get well control fields
        control_fields = self.features.get('control_fields', {})
        bhp_field = control_fields.get('bottom_hole_pressure', 
                                      [[[0.0 for _ in range(nz)] for _ in range(ny)] for _ in range(nx)])
        rate_field = control_fields.get('liquid_production_rate',
                                       [[[0.0 for _ in range(nz)] for _ in range(ny)] for _ in range(nx)])
        
        # Stack input channels - take 2D slice (z=0 for simplicity)
        input_channels = []
        for channel_3d in [saturation, porosity_3d, bhp_field, rate_field]:
            channel_2d = [[channel_3d[i][j][0] for j in range(ny)] for i in range(nx)]
            input_channels.append(channel_2d)
        
        # Forward pass through FNO
        pressure_output = self.fno_model.forward(input_channels)
        
        # Convert back to 3D (replicate across z-direction)
        if pressure_output and len(pressure_output) > 0:
            pressure_2d = pressure_output[0]
            pressure_3d = []
            for i in range(len(pressure_2d)):
                layer_x = []
                for j in range(len(pressure_2d[i])):
                    layer_y = [pressure_2d[i][j] * 5000.0 for _ in range(nz)]  # Denormalize
                    layer_x.append(layer_y)
                pressure_3d.append(layer_x)
            return pressure_3d
        else:
            return self.current_pressure  # Return current if prediction fails
    
    def predict_saturation(self, pressure: List[List[List[float]]], 
                          prev_saturation: List[List[List[float]]]) -> List[List[float]]:
        """
        Use GNN to predict saturation field
        Input: pressure field at current timestep, saturation at previous timestep
        Output: saturation values for graph nodes
        """
        # Update node features with current pressure
        updated_node_features = []
        grid_dims = self.features['grid_dims']
        nx, ny, nz = grid_dims
        
        for node_idx in range(self.graph['num_nodes']):
            # Get 3D coordinates of this node
            k = node_idx % nz
            j = (node_idx // nz) % ny
            i = node_idx // (ny * nz)
            
            # Get current values
            if i < len(pressure) and j < len(pressure[i]) and k < len(pressure[i][j]):
                press_val = pressure[i][j][k] / 5000.0  # Normalize
            else:
                press_val = 0.4  # Default normalized pressure
            
            if i < len(prev_saturation) and j < len(prev_saturation[i]) and k < len(prev_saturation[i][j]):
                sat_val = prev_saturation[i][j][k]
            else:
                sat_val = 0.8  # Default saturation
            
            # Use original node features but update pressure and saturation
            if node_idx < len(self.graph['node_features']):
                node_feat = self.graph['node_features'][node_idx][:]
                node_feat[0] = press_val  # Update pressure
                node_feat[1] = sat_val    # Update saturation
            else:
                # Create default feature vector
                node_feat = [press_val, sat_val, 0.2, 5.0, 5.0, 3.0, float(i)/nx, float(j)/ny, float(k)/nz]
            
            updated_node_features.append(node_feat)
        
        # Forward pass through GNN
        saturation_output = self.gnn_model.forward(updated_node_features, self.graph['edge_index'])
        
        # Extract saturation values (clamp to [0,1])
        saturation_values = []
        for node_output in saturation_output:
            sat_val = max(0.0, min(1.0, node_output[0]))  # Clamp to valid range
            saturation_values.append(sat_val)
        
        return saturation_values
    
    def predict_well_properties(self, pressure: List[List[List[float]]], 
                               saturation: List[List[List[float]]], 
                               timestep: int) -> Dict[str, List[float]]:
        """
        Use well model to predict production rates and BHP
        """
        well_predictions = {}
        well_connections = self.features.get('well_connections', {})
        well_controls = self.features.get('well_controls', {})
        
        for well_name in well_connections.keys():
            if well_name in well_controls:
                connections = well_connections[well_name]
                controls = well_controls[well_name]
                
                if connections:
                    # Average pressure and saturation over perforated cells
                    avg_pressure = 0.0
                    avg_saturation = 0.0
                    
                    for conn in connections:
                        i, j, k = conn['cell']
                        i, j, k = i-1, j-1, k-1  # Convert to 0-based
                        
                        if (0 <= i < len(pressure) and 0 <= j < len(pressure[i]) and 
                            0 <= k < len(pressure[i][j])):
                            avg_pressure += pressure[i][j][k]
                            
                        if (0 <= i < len(saturation) and 0 <= j < len(saturation[i]) and 
                            0 <= k < len(saturation[i][j])):
                            avg_saturation += saturation[i][j][k]
                    
                    avg_pressure /= len(connections)
                    avg_saturation /= len(connections)
                    
                    # Create input features for well model
                    well_input = [
                        avg_pressure / 5000.0,  # Normalized pressure
                        avg_saturation,         # Saturation
                        controls['bottom_hole_ref_depth'] / 10000.0,  # Normalized depth
                        1.0 if controls['is_producer'] else 0.0,     # Producer flag
                        1.0 if controls['has_rate_constraint'] else 0.0,  # Rate constraint
                        1.0 if controls['has_pressure_constraint'] else 0.0,  # Pressure constraint
                        float(len(connections)) / 10.0,  # Number of connections
                        float(timestep) / 100.0,  # Normalized timestep
                        math.sin(timestep * 0.1),  # Seasonal variation
                        math.cos(timestep * 0.1)   # Seasonal variation
                    ]
                    
                    # Predict well properties
                    well_output = self.well_model.forward(well_input)
                    
                    # Interpret outputs
                    oil_rate = well_output[0] * 1000.0 if len(well_output) > 0 else 0.0
                    water_rate = well_output[1] * 500.0 if len(well_output) > 1 else 0.0
                    bhp = well_output[2] * 1000.0 + 2000.0 if len(well_output) > 2 else 2000.0
                    
                    well_predictions[well_name] = {
                        'oil_production_rate': abs(oil_rate) if controls['is_producer'] else 0.0,
                        'water_production_rate': abs(water_rate) if controls['is_producer'] else 0.0,
                        'water_injection_rate': abs(water_rate) if not controls['is_producer'] else 0.0,
                        'bottom_hole_pressure': bhp
                    }
        
        return well_predictions
    
    def run_coupled_simulation(self, num_timesteps: int = 10) -> Dict:
        """
        Run the coupled GNN-FNO simulation workflow
        """
        print(f"Starting coupled simulation for {num_timesteps} timesteps...")
        
        # Initialize if not done
        if self.features is None:
            self.load_data()
        if self.gnn_model is None:
            self.initialize_models()
        
        # Storage for results
        results = {
            'timesteps': [],
            'pressure_fields': [],
            'saturation_fields': [],
            'well_predictions': []
        }
        
        # Main simulation loop
        for t in range(num_timesteps):
            print(f"\n--- Timestep {t+1}/{num_timesteps} ---")
            
            # Step 1: Use FNO to predict pressure at t+1
            next_pressure = self.predict_pressure(self.current_saturation, t)
            
            # Step 2: Use GNN to predict saturation at t+1 using predicted pressure
            saturation_values = self.predict_saturation(next_pressure, self.current_saturation)
            
            # Convert saturation values back to 3D field
            grid_dims = self.features['grid_dims']
            nx, ny, nz = grid_dims
            next_saturation = []
            
            idx = 0
            for i in range(nx):
                layer_x = []
                for j in range(ny):
                    layer_y = []
                    for k in range(nz):
                        if idx < len(saturation_values):
                            layer_y.append(saturation_values[idx])
                        else:
                            layer_y.append(self.current_saturation[i][j][k])
                        idx += 1
                    layer_x.append(layer_y)
                next_saturation.append(layer_x)
            
            # Step 3: Predict well properties
            well_predictions = self.predict_well_properties(next_pressure, next_saturation, t)
            
            # Update state
            self.current_pressure = next_pressure
            self.current_saturation = next_saturation
            self.timestep = t + 1
            
            # Store results
            results['timesteps'].append(t + 1)
            results['pressure_fields'].append(self.get_field_min_max(self.current_pressure))
            results['saturation_fields'].append(self.get_field_min_max(self.current_saturation))
            results['well_predictions'].append(well_predictions)
            
            # Print progress
            print(f"  Pressure range: {self.get_field_min_max(self.current_pressure)}")
            print(f"  Saturation range: {self.get_field_min_max(self.current_saturation)}")
            print(f"  Well predictions: {len(well_predictions)} wells")
            
            for well_name, pred in well_predictions.items():
                print(f"    {well_name}: Oil={pred.get('oil_production_rate', 0):.1f} STB/d, "
                      f"BHP={pred.get('bottom_hole_pressure', 0):.1f} psi")
        
        print(f"\nSimulation completed successfully!")
        return results

def test_coupling_workflow():
    """Test the coupling workflow"""
    print("=== Testing GNN-FNO Coupling Workflow ===")
    
    # Initialize workflow
    workflow = ReservoirSimulationWorkflow("HM", "/workspace/HM")
    
    # Run simulation with fewer timesteps for testing
    results = workflow.run_coupled_simulation(num_timesteps=2)
    
    print(f"\nSimulation Results Summary:")
    print(f"  Total timesteps: {len(results['timesteps'])}")
    print(f"  Pressure evolution: {results['pressure_fields']}")
    print(f"  Saturation evolution: {results['saturation_fields']}")
    print(f"  Well predictions per timestep: {[len(wp) for wp in results['well_predictions']]}")
    
    return results

if __name__ == "__main__":
    test_coupling_workflow()