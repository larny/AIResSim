"""
Feature Extraction Module for Reservoir Simulation ML
Processes reservoir properties into features for GNN and FNO models
"""

import struct
import os
import math
from typing import Dict, List, Tuple, Optional
from data_parser import ReservoirDataParser, SimpleArray

class FeatureExtractor:
    """Extract and process features for ML models"""
    
    def __init__(self, case_name: str, data_dir: str):
        self.parser = ReservoirDataParser(case_name, data_dir)
        self.case_name = case_name
        self.data_dir = data_dir
        
    def extract_grid_features(self) -> Dict[str, SimpleArray]:
        """
        Extract static grid features: permeabilities, porosity, coordinates
        """
        features = {}
        
        print("Extracting permeability fields...")
        try:
            # Read permeability files
            perm_files = [
                (f"{self.case_name}_PERM_I.GSG", "perm_x"),
                (f"{self.case_name}_PERM_J.GSG", "perm_y"), 
                (f"{self.case_name}_PERM_K.GSG", "perm_z")
            ]
            
            for filename, key in perm_files:
                filepath = os.path.join(self.data_dir, filename)
                if os.path.exists(filepath):
                    # Simple binary read for GSG files
                    features[key] = self.read_gsg_simple(filepath)
                    print(f"  Loaded {key}: {len(features[key])} values")
                else:
                    print(f"  Warning: {filename} not found")
                    
        except Exception as e:
            print(f"Error reading permeability: {e}")
        
        print("Extracting porosity...")
        try:
            porosity_file = f"{self.case_name}_POROSITY.GSG"
            filepath = os.path.join(self.data_dir, porosity_file)
            if os.path.exists(filepath):
                features['porosity'] = self.read_gsg_simple(filepath)
                print(f"  Loaded porosity: {len(features['porosity'])} values")
        except Exception as e:
            print(f"Error reading porosity: {e}")
        
        print("Extracting grid coordinates...")
        try:
            coord_file = f"{self.case_name}.GSG"
            filepath = os.path.join(self.data_dir, coord_file)
            if os.path.exists(filepath):
                features['coordinates'] = self.read_gsg_simple(filepath)
                print(f"  Loaded coordinates: {len(features['coordinates'])} values")
        except Exception as e:
            print(f"Error reading coordinates: {e}")
            
        return features
    
    def read_gsg_simple(self, filepath: str) -> SimpleArray:
        """
        Simplified GSG file reader
        """
        data = []
        with open(filepath, 'rb') as f:
            # Skip header (first few bytes)
            f.read(100)  # Skip header
            
            try:
                while True:
                    # Try to read float values
                    bytes_data = f.read(4)
                    if len(bytes_data) < 4:
                        break
                    value = struct.unpack('<f', bytes_data)[0]
                    if not (math.isnan(value) or math.isinf(value)):
                        data.append(value)
            except:
                pass
                
        return SimpleArray(data)
    
    def extract_initial_conditions(self) -> Dict[str, SimpleArray]:
        """
        Extract initial pressure and saturation from INIT file
        """
        features = {}
        
        print("Extracting initial conditions...")
        init_file = os.path.join(self.data_dir, f"{self.case_name}.INIT")
        
        if os.path.exists(init_file):
            # Simple binary read approach
            with open(init_file, 'rb') as f:
                # Skip header
                f.read(1000)
                
                pressure_data = []
                saturation_data = []
                
                try:
                    while True:
                        bytes_data = f.read(4)
                        if len(bytes_data) < 4:
                            break
                        value = struct.unpack('<f', bytes_data)[0]
                        if not (math.isnan(value) or math.isinf(value)):
                            # Heuristic: pressure values are typically > 1000, saturation < 1
                            if value > 100:
                                pressure_data.append(value)
                            elif 0 <= value <= 1:
                                saturation_data.append(value)
                except:
                    pass
            
            if pressure_data:
                features['initial_pressure'] = SimpleArray(pressure_data)
                print(f"  Loaded initial pressure: {len(pressure_data)} values")
            if saturation_data:
                features['initial_saturation'] = SimpleArray(saturation_data)
                print(f"  Loaded initial saturation: {len(saturation_data)} values")
        
        return features
    
    def extract_well_features(self) -> Tuple[Dict, Dict]:
        """
        Extract well connection and control features
        """
        print("Extracting well features...")
        
        # Get well connections and controls
        well_connections = self.parser.parse_well_connections()
        well_controls = self.parser.parse_well_controls()
        
        # Process well features for ML
        processed_connections = {}
        processed_controls = {}
        
        for well_name, connections in well_connections.items():
            if connections:
                # Calculate Phase PI for each connection
                processed_connections[well_name] = []
                for conn in connections:
                    # Calculate Phase PI based on permeability thickness and transmissibility
                    phase_pi = self.calculate_phase_pi(conn)
                    
                    processed_conn = {
                        'cell': conn['cell'],
                        'phase_pi': phase_pi,
                        'wellbore_radius': conn['wellbore_radius'],
                        'skin': conn['skin'],
                        'transmissibility': conn['transmissibility']
                    }
                    processed_connections[well_name].append(processed_conn)
        
        # Process well controls
        for well_name, controls in well_controls.items():
            processed_controls[well_name] = {
                'type': controls['type'],
                'is_producer': controls['type'] == 'PRODUCER',
                'bottom_hole_ref_depth': controls['bottom_hole_ref_depth'],
                'has_rate_constraint': 'LIQUID_PRODUCTION_RATE' in controls['constraints'],
                'has_pressure_constraint': 'BOTTOM_HOLE_PRESSURE' in controls['constraints']
            }
        
        return processed_connections, processed_controls
    
    def calculate_phase_pi(self, connection: Dict) -> float:
        """
        Calculate Phase PI (Productivity Index) for a well connection
        PI = (Permeability * Thickness) / (ln(re/rw) + Skin)
        """
        perm_thickness = connection['permeability_thickness']
        wellbore_radius = connection['wellbore_radius'] 
        pressure_equiv_radius = connection['pressure_equivalent_radius']
        skin = connection['skin']
        
        if wellbore_radius > 0 and pressure_equiv_radius > wellbore_radius:
            ln_term = math.log(pressure_equiv_radius / wellbore_radius)
            phase_pi = perm_thickness / (ln_term + skin)
        else:
            phase_pi = perm_thickness  # Fallback
            
        return phase_pi
    
    def create_3d_field(self, values: SimpleArray, grid_dims: Tuple[int, int, int]) -> List[List[List[float]]]:
        """
        Convert 1D array to 3D grid field
        """
        nx, ny, nz = grid_dims
        field_3d = []
        
        idx = 0
        for i in range(nx):
            layer_x = []
            for j in range(ny):
                layer_y = []
                for k in range(nz):
                    if idx < len(values):
                        layer_y.append(values[idx])
                    else:
                        layer_y.append(0.0)
                    idx += 1
                layer_x.append(layer_y)
            field_3d.append(layer_x)
        
        return field_3d
    
    def create_well_control_fields(self, well_controls: Dict, grid_dims: Tuple[int, int, int]) -> Dict[str, List[List[List[float]]]]:
        """
        Create 3D fields for well controls with logarithmic transformation
        """
        nx, ny, nz = grid_dims
        
        # Initialize control fields
        control_fields = {
            'bottom_hole_pressure': [[[0.0 for _ in range(nz)] for _ in range(ny)] for _ in range(nx)],
            'liquid_production_rate': [[[0.0 for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]
        }
        
        # Get well connections to map controls to grid cells
        well_connections = self.parser.parse_well_connections()
        
        for well_name, controls in well_controls.items():
            if well_name in well_connections:
                connections = well_connections[well_name]
                
                for conn in connections:
                    i, j, k = conn['cell']
                    # Convert to 0-based indexing
                    i, j, k = i-1, j-1, k-1
                    
                    if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                        # Apply logarithmic transformation for better performance
                        if controls['has_pressure_constraint']:
                            # Default BHP value for producers
                            bhp_value = 2000.0  # psi
                            control_fields['bottom_hole_pressure'][i][j][k] = math.log(bhp_value + 1)
                        
                        if controls['has_rate_constraint']:
                            # Default production rate
                            rate_value = 8000.0  # STB/day (from constraints)
                            # Negative for producers, positive for injectors
                            sign = -1 if controls['is_producer'] else 1
                            control_fields['liquid_production_rate'][i][j][k] = math.log(abs(rate_value) + 1) * sign
        
        return control_fields
    
    def extract_all_features(self) -> Dict:
        """
        Extract all features needed for ML models
        """
        print("=== Feature Extraction Started ===")
        
        features = {}
        
        # 1. Grid properties
        grid_features = self.extract_grid_features()
        features.update(grid_features)
        
        # 2. Initial conditions
        initial_features = self.extract_initial_conditions()
        features.update(initial_features)
        
        # 3. Well features
        well_connections, well_controls = self.extract_well_features()
        features['well_connections'] = well_connections
        features['well_controls'] = well_controls
        
        # 4. Grid dimensions
        grid_dims = self.parser.get_grid_dimensions()
        features['grid_dims'] = grid_dims
        
        # 5. Well control fields
        control_fields = self.create_well_control_fields(well_controls, grid_dims)
        features['control_fields'] = control_fields
        
        print("=== Feature Extraction Completed ===")
        return features

def test_feature_extraction():
    """Test feature extraction"""
    extractor = FeatureExtractor("HM", "/workspace/HM")
    
    try:
        features = extractor.extract_all_features()
        
        print(f"\nExtracted features:")
        for key, value in features.items():
            if isinstance(value, SimpleArray):
                print(f"  {key}: {len(value)} values")
            elif isinstance(value, dict):
                print(f"  {key}: {len(value)} items")
            elif isinstance(value, list):
                print(f"  {key}: {len(value)} dimensions")
            else:
                print(f"  {key}: {value}")
        
        return features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_feature_extraction()