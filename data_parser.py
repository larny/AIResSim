"""
Data Parser for Reservoir Simulation Files
Handles binary formats: INIT, GSG, UNRST, UNSMRY files
Based on Eclipse/Petrel reservoir simulation output formats
"""

import struct
import os
from typing import Dict, List, Tuple, Optional, Union

# Simplified numpy-like operations for basic arrays
class SimpleArray:
    def __init__(self, data):
        self.data = list(data) if not isinstance(data, list) else data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def reshape(self, shape):
        # Simple reshape for 3D grids
        return self.data
    
    def mean(self):
        return sum(self.data) / len(self.data)
    
    def std(self):
        mean_val = self.mean()
        variance = sum((x - mean_val) ** 2 for x in self.data) / len(self.data)
        return variance ** 0.5

class ReservoirDataParser:
    """Parser for reservoir simulation binary files"""
    
    def __init__(self, case_name: str, data_dir: str = "."):
        """
        Initialize parser for a specific case
        
        Args:
            case_name: Name of the simulation case (e.g., 'HM')
            data_dir: Directory containing the data files
        """
        self.case_name = case_name
        self.data_dir = data_dir
        self.grid_dims = None  # Will be set when grid is loaded
        
    def read_binary_record(self, file_handle, dtype='f', count=1):
        """Read a binary record with Fortran unformatted structure"""
        # Read record length
        length_bytes = file_handle.read(4)
        if len(length_bytes) < 4:
            return None
        
        record_length = struct.unpack('<I', length_bytes)[0]
        
        # Read the actual data
        if dtype == 'f':  # float32
            data_bytes = file_handle.read(record_length)
            data = struct.unpack(f'<{record_length//4}f', data_bytes)
        elif dtype == 'i':  # int32
            data_bytes = file_handle.read(record_length)
            data = struct.unpack(f'<{record_length//4}i', data_bytes)
        elif dtype == 'd':  # float64
            data_bytes = file_handle.read(record_length)
            data = struct.unpack(f'<{record_length//8}d', data_bytes)
        else:
            data_bytes = file_handle.read(record_length)
            data = data_bytes
        
        # Read trailing record length
        file_handle.read(4)
        
        return SimpleArray(data) if isinstance(data, tuple) else data
    
    def read_gsg_file(self, filename: str) -> SimpleArray:
        """
        Read GSG (Grid Static Geometry) binary files
        These contain 3D property fields like permeability, porosity, coordinates
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GSG file not found: {filepath}")
        
        data_list = []
        
        with open(filepath, 'rb') as f:
            # Skip header information
            header = self.read_binary_record(f, dtype='raw')
            
            while True:
                data_record = self.read_binary_record(f, dtype='f')
                if data_record is None:
                    break
                data_list.extend(data_record)
        
        return SimpleArray(data_list)
    
    def read_init_file(self) -> Dict[str, SimpleArray]:
        """
        Read INIT file containing initial pressure and saturation
        """
        filepath = os.path.join(self.data_dir, f"{self.case_name}.INIT")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"INIT file not found: {filepath}")
        
        properties = {}
        
        with open(filepath, 'rb') as f:
            while True:
                # Try to read property header
                header = self.read_binary_record(f, dtype='raw')
                if header is None:
                    break
                
                # Decode header to identify property type
                if b'PRESSURE' in header:
                    pressure_data = self.read_binary_record(f, dtype='f')
                    if pressure_data is not None:
                        properties['pressure'] = pressure_data
                        
                elif b'SGAS' in header or b'SWAT' in header or b'SOIL' in header:
                    saturation_data = self.read_binary_record(f, dtype='f')
                    if saturation_data is not None:
                        if 'saturation' not in properties:
                            properties['saturation'] = []
                        properties['saturation'].append(saturation_data)
        
        # Convert saturation list to array if exists
        if 'saturation' in properties and isinstance(properties['saturation'], list):
            properties['saturation'] = SimpleArray(properties['saturation'])
        
        return properties
    
    def read_unrst_file(self) -> Dict[str, List[SimpleArray]]:
        """
        Read UNRST file containing dynamic 3D property fields over time
        """
        filepath = os.path.join(self.data_dir, f"{self.case_name}.UNRST")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"UNRST file not found: {filepath}")
        
        time_series_data = {
            'pressure': [],
            'saturation': [],
            'timesteps': []
        }
        
        with open(filepath, 'rb') as f:
            timestep = 0
            while True:
                # Read timestep data
                header = self.read_binary_record(f, dtype='raw')
                if header is None:
                    break
                
                if b'PRESSURE' in header:
                    pressure_data = self.read_binary_record(f, dtype='f')
                    if pressure_data is not None:
                        time_series_data['pressure'].append(pressure_data)
                        
                elif b'SGAS' in header or b'SWAT' in header:
                    saturation_data = self.read_binary_record(f, dtype='f')
                    if saturation_data is not None:
                        time_series_data['saturation'].append(saturation_data)
                
                timestep += 1
                if timestep % 100 == 0:  # Progress indicator
                    print(f"Processed timestep {timestep}")
        
        return time_series_data
    
    def read_unsmry_file(self) -> Dict:
        """
        Read UNSMRY file containing well and field production data
        """
        filepath = os.path.join(self.data_dir, f"{self.case_name}.UNSMRY")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"UNSMRY file not found: {filepath}")
        
        # This is a simplified parser - in practice you might use ecl library
        # For now, we'll create a basic structure
        summary_data = {
            'TIME': [],
            'WOPR': {},  # Well Oil Production Rate
            'WWPR': {},  # Well Water Production Rate  
            'WBHP': {},  # Well Bottom Hole Pressure
            'WWIR': {}   # Well Water Injection Rate
        }
        
        return summary_data
    
    def parse_well_connections(self) -> Dict[str, List[Dict]]:
        """
        Parse well connection data from IXF file
        """
        filepath = os.path.join(self.data_dir, f"{self.case_name}_WELL_CONNECTIONS.ixf")
        
        wells_data = {}
        current_well = None
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('WellDef'):
                # Extract well name
                well_name = line.split('"')[1]
                current_well = well_name
                wells_data[well_name] = []
                
            elif 'WellToCellConnections' in line and current_well:
                # Skip to the actual data lines
                i += 1
                # Skip header line if it exists
                if i < len(lines) and 'Cell' in lines[i]:
                    i += 1
                
                # Read connection data
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith('}') or line == '' or line.startswith(']'):
                        break
                    
                    if line.startswith('('):
                        # Parse connection data
                        parts = line.split()
                        if len(parts) >= 11:
                            # Extract cell coordinates from format "(13 8 1)"
                            coord_str = ""
                            for part in parts[:4]:  # Coordinates might span multiple parts
                                coord_str += part + " "
                            
                            # Find the coordinate pattern
                            import re
                            coord_match = re.search(r'\((\d+)\s+(\d+)\s+(\d+)\)', coord_str)
                            if coord_match:
                                cell_coords = (int(coord_match.group(1)), int(coord_match.group(2)), int(coord_match.group(3)))
                                
                                # Find the completion name (quoted string)
                                completion_name = ""
                                for part in parts:
                                    if '"' in part:
                                        completion_name = part.replace('"', '')
                                        break
                                
                                # Extract numeric values from the end of the parts list
                                numeric_parts = []
                                for part in parts:
                                    try:
                                        if '.' in part or part.isdigit():
                                            numeric_parts.append(float(part))
                                    except ValueError:
                                        continue
                                
                                if len(numeric_parts) >= 6:  # Need at least 6 numeric values
                                    connection = {
                                        'cell': cell_coords,
                                        'completion': completion_name,
                                        'segment_node': 1,
                                        'status': 'OPEN',
                                        'measured_depth': numeric_parts[0],
                                        'wellbore_radius': numeric_parts[1],
                                        'skin': numeric_parts[2],
                                        'pi_multiplier': numeric_parts[3],
                                        'pressure_equivalent_radius': numeric_parts[4],
                                        'permeability_thickness': numeric_parts[5],
                                        'transmissibility': numeric_parts[6] if len(numeric_parts) > 6 else 0.0
                                    }
                                    wells_data[current_well].append(connection)
                    
                    i += 1
                continue
            
            i += 1
        
        return wells_data
    
    def parse_well_controls(self) -> Dict[str, Dict]:
        """
        Parse well control data from FM files
        """
        filepath = os.path.join(self.data_dir, f"{self.case_name}_PRED_FM.ixf")
        
        wells_controls = {}
        current_well = None
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Well "'):
                well_name = line.split('"')[1]
                current_well = well_name
                wells_controls[well_name] = {
                    'type': None,
                    'constraints': [],
                    'bottom_hole_ref_depth': None
                }
                
            elif 'Type=' in line and current_well:
                well_type = line.split('=')[1]
                wells_controls[current_well]['type'] = well_type
                
            elif 'BottomHoleRefDepth=' in line and current_well:
                depth = float(line.split('=')[1])
                wells_controls[current_well]['bottom_hole_ref_depth'] = depth
                
            elif 'Constraints=' in line and current_well:
                # Parse constraints - simplified for now
                if 'LIQUID_PRODUCTION_RATE' in line:
                    wells_controls[current_well]['constraints'].append('LIQUID_PRODUCTION_RATE')
                if 'BOTTOM_HOLE_PRESSURE' in line:
                    wells_controls[current_well]['constraints'].append('BOTTOM_HOLE_PRESSURE')
        
        return wells_controls
    
    def get_grid_dimensions(self) -> Tuple[int, int, int]:
        """
        Determine grid dimensions from coordinate file
        """
        if self.grid_dims is not None:
            return self.grid_dims
        
        # Try to read from GSG file or estimate from data
        coord_file = os.path.join(self.data_dir, f"{self.case_name}.GSG")
        if os.path.exists(coord_file):
            # Read coordinate data to determine dimensions
            coords = self.read_gsg_file(f"{self.case_name}.GSG")
            # Estimate dimensions - this is simplified
            # In practice, you'd need to parse the actual grid structure
            total_cells = len(coords) // 3  # Assuming x,y,z coordinates
            # Default assumption for a typical reservoir grid
            nx, ny, nz = 30, 30, 15  # This should be determined from actual data
            self.grid_dims = (nx, ny, nz)
        else:
            # Default grid dimensions
            self.grid_dims = (30, 30, 15)
        
        return self.grid_dims
    
    def load_all_properties(self) -> Dict[str, SimpleArray]:
        """
        Load all required properties for the ML workflow
        """
        properties = {}
        
        print("Loading initial properties...")
        init_data = self.read_init_file()
        properties.update(init_data)
        
        print("Loading permeability fields...")
        try:
            properties['perm_x'] = self.read_gsg_file(f"{self.case_name}_PERM_I.GSG")
            properties['perm_y'] = self.read_gsg_file(f"{self.case_name}_PERM_J.GSG") 
            properties['perm_z'] = self.read_gsg_file(f"{self.case_name}_PERM_K.GSG")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
        
        print("Loading porosity...")
        try:
            properties['porosity'] = self.read_gsg_file(f"{self.case_name}_POROSITY.GSG")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
        
        print("Loading grid coordinates...")
        try:
            properties['coordinates'] = self.read_gsg_file(f"{self.case_name}.GSG")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
        
        print("Loading well data...")
        properties['well_connections'] = self.parse_well_connections()
        properties['well_controls'] = self.parse_well_controls()
        
        print("Loading dynamic data...")
        try:
            properties['dynamic_data'] = self.read_unrst_file()
        except FileNotFoundError as e:
            print(f"Warning: {e}")
        
        return properties

def test_parser():
    """Test the data parser with HM case"""
    parser = ReservoirDataParser("HM", "/workspace/HM")
    
    try:
        # Test individual components
        print("Testing well connections parsing...")
        well_connections = parser.parse_well_connections()
        print(f"Found {len(well_connections)} wells:")
        for well_name, connections in well_connections.items():
            print(f"  {well_name}: {len(connections)} connections")
            if connections:  # Show first connection as example
                print(f"    Example connection: {connections[0]}")
        
        print("\nTesting well controls parsing...")
        well_controls = parser.parse_well_controls()
        print(f"Well controls: {list(well_controls.keys())}")
        for well_name, controls in well_controls.items():
            print(f"  {well_name}: {controls}")
        
        # Test reading a simple text-based property
        print("\nTesting file existence...")
        hm_dir = "/workspace/HM"
        files_to_check = ["HM_WELL_CONNECTIONS.ixf", "HM_PRED_FM.ixf", "HM.GSG"]
        for filename in files_to_check:
            filepath = os.path.join(hm_dir, filename)
            print(f"  {filename}: {'EXISTS' if os.path.exists(filepath) else 'NOT FOUND'}")
        
        return True
    except Exception as e:
        print(f"Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_parser()