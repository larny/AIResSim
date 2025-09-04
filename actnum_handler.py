"""
ACTNUM Handler for Active Cell Mapping
Handles the mapping between grid coordinates and active cells
"""

import struct
import os
from typing import Dict, List, Tuple, Optional, Set
from data_parser import SimpleArray

class ACTNUMHandler:
    """Handle active cell mapping using ACTNUM data"""
    
    def __init__(self, case_name: str, data_dir: str):
        self.case_name = case_name
        self.data_dir = data_dir
        self.actnum = None
        self.active_cells = []  # List of (i,j,k) coordinates for active cells
        self.grid_to_active = {}  # Map (i,j,k) -> active_index
        self.active_to_grid = {}  # Map active_index -> (i,j,k)
        self.grid_dims = None
        
    def load_actnum(self, grid_dims: Tuple[int, int, int]) -> SimpleArray:
        """
        Load ACTNUM data from INIT file or create default
        ACTNUM: 1 for active cells, 0 for inactive cells
        """
        self.grid_dims = grid_dims
        nx, ny, nz = grid_dims
        total_cells = nx * ny * nz
        
        # Try to read ACTNUM from INIT file
        init_file = os.path.join(self.data_dir, f"{self.case_name}.INIT")
        
        if os.path.exists(init_file):
            print("Reading ACTNUM from INIT file...")
            actnum_data = self.read_actnum_from_init(init_file, total_cells)
        else:
            print("INIT file not found, creating default ACTNUM...")
            # Default: assume all cells are active except boundary cells
            actnum_data = []
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        # Make boundary cells inactive for more realistic reservoir
                        if (i == 0 or i == nx-1 or j == 0 or j == ny-1 or 
                            k == 0 or k == nz-1):
                            actnum_data.append(0)
                        else:
                            actnum_data.append(1)
        
        self.actnum = SimpleArray(actnum_data)
        self.build_active_cell_mapping()
        
        return self.actnum
    
    def read_actnum_from_init(self, filepath: str, expected_size: int) -> List[int]:
        """Read ACTNUM from INIT file"""
        actnum_data = []
        
        try:
            with open(filepath, 'rb') as f:
                # Skip header
                f.read(1000)
                
                # Look for integer data (ACTNUM is typically integer)
                while len(actnum_data) < expected_size:
                    try:
                        bytes_data = f.read(4)
                        if len(bytes_data) < 4:
                            break
                        
                        # Try as integer first
                        value = struct.unpack('<i', bytes_data)[0]
                        if value in [0, 1]:  # Valid ACTNUM values
                            actnum_data.append(value)
                        else:
                            # Try as float and convert
                            f.seek(-4, 1)  # Go back
                            bytes_data = f.read(4)
                            float_val = struct.unpack('<f', bytes_data)[0]
                            if 0.0 <= float_val <= 1.0:
                                actnum_data.append(int(round(float_val)))
                    except:
                        continue
                        
        except Exception as e:
            print(f"Error reading ACTNUM: {e}")
        
        # If we didn't get enough data, fill with defaults
        while len(actnum_data) < expected_size:
            actnum_data.append(1)  # Default to active
        
        return actnum_data[:expected_size]
    
    def build_active_cell_mapping(self):
        """Build mapping between grid coordinates and active cell indices"""
        if not self.actnum or not self.grid_dims:
            return
        
        nx, ny, nz = self.grid_dims
        active_index = 0
        
        self.active_cells = []
        self.grid_to_active = {}
        self.active_to_grid = {}
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    cell_index = i * (ny * nz) + j * nz + k
                    
                    if cell_index < len(self.actnum) and self.actnum[cell_index] == 1:
                        # This cell is active
                        self.active_cells.append((i, j, k))
                        self.grid_to_active[(i, j, k)] = active_index
                        self.active_to_grid[active_index] = (i, j, k)
                        active_index += 1
        
        print(f"Found {len(self.active_cells)} active cells out of {nx*ny*nz} total cells")
    
    def is_active(self, i: int, j: int, k: int) -> bool:
        """Check if a grid cell is active"""
        return (i, j, k) in self.grid_to_active
    
    def get_active_index(self, i: int, j: int, k: int) -> Optional[int]:
        """Get active cell index for grid coordinates"""
        return self.grid_to_active.get((i, j, k))
    
    def get_grid_coords(self, active_index: int) -> Optional[Tuple[int, int, int]]:
        """Get grid coordinates for active cell index"""
        return self.active_to_grid.get(active_index)
    
    def get_active_neighbors(self, active_index: int) -> List[int]:
        """Get active neighbors of an active cell"""
        coords = self.get_grid_coords(active_index)
        if not coords:
            return []
        
        i, j, k = coords
        neighbors = []
        
        # Check 6-connected neighbors
        for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            ni, nj, nk = i + di, j + dj, k + dk
            neighbor_active_idx = self.get_active_index(ni, nj, nk)
            if neighbor_active_idx is not None:
                neighbors.append(neighbor_active_idx)
        
        return neighbors
    
    def map_property_to_active_cells(self, property_data: SimpleArray) -> SimpleArray:
        """Map a property from all cells to only active cells"""
        if not self.active_cells:
            return property_data
        
        active_property = []
        nx, ny, nz = self.grid_dims
        
        for i, j, k in self.active_cells:
            cell_index = i * (ny * nz) + j * nz + k
            if cell_index < len(property_data):
                active_property.append(property_data[cell_index])
            else:
                active_property.append(0.0)  # Default value
        
        return SimpleArray(active_property)
    
    def get_active_cell_count(self) -> int:
        """Get number of active cells"""
        return len(self.active_cells)

def test_actnum_handler():
    """Test ACTNUM handler"""
    print("=== Testing ACTNUM Handler ===")
    
    handler = ACTNUMHandler("HM", "/workspace/HM")
    
    # Test with default grid dimensions
    grid_dims = (30, 30, 15)
    actnum = handler.load_actnum(grid_dims)
    
    print(f"Grid dimensions: {grid_dims}")
    print(f"Total cells: {grid_dims[0] * grid_dims[1] * grid_dims[2]}")
    print(f"Active cells: {handler.get_active_cell_count()}")
    print(f"Activity ratio: {handler.get_active_cell_count() / (grid_dims[0] * grid_dims[1] * grid_dims[2]):.3f}")
    
    # Test neighbor finding
    if handler.get_active_cell_count() > 0:
        test_active_idx = 0
        neighbors = handler.get_active_neighbors(test_active_idx)
        coords = handler.get_grid_coords(test_active_idx)
        print(f"Active cell {test_active_idx} at {coords} has {len(neighbors)} active neighbors")
    
    return handler

if __name__ == "__main__":
    test_actnum_handler()