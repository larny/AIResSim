"""
Graph Construction Module for Reservoir Simulation
Builds graph structure with cells as nodes and connections as edges
Computes harmonic mean permeabilities for edge features
"""

import math
from typing import Dict, List, Tuple, Optional, Set
from data_parser import SimpleArray
from feature_extractor import FeatureExtractor

class GraphConstructor:
    """Construct graph representation of reservoir for GNN"""
    
    def __init__(self, features: Dict):
        self.features = features
        self.grid_dims = features['grid_dims']
        self.nx, self.ny, self.nz = self.grid_dims
        
    def get_cell_index(self, i: int, j: int, k: int) -> int:
        """Convert 3D coordinates to 1D cell index"""
        return i * (self.ny * self.nz) + j * self.nz + k
    
    def get_3d_coords(self, cell_idx: int) -> Tuple[int, int, int]:
        """Convert 1D cell index to 3D coordinates"""
        k = cell_idx % self.nz
        j = (cell_idx // self.nz) % self.ny
        i = cell_idx // (self.ny * self.nz)
        return i, j, k
    
    def get_neighbors(self, i: int, j: int, k: int) -> List[Tuple[int, int, int]]:
        """Get 6-connected neighbors of a cell"""
        neighbors = []
        
        # Check all 6 directions (±x, ±y, ±z)
        directions = [
            (-1, 0, 0), (1, 0, 0),  # x direction
            (0, -1, 0), (0, 1, 0),  # y direction
            (0, 0, -1), (0, 0, 1)   # z direction
        ]
        
        for di, dj, dk in directions:
            ni, nj, nk = i + di, j + dj, k + dk
            
            # Check bounds
            if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                neighbors.append((ni, nj, nk))
        
        return neighbors
    
    def compute_harmonic_mean_permeability(self, cell1: Tuple[int, int, int], 
                                         cell2: Tuple[int, int, int]) -> float:
        """
        Compute harmonic mean permeability between two cells
        Weighted by distance in each direction
        """
        i1, j1, k1 = cell1
        i2, j2, k2 = cell2
        
        # Get cell indices
        idx1 = self.get_cell_index(i1, j1, k1)
        idx2 = self.get_cell_index(i2, j2, k2)
        
        # Get permeabilities for both cells
        perm_x = self.features.get('perm_x', SimpleArray([1.0] * (self.nx * self.ny * self.nz)))
        perm_y = self.features.get('perm_y', SimpleArray([1.0] * (self.nx * self.ny * self.nz)))
        perm_z = self.features.get('perm_z', SimpleArray([1.0] * (self.nx * self.ny * self.nz)))
        
        # Handle index bounds
        if idx1 >= len(perm_x) or idx2 >= len(perm_x):
            return 1.0  # Default permeability
        
        # Get permeability values
        kx1, ky1, kz1 = perm_x[idx1], perm_y[idx1], perm_z[idx1]
        kx2, ky2, kz2 = perm_x[idx2], perm_y[idx2], perm_z[idx2]
        
        # Determine flow direction
        if abs(i2 - i1) == 1:  # x-direction flow
            k1_eff, k2_eff = kx1, kx2
            distance = abs(i2 - i1)
        elif abs(j2 - j1) == 1:  # y-direction flow
            k1_eff, k2_eff = ky1, ky2
            distance = abs(j2 - j1)
        elif abs(k2 - k1) == 1:  # z-direction flow
            k1_eff, k2_eff = kz1, kz2
            distance = abs(k2 - k1)
        else:
            # Diagonal connection - use geometric mean of all directions
            k1_eff = (kx1 * ky1 * kz1) ** (1/3)
            k2_eff = (kx2 * ky2 * kz2) ** (1/3)
            distance = math.sqrt((i2-i1)**2 + (j2-j1)**2 + (k2-k1)**2)
        
        # Compute harmonic mean weighted by distance
        if k1_eff > 0 and k2_eff > 0:
            harmonic_mean = 2 * k1_eff * k2_eff / (k1_eff + k2_eff)
        else:
            harmonic_mean = max(k1_eff, k2_eff)
        
        # Weight by inverse distance
        if distance > 0:
            weighted_perm = harmonic_mean / distance
        else:
            weighted_perm = harmonic_mean
        
        return weighted_perm
    
    def build_node_features(self) -> List[List[float]]:
        """
        Build node features for each cell
        Features: [pressure, saturation, porosity, perm_x, perm_y, perm_z, x_coord, y_coord, z_coord]
        """
        total_cells = self.nx * self.ny * self.nz
        node_features = []
        
        # Get feature arrays
        pressure = self.features.get('initial_pressure', SimpleArray([2000.0] * total_cells))
        saturation = self.features.get('initial_saturation', SimpleArray([0.8] * total_cells))
        porosity = self.features.get('porosity', SimpleArray([0.2] * total_cells))
        perm_x = self.features.get('perm_x', SimpleArray([100.0] * total_cells))
        perm_y = self.features.get('perm_y', SimpleArray([100.0] * total_cells))
        perm_z = self.features.get('perm_z', SimpleArray([10.0] * total_cells))
        coordinates = self.features.get('coordinates', SimpleArray([0.0] * (total_cells * 3)))
        
        print(f"Building node features for {total_cells} cells...")
        
        for cell_idx in range(total_cells):
            i, j, k = self.get_3d_coords(cell_idx)
            
            # Get feature values (with bounds checking)
            press_val = pressure[cell_idx] if cell_idx < len(pressure) else 2000.0
            sat_val = saturation[cell_idx] if cell_idx < len(saturation) else 0.8
            poro_val = porosity[cell_idx] if cell_idx < len(porosity) else 0.2
            kx_val = perm_x[cell_idx] if cell_idx < len(perm_x) else 100.0
            ky_val = perm_y[cell_idx] if cell_idx < len(perm_y) else 100.0
            kz_val = perm_z[cell_idx] if cell_idx < len(perm_z) else 10.0
            
            # Get coordinates (x, y, z for each cell)
            coord_idx = cell_idx * 3
            x_coord = coordinates[coord_idx] if coord_idx < len(coordinates) else float(i)
            y_coord = coordinates[coord_idx + 1] if coord_idx + 1 < len(coordinates) else float(j)
            z_coord = coordinates[coord_idx + 2] if coord_idx + 2 < len(coordinates) else float(k)
            
            # Normalize features
            node_feature = [
                press_val / 5000.0,  # Normalize pressure
                sat_val,  # Saturation already in [0,1]
                poro_val,  # Porosity already in [0,1]
                math.log(kx_val + 1) / 10.0,  # Log-normalize permeability
                math.log(ky_val + 1) / 10.0,
                math.log(kz_val + 1) / 10.0,
                x_coord / 1000.0,  # Normalize coordinates
                y_coord / 1000.0,
                z_coord / 1000.0
            ]
            
            node_features.append(node_feature)
        
        print(f"Created {len(node_features)} node features with {len(node_features[0])} dimensions each")
        return node_features
    
    def build_edges(self) -> Tuple[List[Tuple[int, int]], List[List[float]]]:
        """
        Build edge list and edge features
        Returns: (edge_list, edge_features)
        """
        edges = []
        edge_features = []
        
        print("Building graph edges...")
        
        # Build regular grid connections
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    cell_idx = self.get_cell_index(i, j, k)
                    neighbors = self.get_neighbors(i, j, k)
                    
                    for ni, nj, nk in neighbors:
                        neighbor_idx = self.get_cell_index(ni, nj, nk)
                        
                        # Add bidirectional edges
                        edges.append((cell_idx, neighbor_idx))
                        
                        # Compute edge features
                        harmonic_perm = self.compute_harmonic_mean_permeability(
                            (i, j, k), (ni, nj, nk)
                        )
                        
                        # Distance between cells
                        distance = math.sqrt((ni-i)**2 + (nj-j)**2 + (nk-k)**2)
                        
                        # Edge features: [harmonic_permeability, distance, direction_x, direction_y, direction_z]
                        direction_x = (ni - i) / max(distance, 1e-6)
                        direction_y = (nj - j) / max(distance, 1e-6)
                        direction_z = (nk - k) / max(distance, 1e-6)
                        
                        edge_feature = [
                            math.log(harmonic_perm + 1) / 10.0,  # Normalized log permeability
                            distance,  # Distance
                            direction_x,  # Direction vector
                            direction_y,
                            direction_z
                        ]
                        
                        edge_features.append(edge_feature)
        
        # Add well connections as additional edges
        well_connections = self.features.get('well_connections', {})
        
        print("Adding well connection edges...")
        for well_name, connections in well_connections.items():
            if len(connections) > 1:
                # Connect all perforated cells within the same well
                for i, conn1 in enumerate(connections):
                    for j, conn2 in enumerate(connections):
                        if i != j:
                            cell1 = conn1['cell']
                            cell2 = conn2['cell']
                            
                            # Convert to 0-based indexing
                            i1, j1, k1 = cell1[0]-1, cell1[1]-1, cell1[2]-1
                            i2, j2, k2 = cell2[0]-1, cell2[1]-1, cell2[2]-1
                            
                            # Check bounds
                            if (0 <= i1 < self.nx and 0 <= j1 < self.ny and 0 <= k1 < self.nz and
                                0 <= i2 < self.nx and 0 <= j2 < self.ny and 0 <= k2 < self.nz):
                                
                                cell_idx1 = self.get_cell_index(i1, j1, k1)
                                cell_idx2 = self.get_cell_index(i2, j2, k2)
                                
                                edges.append((cell_idx1, cell_idx2))
                                
                                # Well connection edge features
                                phase_pi1 = conn1.get('phase_pi', 1.0)
                                phase_pi2 = conn2.get('phase_pi', 1.0)
                                avg_phase_pi = (phase_pi1 + phase_pi2) / 2.0
                                
                                distance = math.sqrt((i2-i1)**2 + (j2-j1)**2 + (k2-k1)**2)
                                
                                edge_feature = [
                                    math.log(avg_phase_pi + 1) / 10.0,  # Well PI
                                    distance,
                                    (i2-i1) / max(distance, 1e-6),
                                    (j2-j1) / max(distance, 1e-6),
                                    (k2-k1) / max(distance, 1e-6)
                                ]
                                
                                edge_features.append(edge_feature)
        
        print(f"Created {len(edges)} edges with {len(edge_features[0])} features each")
        return edges, edge_features
    
    def build_graph(self) -> Dict:
        """
        Build complete graph representation
        """
        print("=== Building Graph Representation ===")
        
        # Build node features
        node_features = self.build_node_features()
        
        # Build edges
        edge_list, edge_features = self.build_edges()
        
        # Convert edge list to source/target format
        edge_index = [[], []]
        for src, tgt in edge_list:
            edge_index[0].append(src)
            edge_index[1].append(tgt)
        
        graph = {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'num_nodes': len(node_features),
            'num_edges': len(edge_list),
            'grid_dims': self.grid_dims
        }
        
        print(f"Graph created: {graph['num_nodes']} nodes, {graph['num_edges']} edges")
        print("=== Graph Construction Completed ===")
        
        return graph

def test_graph_construction():
    """Test graph construction"""
    from feature_extractor import FeatureExtractor
    
    print("Testing graph construction...")
    
    # Extract features first
    extractor = FeatureExtractor("HM", "/workspace/HM")
    features = extractor.extract_all_features()
    
    if features:
        # Build graph
        constructor = GraphConstructor(features)
        graph = constructor.build_graph()
        
        print(f"\nGraph statistics:")
        print(f"  Nodes: {graph['num_nodes']}")
        print(f"  Edges: {graph['num_edges']}")
        print(f"  Node feature dim: {len(graph['node_features'][0])}")
        print(f"  Edge feature dim: {len(graph['edge_features'][0])}")
        print(f"  Grid dimensions: {graph['grid_dims']}")
        
        # Show sample features
        print(f"\nSample node features (first node): {graph['node_features'][0]}")
        print(f"Sample edge features (first edge): {graph['edge_features'][0]}")
        
        return graph
    else:
        print("Failed to extract features")
        return None

if __name__ == "__main__":
    test_graph_construction()