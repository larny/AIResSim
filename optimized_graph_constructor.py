"""
Optimized Graph Constructor using ACTNUM for Active Cells Only
Builds graph with only active cells as nodes and realistic edge connections
"""

import math
from typing import Dict, List, Tuple, Optional, Set
from data_parser import SimpleArray
from feature_extractor import FeatureExtractor
from actnum_handler import ACTNUMHandler

class OptimizedGraphConstructor:
    """Construct graph representation using only active reservoir cells"""
    
    def __init__(self, features: Dict):
        self.features = features
        self.grid_dims = features['grid_dims']
        self.nx, self.ny, self.nz = self.grid_dims
        
        # Initialize ACTNUM handler
        case_name = "HM"  # Should be passed as parameter in real implementation
        data_dir = "/workspace/HM"
        self.actnum_handler = ACTNUMHandler(case_name, data_dir)
        self.actnum_handler.load_actnum(self.grid_dims)
        
        print(f"Graph will use {self.actnum_handler.get_active_cell_count()} active cells")
        
    def compute_harmonic_mean_permeability(self, active_idx1: int, active_idx2: int) -> float:
        """
        Compute harmonic mean permeability between two active cells
        """
        coords1 = self.actnum_handler.get_grid_coords(active_idx1)
        coords2 = self.actnum_handler.get_grid_coords(active_idx2)
        
        if not coords1 or not coords2:
            return 1.0
        
        i1, j1, k1 = coords1
        i2, j2, k2 = coords2
        
        # Get permeabilities for both cells
        perm_x = self.features.get('perm_x', SimpleArray([1.0] * (self.nx * self.ny * self.nz)))
        perm_y = self.features.get('perm_y', SimpleArray([1.0] * (self.nx * self.ny * self.nz)))
        perm_z = self.features.get('perm_z', SimpleArray([1.0] * (self.nx * self.ny * self.nz)))
        
        # Convert to grid indices
        grid_idx1 = i1 * (self.ny * self.nz) + j1 * self.nz + k1
        grid_idx2 = i2 * (self.ny * self.nz) + j2 * self.nz + k2
        
        # Handle index bounds
        if grid_idx1 >= len(perm_x) or grid_idx2 >= len(perm_x):
            return 1.0
        
        # Get permeability values
        kx1, ky1, kz1 = perm_x[grid_idx1], perm_y[grid_idx1], perm_z[grid_idx1]
        kx2, ky2, kz2 = perm_x[grid_idx2], perm_y[grid_idx2], perm_z[grid_idx2]
        
        # Determine flow direction and effective permeability
        if abs(i2 - i1) == 1:  # x-direction flow
            k1_eff, k2_eff = kx1, kx2
        elif abs(j2 - j1) == 1:  # y-direction flow
            k1_eff, k2_eff = ky1, ky2
        elif abs(k2 - k1) == 1:  # z-direction flow
            k1_eff, k2_eff = kz1, kz2
        else:
            # Diagonal connection - use geometric mean
            k1_eff = (kx1 * ky1 * kz1) ** (1/3)
            k2_eff = (kx2 * ky2 * kz2) ** (1/3)
        
        # Compute harmonic mean
        if k1_eff > 0 and k2_eff > 0:
            harmonic_mean = 2 * k1_eff * k2_eff / (k1_eff + k2_eff)
        else:
            harmonic_mean = max(k1_eff, k2_eff)
        
        return harmonic_mean
    
    def build_node_features(self) -> List[List[float]]:
        """
        Build node features for active cells only
        Features: [pressure, saturation, porosity, perm_x, perm_y, perm_z, x_coord, y_coord, z_coord]
        """
        active_count = self.actnum_handler.get_active_cell_count()
        node_features = []
        
        # Map properties to active cells
        pressure = self.actnum_handler.map_property_to_active_cells(
            self.features.get('initial_pressure', SimpleArray([2000.0] * (self.nx * self.ny * self.nz)))
        )
        saturation = self.actnum_handler.map_property_to_active_cells(
            self.features.get('initial_saturation', SimpleArray([0.8] * (self.nx * self.ny * self.nz)))
        )
        porosity = self.actnum_handler.map_property_to_active_cells(
            self.features.get('porosity', SimpleArray([0.2] * (self.nx * self.ny * self.nz)))
        )
        perm_x = self.actnum_handler.map_property_to_active_cells(
            self.features.get('perm_x', SimpleArray([100.0] * (self.nx * self.ny * self.nz)))
        )
        perm_y = self.actnum_handler.map_property_to_active_cells(
            self.features.get('perm_y', SimpleArray([100.0] * (self.nx * self.ny * self.nz)))
        )
        perm_z = self.actnum_handler.map_property_to_active_cells(
            self.features.get('perm_z', SimpleArray([10.0] * (self.nx * self.ny * self.nz)))
        )
        
        print(f"Building node features for {active_count} active cells...")
        
        for active_idx in range(active_count):
            coords = self.actnum_handler.get_grid_coords(active_idx)
            if not coords:
                continue
            
            i, j, k = coords
            
            # Get feature values with bounds checking
            press_val = pressure[active_idx] if active_idx < len(pressure) else 2000.0
            sat_val = saturation[active_idx] if active_idx < len(saturation) else 0.8
            poro_val = porosity[active_idx] if active_idx < len(porosity) else 0.2
            kx_val = perm_x[active_idx] if active_idx < len(perm_x) else 100.0
            ky_val = perm_y[active_idx] if active_idx < len(perm_y) else 100.0
            kz_val = perm_z[active_idx] if active_idx < len(perm_z) else 10.0
            
            # Normalize features properly
            node_feature = [
                max(0.0, min(1.0, press_val / 5000.0)),  # Normalize pressure [0,1]
                max(0.0, min(1.0, sat_val)),  # Saturation already in [0,1]
                max(0.0, min(1.0, poro_val)),  # Porosity already in [0,1]
                max(0.0, min(1.0, math.log(max(kx_val, 0.1) + 1) / 10.0)),  # Log-normalize permeability
                max(0.0, min(1.0, math.log(max(ky_val, 0.1) + 1) / 10.0)),
                max(0.0, min(1.0, math.log(max(kz_val, 0.1) + 1) / 10.0)),
                i / float(self.nx),  # Normalized coordinates [0,1]
                j / float(self.ny),
                k / float(self.nz)
            ]
            
            node_features.append(node_feature)
        
        print(f"Created {len(node_features)} node features with {len(node_features[0]) if node_features else 0} dimensions each")
        return node_features
    
    def build_edges(self) -> Tuple[List[Tuple[int, int]], List[List[float]]]:
        """
        Build edge list and edge features for active cells only
        Much more efficient - only connects neighboring active cells
        """
        edges = []
        edge_features = []
        active_count = self.actnum_handler.get_active_cell_count()
        
        print("Building graph edges for active cells...")
        
        # Build connections between neighboring active cells
        neighbor_pairs_found = 0
        for active_idx in range(active_count):
            neighbors = self.actnum_handler.get_active_neighbors(active_idx)
            
            for neighbor_idx in neighbors:
                # Add bidirectional edge (but avoid duplicates)
                if (active_idx, neighbor_idx) not in [(e[1], e[0]) for e in edges]:
                    edges.append((active_idx, neighbor_idx))
                    neighbor_pairs_found += 1
                
                # Compute edge features
                harmonic_perm = self.compute_harmonic_mean_permeability(active_idx, neighbor_idx)
                
                # Get coordinates for distance calculation
                coords1 = self.actnum_handler.get_grid_coords(active_idx)
                coords2 = self.actnum_handler.get_grid_coords(neighbor_idx)
                
                if coords1 and coords2:
                    i1, j1, k1 = coords1
                    i2, j2, k2 = coords2
                    distance = math.sqrt((i2-i1)**2 + (j2-j1)**2 + (k2-k1)**2)
                    
                    # Direction vector
                    direction_x = (i2 - i1) / max(distance, 1e-6)
                    direction_y = (j2 - j1) / max(distance, 1e-6)
                    direction_z = (k2 - k1) / max(distance, 1e-6)
                else:
                    distance = 1.0
                    direction_x = direction_y = direction_z = 0.0
                
                # Edge features: [harmonic_permeability, distance, direction_x, direction_y, direction_z]
                edge_feature = [
                    max(0.0, min(1.0, math.log(max(harmonic_perm, 0.1) + 1) / 10.0)),  # Normalized log permeability
                    min(distance, 2.0),  # Clamp distance
                    direction_x,
                    direction_y,
                    direction_z
                ]
                
                edge_features.append(edge_feature)
        
        # Add well connections as additional edges (only for active cells)
        well_connections = self.features.get('well_connections', {})
        
        print("Adding well connection edges...")
        well_edges_added = 0
        
        for well_name, connections in well_connections.items():
            active_well_cells = []
            
            # Find which well perforations are in active cells
            for conn in connections:
                i, j, k = conn['cell']
                i, j, k = i-1, j-1, k-1  # Convert to 0-based indexing
                
                active_idx = self.actnum_handler.get_active_index(i, j, k)
                if active_idx is not None:
                    active_well_cells.append((active_idx, conn))
            
            # Connect active well cells to each other
            for idx1, (active_idx1, conn1) in enumerate(active_well_cells):
                for idx2, (active_idx2, conn2) in enumerate(active_well_cells):
                    if idx1 != idx2:
                        edges.append((active_idx1, active_idx2))
                        
                        # Well connection edge features
                        phase_pi1 = conn1.get('phase_pi', 1.0) if hasattr(conn1, 'get') else 1.0
                        phase_pi2 = conn2.get('phase_pi', 1.0) if hasattr(conn2, 'get') else 1.0
                        avg_phase_pi = (phase_pi1 + phase_pi2) / 2.0
                        
                        # Distance between well cells
                        coords1 = self.actnum_handler.get_grid_coords(active_idx1)
                        coords2 = self.actnum_handler.get_grid_coords(active_idx2)
                        
                        if coords1 and coords2:
                            i1, j1, k1 = coords1
                            i2, j2, k2 = coords2
                            distance = math.sqrt((i2-i1)**2 + (j2-j1)**2 + (k2-k1)**2)
                            
                            edge_feature = [
                                max(0.0, min(1.0, math.log(max(avg_phase_pi, 0.1) + 1) / 10.0)),  # Well PI
                                min(distance, 2.0),
                                (i2-i1) / max(distance, 1e-6),
                                (j2-j1) / max(distance, 1e-6),
                                (k2-k1) / max(distance, 1e-6)
                            ]
                            
                            edge_features.append(edge_feature)
                            well_edges_added += 1
        
        # If no natural neighbors exist, create a few artificial connections for GNN to work
        if len(edges) == 0 and active_count > 1:
            print("No natural neighbors found, creating artificial connections...")
            # Connect first few cells in a chain
            for i in range(min(active_count - 1, 5)):
                edges.append((i, i + 1))
                
                # Create simple edge features
                edge_feature = [0.5, 1.0, 0.0, 0.0, 0.0]  # Default features
                edge_features.append(edge_feature)
        
        print(f"Created {len(edges)} edges ({well_edges_added} well edges, {neighbor_pairs_found} neighbor edges)")
        print(f"Edge features dim: {len(edge_features[0]) if edge_features else 0}")
        return edges, edge_features
    
    def build_graph(self) -> Dict:
        """
        Build complete optimized graph representation
        """
        print("=== Building Optimized Graph Representation ===")
        
        # Build node features for active cells only
        node_features = self.build_node_features()
        
        # Build edges between active cells only
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
            'grid_dims': self.grid_dims,
            'active_cells': self.actnum_handler.active_cells,
            'actnum_handler': self.actnum_handler
        }
        
        print(f"Optimized Graph created: {graph['num_nodes']} active nodes, {graph['num_edges']} edges")
        print(f"Edge density: {graph['num_edges'] / max(graph['num_nodes'], 1):.2f} edges per node")
        print("=== Optimized Graph Construction Completed ===")
        
        return graph

def test_optimized_graph_construction():
    """Test optimized graph construction with ACTNUM"""
    from feature_extractor import FeatureExtractor
    
    print("Testing optimized graph construction...")
    
    # Extract features first
    extractor = FeatureExtractor("HM", "/workspace/HM")
    features = extractor.extract_all_features()
    
    if features:
        # Build optimized graph
        constructor = OptimizedGraphConstructor(features)
        graph = constructor.build_graph()
        
        print(f"\nOptimized Graph statistics:")
        print(f"  Active Nodes: {graph['num_nodes']}")
        print(f"  Edges: {graph['num_edges']}")
        print(f"  Edge/Node Ratio: {graph['num_edges'] / max(graph['num_nodes'], 1):.2f}")
        print(f"  Node feature dim: {len(graph['node_features'][0]) if graph['node_features'] else 0}")
        print(f"  Edge feature dim: {len(graph['edge_features'][0]) if graph['edge_features'] else 0}")
        print(f"  Grid dimensions: {graph['grid_dims']}")
        print(f"  Active cell ratio: {graph['num_nodes'] / (graph['grid_dims'][0] * graph['grid_dims'][1] * graph['grid_dims'][2]):.4f}")
        
        # Show sample features
        if graph['node_features']:
            print(f"\nSample node features (first active cell): {[f'{x:.4f}' for x in graph['node_features'][0]]}")
        if graph['edge_features']:
            print(f"Sample edge features (first edge): {[f'{x:.4f}' for x in graph['edge_features'][0]]}")
        
        return graph
    else:
        print("Failed to extract features")
        return None

if __name__ == "__main__":
    test_optimized_graph_construction()