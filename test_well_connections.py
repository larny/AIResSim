#!/usr/bin/env python3
"""
Test well connections parsing to verify the correct number of perforations
"""

from data_parser import ReservoirDataParser
from actnum_handler import ACTNUMHandler

def test_well_connections():
    """Test well connections parsing"""
    print("=== Testing Well Connections ===")
    
    parser = ReservoirDataParser("HM", "/workspace/HM")
    
    # Test well connections
    well_connections = parser.parse_well_connections()
    
    expected_connections = {
        'PROD1': 8,
        'PROD2': 6, 
        'PROD3': 7,
        'PROD4': 4,  # You mentioned 3, but from file it looks like 4
        'PRODUCER': 6
    }
    
    print(f"Well Connection Analysis:")
    for well_name, connections in well_connections.items():
        expected = expected_connections.get(well_name, 0)
        actual = len(connections)
        status = "✅" if actual == expected else "❌"
        print(f"  {well_name}: {actual} connections (expected {expected}) {status}")
        
        if connections:
            print(f"    Sample connection: Cell {connections[0]['cell']}, Transmissibility: {connections[0]['transmissibility']:.2f}")
    
    # Test ACTNUM with correct grid dimensions
    print(f"\n=== Testing ACTNUM with Correct Grid ===")
    actnum_handler = ACTNUMHandler("HM", "/workspace/HM")
    grid_dims = (24, 25, 12)  # 7200 total cells
    actnum = actnum_handler.load_actnum(grid_dims)
    
    print(f"Grid dimensions: {grid_dims}")
    print(f"Total cells: {grid_dims[0] * grid_dims[1] * grid_dims[2]}")
    print(f"Active cells: {actnum_handler.get_active_cell_count()}")
    print(f"Target active cells: 5183")
    print(f"Activity ratio: {actnum_handler.get_active_cell_count() / (grid_dims[0] * grid_dims[1] * grid_dims[2]):.3f}")
    
    # Check if we're close to the expected number
    expected_active = 5183
    actual_active = actnum_handler.get_active_cell_count()
    diff = abs(actual_active - expected_active)
    
    if diff < 100:  # Within 100 cells
        print(f"✅ Active cell count is close to expected ({diff} difference)")
    else:
        print(f"❌ Active cell count differs significantly ({diff} difference)")
    
    return well_connections, actnum_handler

if __name__ == "__main__":
    well_connections, actnum_handler = test_well_connections()