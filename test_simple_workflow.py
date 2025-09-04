#!/usr/bin/env python3
"""
Simple test of the coupling workflow
"""

from coupling_workflow import ReservoirSimulationWorkflow

def test_simple():
    """Simple test with minimal computation"""
    print("=== Simple Workflow Test ===")
    
    # Initialize workflow
    workflow = ReservoirSimulationWorkflow("HM", "/workspace/HM")
    
    # Override config for faster testing
    workflow.config.update({
        'gnn_hidden_dim': 16,
        'gnn_num_layers': 2,
        'fno_hidden_channels': 8,
        'well_hidden_dims': [16, 8]
    })
    
    # Load data
    print("Loading data...")
    workflow.load_data()
    
    # Initialize models
    print("Initializing models...")
    workflow.initialize_models()
    
    print("Testing individual components...")
    
    # Test pressure prediction
    print("1. Testing FNO pressure prediction...")
    try:
        next_pressure = workflow.predict_pressure(workflow.current_saturation, 0)
        print(f"   Pressure prediction: {workflow.get_field_min_max(next_pressure)}")
    except Exception as e:
        print(f"   Error in pressure prediction: {e}")
    
    # Test saturation prediction with smaller subset
    print("2. Testing GNN saturation prediction...")
    try:
        # Limit to first 100 nodes for testing
        limited_graph = {
            'node_features': workflow.graph['node_features'][:100],
            'edge_index': [[], []],  # Empty edges for simplicity
            'num_nodes': 100
        }
        workflow.graph = limited_graph
        
        saturation_values = workflow.predict_saturation(workflow.current_pressure, workflow.current_saturation)
        print(f"   Saturation prediction: {len(saturation_values)} values, range [{min(saturation_values):.3f}, {max(saturation_values):.3f}]")
    except Exception as e:
        print(f"   Error in saturation prediction: {e}")
    
    # Test well prediction
    print("3. Testing well property prediction...")
    try:
        well_pred = workflow.predict_well_properties(workflow.current_pressure, workflow.current_saturation, 0)
        print(f"   Well predictions: {len(well_pred)} wells")
        for well_name, pred in well_pred.items():
            print(f"     {well_name}: Oil={pred.get('oil_production_rate', 0):.1f} STB/d")
    except Exception as e:
        print(f"   Error in well prediction: {e}")
    
    print("=== Simple Test Completed ===")

if __name__ == "__main__":
    test_simple()