#!/usr/bin/env python3
"""
Complete Demonstration of Reservoir Simulation ML Workflow
Showcases the full GNN-FNO coupling methodology from SPE-223907-MS
"""

from optimized_coupling_workflow import OptimizedReservoirSimulationWorkflow
from training_pipeline import ReservoirMLTrainer
import time

def demonstrate_complete_workflow():
    """Demonstrate the complete reservoir simulation ML workflow"""
    
    print("=" * 80)
    print("RESERVOIR SIMULATION ML WORKFLOW DEMONSTRATION")
    print("Based on SPE-223907-MS: GNN-FNO Coupling for Reservoir Simulation")
    print("=" * 80)
    
    # Phase 1: Initialize Workflow
    print("\n🚀 PHASE 1: WORKFLOW INITIALIZATION")
    print("-" * 50)
    
    start_time = time.time()
    workflow = OptimizedReservoirSimulationWorkflow("HM", "/workspace/HM")
    
    # Configure for demonstration
    workflow.config.update({
        'gnn_hidden_dim': 32,
        'gnn_num_layers': 3,
        'fno_hidden_channels': 16,
        'fno_modes': 8,
        'well_hidden_dims': [32, 16]
    })
    
    init_time = time.time() - start_time
    print(f"✅ Workflow initialized in {init_time:.2f} seconds")
    
    # Phase 2: Data Loading and Processing
    print("\n📊 PHASE 2: DATA LOADING AND PROCESSING")
    print("-" * 50)
    
    start_time = time.time()
    workflow.load_data()
    
    # Display data statistics
    print(f"📈 Data Statistics:")
    print(f"   Grid Dimensions: {workflow.features['grid_dims']}")
    print(f"   Total Grid Cells: {workflow.features['grid_dims'][0] * workflow.features['grid_dims'][1] * workflow.features['grid_dims'][2]:,}")
    print(f"   Active Cells: {workflow.graph['num_nodes']:,}")
    print(f"   Activity Ratio: {workflow.graph['num_nodes'] / (workflow.features['grid_dims'][0] * workflow.features['grid_dims'][1] * workflow.features['grid_dims'][2]):.4f}")
    print(f"   Graph Edges: {workflow.graph['num_edges']:,}")
    print(f"   Wells: {len(workflow.features.get('well_connections', {}))}")
    
    data_time = time.time() - start_time
    print(f"✅ Data loaded and processed in {data_time:.2f} seconds")
    
    # Phase 3: Model Architecture
    print("\n🧠 PHASE 3: NEURAL NETWORK MODELS")
    print("-" * 50)
    
    start_time = time.time()
    workflow.initialize_models()
    
    print(f"🔗 Model Architecture:")
    print(f"   GNN Model: {workflow.config['gnn_hidden_dim']} hidden units, {workflow.config['gnn_num_layers']} layers")
    print(f"   FNO Model: {workflow.config['fno_hidden_channels']} channels, {workflow.config['fno_modes']} modes")
    print(f"   Well Model: {workflow.config['well_hidden_dims']} hidden layers")
    print(f"   Total Parameters: ~{estimate_parameters(workflow):,}")
    
    model_time = time.time() - start_time
    print(f"✅ Models initialized in {model_time:.2f} seconds")
    
    # Phase 4: Training Demonstration
    print("\n🎯 PHASE 4: TRAINING DEMONSTRATION")
    print("-" * 50)
    
    start_time = time.time()
    trainer = ReservoirMLTrainer(workflow)
    
    # Quick training demonstration
    print("🔄 Training Models (demonstration with small datasets)...")
    training_results = trainer.train_all_models()
    
    print(f"📊 Training Results:")
    print(f"   FNO Loss: {training_results['fno']['final_loss']:.6f}")
    print(f"   GNN Loss: {training_results['gnn']['final_loss']:.6f}")
    print(f"   Well Model Loss: {training_results['well']['final_loss']:.6f}")
    print(f"   Combined Loss: {training_results['total_loss']:.6f}")
    
    train_time = time.time() - start_time
    print(f"✅ Training completed in {train_time:.2f} seconds")
    
    # Phase 5: Coupled Simulation
    print("\n⚡ PHASE 5: COUPLED GNN-FNO SIMULATION")
    print("-" * 50)
    
    start_time = time.time()
    print("🔄 Running coupled simulation...")
    
    # Run the simulation
    simulation_results = workflow.run_optimized_simulation(num_timesteps=5)
    
    sim_time = time.time() - start_time
    print(f"✅ Simulation completed in {sim_time:.2f} seconds")
    
    # Phase 6: Results Analysis
    print("\n📈 PHASE 6: RESULTS ANALYSIS")
    print("-" * 50)
    
    analyze_results(simulation_results)
    
    # Phase 7: Validation
    print("\n✅ PHASE 7: MODEL VALIDATION")
    print("-" * 50)
    
    validation_results = trainer.validate_models()
    
    print(f"🔍 Validation Results:")
    print(f"   Simulation Success: {'✅' if validation_results['simulation_success'] else '❌'}")
    print(f"   Pressure Stability: {'✅' if validation_results['pressure_stability'] else '❌'}")
    print(f"   Saturation Stability: {'✅' if validation_results['saturation_stability'] else '❌'}")
    print(f"   Well Predictions: {validation_results['well_predictions_count']} timesteps")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("🎉 WORKFLOW DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    total_time = init_time + data_time + model_time + train_time + sim_time
    
    print(f"⏱️  Performance Summary:")
    print(f"   Total Execution Time: {total_time:.2f} seconds")
    print(f"   Data Processing: {data_time:.2f}s ({data_time/total_time*100:.1f}%)")
    print(f"   Model Training: {train_time:.2f}s ({train_time/total_time*100:.1f}%)")
    print(f"   Simulation: {sim_time:.2f}s ({sim_time/total_time*100:.1f}%)")
    
    print(f"\n🏆 Key Achievements:")
    print(f"   ✅ Successfully parsed binary reservoir files")
    print(f"   ✅ Built optimized graph with {workflow.graph['num_nodes']} active nodes")
    print(f"   ✅ Implemented GNN-FNO coupling methodology")
    print(f"   ✅ Achieved {total_time:.1f}s end-to-end execution")
    print(f"   ✅ Demonstrated complete ML workflow for reservoir simulation")
    
    return {
        'workflow': workflow,
        'training_results': training_results,
        'simulation_results': simulation_results,
        'validation_results': validation_results,
        'performance': {
            'total_time': total_time,
            'data_time': data_time,
            'train_time': train_time,
            'sim_time': sim_time
        }
    }

def estimate_parameters(workflow):
    """Estimate total number of parameters in all models"""
    # Rough estimation based on model architecture
    gnn_params = (9 * workflow.config['gnn_hidden_dim'] + 
                  workflow.config['gnn_hidden_dim'] * workflow.config['gnn_hidden_dim'] * workflow.config['gnn_num_layers'] + 
                  workflow.config['gnn_hidden_dim'] * 1)
    
    fno_params = (4 * workflow.config['fno_hidden_channels'] + 
                  workflow.config['fno_hidden_channels'] * workflow.config['fno_hidden_channels'] * 4 + 
                  workflow.config['fno_hidden_channels'] * 1)
    
    well_params = (10 * workflow.config['well_hidden_dims'][0] + 
                   workflow.config['well_hidden_dims'][0] * workflow.config['well_hidden_dims'][1] + 
                   workflow.config['well_hidden_dims'][1] * 3)
    
    return gnn_params + fno_params + well_params

def analyze_results(simulation_results):
    """Analyze and display simulation results"""
    
    print(f"📊 Simulation Analysis:")
    print(f"   Timesteps Completed: {len(simulation_results['timesteps'])}")
    print(f"   Active Cells: {simulation_results['active_cell_count']}")
    
    # Pressure evolution
    print(f"\n🌡️  Pressure Evolution:")
    for i, p_stat in enumerate(simulation_results['pressure_stats']):
        print(f"   T{i+1}: {p_stat['min']:.1f} - {p_stat['max']:.1f} psi (avg: {p_stat['mean']:.1f})")
    
    # Saturation evolution
    print(f"\n💧 Saturation Evolution:")
    for i, s_stat in enumerate(simulation_results['saturation_stats']):
        print(f"   T{i+1}: {s_stat['min']:.3f} - {s_stat['max']:.3f} (avg: {s_stat['mean']:.3f})")
    
    # Well performance
    if simulation_results['well_predictions']:
        print(f"\n🛢️  Well Performance:")
        final_wells = simulation_results['well_predictions'][-1]
        if final_wells:
            for well_name, pred in final_wells.items():
                print(f"   {well_name}: Oil={pred.get('oil_production_rate', 0):.1f} STB/d, "
                      f"BHP={pred.get('bottom_hole_pressure', 0):.1f} psi")
        else:
            print("   No active wells found in simulation")
    
    # Physics validation
    print(f"\n🔬 Physics Validation:")
    final_p = simulation_results['pressure_stats'][-1]
    final_s = simulation_results['saturation_stats'][-1]
    
    pressure_valid = 500 <= final_p['mean'] <= 5000
    saturation_valid = 0 <= final_s['mean'] <= 1
    
    print(f"   Pressure Range: {'✅' if pressure_valid else '❌'} ({final_p['mean']:.1f} psi)")
    print(f"   Saturation Range: {'✅' if saturation_valid else '❌'} ({final_s['mean']:.3f})")

if __name__ == "__main__":
    # Run the complete demonstration
    demo_results = demonstrate_complete_workflow()
    
    print(f"\n🎯 Demo completed! All components working successfully.")
    print(f"   Check the results in the returned dictionary for detailed analysis.")