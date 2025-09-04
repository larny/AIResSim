# Reservoir Simulation ML Workflow

A complete machine learning workflow for reservoir simulation using Graph Neural Networks (GNN) and Fourier Neural Operators (FNO), based on the methodology from SPE-223907-MS paper.

## Overview

This implementation provides a coupled GNN-FNO framework for reservoir simulation with the following key features:

- **Data Parsing**: Reads binary reservoir files (INIT, GSG, UNRST, UNSMRY)
- **ACTNUM Support**: Uses active cell mapping for realistic reservoir representation
- **Graph Neural Networks**: Predicts saturation evolution using GCN II architecture
- **Fourier Neural Operators**: Predicts pressure fields using spectral methods
- **Well Models**: Forecasts production rates and bottom hole pressures
- **Coupled Workflow**: Iterative pressure-saturation prediction cycle

## Architecture

### Core Components

1. **Data Parser** (`data_parser.py`)
   - Handles binary reservoir simulation files
   - Extracts initial conditions, properties, and well data

2. **Feature Extractor** (`feature_extractor.py`)
   - Processes 8 key channels: pressure, saturation, permeabilities (x,y,z), porosity, coordinates, well definitions, well controls
   - Applies logarithmic transformation for well controls

3. **ACTNUM Handler** (`actnum_handler.py`)
   - Maps between grid coordinates and active cells
   - Reduces computational complexity by using only active reservoir cells

4. **Graph Constructor** (`optimized_graph_constructor.py`)
   - Builds graph with active cells as nodes
   - Computes harmonic mean permeabilities for edge features
   - Creates well perforation connections

5. **Neural Networks** (`neural_networks.py`)
   - **GNN Model**: Encoder-decoder with GCN II layers for saturation prediction
   - **FNO Model**: Fourier Neural Operator for pressure field prediction
   - **Well Model**: Feedforward network for production/BHP prediction

6. **Coupling Workflow** (`optimized_coupling_workflow.py`)
   - Implements iterative GNN-FNO simulation cycle
   - Handles state management for active cells

7. **Training Pipeline** (`training_pipeline.py`)
   - Complete training framework with rollout losses
   - Data generation and model optimization

## Key Features

### Efficiency Optimizations
- **Active Cell Mapping**: Reduces from 13,500 to ~57 active nodes (0.42% of grid)
- **Sparse Connectivity**: Only 5-10 edges instead of 77,570+ connections
- **Optimized Data Structures**: Memory-efficient representation

### Reservoir Physics
- **Harmonic Mean Permeabilities**: Proper flow calculations between cells
- **Well Perforation Modeling**: Phase PI calculations for production allocation
- **Control Constraints**: Rate and pressure constraint handling

### Machine Learning
- **GCN II Architecture**: Advanced graph convolution with residual connections
- **Spectral Methods**: FNO for efficient PDE solving
- **Rollout Training**: Temporal sequence optimization

## Usage

### Basic Simulation

```python
from optimized_coupling_workflow import OptimizedReservoirSimulationWorkflow

# Initialize workflow
workflow = OptimizedReservoirSimulationWorkflow("HM", "/workspace/HM")

# Run simulation
results = workflow.run_optimized_simulation(num_timesteps=10)

# Access results
print(f"Active cells: {results['active_cell_count']}")
print(f"Final pressure: {results['pressure_stats'][-1]}")
print(f"Well predictions: {results['well_predictions'][-1]}")
```

### Training Models

```python
from training_pipeline import ReservoirMLTrainer

# Initialize trainer
trainer = ReservoirMLTrainer(workflow)

# Train all models
results = trainer.train_all_models()

# Validate
validation = trainer.validate_models()
```

## Data Requirements

The workflow expects the following files in the case directory:

- `CASE_NAME.INIT` - Initial pressure and saturation (binary)
- `CASE_NAME_PERM_I.GSG` - X-direction permeability (binary)
- `CASE_NAME_PERM_J.GSG` - Y-direction permeability (binary)  
- `CASE_NAME_PERM_K.GSG` - Z-direction permeability (binary)
- `CASE_NAME_POROSITY.GSG` - Porosity field (binary)
- `CASE_NAME.GSG` - Grid coordinates (binary)
- `CASE_NAME_WELL_CONNECTIONS.ixf` - Well perforation data (text)
- `CASE_NAME_PRED_FM.ixf` - Well control constraints (text)
- `CASE_NAME.UNRST` - Dynamic simulation results (binary)
- `CASE_NAME.UNSMRY` - Well and field summary data (binary)

## Results

### HM Test Case Performance

- **Grid**: 30×30×15 = 13,500 total cells
- **Active Cells**: 57 (0.42% activity ratio)
- **Graph**: 57 nodes, 5 edges
- **Wells**: 5 producers (PROD1-4, PRODUCER)
- **Execution Time**: <10 seconds for 3 timesteps

### Simulation Output

```
Timestep 3/3:
  Pressure: min=980.6, max=982.9, mean=981.1 psi
  Saturation: min=0.000, max=0.012, mean=0.003
  Wells tracked: 5 wells with production forecasts
```

## Implementation Notes

### Methodology
- Based on SPE-223907-MS paper methodology
- Follows coupled GNN-FNO workflow:
  1. FNO predicts pressure at t+1
  2. GNN predicts saturation using predicted pressure
  3. Well model predicts production using pressure/saturation
  4. Iterate for T timesteps

### Limitations
- Simplified binary file parsing (production use would benefit from ecl/deepfield libraries)
- Basic neural network implementations (production use should leverage PyTorch/PyG)
- No GPU acceleration in current implementation
- Limited to demonstration scale

### Extensions
- Multi-phase flow modeling
- Advanced well control strategies  
- Uncertainty quantification
- History matching capabilities
- Real-time optimization

## Files

- `data_parser.py` - Binary file parsing utilities
- `feature_extractor.py` - Property extraction and processing
- `actnum_handler.py` - Active cell mapping
- `optimized_graph_constructor.py` - Graph building with ACTNUM
- `neural_networks.py` - GNN, FNO, and Well models
- `optimized_coupling_workflow.py` - Main simulation workflow
- `training_pipeline.py` - Model training framework
- `requirements.txt` - Python dependencies

## Citation

Implementation based on methodology from:
"Graph Neural Networks and Fourier Neural Operators for Reservoir Simulation" (SPE-223907-MS)

## License

Research and educational use. See individual file headers for specific licensing terms.