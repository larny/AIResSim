"""
Visualization and Quantification Module for Reservoir Simulation Results
Includes 2D/3D visualization and error analysis functions
"""

import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from optimized_coupling_workflow import OptimizedReservoirSimulationWorkflow

class ReservoirVisualizer:
    """Visualization and analysis tools for reservoir simulation results"""
    
    def __init__(self, workflow: OptimizedReservoirSimulationWorkflow):
        self.workflow = workflow
        self.grid_dims = workflow.features['grid_dims'] if workflow.features else (24, 25, 12)
        self.nx, self.ny, self.nz = self.grid_dims
        self.actnum_handler = workflow.graph['actnum_handler'] if workflow.graph else None
        
        # Create custom colormaps
        self.pressure_cmap = LinearSegmentedColormap.from_list(
            'pressure', ['blue', 'cyan', 'yellow', 'red']
        )
        self.saturation_cmap = LinearSegmentedColormap.from_list(
            'saturation', ['white', 'lightblue', 'blue', 'darkblue']
        )
        self.error_cmap = LinearSegmentedColormap.from_list(
            'error', ['green', 'yellow', 'orange', 'red']
        )
        
        print("Reservoir Visualizer initialized")
    
    def active_to_3d_field(self, active_values: List[float], default_value: float = 0.0) -> List[List[List[float]]]:
        """Convert active cell values to 3D field"""
        field_3d = [[[default_value for _ in range(self.nz)] 
                     for _ in range(self.ny)] 
                    for _ in range(self.nx)]
        
        if self.actnum_handler:
            for active_idx, value in enumerate(active_values):
                coords = self.actnum_handler.get_grid_coords(active_idx)
                if coords:
                    i, j, k = coords
                    if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
                        field_3d[i][j][k] = value
        
        return field_3d
    
    def extract_2d_slice(self, field_3d: List[List[List[float]]], layer: int) -> List[List[float]]:
        """Extract 2D slice from 3D field at specified Z layer"""
        if 0 <= layer < self.nz:
            return [[field_3d[i][j][layer] for j in range(self.ny)] for i in range(self.nx)]
        else:
            return [[0.0 for _ in range(self.ny)] for _ in range(self.nx)]
    
    def plot_2d_property(self, field_2d: List[List[float]], title: str, 
                        property_name: str, layer: int, 
                        colormap=None, vmin=None, vmax=None, 
                        save_path: str = None) -> str:
        """Plot 2D property distribution"""
        if colormap is None:
            colormap = plt.cm.viridis
        
        # Convert to numpy array for plotting
        field_array = np.array(field_2d)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the plot
        im = ax.imshow(field_array.T, origin='lower', cmap=colormap, 
                      vmin=vmin, vmax=vmax, aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(property_name, rotation=270, labelpad=20)
        
        # Labels and title
        ax.set_xlabel('Grid X Index')
        ax.set_ylabel('Grid Y Index')
        ax.set_title(f'{title} - Layer {layer+1}/{self.nz}')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add well locations if available
        self.add_well_markers_2d(ax, layer)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 2D plot: {save_path}")
        
        return save_path or f"2d_{property_name}_layer_{layer}.png"
    
    def plot_3d_property(self, field_3d: List[List[List[float]]], title: str,
                        property_name: str, colormap=None, 
                        save_path: str = None) -> str:
        """Plot 3D property distribution (multiple 2D slices)"""
        if colormap is None:
            colormap = plt.cm.viridis
        
        # Calculate global min/max for consistent color scale
        all_values = []
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if field_3d[i][j][k] != 0.0:  # Skip inactive cells
                        all_values.append(field_3d[i][j][k])
        
        if not all_values:
            print("No active values found in 3D field")
            return ""
        
        vmin, vmax = min(all_values), max(all_values)
        
        # Create subplot for multiple layers
        n_layers = min(self.nz, 6)  # Show at most 6 layers
        layer_indices = [int(i * (self.nz - 1) / (n_layers - 1)) for i in range(n_layers)]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, layer in enumerate(layer_indices):
            field_2d = self.extract_2d_slice(field_3d, layer)
            field_array = np.array(field_2d)
            
            ax = axes[idx]
            im = ax.imshow(field_array.T, origin='lower', cmap=colormap,
                          vmin=vmin, vmax=vmax, aspect='equal')
            
            ax.set_title(f'Layer {layer+1}')
            ax.set_xlabel('X Index')
            ax.set_ylabel('Y Index')
            
            # Add well markers
            self.add_well_markers_2d(ax, layer)
        
        # Add colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(property_name, rotation=270, labelpad=20)
        
        fig.suptitle(f'{title} - 3D Distribution', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D plot: {save_path}")
        
        return save_path or f"3d_{property_name}.png"
    
    def add_well_markers_2d(self, ax, layer: int):
        """Add well location markers to 2D plot"""
        if not self.workflow.features or 'well_connections' not in self.workflow.features:
            return
        
        well_connections = self.workflow.features['well_connections']
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for idx, (well_name, connections) in enumerate(well_connections.items()):
            color = colors[idx % len(colors)]
            
            for conn in connections:
                i, j, k = conn['cell']
                i, j, k = i-1, j-1, k-1  # Convert to 0-based
                
                if k == layer:  # Well perforation at this layer
                    ax.plot(i, j, 'o', color=color, markersize=8, 
                           markeredgecolor='white', markeredgewidth=1)
                    ax.text(i+0.5, j+0.5, well_name[:4], fontsize=8, 
                           color=color, fontweight='bold')
    
    def visualize_static_properties(self, save_dir: str = "/workspace") -> Dict[str, List[str]]:
        """Visualize all static reservoir properties"""
        print("Creating static property visualizations...")
        
        saved_files = {
            'pressure_2d': [],
            'pressure_3d': [],
            'saturation_2d': [],
            'saturation_3d': [],
            'porosity_2d': [],
            'porosity_3d': [],
            'permeability_2d': [],
            'permeability_3d': []
        }
        
        # Get current simulation state
        if hasattr(self.workflow, 'current_pressure_active'):
            pressure_3d = self.active_to_3d_field(self.workflow.current_pressure_active, 0.0)
            saturation_3d = self.active_to_3d_field(self.workflow.current_saturation_active, 0.0)
        else:
            # Use initial values
            pressure_3d = [[[2000.0 for _ in range(self.nz)] for _ in range(self.ny)] for _ in range(self.nx)]
            saturation_3d = [[[0.8 for _ in range(self.nz)] for _ in range(self.ny)] for _ in range(self.nx)]
        
        # Create porosity and permeability fields
        if self.workflow.features:
            porosity_active = self.workflow.graph['actnum_handler'].map_property_to_active_cells(
                self.workflow.features.get('porosity', [])
            )
            perm_x_active = self.workflow.graph['actnum_handler'].map_property_to_active_cells(
                self.workflow.features.get('perm_x', [])
            )
            
            porosity_3d = self.active_to_3d_field(porosity_active.data if porosity_active else [], 0.0)
            perm_x_3d = self.active_to_3d_field(perm_x_active.data if perm_x_active else [], 0.0)
        else:
            porosity_3d = [[[0.2 for _ in range(self.nz)] for _ in range(self.ny)] for _ in range(self.nx)]
            perm_x_3d = [[[100.0 for _ in range(self.nz)] for _ in range(self.ny)] for _ in range(self.nx)]
        
        # Plot 2D slices for middle layer
        middle_layer = self.nz // 2
        
        # Pressure 2D
        pressure_2d = self.extract_2d_slice(pressure_3d, middle_layer)
        file_path = f"{save_dir}/pressure_2d_layer_{middle_layer}.png"
        self.plot_2d_property(pressure_2d, "Pressure Distribution", "Pressure (psi)", 
                             middle_layer, self.pressure_cmap, save_path=file_path)
        saved_files['pressure_2d'].append(file_path)
        
        # Saturation 2D
        saturation_2d = self.extract_2d_slice(saturation_3d, middle_layer)
        file_path = f"{save_dir}/saturation_2d_layer_{middle_layer}.png"
        self.plot_2d_property(saturation_2d, "Saturation Distribution", "Saturation", 
                             middle_layer, self.saturation_cmap, save_path=file_path)
        saved_files['saturation_2d'].append(file_path)
        
        # Porosity 2D
        porosity_2d = self.extract_2d_slice(porosity_3d, middle_layer)
        file_path = f"{save_dir}/porosity_2d_layer_{middle_layer}.png"
        self.plot_2d_property(porosity_2d, "Porosity Distribution", "Porosity", 
                             middle_layer, plt.cm.YlOrRd, save_path=file_path)
        saved_files['porosity_2d'].append(file_path)
        
        # Permeability 2D
        perm_2d = self.extract_2d_slice(perm_x_3d, middle_layer)
        file_path = f"{save_dir}/permeability_2d_layer_{middle_layer}.png"
        self.plot_2d_property(perm_2d, "Permeability X Distribution", "Permeability (mD)", 
                             middle_layer, plt.cm.plasma, save_path=file_path)
        saved_files['permeability_2d'].append(file_path)
        
        # Plot 3D distributions
        file_path = f"{save_dir}/pressure_3d.png"
        self.plot_3d_property(pressure_3d, "Pressure", "Pressure (psi)", 
                             self.pressure_cmap, save_path=file_path)
        saved_files['pressure_3d'].append(file_path)
        
        file_path = f"{save_dir}/saturation_3d.png"
        self.plot_3d_property(saturation_3d, "Saturation", "Saturation", 
                             self.saturation_cmap, save_path=file_path)
        saved_files['saturation_3d'].append(file_path)
        
        print("Static property visualizations completed")
        return saved_files
    
    def calculate_prediction_errors(self, predicted: List[float], actual: List[float]) -> Dict[str, float]:
        """Calculate prediction errors"""
        if len(predicted) != len(actual):
            print(f"Warning: Length mismatch - predicted: {len(predicted)}, actual: {len(actual)}")
            min_len = min(len(predicted), len(actual))
            predicted = predicted[:min_len]
            actual = actual[:min_len]
        
        if not predicted or not actual:
            return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'r2': 0.0}
        
        # Mean Absolute Error
        mae = sum(abs(p - a) for p, a in zip(predicted, actual)) / len(predicted)
        
        # Root Mean Square Error
        rmse = math.sqrt(sum((p - a)**2 for p, a in zip(predicted, actual)) / len(predicted))
        
        # Mean Absolute Percentage Error
        mape = 0.0
        valid_count = 0
        for p, a in zip(predicted, actual):
            if abs(a) > 1e-6:  # Avoid division by zero
                mape += abs((p - a) / a)
                valid_count += 1
        mape = (mape / valid_count * 100) if valid_count > 0 else 0.0
        
        # R-squared
        actual_mean = sum(actual) / len(actual)
        ss_tot = sum((a - actual_mean)**2 for a in actual)
        ss_res = sum((a - p)**2 for a, p in zip(actual, predicted))
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    def plot_error_analysis_2d(self, predicted_3d: List[List[List[float]]], 
                              actual_3d: List[List[List[float]]], 
                              property_name: str, layer: int,
                              save_path: str = None) -> str:
        """Plot 2D error analysis for a specific layer"""
        
        # Extract 2D slices
        pred_2d = self.extract_2d_slice(predicted_3d, layer)
        actual_2d = self.extract_2d_slice(actual_3d, layer)
        
        # Calculate absolute and relative errors
        abs_error_2d = [[abs(pred_2d[i][j] - actual_2d[i][j]) 
                        for j in range(self.ny)] for i in range(self.nx)]
        
        rel_error_2d = []
        for i in range(self.nx):
            row = []
            for j in range(self.ny):
                if abs(actual_2d[i][j]) > 1e-6:
                    rel_err = abs(pred_2d[i][j] - actual_2d[i][j]) / abs(actual_2d[i][j]) * 100
                else:
                    rel_err = 0.0
                row.append(rel_err)
            rel_error_2d.append(row)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot predicted
        pred_array = np.array(pred_2d)
        im1 = ax1.imshow(pred_array.T, origin='lower', cmap=self.pressure_cmap, aspect='equal')
        ax1.set_title(f'Predicted {property_name}')
        ax1.set_xlabel('X Index')
        ax1.set_ylabel('Y Index')
        plt.colorbar(im1, ax=ax1)
        
        # Plot actual
        actual_array = np.array(actual_2d)
        im2 = ax2.imshow(actual_array.T, origin='lower', cmap=self.pressure_cmap, aspect='equal')
        ax2.set_title(f'Actual {property_name}')
        ax2.set_xlabel('X Index')
        ax2.set_ylabel('Y Index')
        plt.colorbar(im2, ax=ax2)
        
        # Plot absolute error
        abs_err_array = np.array(abs_error_2d)
        im3 = ax3.imshow(abs_err_array.T, origin='lower', cmap=self.error_cmap, aspect='equal')
        ax3.set_title(f'Absolute Error')
        ax3.set_xlabel('X Index')
        ax3.set_ylabel('Y Index')
        plt.colorbar(im3, ax=ax3)
        
        # Plot relative error
        rel_err_array = np.array(rel_error_2d)
        im4 = ax4.imshow(rel_err_array.T, origin='lower', cmap=self.error_cmap, aspect='equal')
        ax4.set_title(f'Relative Error (%)')
        ax4.set_xlabel('X Index')
        ax4.set_ylabel('Y Index')
        plt.colorbar(im4, ax=ax4)
        
        # Add well markers to all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            self.add_well_markers_2d(ax, layer)
        
        fig.suptitle(f'{property_name} Error Analysis - Layer {layer+1}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved error analysis: {save_path}")
        
        return save_path or f"error_analysis_2d_{property_name}_layer_{layer}.png"
    
    def plot_well_performance_analysis(self, simulation_results: Dict, 
                                     save_path: str = None) -> str:
        """Plot well performance over time"""
        
        if not simulation_results.get('well_predictions'):
            print("No well prediction data available")
            return ""
        
        # Extract well data over time
        timesteps = simulation_results['timesteps']
        well_data = {}
        
        # Initialize well data structure
        for timestep_data in simulation_results['well_predictions']:
            for well_name in timestep_data.keys():
                if well_name not in well_data:
                    well_data[well_name] = {
                        'oil_rate': [],
                        'water_rate': [],
                        'liquid_rate': [],
                        'bhp': []
                    }
        
        # Collect data over time
        for timestep_data in simulation_results['well_predictions']:
            for well_name, well_info in well_data.items():
                if well_name in timestep_data:
                    pred = timestep_data[well_name]
                    well_info['oil_rate'].append(pred.get('oil_production_rate', 0.0))
                    well_info['water_rate'].append(pred.get('water_production_rate', 0.0))
                    well_info['liquid_rate'].append(
                        pred.get('oil_production_rate', 0.0) + pred.get('water_production_rate', 0.0)
                    )
                    well_info['bhp'].append(pred.get('bottom_hole_pressure', 2000.0))
                else:
                    # Fill with zeros if well data not available
                    well_info['oil_rate'].append(0.0)
                    well_info['water_rate'].append(0.0)
                    well_info['liquid_rate'].append(0.0)
                    well_info['bhp'].append(2000.0)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Plot oil production rates
        for idx, (well_name, data) in enumerate(well_data.items()):
            color = colors[idx % len(colors)]
            ax1.plot(timesteps, data['oil_rate'], 'o-', color=color, label=well_name, linewidth=2)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Oil Production Rate (STB/d)')
        ax1.set_title('Oil Production Rate vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot water production rates
        for idx, (well_name, data) in enumerate(well_data.items()):
            color = colors[idx % len(colors)]
            ax2.plot(timesteps, data['water_rate'], 's-', color=color, label=well_name, linewidth=2)
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Water Production Rate (STB/d)')
        ax2.set_title('Water Production Rate vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot liquid production rates
        for idx, (well_name, data) in enumerate(well_data.items()):
            color = colors[idx % len(colors)]
            ax3.plot(timesteps, data['liquid_rate'], '^-', color=color, label=well_name, linewidth=2)
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Liquid Production Rate (STB/d)')
        ax3.set_title('Total Liquid Production Rate vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot bottom hole pressure
        for idx, (well_name, data) in enumerate(well_data.items()):
            color = colors[idx % len(colors)]
            ax4.plot(timesteps, data['bhp'], 'd-', color=color, label=well_name, linewidth=2)
        ax4.set_xlabel('Timestep')
        ax4.set_ylabel('Bottom Hole Pressure (psi)')
        ax4.set_title('Bottom Hole Pressure vs Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle('Well Performance Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved well performance analysis: {save_path}")
        
        return save_path or "well_performance_analysis.png"
    
    def plot_error_trends(self, error_history: Dict[str, List[float]], 
                         save_path: str = None) -> str:
        """Plot error trends over time"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        timesteps = list(range(1, len(list(error_history.values())[0]) + 1))
        
        # Mean Absolute Error
        if 'mae' in error_history:
            ax1.plot(timesteps, error_history['mae'], 'o-', color='blue', linewidth=2)
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Mean Absolute Error')
            ax1.set_title('MAE Trend')
            ax1.grid(True, alpha=0.3)
        
        # Root Mean Square Error
        if 'rmse' in error_history:
            ax2.plot(timesteps, error_history['rmse'], 's-', color='red', linewidth=2)
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Root Mean Square Error')
            ax2.set_title('RMSE Trend')
            ax2.grid(True, alpha=0.3)
        
        # Mean Absolute Percentage Error
        if 'mape' in error_history:
            ax3.plot(timesteps, error_history['mape'], '^-', color='green', linewidth=2)
            ax3.set_xlabel('Timestep')
            ax3.set_ylabel('Mean Absolute Percentage Error (%)')
            ax3.set_title('MAPE Trend')
            ax3.grid(True, alpha=0.3)
        
        # R-squared
        if 'r2' in error_history:
            ax4.plot(timesteps, error_history['r2'], 'd-', color='purple', linewidth=2)
            ax4.set_xlabel('Timestep')
            ax4.set_ylabel('R-squared')
            ax4.set_title('R² Trend')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])
        
        fig.suptitle('Prediction Error Trends', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved error trends: {save_path}")
        
        return save_path or "error_trends.png"

def test_visualization():
    """Test visualization functions"""
    print("=== Testing Visualization Functions ===")
    
    # Initialize workflow
    from optimized_coupling_workflow import OptimizedReservoirSimulationWorkflow
    workflow = OptimizedReservoirSimulationWorkflow("HM", "/workspace/HM")
    
    # Run a short simulation
    results = workflow.run_optimized_simulation(num_timesteps=3)
    
    # Initialize visualizer
    visualizer = ReservoirVisualizer(workflow)
    
    # Test static property visualization
    print("Creating static property visualizations...")
    static_files = visualizer.visualize_static_properties()
    
    # Test well performance visualization
    print("Creating well performance visualization...")
    well_file = visualizer.plot_well_performance_analysis(results, "/workspace/well_performance.png")
    
    # Test error analysis (using dummy data)
    print("Creating error analysis...")
    
    # Create dummy predicted vs actual data
    predicted_3d = [[[900.0 + i*10 + j*5 for _ in range(12)] for j in range(25)] for i in range(24)]
    actual_3d = [[[950.0 + i*8 + j*3 for _ in range(12)] for j in range(25)] for i in range(24)]
    
    error_file = visualizer.plot_error_analysis_2d(
        predicted_3d, actual_3d, "Pressure", 6, "/workspace/pressure_error_analysis.png"
    )
    
    # Test error trends
    error_history = {
        'mae': [50.0, 45.0, 40.0],
        'rmse': [65.0, 58.0, 52.0],
        'mape': [5.2, 4.8, 4.3],
        'r2': [0.85, 0.88, 0.91]
    }
    
    trend_file = visualizer.plot_error_trends(error_history, "/workspace/error_trends.png")
    
    print("✅ All visualization tests completed!")
    print(f"Generated files:")
    for category, files in static_files.items():
        print(f"  {category}: {len(files)} files")
    print(f"  Well performance: {well_file}")
    print(f"  Error analysis: {error_file}")
    print(f"  Error trends: {trend_file}")
    
    return visualizer, static_files

if __name__ == "__main__":
    visualizer, files = test_visualization()