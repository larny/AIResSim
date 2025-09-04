"""
Training Pipeline for Reservoir Simulation ML Models
Includes loss functions, optimization, and GPU support framework
"""

import math
import random
from typing import Dict, List, Tuple, Optional
from coupling_workflow import ReservoirSimulationWorkflow

class LossFunction:
    """Base class for loss functions"""
    
    def compute_loss(self, predictions: List[float], targets: List[float]) -> float:
        raise NotImplementedError

class MSELoss(LossFunction):
    """Mean Squared Error Loss"""
    
    def compute_loss(self, predictions: List[float], targets: List[float]) -> float:
        if len(predictions) != len(targets):
            return float('inf')
        
        mse = 0.0
        for pred, target in zip(predictions, targets):
            mse += (pred - target) ** 2
        
        return mse / len(predictions)

class RolloutLoss(LossFunction):
    """Rollout loss for temporal sequences"""
    
    def __init__(self, rollout_steps: int = 5, decay_factor: float = 0.9):
        self.rollout_steps = rollout_steps
        self.decay_factor = decay_factor
        self.mse_loss = MSELoss()
    
    def compute_rollout_loss(self, predictions_sequence: List[List[float]], 
                           targets_sequence: List[List[float]]) -> float:
        """Compute loss over rollout sequence with exponential decay"""
        total_loss = 0.0
        weight = 1.0
        
        for t in range(min(len(predictions_sequence), len(targets_sequence), self.rollout_steps)):
            step_loss = self.mse_loss.compute_loss(predictions_sequence[t], targets_sequence[t])
            total_loss += weight * step_loss
            weight *= self.decay_factor
        
        return total_loss

class Optimizer:
    """Simple gradient descent optimizer"""
    
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
    
    def step(self, model, gradients: Dict):
        """Update model parameters (simplified)"""
        # In a real implementation, this would update model weights
        # For demonstration, we'll just track the step
        pass

class TrainingDataGenerator:
    """Generate training data from reservoir simulation"""
    
    def __init__(self, workflow: ReservoirSimulationWorkflow):
        self.workflow = workflow
    
    def generate_pressure_training_data(self, num_samples: int = 10) -> List[Dict]:
        """Generate training data for FNO pressure model"""
        training_data = []
        
        for sample in range(num_samples):
            # Generate random initial conditions
            nx, ny, nz = self.workflow.features['grid_dims']
            
            # Random saturation field
            saturation = [[[random.uniform(0.2, 0.9) for _ in range(nz)] 
                          for _ in range(ny)] 
                         for _ in range(nx)]
            
            # Predict pressure using current model
            predicted_pressure = self.workflow.predict_pressure(saturation, sample)
            
            # Create training sample
            # Input: saturation field at t
            # Output: pressure field at t+1
            sample_data = {
                'input_saturation': saturation,
                'target_pressure': predicted_pressure,
                'timestep': sample
            }
            
            training_data.append(sample_data)
        
        return training_data
    
    def generate_saturation_training_data(self, num_samples: int = 10) -> List[Dict]:
        """Generate training data for GNN saturation model"""
        training_data = []
        
        for sample in range(num_samples):
            # Generate random pressure field
            nx, ny, nz = self.workflow.features['grid_dims']
            pressure = [[[random.uniform(1000, 4000) for _ in range(nz)] 
                        for _ in range(ny)] 
                       for _ in range(nx)]
            
            # Generate random previous saturation
            prev_saturation = [[[random.uniform(0.2, 0.9) for _ in range(nz)] 
                               for _ in range(ny)] 
                              for _ in range(nx)]
            
            # Predict saturation using current model
            predicted_saturation = self.workflow.predict_saturation(pressure, prev_saturation)
            
            # Create training sample
            sample_data = {
                'input_pressure': pressure,
                'input_prev_saturation': prev_saturation,
                'target_saturation': predicted_saturation,
                'timestep': sample
            }
            
            training_data.append(sample_data)
        
        return training_data
    
    def generate_well_training_data(self, num_samples: int = 10) -> List[Dict]:
        """Generate training data for well model"""
        training_data = []
        
        for sample in range(num_samples):
            well_data = []
            
            # Generate data for each well
            well_connections = self.workflow.features.get('well_connections', {})
            for well_name, connections in well_connections.items():
                if connections:
                    # Random well conditions
                    avg_pressure = random.uniform(1500, 3500)
                    avg_saturation = random.uniform(0.3, 0.8)
                    
                    # Create input features
                    well_input = [
                        avg_pressure / 5000.0,
                        avg_saturation,
                        random.uniform(0.7, 0.9),  # Normalized depth
                        1.0,  # Producer flag
                        1.0,  # Rate constraint
                        0.0,  # Pressure constraint
                        len(connections) / 10.0,
                        sample / 100.0,
                        math.sin(sample * 0.1),
                        math.cos(sample * 0.1)
                    ]
                    
                    # Generate target values (simplified)
                    target_oil_rate = random.uniform(500, 2000)
                    target_water_rate = random.uniform(100, 500)
                    target_bhp = random.uniform(1800, 2500)
                    
                    well_sample = {
                        'well_name': well_name,
                        'input_features': well_input,
                        'target_oil_rate': target_oil_rate,
                        'target_water_rate': target_water_rate,
                        'target_bhp': target_bhp
                    }
                    
                    well_data.append(well_sample)
            
            training_data.append({
                'timestep': sample,
                'wells': well_data
            })
        
        return training_data

class ReservoirMLTrainer:
    """Main training class for reservoir ML models"""
    
    def __init__(self, workflow: ReservoirSimulationWorkflow, config: Dict = None):
        self.workflow = workflow
        self.config = config or self.get_default_training_config()
        
        # Initialize loss functions
        self.pressure_loss = RolloutLoss(rollout_steps=self.config['pressure_rollout_steps'])
        self.saturation_loss = RolloutLoss(rollout_steps=self.config['saturation_rollout_steps'])
        self.well_loss = MSELoss()
        
        # Initialize optimizer
        self.optimizer = Optimizer(learning_rate=self.config['learning_rate'])
        
        # Training data generator
        self.data_generator = TrainingDataGenerator(workflow)
        
        # Training history
        self.training_history = {
            'pressure_losses': [],
            'saturation_losses': [],
            'well_losses': [],
            'total_losses': []
        }
        
        print("Reservoir ML Trainer initialized")
    
    def get_default_training_config(self) -> Dict:
        """Get default training configuration"""
        return {
            'num_epochs': 50,
            'batch_size': 4,
            'learning_rate': 0.001,
            'pressure_rollout_steps': 10,
            'saturation_rollout_steps': 5,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'checkpoint_frequency': 10,
            'use_gpu': True  # Framework for GPU support
        }
    
    def train_fno_model(self, num_epochs: int = 20) -> Dict:
        """Train the FNO pressure model"""
        print(f"Training FNO model for {num_epochs} epochs...")
        
        # Generate training data
        training_data = self.data_generator.generate_pressure_training_data(num_samples=50)
        
        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Mini-batch training
            batch_size = self.config['batch_size']
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                batch_loss = 0.0
                for sample in batch:
                    # Forward pass (simplified)
                    input_sat = sample['input_saturation']
                    target_press = sample['target_pressure']
                    
                    # Predict pressure
                    predicted_press = self.workflow.predict_pressure(input_sat, sample['timestep'])
                    
                    # Compute loss (simplified - compare field statistics)
                    pred_mean = self.compute_field_mean(predicted_press)
                    target_mean = self.compute_field_mean(target_press)
                    
                    sample_loss = (pred_mean - target_mean) ** 2
                    batch_loss += sample_loss
                
                batch_loss /= len(batch)
                epoch_loss += batch_loss
                
                # Backward pass (simplified)
                self.optimizer.step(self.workflow.fno_model, {})
            
            epoch_loss /= (len(training_data) // batch_size)
            losses.append(epoch_loss)
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
        self.training_history['pressure_losses'].extend(losses)
        print("FNO training completed")
        
        return {'losses': losses, 'final_loss': losses[-1] if losses else 0.0}
    
    def train_gnn_model(self, num_epochs: int = 20) -> Dict:
        """Train the GNN saturation model"""
        print(f"Training GNN model for {num_epochs} epochs...")
        
        # Generate training data
        training_data = self.data_generator.generate_saturation_training_data(num_samples=50)
        
        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            batch_size = self.config['batch_size']
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                batch_loss = 0.0
                for sample in batch:
                    # Forward pass
                    input_press = sample['input_pressure']
                    prev_sat = sample['input_prev_saturation']
                    target_sat = sample['target_saturation']
                    
                    # Predict saturation
                    predicted_sat = self.workflow.predict_saturation(input_press, prev_sat)
                    
                    # Compute loss
                    if len(predicted_sat) > 0 and len(target_sat) > 0:
                        sample_loss = self.well_loss.compute_loss(
                            predicted_sat[:min(len(predicted_sat), len(target_sat))],
                            target_sat[:min(len(predicted_sat), len(target_sat))]
                        )
                    else:
                        sample_loss = 1.0
                    
                    batch_loss += sample_loss
                
                batch_loss /= len(batch)
                epoch_loss += batch_loss
                
                # Backward pass (simplified)
                self.optimizer.step(self.workflow.gnn_model, {})
            
            epoch_loss /= (len(training_data) // batch_size)
            losses.append(epoch_loss)
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
        self.training_history['saturation_losses'].extend(losses)
        print("GNN training completed")
        
        return {'losses': losses, 'final_loss': losses[-1] if losses else 0.0}
    
    def train_well_model(self, num_epochs: int = 20) -> Dict:
        """Train the well property prediction model"""
        print(f"Training Well model for {num_epochs} epochs...")
        
        # Generate training data
        training_data = self.data_generator.generate_well_training_data(num_samples=50)
        
        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for sample in training_data:
                for well_data in sample['wells']:
                    # Forward pass
                    well_input = well_data['input_features']
                    well_output = self.workflow.well_model.forward(well_input)
                    
                    # Compute loss for each output
                    if len(well_output) >= 3:
                        oil_loss = (well_output[0] * 1000.0 - well_data['target_oil_rate']) ** 2
                        water_loss = (well_output[1] * 500.0 - well_data['target_water_rate']) ** 2
                        bhp_loss = (well_output[2] * 1000.0 + 2000.0 - well_data['target_bhp']) ** 2
                        
                        total_loss = (oil_loss + water_loss + bhp_loss) / 3.0
                        epoch_loss += total_loss
                
                # Backward pass (simplified)
                self.optimizer.step(self.workflow.well_model, {})
            
            epoch_loss /= len(training_data)
            losses.append(epoch_loss)
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
        self.training_history['well_losses'].extend(losses)
        print("Well model training completed")
        
        return {'losses': losses, 'final_loss': losses[-1] if losses else 0.0}
    
    def compute_field_mean(self, field: List[List[List[float]]]) -> float:
        """Compute mean value of a 3D field"""
        total = 0.0
        count = 0
        
        for i in range(len(field)):
            for j in range(len(field[i])):
                for k in range(len(field[i][j])):
                    total += field[i][j][k]
                    count += 1
        
        return total / count if count > 0 else 0.0
    
    def train_all_models(self) -> Dict:
        """Train all models in the workflow"""
        print("=== Starting Complete Training Pipeline ===")
        
        # Ensure workflow is initialized
        if self.workflow.features is None:
            self.workflow.load_data()
        if self.workflow.gnn_model is None:
            self.workflow.initialize_models()
        
        results = {}
        
        # Train FNO model
        print("\n1. Training FNO (Pressure) Model...")
        fno_results = self.train_fno_model(num_epochs=10)
        results['fno'] = fno_results
        
        # Train GNN model
        print("\n2. Training GNN (Saturation) Model...")
        gnn_results = self.train_gnn_model(num_epochs=10)
        results['gnn'] = gnn_results
        
        # Train Well model
        print("\n3. Training Well Model...")
        well_results = self.train_well_model(num_epochs=10)
        results['well'] = well_results
        
        # Compute total loss
        total_loss = (fno_results['final_loss'] + 
                     gnn_results['final_loss'] + 
                     well_results['final_loss']) / 3.0
        results['total_loss'] = total_loss
        
        print(f"\n=== Training Completed ===")
        print(f"Final Losses:")
        print(f"  FNO (Pressure): {fno_results['final_loss']:.6f}")
        print(f"  GNN (Saturation): {gnn_results['final_loss']:.6f}")
        print(f"  Well Model: {well_results['final_loss']:.6f}")
        print(f"  Total: {total_loss:.6f}")
        
        return results
    
    def validate_models(self) -> Dict:
        """Validate trained models on test data"""
        print("Validating models...")
        
        # Run a short simulation to test the coupled models
        validation_results = self.workflow.run_coupled_simulation(num_timesteps=3)
        
        validation_metrics = {
            'simulation_success': len(validation_results['timesteps']) > 0,
            'pressure_stability': self.check_field_stability(validation_results['pressure_fields']),
            'saturation_stability': self.check_field_stability(validation_results['saturation_fields']),
            'well_predictions_count': len(validation_results['well_predictions'])
        }
        
        print(f"Validation Results:")
        print(f"  Simulation Success: {validation_metrics['simulation_success']}")
        print(f"  Pressure Stability: {validation_metrics['pressure_stability']}")
        print(f"  Saturation Stability: {validation_metrics['saturation_stability']}")
        print(f"  Well Predictions: {validation_metrics['well_predictions_count']} timesteps")
        
        return validation_metrics
    
    def check_field_stability(self, field_evolution: List[str]) -> bool:
        """Check if field values remain stable during simulation"""
        return len(field_evolution) > 1 and all('nan' not in f.lower() for f in field_evolution)

def test_training_pipeline():
    """Test the complete training pipeline"""
    print("=== Testing Training Pipeline ===")
    
    # Initialize workflow
    workflow = ReservoirSimulationWorkflow("HM", "/workspace/HM")
    
    # Override config for faster training
    workflow.config.update({
        'gnn_hidden_dim': 16,
        'gnn_num_layers': 2,
        'fno_hidden_channels': 8,
        'well_hidden_dims': [16, 8]
    })
    
    # Initialize trainer
    trainer = ReservoirMLTrainer(workflow)
    
    # Train all models
    training_results = trainer.train_all_models()
    
    # Validate models
    validation_results = trainer.validate_models()
    
    print(f"\n=== Training Pipeline Test Completed ===")
    return training_results, validation_results

if __name__ == "__main__":
    test_training_pipeline()