"""
Execution engine for running hyperparameter optimization trials.
This module handles model instantiation, training, and evaluation.
"""

from typing import Dict, Any, Tuple, Optional
import time
import numpy as np
from pyreco.core.custom_models import RC
from pyreco.core.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.core.optimizers import RidgeSK
import logging

logger = logging.getLogger(__name__)

# Optional import of psutil for resource tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not found. Resource tracking will be limited to runtime only.")

class ExecutionEngine:
    """
    Executes hyperparameter optimization trials.
    """
    
    def __init__(self, task_config: Dict[str, Any]):
        """
        Initialize the execution engine with task configuration.
        
        Args:
            task_config: Task configuration dictionary containing:
                - input_shape: Tuple of (n_time, n_features)
                - output_shape: Tuple of (n_time_out, n_features_out)
                - train_data: Tuple of (X_train, y_train)
                - val_data: Tuple of (X_val, y_val)
        """
        self.task_config = task_config
        self.input_shape = task_config['input_shape']
        self.output_shape = task_config['output_shape']
        self.X_train, self.y_train = task_config['train_data']
        self.X_val, self.y_val = task_config['val_data']
    
    def _create_model(self, params: Dict[str, Any]) -> RC:
        """
        Create a reservoir computing model with the given hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters including:
                - nodes: Number of reservoir nodes
                - density: Network density
                - activation: Activation function ('tanh' or 'sigmoid')
                - leakage_rate: Leakage rate
                
        Returns:
            RC: Configured reservoir computing model
        """
        # Convert numpy types to Python types
        params = {
            'nodes': int(params['nodes']),  # Convert to Python int
            'density': float(params['density']),  # Convert to Python float
            'activation': str(params['activation']),  # Convert to Python str
            'leakage_rate': float(params['leakage_rate']),  # Convert to Python float
            'alpha': float(params['alpha'])  # Convert to Python float
        }
        
        model = RC()
        
        # Add input layer
        model.add(InputLayer(input_shape=self.input_shape))
        
        # Add reservoir layer
        model.add(RandomReservoirLayer(
            nodes=params['nodes'],
            density=params['density'],
            activation=params['activation'],
            leakage_rate=params['leakage_rate']
        ))
        
        # Add readout layer
        model.add(ReadoutLayer(
            output_shape=self.output_shape
        ))
        
        # Compile model
        model.compile(
            optimizer=RidgeSK(alpha=params['alpha']),
            metrics=['mean_squared_error']
        )
        
        return model
    
    def _train_model(self, model: RC, n_init: int = 1) -> Dict[str, Any]:
        """
        Train the model and collect training metrics.
        
        Args:
            model: Reservoir computing model to train
            n_init: Number of initializations to try
            
        Returns:
            Dict[str, Any]: Training history and metrics
        """
        history = model.fit(
            self.X_train,
            self.y_train,
            n_init=n_init,
            store_states=True
        )
        return history
    
    def _evaluate_model(self, model: RC) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            model: Trained reservoir computing model
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        metrics = model.evaluate(
            self.X_val,
            self.y_val,
            metrics=['mean_squared_error', 'mean_absolute_error']
        )
        return {
            'mse': metrics[0],
            'mae': metrics[1]
        }
    
    def run_trial(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single trial with the given hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters to try
            
        Returns:
            Dict[str, Any]: Trial results including:
                - params: Original hyperparameters
                - metrics: Evaluation metrics
                - history: Training history
                - resources: Resource usage statistics
                - model: Trained model (if successful)
                - error: Any error that occurred (if applicable)
        """
        start_time = time.time()
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            start_memory = process.memory_info().rss
        
        try:
            # Create and train model
            model = self._create_model(params)
            history = self._train_model(model)
            metrics = self._evaluate_model(model)
            
            end_time = time.time()
            resources = {
                'runtime': end_time - start_time
            }
            
            if PSUTIL_AVAILABLE:
                end_memory = process.memory_info().rss
                resources.update({
                    'memory_usage': end_memory - start_memory,
                    'cpu_percent': process.cpu_percent()
                })
            else:
                resources.update({
                    'memory_usage': 0,
                    'cpu_percent': 0
                })
            
            return {
                'hyperparameters': params,
                'metrics': metrics,
                'history': history,
                'model': model,
                'resources': resources
            }
            
        except Exception as e:
            end_time = time.time()
            resources = {
                'runtime': end_time - start_time
            }
            
            if PSUTIL_AVAILABLE:
                end_memory = process.memory_info().rss
                resources.update({
                    'memory_usage': end_memory - start_memory,
                    'cpu_percent': process.cpu_percent()
                })
            else:
                resources.update({
                    'memory_usage': 0,
                    'cpu_percent': 0
                })
            
            # Log the error for debugging
            logger.error(f"Error in trial with params {params}: {str(e)}")
            
            return {
                'hyperparameters': params,
                'error': str(e),
                'metrics': {'mse': float('inf'), 'mae': float('inf')},
                'resources': resources
            } 