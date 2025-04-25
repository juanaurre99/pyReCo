"""
Experiment manager for hyperparameter optimization.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import os
import yaml
import json
import pickle

from .config import ExperimentConfig
from .tasks import TaskFactory
from ..search.strategies import SearchStrategy
from ..search.composite import CompositeSearch
from ..search.random import RandomSearch
from .engine import ExecutionEngine

logger = logging.getLogger(__name__)

# Lazy import of BayesianSearch
def _get_bayesian_search():
    try:
        from ..search.bayesian import BayesianSearch
        return BayesianSearch
    except ImportError:
        logger.warning("scikit-optimize not found. BayesianSearch will fall back to RandomSearch. Install scikit-optimize for Bayesian optimization support.")
        return RandomSearch


class ExperimentManager:
    """
    Manages experiment configuration and execution.
    
    This class is responsible for:
    - Loading and validating experiment configurations
    - Creating search strategies based on configuration
    - Managing the experiment lifecycle
    - Coordinating between search strategies and execution engine
    """
    
    def __init__(self, config: Union[str, Path, Dict[str, Any], ExperimentConfig]):
        """
        Initialize the experiment manager.
        
        Args:
            config: Either a path to a YAML/JSON config file,
                   a dictionary containing the config,
                   or an ExperimentConfig object
        """
        self.config = self._load_config(config)
        self._validate_config()
        self.search_strategy = self._create_search_strategy()
        self.task_data = self._create_task()
        self.engine = None
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentManager':
        """
        Create ExperimentManager from configuration dictionary.
        
        Args:
            config_dict: Dictionary containing experiment configuration
            
        Returns:
            ExperimentManager instance
        """
        config = ExperimentConfig.from_dict(config_dict)
        return cls(config)
    
    def _load_config(self, config: Union[str, Path, Dict[str, Any], ExperimentConfig]) -> ExperimentConfig:
        """Load configuration from various input types."""
        if isinstance(config, ExperimentConfig):
            return config
        elif isinstance(config, (str, Path)):
            path = Path(config)
            if path.suffix.lower() == '.yaml':
                return ExperimentConfig.from_yaml(path)
            elif path.suffix.lower() == '.json':
                return ExperimentConfig.from_json(path)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        elif isinstance(config, dict):
            try:
                return ExperimentConfig.from_dict(config)
            except KeyError as e:
                raise ValueError(f"Missing required configuration field: {e}")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid configuration: {e}")
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")
    
    def _validate_config(self):
        """Validate the experiment configuration."""
        # Validate task configuration
        if self.config.task.train_ratio + self.config.task.validation_ratio > 1.0:
            raise ValueError("Train and validation ratios sum to more than 1.0")
        
        # Validate model configuration
        if self.config.model.type != "reservoir":
            raise ValueError(f"Unsupported model type: {self.config.model.type}")
        
        # Validate search space
        for param_name, param_config in self.config.search_space.parameters.items():
            if param_config.type not in ["float", "int", "categorical"]:
                raise ValueError(f"Unsupported parameter type for {param_name}: {param_config.type}")
            if param_config.type in ["float", "int"]:
                if len(param_config.range) != 2:
                    raise ValueError(f"Range for {param_name} must be [min, max]")
                if param_config.range[0] >= param_config.range[1]:
                    raise ValueError(f"Invalid range for {param_name}: min >= max")
    
    def _create_search_strategy(self) -> SearchStrategy:
        """Create the search strategy based on configuration."""
        if self.config.optimization.strategy == "random":
            return RandomSearch(self._get_param_ranges())
        elif self.config.optimization.strategy == "bayesian":
            BayesianSearch = _get_bayesian_search()
            return BayesianSearch(self._get_param_ranges())
        elif self.config.optimization.strategy == "composite":
            strategies = []
            for strategy_config in self.config.optimization.strategies:
                if strategy_config.type == "random":
                    strategies.append(RandomSearch(self._get_param_ranges()))
                elif strategy_config.type == "bayesian":
                    BayesianSearch = _get_bayesian_search()
                    strategies.append(BayesianSearch(self._get_param_ranges()))
                else:
                    raise ValueError(f"Unsupported strategy type: {strategy_config.type}")
            return CompositeSearch(strategies)
        else:
            raise ValueError(f"Unsupported optimization strategy: {self.config.optimization.strategy}")
    
    def _get_param_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Convert search space configuration to parameter ranges."""
        param_ranges = {}
        for name, param in self.config.search_space.parameters.items():
            if param.type in ['float', 'int']:
                param_ranges[name] = {
                    'type': param.type,
                    'range': tuple(param.range)
                }
            elif param.type == 'categorical':
                param_ranges[name] = {
                    'type': param.type,
                    'values': param.values
                }
        return param_ranges
    
    def _create_task(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create the task dataset based on configuration."""
        return TaskFactory.create_task(self.config.task)
    
    def get_next_trial_config(self) -> Dict[str, Any]:
        """Get configuration for the next trial from the search strategy."""
        suggested_params = self.search_strategy.suggest()
        return {
            "model": self.config.model,
            "task": self.config.task,
            "hyperparameters": suggested_params,
            "data": {
                "X_train": self.task_data[0],
                "X_test": self.task_data[1],
                "y_train": self.task_data[2],
                "y_test": self.task_data[3]
            }
        }
    
    def update_search_strategy(self, params: Dict[str, Any], score: float):
        """
        Update the search strategy with trial results.
        
        For failed trials (score = inf), we use a large finite value instead
        to ensure the optimizer can handle it properly.
        
        Args:
            params: Dictionary containing trial configuration
            score: Performance score (lower is better)
        """
        if np.isinf(score):
            # Use a large finite value instead of infinity
            score = 1e10  # Large enough to be considered a bad result
        
        # Convert numpy types to Python types for the search strategy
        config = {
            'nodes': int(params['nodes']),
            'density': float(params['density']),
            'activation': str(params['activation']),
            'leakage_rate': float(params['leakage_rate']),
            'alpha': float(params['alpha'])
        }
        self.search_strategy.observe(config, score)
    
    @property
    def max_trials(self) -> int:
        """Get the maximum number of trials for this experiment."""
        return self.config.optimization.max_trials
    
    @property
    def experiment_name(self) -> str:
        """Get the experiment name."""
        return self.config.name
    
    @property
    def metrics_config(self) -> Dict[str, Any]:
        """Get the metrics configuration."""
        return {
            "primary": self.config.metrics.primary,
            "secondary": self.config.metrics.secondary,
            "resource": self.config.metrics.resource
        }
    
    def set_task_data(self, train_data: Tuple, val_data: Tuple) -> None:
        """
        Set task data for the experiment.
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
        """
        task_config = {
            'input_shape': [self.config.task.sequence_length, self.config.task.input_dim],
            'output_shape': [self.config.task.sequence_length, self.config.task.output_dim],
            'train_data': train_data,
            'val_data': val_data
        }
        self.engine = ExecutionEngine(task_config)
    
    def run_trial(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a trial with the given configuration.
        
        Args:
            config: Trial configuration
            
        Returns:
            Dictionary containing trial results
        """
        if self.engine is None:
            raise RuntimeError("Task data not set. Call set_task_data first.")
        
        # Map hyperparameters to model parameters
        params = {
            'nodes': config['hyperparameters']['nodes'],
            'density': config['hyperparameters']['density'],
            'activation': config['hyperparameters']['activation'],
            'leakage_rate': config['hyperparameters']['leakage_rate'],
            'alpha': config['hyperparameters']['alpha']
        }
        
        # Run trial
        return self.engine.run_trial(params)
    
    def _save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Save experiment results to a JSON file.
        
        Args:
            results: Dictionary containing experiment results
            output_dir: Directory to save results in
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy arrays and types to Python types for JSON serialization
        def convert_numpy(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'RC':
                # Convert RC model to a string representation
                return f"RC_Model(nodes={obj.reservoir_layer.nodes})"
            return obj
        
        # Save results to JSON file
        output_file = os.path.join(output_dir, "results.json")
        with open(output_file, "w") as f:
            json.dump(convert_numpy(results), f, indent=2)
    
    def _save_best_model(self, model: Any, output_dir: str) -> None:
        """
        Save the best model to a pickle file.
        
        Args:
            model: Best performing model
            output_dir: Directory to save model in
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model to pickle file
        output_file = os.path.join(output_dir, "best_model.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(model, f)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete experiment.
        
        Returns:
            Dictionary containing:
                - trials: List of trial results
                - best_config: Best hyperparameter configuration
                - best_score: Best score achieved
                - resources: Aggregate resource usage
        """
        trials = []
        best_score = float('inf')
        best_config = None
        best_model = None
        total_runtime = 0
        total_memory = 0
        
        for _ in range(self.max_trials):
            # Get next trial configuration
            trial_config = self.get_next_trial_config()
            
            # Run trial
            result = self.run_trial(trial_config)
            trials.append(result)
            
            # Update search strategy if trial was successful
            if 'metrics' in result:
                score = result['metrics']['mse']
                
                # Track best result
                if score < best_score:
                    best_score = score
                    best_config = trial_config['hyperparameters']
                    best_model = result.get('model')
                
                # Update search strategy after tracking best result
                self.update_search_strategy(trial_config['hyperparameters'], score)
            
            # Accumulate resource usage
            total_runtime += result['resources']['runtime']
            total_memory += result['resources']['memory_usage']
        
        # Calculate average resource usage
        n_trials = len(trials)
        results = {
            'trials': trials,
            'best_config': best_config,
            'best_score': best_score,
            'resources': {
                'total_runtime': total_runtime,
                'total_memory': total_memory,
                'avg_runtime': total_runtime / n_trials if n_trials > 0 else 0,
                'avg_memory': total_memory / n_trials if n_trials > 0 else 0
            }
        }
        
        # Save best model if available
        if best_model is not None:
            self._save_best_model(best_model, str(self.config.output_dir))
        
        # Save results
        self._save_results(results, str(self.config.output_dir))
        
        return results 