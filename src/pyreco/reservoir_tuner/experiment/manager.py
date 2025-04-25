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
    
    def __init__(self, config: Dict[str, Any] or ExperimentConfig):
        """
        Initialize the experiment manager.
        
        Args:
            config: Experiment configuration dictionary or ExperimentConfig object
        """
        self.config = self._load_config(config)
        self._validate_config()
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize search strategy
        self.search_strategy = self._create_search_strategy()
        
        # Create task data
        X_train, X_test, y_train, y_test = self._create_task()
        
        # Create task configuration for execution engine
        task_config = {
            'input_shape': (self.config.task.sequence_length, self.config.task.input_dim),
            'output_shape': (self.config.task.sequence_length, self.config.task.output_dim),
            'train_data': (X_train, y_train),
            'val_data': (X_test, y_test)
        }
        
        # Initialize execution engine with task configuration
        self.engine = ExecutionEngine(task_config)
        
        # Initialize results tracking
        self.trials = []
        self.best_score = float('inf')
        self.best_config = None
        self.best_model = None
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentManager':
        """Create an ExperimentManager from a dictionary."""
        return cls(config_dict)
    
    def _load_config(self, config: Dict[str, Any] or ExperimentConfig) -> ExperimentConfig:
        """Load configuration from dictionary or file."""
        if isinstance(config, ExperimentConfig):
            return config
            
        if isinstance(config, dict):
            return ExperimentConfig.from_dict(config)
            
        raise ValueError("Config must be either a dictionary or ExperimentConfig object")
    
    def _validate_config(self):
        """Validate the configuration."""
        required_sections = [
            "task", "model", "search_space", "optimization", "metrics"
        ]
        
        for section in required_sections:
            if not hasattr(self.config, section):
                raise ValueError(f"Missing required section: {section}")
    
    def _create_search_strategy(self) -> SearchStrategy:
        """Create search strategy based on configuration."""
        strategy_type = self.config.optimization.strategy
        
        # Convert SearchSpaceConfig to dictionary format
        param_ranges = {}
        for name, param_config in self.config.search_space.parameters.items():
            param_ranges[name] = {
                'type': param_config.type,
                'range': param_config.range,
                'values': param_config.values
            }
        
        if strategy_type == "random":
            return RandomSearch(
                parameter_space=param_ranges,
                seed=self.config.seed
            )
        elif strategy_type == "bayesian":
            return _get_bayesian_search()(
                parameter_space=param_ranges,
                seed=self.config.seed
            )
        elif strategy_type == "composite":
            # Create individual strategies first
            strategies = []
            for strategy_config in self.config.optimization.strategies:
                if strategy_config.type == "random":
                    strategies.append(RandomSearch(
                        parameter_space=param_ranges,
                        seed=self.config.seed
                    ))
                elif strategy_config.type == "bayesian":
                    strategies.append(_get_bayesian_search()(
                        parameter_space=param_ranges,
                        seed=self.config.seed
                    ))
                else:
                    raise ValueError(f"Unsupported strategy type: {strategy_config.type}")
            
            # Get exploration factor from the first strategy config
            exploration_factor = self.config.optimization.strategies[0].exploration_factor if self.config.optimization.strategies else 2.0
            
            return CompositeSearch(
                strategies=strategies,
                exploration_factor=exploration_factor
            )
        else:
            raise ValueError(f"Unsupported search strategy: {strategy_type}")
    
    def _get_param_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter ranges from configuration."""
        if isinstance(self.config.search_space, dict):
            return self.config.search_space
        else:
            # Convert SearchSpaceConfig to dictionary format
            param_ranges = {}
            for name, param_config in self.config.search_space.parameters.items():
                param_ranges[name] = {
                    'type': param_config.type,
                    'range': param_config.range,
                    'values': param_config.values
                }
            return param_ranges
    
    def _create_task(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create the task dataset based on configuration."""
        return TaskFactory.create_task(self.config.task)
    
    def get_next_trial_config(self) -> Dict[str, Any]:
        """Get next trial configuration from search strategy."""
        return self.search_strategy.suggest()
    
    def update_search_strategy(self, params: Dict[str, Any], score: float):
        """Update search strategy with trial results."""
        self.search_strategy.observe(params, score)
    
    @property
    def max_trials(self) -> int:
        """Get maximum number of trials."""
        return self.config.optimization.max_trials
    
    @property
    def experiment_name(self) -> str:
        """Get experiment name."""
        return self.config.experiment.name
    
    @property
    def metrics_config(self) -> Dict[str, Any]:
        """Get metrics configuration."""
        return self.config.metrics
    
    def run_trial(self, trial_config: Dict[str, Any]) -> Dict[str, float]:
        """Run a single optimization trial."""
        return self.engine.run_trial(trial_config)
    
    def run_experiment(self, n_trials: int):
        """Run the full optimization experiment."""
        best_score = float("-inf")
        best_params = None
        
        for trial in range(n_trials):
            logger.info(f"Running trial {trial + 1}/{n_trials}")
            
            # Get next trial configuration
            trial_config = self.get_next_trial_config()
            
            # Run trial
            results = self.run_trial(trial_config)
            
            # Update search strategy
            self.update_search_strategy(trial_config, results["score"])
            
            # Update best results
            if results["score"] > best_score:
                best_score = results["score"]
                best_params = trial_config
                
                # Save best results
                self._save_results(best_params, best_score)
    
    def _save_results(self, params: Dict[str, Any], score: float):
        """Save best results to file."""
        # Convert numpy types to Python native types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results = {
            "params": convert_numpy(params),
            "score": convert_numpy(score)
        }
        
        with open(self.output_dir / "best_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    def _save_best_model(self, model: Any, output_dir: str) -> None:
        """Save the best model to disk."""
        output_path = Path(output_dir)
        model_path = output_path / "best_model.pkl"
        with open(model_path, 'wb') as f:
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
        max_memory = 0
        
        for _ in range(self.max_trials):
            # Get next trial configuration
            trial_config = self.get_next_trial_config()
            
            # Run trial
            result = self.run_trial(trial_config)
            
            # Update best results
            if result['metrics']['mse'] < best_score:
                best_score = result['metrics']['mse']
                best_config = trial_config
                best_model = result.get('model')
            
            # Update search strategy
            self.update_search_strategy(trial_config, result['metrics']['mse'])
            
            # Track resources
            total_runtime += result['resources']['runtime']
            max_memory = max(max_memory, result['resources']['memory_usage'])
            
            # Store trial results
            trials.append(result)
        
        # Prepare final results
        results = {
            'trials': trials,
            'best_config': best_config,
            'best_score': best_score,
            'resources': {
                'total_runtime': total_runtime,
                'max_memory_usage': max_memory
            }
        }
        
        # Save results
        self._save_results(best_config, best_score)
        
        # Save best model if available
        if best_model is not None:
            self._save_best_model(best_model, str(self.output_dir))
        
        return results 