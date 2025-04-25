"""
Bayesian optimization search strategy implementation.
"""
from typing import Dict, Any, List, Optional
import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical

from .strategies import SearchStrategy

class BayesianSearch(SearchStrategy):
    """
    Implements Bayesian optimization for hyperparameter search using scikit-optimize.
    """
    def __init__(self, parameter_space: Dict[str, Any], seed: Optional[int] = None):
        """
        Initialize the Bayesian optimization search strategy.
        
        Args:
            parameter_space: Dictionary defining the search space for each parameter
            seed: Random seed for reproducibility
        """
        super().__init__(parameter_space)
        self.dimensions = self._create_dimensions()
        self.optimizer = Optimizer(
            dimensions=self.dimensions,
            random_state=seed,
            base_estimator="GP",
            n_initial_points=3
        )
        self.param_names = list(parameter_space.keys())
        
    def _create_dimensions(self) -> List[Any]:
        """Create scikit-optimize dimensions from parameter space."""
        dimensions = []
        for name, config in self.parameter_space.items():
            if isinstance(config, tuple):
                # Handle tuple format (min, max)
                dimensions.append(Real(config[0], config[1], name=name))
            else:
                # Handle dict format with type information
                if config['type'] == 'float':
                    dimensions.append(Real(config['range'][0], config['range'][1], name=name))
                elif config['type'] == 'int':
                    dimensions.append(Integer(config['range'][0], config['range'][1], name=name))
                elif config['type'] == 'categorical':
                    dimensions.append(Categorical(config['values'], name=name))
                else:
                    raise ValueError(f"Unsupported parameter type: {config['type']}")
        return dimensions
        
    def suggest(self) -> Dict[str, Any]:
        """
        Suggest a configuration using Bayesian optimization.
        
        Returns:
            Dict containing suggested parameter values
        """
        suggestion = self.optimizer.ask()
        return dict(zip(self.param_names, suggestion))
        
    def observe(self, params: Dict[str, Any], score: float) -> None:
        """
        Record an observation of the performance for a set of parameters.
        
        Args:
            params: Parameter configuration that was evaluated
            score: Performance score for the configuration (lower is better)
        """
        x = [params[name] for name in self.param_names]
        self.optimizer.tell(x, score) 