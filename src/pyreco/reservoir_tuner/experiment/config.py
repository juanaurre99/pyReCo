"""
Configuration classes for experiment management.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    type: str
    activation: Optional[str] = None
    size: Optional[int] = None
    connectivity: Optional[float] = None


@dataclass
class ModelConfig:
    """Configuration for the reservoir computing model."""
    type: str
    input_layer: LayerConfig
    reservoir_layer: LayerConfig
    output_layer: LayerConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary."""
        return cls(
            type=data['type'],
            input_layer=LayerConfig(type=data['input_layer']['type'], 
                                  activation=data['input_layer'].get('activation')),
            reservoir_layer=LayerConfig(type=data['reservoir_layer']['type'], 
                                      activation=data['reservoir_layer'].get('activation')),
            output_layer=LayerConfig(type=data['output_layer']['type'], 
                                   activation=data['output_layer'].get('activation'))
        )
    
    def set_layer_sizes(self, task_config: 'TaskConfig', search_space: Dict[str, Any]) -> None:
        """Set layer sizes based on task dimensions and search space."""
        # Input layer size from task input dimension
        self.input_layer.size = task_config.input_dim
        
        # Reservoir layer size from search space if available
        if 'nodes' in search_space:
            self.reservoir_layer.size = search_space['nodes'].get('range', [100])[0]
        else:
            self.reservoir_layer.size = 100  # Default size
            
        # Output layer size from task output dimension
        self.output_layer.size = task_config.output_dim


@dataclass
class TaskConfig:
    """Configuration for the learning task."""
    type: str
    name: str
    input_dim: int
    output_dim: int
    sequence_length: int
    train_ratio: float
    validation_ratio: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskConfig':
        """Create TaskConfig from dictionary."""
        return cls(**data)


@dataclass
class ParameterConfig:
    """Configuration for a hyperparameter search space."""
    type: str
    range: Optional[Union[Tuple[float, float], List[Any]]] = None
    values: Optional[List[Any]] = None

    def __post_init__(self):
        """Validate parameter configuration."""
        if self.type in ['float', 'int']:
            if self.range is None:
                raise ValueError(f"Parameter of type {self.type} must have a range")
            if self.values is not None:
                raise ValueError(f"Parameter of type {self.type} cannot have values")
        elif self.type == 'categorical':
            if self.values is None:
                raise ValueError("Categorical parameter must have values")
            if self.range is not None:
                raise ValueError("Categorical parameter cannot have range")


@dataclass
class SearchSpaceConfig:
    """Configuration for the hyperparameter search space."""
    parameters: Dict[str, ParameterConfig]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchSpaceConfig':
        """Create SearchSpaceConfig from dictionary."""
        parameters = {
            name: ParameterConfig(**param_data)
            for name, param_data in data.items()
        }
        return cls(parameters=parameters)


@dataclass
class StrategyConfig:
    """Configuration for a search strategy."""
    type: str
    weight: float
    exploration_factor: Optional[float] = None


@dataclass
class OptimizationConfig:
    """Configuration for the optimization process."""
    strategy: str
    max_trials: int
    strategies: List[StrategyConfig]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Create OptimizationConfig from dictionary."""
        return cls(
            strategy=data['strategy'],
            max_trials=data['max_trials'],
            strategies=[StrategyConfig(**s) for s in data['strategies']]
        )


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    primary: str
    secondary: List[str]
    resource: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricsConfig':
        """Create MetricsConfig from dictionary."""
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Root configuration for an experiment."""
    name: str
    description: str
    seed: int
    task: TaskConfig
    model: ModelConfig
    search_space: SearchSpaceConfig
    optimization: OptimizationConfig
    metrics: MetricsConfig
    output_dir: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create ExperimentConfig from dictionary."""
        experiment_data = data['experiment']
        task_config = TaskConfig.from_dict(data['task'])
        model_config = ModelConfig.from_dict(data['model'])
        search_space = data['search_space']
        
        # Set layer sizes based on task dimensions and search space
        model_config.set_layer_sizes(task_config, search_space)
        
        return cls(
            name=experiment_data['name'],
            description=experiment_data['description'],
            seed=experiment_data['seed'],
            task=task_config,
            model=model_config,
            search_space=SearchSpaceConfig.from_dict(search_space),
            optimization=OptimizationConfig.from_dict(data['optimization']),
            metrics=MetricsConfig.from_dict(data['metrics']),
            output_dir=Path(experiment_data.get('output_dir', '.'))
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data) 