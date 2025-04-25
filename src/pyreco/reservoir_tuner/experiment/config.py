"""
Configuration classes for experiment management.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    size: int
    type: Optional[str] = None
    activation: Optional[str] = None
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
            input_layer=LayerConfig(**data['input_layer']),
            reservoir_layer=LayerConfig(**data['reservoir_layer']),
            output_layer=LayerConfig(**data['output_layer'])
        )


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
        return cls(
            name=experiment_data['name'],
            description=experiment_data['description'],
            seed=experiment_data['seed'],
            task=TaskConfig.from_dict(data['task']),
            model=ModelConfig.from_dict(data['model']),
            search_space=SearchSpaceConfig.from_dict(data['search_space']),
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