"""
Task factory for creating experiment tasks.
"""

from typing import Dict, Any, Tuple
import numpy as np
from ...utils.utils_data import (
    vector_to_vector,
    sequence_to_scalar,
    sequence_to_sequence,
)
from .config import TaskConfig


class TaskFactory:
    """Factory class for creating experiment tasks."""
    
    TASK_TYPES = {
        "vector_to_vector": vector_to_vector,
        "sequence_to_scalar": sequence_to_scalar,
        "sequence_to_sequence": sequence_to_sequence,
    }
    
    TASK_NAMES = {
        "sine_prediction",
        "sine_to_cosine",
        "sin_to_cos2",
    }
    
    @classmethod
    def create_task(cls, config: TaskConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a task based on configuration.
        
        Args:
            config: Task configuration containing:
                - type: Task type (vector_to_vector, sequence_to_scalar, sequence_to_sequence)
                - name: Task name (sine_prediction, sine_to_cosine, sin_to_cos2)
                - input_dim: Number of input dimensions
                - sequence_length: Length of sequences (for sequence tasks)
                
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if config.type not in cls.TASK_TYPES:
            raise ValueError(f"Unsupported task type: {config.type}")
        
        if config.name not in cls.TASK_NAMES:
            raise ValueError(f"Unsupported task name: {config.name}")
        
        task_fn = cls.TASK_TYPES[config.type]
        
        # Generate task data based on task type
        if config.type == "vector_to_vector":
            X_train, X_test, y_train, y_test = task_fn(
                name=config.name,
                n_batch=100,  # Default batch size
                n_states=config.input_dim
            )
        elif config.type == "sequence_to_scalar":
            X_train, X_test, y_train, y_test = task_fn(
                name=config.name,
                n_batch=100,  # Default batch size
                n_states=config.input_dim,
                n_time=config.sequence_length
            )
        else:  # sequence_to_sequence
            X_train, X_test, y_train, y_test = task_fn(
                name=config.name,
                n_batch=100,  # Default batch size
                n_states=config.input_dim,
                n_time=config.sequence_length
            )
        
        # Reshape data if needed
        if config.type == "vector_to_vector":
            X_train = X_train.reshape(-1, config.input_dim)
            X_test = X_test.reshape(-1, config.input_dim)
            y_train = y_train.reshape(-1, config.output_dim)
            y_test = y_test.reshape(-1, config.output_dim)
        
        return X_train, X_test, y_train, y_test 