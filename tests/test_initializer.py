import pytest
import numpy as np
from pyreco.core.initializer import NetworkInitializer

class TestNetworkInitializer:
    @pytest.fixture
    def initializer(self):
        return NetworkInitializer()

    def test_initialization(self):
        # Test default initialization
        init = NetworkInitializer()
        assert init.method == "random"

        # Test custom initialization
        init = NetworkInitializer(method="random_normal")
        assert init.method == "random_normal"

    def test_random_initialization(self):
        init = NetworkInitializer(method="random")
        shape = (5, 3)
        states = init.gen_initial_states(shape)
        
        # Test shape
        assert states.shape == shape
        
        # Test values are between 0 and 1
        assert np.all(states >= 0)
        assert np.all(states <= 1)
        
        # Test normalization
        assert np.max(np.abs(states)) == 1.0

    def test_random_normal_initialization(self):
        init = NetworkInitializer(method="random_normal")
        shape = (5, 3)
        states = init.gen_initial_states(shape)
        
        # Test shape
        assert states.shape == shape
        
        # Test normalization
        assert np.max(np.abs(states)) == 1.0

    def test_ones_initialization(self):
        init = NetworkInitializer(method="ones")
        shape = (5, 3)
        states = init.gen_initial_states(shape)
        
        # Test shape
        assert states.shape == shape
        
        # Test all values are 1
        assert np.all(states == 1.0)

    def test_zeros_initialization(self):
        init = NetworkInitializer(method="zeros")
        shape = (5, 3)
        states = init.gen_initial_states(shape)
        
        # Test shape
        assert states.shape == shape
        
        # Test all values are 0
        assert np.all(states == 0.0)

    def test_invalid_method(self):
        init = NetworkInitializer(method="invalid_method")
        with pytest.raises(ValueError):
            init.gen_initial_states((5, 3))

    def test_different_shapes(self):
        init = NetworkInitializer()
        
        # Test scalar shape
        states = init.gen_initial_states(5)
        assert states.shape == (5,)
        
        # Test list shape
        states = init.gen_initial_states([5, 3])
        assert states.shape == (5, 3)
        
        # Test tuple shape
        states = init.gen_initial_states((5, 3))
        assert states.shape == (5, 3)

    def test_normalization(self):
        init = NetworkInitializer(method="random")
        shape = (5, 3)
        states = init.gen_initial_states(shape)
        
        # Test that values are normalized to max absolute value of 1
        assert np.max(np.abs(states)) == 1.0
        
        # Test that zeros method is not normalized
        init = NetworkInitializer(method="zeros")
        states = init.gen_initial_states(shape)
        assert np.max(np.abs(states)) == 0.0 