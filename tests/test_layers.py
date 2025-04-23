import pytest
import numpy as np
from pyreco.core.layers import (
    Layer,
    InputLayer,
    ReadoutLayer,
    ReservoirLayer,
    RandomReservoirLayer,
)

# Test Layer (Abstract Base Class)
def test_layer_abstract_class():
    with pytest.raises(TypeError):
        layer = Layer()  # Should raise TypeError as Layer is abstract

# Test InputLayer
class TestInputLayer:
    @pytest.fixture
    def input_layer(self):
        return InputLayer(input_shape=(10, 3))

    def test_input_layer_initialization(self, input_layer):
        assert input_layer.name == "input_layer"
        assert input_layer.shape == (10, 3)
        assert input_layer.n_time == 10
        assert input_layer.n_states == 3
        assert input_layer.weights is None

    def test_input_layer_remove_nodes(self, input_layer):
        # First set some weights
        input_layer.weights = np.random.rand(5, 3)
        
        # Test valid node removal
        input_layer.remove_nodes([0, 2])
        assert input_layer.weights.shape[0] == 3  # 5 - 2 nodes removed

        # Test invalid node removal
        with pytest.raises(ValueError):
            input_layer.remove_nodes([-1])  # Negative index
        with pytest.raises(ValueError):
            input_layer.remove_nodes([10])  # Index out of bounds
        with pytest.raises(TypeError):
            input_layer.remove_nodes("not_a_list")  # Wrong type

    def test_input_layer_update_properties(self, input_layer):
        # Set weights and test property updates
        input_layer.weights = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        input_layer.update_layer_properties()
        assert input_layer.fraction_nonzero_entries == 3/9  # 3 non-zero entries out of 9 total

# Test ReadoutLayer
class TestReadoutLayer:
    @pytest.fixture
    def readout_layer(self):
        return ReadoutLayer(output_shape=(10, 2), fraction_out=0.8)

    def test_readout_layer_initialization(self, readout_layer):
        assert readout_layer.name == "readout_layer"
        assert readout_layer.output_shape == (10, 2)
        assert readout_layer.n_time == 10
        assert readout_layer.n_states == 2
        assert readout_layer.fraction_out == 0.8
        assert readout_layer.weights is None
        assert readout_layer.readout_nodes == []

    def test_readout_layer_remove_nodes(self, readout_layer):
        with pytest.raises(NotImplementedError):
            readout_layer.remove_nodes([0])

# Test ReservoirLayer
class TestReservoirLayer:
    @pytest.fixture
    def reservoir_layer(self):
        return ReservoirLayer(
            nodes=5,
            density=0.5,
            activation="tanh",
            leakage_rate=0.5,
            fraction_input=0.8,
            init_res_sampling="random_normal",
            seed=42
        )

    def test_reservoir_layer_initialization(self, reservoir_layer):
        assert reservoir_layer.name == "reservoir_layer"
        assert reservoir_layer.nodes == 5
        assert reservoir_layer.density == 0.5
        assert reservoir_layer.activation == "tanh"
        assert reservoir_layer.leakage_rate == 0.5
        assert reservoir_layer.fraction_input == 0.8
        assert reservoir_layer.weights is None
        assert reservoir_layer.initial_res_states is None

    def test_reservoir_activation_functions(self, reservoir_layer):
        # Test tanh activation
        x = np.array([0, 1, -1])
        tanh_result = reservoir_layer.activation_fun(x)
        assert np.allclose(tanh_result, np.tanh(x))

        # Test sigmoid activation
        reservoir_layer.activation = "sigmoid"
        sigmoid_result = reservoir_layer.activation_fun(x)
        assert np.allclose(sigmoid_result, 1 / (1 + np.exp(-x)))

        # Test invalid activation
        reservoir_layer.activation = "invalid"
        with pytest.raises(ValueError):
            reservoir_layer.activation_fun(x)

    def test_set_weights(self, reservoir_layer):
        weights = np.random.rand(5, 5)
        reservoir_layer.set_weights(weights)
        assert np.array_equal(reservoir_layer.weights, weights)
        assert reservoir_layer.nodes == 5
        assert 0 <= reservoir_layer.density <= 1
        assert reservoir_layer.spec_rad is not None

    def test_set_initial_state(self, reservoir_layer):
        initial_state = np.random.rand(5)
        reservoir_layer.set_initial_state(initial_state)
        assert np.array_equal(reservoir_layer.initial_res_states, initial_state)

        # Test invalid initial state
        with pytest.raises(ValueError):
            reservoir_layer.set_initial_state(np.random.rand(3))  # Wrong size

    def test_remove_nodes(self, reservoir_layer):
        # First set some weights and initial state
        weights = np.random.rand(5, 5)
        initial_state = np.random.rand(5)
        reservoir_layer.set_weights(weights)
        reservoir_layer.set_initial_state(initial_state)

        # Test valid node removal
        reservoir_layer.remove_nodes([0, 2])
        assert reservoir_layer.weights.shape == (3, 3)
        assert reservoir_layer.initial_res_states.shape == (3,)

        # Test invalid node removal
        with pytest.raises(ValueError):
            reservoir_layer.remove_nodes([-1])  # Negative index
        with pytest.raises(ValueError):
            reservoir_layer.remove_nodes([10])  # Index out of bounds
        with pytest.raises(TypeError):
            reservoir_layer.remove_nodes("not_a_list")  # Wrong type

# Test RandomReservoirLayer
class TestRandomReservoirLayer:
    @pytest.fixture
    def random_reservoir(self):
        return RandomReservoirLayer(
            nodes=5,
            density=0.5,
            activation="tanh",
            leakage_rate=0.5,
            fraction_input=0.8,
            spec_rad=0.9,
            init_res_sampling="random_normal",
            seed=42
        )

    def test_random_reservoir_initialization(self, random_reservoir):
        assert random_reservoir.name == "reservoir_layer"
        assert random_reservoir.nodes == 5
        assert random_reservoir.density == 0.5
        assert random_reservoir.spec_rad == 0.9
        assert random_reservoir.weights is not None
        assert random_reservoir.weights.shape == (5, 5)

    def test_random_reservoir_properties(self, random_reservoir):
        # Test that the generated network has the expected properties
        assert 0 <= random_reservoir.density <= 1
        assert random_reservoir.spec_rad is not None
        assert random_reservoir.nodes == 5

    def test_random_reservoir_seed(self):
        # Test that the same seed produces the same network
        reservoir1 = RandomReservoirLayer(nodes=5, seed=42)
        reservoir2 = RandomReservoirLayer(nodes=5, seed=42)
        assert np.array_equal(reservoir1.weights, reservoir2.weights)

        # Different seeds should produce different networks
        reservoir3 = RandomReservoirLayer(nodes=5, seed=43)
        assert not np.array_equal(reservoir1.weights, reservoir3.weights) 