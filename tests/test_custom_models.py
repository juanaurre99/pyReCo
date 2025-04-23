import pytest
import numpy as np
from pyreco.core.custom_models import CustomModel, RC, AutoRC, HybridRC
from pyreco.core.layers import InputLayer, ReservoirLayer, ReadoutLayer


@pytest.fixture
def sample_data():
    # Generate sample input data
    np.random.seed(42)
    X = np.random.randn(1, 10, 3)  # (batch_size, timesteps, features)
    y = np.random.randn(1, 10, 2)  # (batch_size, timesteps, output_features)
    return X, y

@pytest.fixture
def basic_model():
    model = RC()
    # Add required layers
    model.add(InputLayer(input_shape=(10, 3)))
    model.add(ReservoirLayer(
        nodes=50,
        density=0.2,
        activation='tanh',
        leakage_rate=0.5,
        fraction_input=0.8,
        init_res_sampling='random_normal'
    ))
    model.add(ReadoutLayer(output_shape=(10, 2)))
    return model


class TestCustomModel:
    def test_initialization(self):
        model = CustomModel()
        assert isinstance(model, CustomModel)
        assert model.input_layer is None
        assert model.reservoir_layer is None
        assert model.readout_layer is None

    def test_add_layers(self):
        model = CustomModel()

        # Test adding input layer
        input_layer = InputLayer(input_shape=(10, 3))
        model.add(input_layer)
        assert model.input_layer == input_layer

        # Test adding reservoir layer
        reservoir_layer = ReservoirLayer(
            nodes=50,
            density=0.2,
            activation='tanh',
            leakage_rate=0.5,
            fraction_input=0.8,
            init_res_sampling='random_normal'
        )
        model.add(reservoir_layer)
        assert model.reservoir_layer == reservoir_layer

        # Test adding readout layer
        readout_layer = ReadoutLayer(output_shape=(10, 2))
        model.add(readout_layer)
        assert model.readout_layer == readout_layer

    def test_compile(self, basic_model):
        basic_model.compile(optimizer='ridge')
        assert basic_model.optimizer is not None

    def test_fit(self, basic_model, sample_data):
        X, y = sample_data
        basic_model.compile(optimizer='ridge')
        basic_model.fit(X, y)
        assert basic_model.is_fitted

    def test_predict(self, basic_model, sample_data):
        X, y = sample_data
        basic_model.compile(optimizer='ridge')
        basic_model.fit(X, y)
        predictions = basic_model.predict(X)
        assert predictions.shape == y.shape

    def test_evaluate(self, basic_model, sample_data):
        X, y = sample_data
        basic_model.compile(optimizer='ridge')
        basic_model.fit(X, y)
        metrics = basic_model.evaluate(X, y)
        assert isinstance(metrics, dict)

    def test_remove_reservoir_nodes(self, basic_model):
        nodes_to_remove = [0, 1]
        initial_nodes = basic_model.reservoir_layer.nodes
        basic_model.remove_reservoir_nodes(nodes_to_remove)
        assert basic_model.reservoir_layer.nodes == initial_nodes - len(nodes_to_remove)

    def test_compute_reservoir_state(self, basic_model, sample_data):
        X, _ = sample_data
        basic_model.compile(optimizer='ridge')
        
        # Test reservoir state computation
        reservoir_states = basic_model.compute_reservoir_state(X)
        
        # Check output shape
        n_batch, n_time, n_features = X.shape
        n_nodes = basic_model.reservoir_layer.nodes
        assert reservoir_states.shape == (n_batch, n_time + 1, n_nodes)
        
        # Check initial state
        assert np.array_equal(reservoir_states[:, 0], basic_model.reservoir_layer.initial_res_states)
        
        # Check that states are within expected range for tanh activation
        assert np.all(reservoir_states >= -1) and np.all(reservoir_states <= 1)

    def test_compute_reservoir_state_with_different_activations(self, sample_data):
        X, _ = sample_data
        
        # Test with tanh activation
        model_tanh = RC()
        model_tanh.add(InputLayer(input_shape=(10, 3)))
        model_tanh.add(ReservoirLayer(
            nodes=50,
            density=0.2,
            activation='tanh',
            leakage_rate=0.5,
            fraction_input=0.8,
            init_res_sampling='random_normal'
        ))
        model_tanh.add(ReadoutLayer(output_shape=(10, 2)))
        model_tanh.compile(optimizer='ridge')
        
        states_tanh = model_tanh.compute_reservoir_state(X)
        assert np.all(states_tanh >= -1) and np.all(states_tanh <= 1)
        
        # Test with sigmoid activation
        model_sigmoid = RC()
        model_sigmoid.add(InputLayer(input_shape=(10, 3)))
        model_sigmoid.add(ReservoirLayer(
            nodes=50,
            density=0.2,
            activation='sigmoid',
            leakage_rate=0.5,
            fraction_input=0.8,
            init_res_sampling='random_normal'
        ))
        model_sigmoid.add(ReadoutLayer(output_shape=(10, 2)))
        model_sigmoid.compile(optimizer='ridge')
        
        states_sigmoid = model_sigmoid.compute_reservoir_state(X)
        assert np.all(states_sigmoid >= 0) and np.all(states_sigmoid <= 1)

    def test_compute_reservoir_state_with_different_leakage(self, basic_model, sample_data):
        X, _ = sample_data
        basic_model.compile(optimizer='ridge')
        
        # Test with different leakage rates
        leakage_rates = [0.1, 0.5, 0.9]
        previous_states = None
        
        for lr in leakage_rates:
            basic_model.reservoir_layer.leakage_rate = lr
            states = basic_model.compute_reservoir_state(X)
            
            # Check shape consistency
            assert states.shape == (X.shape[0], X.shape[1] + 1, basic_model.reservoir_layer.nodes)
            
            # For higher leakage rates, states should change more between time steps
            if previous_states is not None:
                state_changes = np.abs(states[:, 1:] - states[:, :-1])
                prev_changes = np.abs(previous_states[:, 1:] - previous_states[:, :-1])
                if lr > basic_model.reservoir_layer.leakage_rate:
                    assert np.mean(state_changes) > np.mean(prev_changes)
            
            previous_states = states


class TestRC:
    def test_initialization(self):
        model = RC()
        assert isinstance(model, RC)
        assert model.input_layer is None
        assert model.reservoir_layer is None
        assert model.readout_layer is None

    def test_basic_workflow(self, sample_data):
        X, y = sample_data
        model = RC()

        # Add layers
        model.add(InputLayer(input_shape=(10, 3)))
        model.add(ReservoirLayer(
            nodes=50,
            density=0.2,
            activation='tanh',
            leakage_rate=0.5,
            fraction_input=0.8,
            init_res_sampling='random_normal'
        ))
        model.add(ReadoutLayer(output_shape=(10, 2)))

        # Compile and train
        model.compile(optimizer='ridge')
        model.fit(X, y)

        # Make predictions
        predictions = model.predict(X)
        assert predictions.shape == y.shape


class TestAutoRC:
    def test_initialization(self):
        model = AutoRC()
        assert isinstance(model, AutoRC)
        assert model.input_layer is None
        assert model.reservoir_layer is None
        assert model.readout_layer is None

    def test_predict_ar(self, sample_data):
        X, y = sample_data
        model = AutoRC()

        # Add layers
        model.add(InputLayer(input_shape=(10, 3)))
        model.add(ReservoirLayer(
            nodes=50,
            density=0.2,
            activation='tanh',
            leakage_rate=0.5,
            fraction_input=0.8,
            init_res_sampling='random_normal'
        ))
        model.add(ReadoutLayer(output_shape=(10, 2)))

        # Compile and train
        model.compile(optimizer='ridge')
        model.fit(X, y)

        # Test autoregressive prediction
        n_steps = 5
        predictions = model.predict_ar(X, n_steps)
        assert predictions.shape == (X.shape[0], n_steps, y.shape[2])


class TestHybridRC:
    def test_initialization(self):
        model = HybridRC()
        assert isinstance(model, HybridRC)
        assert model.input_layer is None
        assert model.reservoir_layer is None
        assert model.readout_layer is None

    def test_basic_workflow(self, sample_data):
        X, y = sample_data
        model = HybridRC()

        # Add layers
        model.add(InputLayer(input_shape=(10, 3)))
        model.add(ReservoirLayer(
            nodes=50,
            density=0.2,
            activation='tanh',
            leakage_rate=0.5,
            fraction_input=0.8,
            init_res_sampling='random_normal'
        ))
        model.add(ReadoutLayer(output_shape=(10, 2)))

        # Compile and train
        model.compile(optimizer='ridge')
        model.fit(X, y)

        # Make predictions
        predictions = model.predict(X)
        assert predictions.shape == y.shape 