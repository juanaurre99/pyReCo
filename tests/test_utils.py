import pytest
import numpy as np
import networkx as nx
from pyreco.utils.utils_networks import (
    gen_ER_graph,
    compute_density,
    get_num_nodes,
    compute_spec_rad,
    remove_nodes_from_graph,
)
from pyreco.utils.utils_data import (
    gen_sine,
    gen_cos,
    gen_sincos,
    split_sequence,
    train_test_split,
    sine_pred,
    sine_to_cosine,
    sincos2
)

class TestNetworkUtils:
    def test_gen_ER_graph(self):
        # Test ER graph generation with different parameters
        n_nodes = 100
        density = 0.1
        
        # Test basic generation
        graph = gen_ER_graph(n_nodes, density)
        assert isinstance(graph, np.ndarray)
        assert graph.shape == (n_nodes, n_nodes)
        assert np.all(np.diag(graph) == 0)  # No self-loops
        
        # Test density
        actual_density = compute_density(graph)
        assert abs(actual_density - density) < 0.1  # Allow some variance
        
        # Test with different parameters
        graph_sparse = gen_ER_graph(n_nodes, 0.01)
        graph_dense = gen_ER_graph(n_nodes, 0.5)
        assert compute_density(graph_sparse) < compute_density(graph_dense)

    def test_compute_density(self):
        # Test density computation
        # Create a known graph
        graph = np.zeros((4, 4))
        graph[0, 1] = 1
        graph[1, 2] = 1
        graph[2, 3] = 1
        graph[3, 0] = 1
        
        # 4 edges out of possible 16 (including diagonal)
        expected_density = 4 / 16
        assert abs(compute_density(graph) - expected_density) < 1e-10

    def test_get_num_nodes(self):
        # Test node counting
        graph = np.zeros((10, 10))
        assert get_num_nodes(graph) == 10
        
        # Test with non-square matrix
        with pytest.raises(ValueError):
            get_num_nodes(np.zeros((10, 5)))

    def test_compute_spec_rad(self):
        # Test spectral radius computation
        # Create a known graph with known spectral radius
        graph = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        # For this simple graph, spectral radius should be sqrt(2)
        assert abs(compute_spec_rad(graph) - np.sqrt(2)) < 1e-10

    def test_remove_nodes_from_graph(self):
        # Test node removal
        graph = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        
        # Remove node 1
        new_graph = remove_nodes_from_graph(graph, [1])
        assert new_graph.shape == (3, 3)
        assert np.array_equal(new_graph, np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]))

class TestDataUtils:
    def test_gen_sine(self):
        # Test sine wave generation
        n = 10
        omega = np.pi
        sine_wave = gen_sine(n, omega)
        
        assert len(sine_wave) == n
        assert np.all(sine_wave >= -1) and np.all(sine_wave <= 1)
        assert abs(sine_wave[0]) < 1e-10  # Should start at 0

    def test_gen_cos(self):
        # Test cosine wave generation
        n = 10
        omega = np.pi
        cos_wave = gen_cos(n, omega)
        
        assert len(cos_wave) == n
        assert np.all(cos_wave >= -1) and np.all(cos_wave <= 1)
        assert abs(cos_wave[0] - 1) < 1e-10  # Should start at 1

    def test_gen_sincos(self):
        # Test combined sine-cosine generation
        n = 10
        omega = np.pi
        a_sc = 1
        b_sc = 0.25
        P_sc = 3
        
        sincos_wave = gen_sincos(n, omega, a_sc, b_sc, P_sc)
        
        assert len(sincos_wave) == n
        # Test that the wave is a combination of sine and cosine
        sine_wave = gen_sine(n, omega)
        cos_wave = gen_cos(n, omega)
        expected = a_sc * sine_wave**P_sc + b_sc * cos_wave**P_sc
        assert np.allclose(sincos_wave, expected)

    def test_split_sequence(self):
        # Test sequence splitting
        signal = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10]
        ])
        n_batch = 2
        n_time_in = 2
        n_time_out = 1
        
        x, y = split_sequence(signal, n_batch, n_time_in, n_time_out)
        
        assert x.shape == (n_batch, n_time_in, 2)
        assert y.shape == (n_batch, n_time_out, 2)
        assert np.array_equal(x[0], np.array([[1, 2], [3, 4]]))
        assert np.array_equal(y[0], np.array([[5, 6]]))

    def test_train_test_split(self):
        # Test train-test splitting
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        
        X_train, X_test, y_train, y_test = train_test_split(x, y)
        
        assert len(X_train) == 8  # 80% of 10
        assert len(X_test) == 2   # 20% of 10
        assert len(y_train) == 8
        assert len(y_test) == 2
        assert len(set(X_train) & set(X_test)) == 0  # No overlap

    def test_sine_pred(self):
        # Test sine prediction data generation
        n_batch = 5
        n_time_in = 2
        n_time_out = 1
        n_states = 2
        
        X_train, X_test, y_train, y_test = sine_pred(n_batch, n_time_in, n_time_out, n_states)
        
        assert X_train.shape[0] == 4  # 80% of n_batch
        assert X_test.shape[0] == 1   # 20% of n_batch
        assert X_train.shape[1] == n_time_in
        assert y_train.shape[1] == n_time_out
        assert X_train.shape[2] == n_states
        assert y_train.shape[2] == n_states

    def test_sine_to_cosine(self):
        # Test sine to cosine mapping data generation
        n_batch = 5
        n_time_in = 2
        n_time_out = 1
        n_states = 2
        
        X_train, X_test, y_train, y_test = sine_to_cosine(n_batch, n_time_in, n_time_out, n_states)
        
        assert X_train.shape[0] == 4  # 80% of n_batch
        assert X_test.shape[0] == 1   # 20% of n_batch
        assert X_train.shape[1] == n_time_in
        assert y_train.shape[1] == n_time_out
        assert X_train.shape[2] == n_states
        assert y_train.shape[2] == n_states

    def test_sincos2(self):
        # Test sincos2 data generation
        n_batch = 5
        n_time_in = 2
        n_time_out = 1
        n_states = 2
        
        X_train, X_test, y_train, y_test = sincos2(n_batch, n_time_in, n_time_out, n_states)
        
        assert X_train.shape[0] == 4  # 80% of n_batch
        assert X_test.shape[0] == 1   # 20% of n_batch
        assert X_train.shape[1] == n_time_in
        assert y_train.shape[1] == n_time_out
        assert X_train.shape[2] == n_states
        assert y_train.shape[2] == n_states 