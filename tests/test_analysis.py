import pytest
import numpy as np
import networkx as nx
from pyreco.analysis.node_analyzer import NodeAnalyzer, available_extractors as node_extractors, map_extractor_names as map_node_extractors
from pyreco.analysis.graph_analyzer import GraphAnalyzer, available_extractors as graph_extractors, map_extractor_names as map_graph_extractors
from pyreco.analysis.remove_transients import (
    RemoveTransients_Res,
    RemoveTransient_Inps,
    RemoveTransient_Outs,
    TransientRemover
)

class TestNodeAnalyzer:
    @pytest.fixture
    def sample_graph(self):
        # Create a sample directed graph for testing
        G = nx.DiGraph()
        G.add_edges_from([
            (0, 1), (1, 2), (2, 0),  # Create a cycle
            (1, 3), (2, 3), (3, 1)   # Add some additional edges
        ])
        return G
    
    @pytest.fixture
    def sample_undirected_graph(self):
        # Create a sample undirected graph for testing
        G = nx.Graph()
        G.add_edges_from([
            (0, 1), (1, 2), (2, 0),  # Create a cycle
            (1, 3), (2, 3)           # Add some additional edges
        ])
        return G

    def test_node_analyzer_init(self):
        # Test initialization with default parameters
        analyzer = NodeAnalyzer()
        assert set(analyzer.quantities) == set(node_extractors().keys())
        
        # Test initialization with specific quantities
        quantities = ['degree', 'in_degree']
        analyzer = NodeAnalyzer(quantities=quantities)
        assert set(analyzer.quantities) == set(quantities)
        
        # Test initialization with invalid quantity
        quantities = ['degree', 'invalid_property']
        analyzer = NodeAnalyzer(quantities=quantities)
        assert set(analyzer.quantities) == {'degree'}  # Invalid property should be removed

    def test_extract_properties_directed(self, sample_graph):
        analyzer = NodeAnalyzer()
        properties = analyzer.extract_properties(sample_graph, node=1)
        
        # Check if all properties are present
        assert set(properties.keys()) == set(analyzer.quantities)
        
        # Check specific property values
        assert properties['in_degree'] == 2  # Node 1 has two incoming edges (from 0 and 3)
        assert properties['out_degree'] == 2  # Node 1 has two outgoing edges (to 2 and 3)
        assert properties['degree'] == 4  # Total degree should be 4 (2 in + 2 out)
        assert 0 <= properties['clustering_coefficient'] <= 1
        assert 0 <= properties['betweenness_centrality'] <= 1
        assert 0 <= properties['pagerank'] <= 1

    def test_extract_properties_undirected(self, sample_undirected_graph):
        analyzer = NodeAnalyzer(['degree'])  # Only test degree for undirected graph
        properties = analyzer.extract_properties(sample_undirected_graph, node=1)
        
        # Check if all properties are present
        assert set(properties.keys()) == {'degree'}
        
        # Check specific property values
        assert properties['degree'] == 3  # Node 1 has three edges in undirected graph

    def test_extract_properties_invalid_node(self, sample_graph):
        analyzer = NodeAnalyzer()
        
        # Test with non-integer node
        with pytest.raises(ValueError):
            analyzer.extract_properties(sample_graph, node=1.5)
        
        # Test with node index out of range
        with pytest.raises(KeyError):  # NetworkX raises KeyError for non-existent nodes
            analyzer.extract_properties(sample_graph, node=10)

    def test_available_extractors(self):
        extractors = node_extractors()
        
        # Check if all expected extractors are present
        expected_extractors = {
            'degree', 'in_degree', 'out_degree',
            'clustering_coefficient', 'betweenness_centrality', 'pagerank'
        }
        assert set(extractors.keys()) == expected_extractors
        
        # Check if all extractors are callable
        for extractor in extractors.values():
            assert callable(extractor)

    def test_map_extractor_names(self):
        # Test with valid property names
        prop_names = ['degree', 'in_degree']
        extractor_dict, extractor_funs = map_node_extractors(prop_names)
        
        assert set(extractor_dict.keys()) == set(prop_names)
        assert len(extractor_funs) == len(prop_names)
        
        # Test with invalid property name
        prop_names = ['degree', 'invalid_property']
        extractor_dict, extractor_funs = map_node_extractors(prop_names)
        
        assert set(extractor_dict.keys()) == {'degree'}
        assert len(extractor_funs) == 1

    def test_list_properties(self):
        analyzer = NodeAnalyzer()
        properties = analyzer.list_properties()
        
        # Check if all expected properties are present
        expected_properties = {
            'degree', 'in_degree', 'out_degree',
            'clustering_coefficient', 'betweenness_centrality', 'pagerank'
        }
        assert set(properties) == expected_properties

class TestGraphAnalyzer:
    @pytest.fixture
    def sample_graph(self):
        # Create a sample directed graph for testing
        G = nx.DiGraph()
        G.add_edges_from([
            (0, 1), (1, 2), (2, 0),  # Create a cycle
            (1, 3), (2, 3), (3, 1)   # Add some additional edges
        ])
        return G
    
    @pytest.fixture
    def sample_undirected_graph(self):
        # Create a sample undirected graph for testing
        G = nx.Graph()
        G.add_edges_from([
            (0, 1), (1, 2), (2, 0),  # Create a cycle
            (1, 3), (2, 3)           # Add some additional edges
        ])
        return G

    def test_graph_analyzer_init(self):
        # Test initialization with default parameters
        analyzer = GraphAnalyzer()
        assert set(analyzer.quantities) == set(graph_extractors().keys())
        
        # Test initialization with specific quantities
        quantities = ['density', 'spectral_radius']
        analyzer = GraphAnalyzer(quantities=quantities)
        assert set(analyzer.quantities) == set(quantities)
        
        # Test initialization with invalid quantity
        quantities = ['density', 'invalid_property']
        analyzer = GraphAnalyzer(quantities=quantities)
        assert set(analyzer.quantities) == {'density'}  # Invalid property should be removed

    def test_extract_properties_directed(self, sample_graph):
        analyzer = GraphAnalyzer()
        properties = analyzer.extract_properties(sample_graph)
        
        # Check if all properties are present
        assert set(properties.keys()) == set(analyzer.quantities)
        
        # Check specific property values
        assert properties['density'] == 6 / (4 * 3)  # 6 edges out of 12 possible edges
        assert 0 < properties['spectral_radius'] <= 2  # For this graph structure
        assert properties['av_in_degree'] == 1.5  # 6 edges / 4 nodes
        assert properties['av_out_degree'] == 1.5  # 6 edges / 4 nodes
        assert 0 <= properties['clustering_coefficient'] <= 1

    def test_extract_properties_undirected(self, sample_undirected_graph):
        analyzer = GraphAnalyzer(['density', 'clustering_coefficient'])  # Only test properties that make sense for undirected graphs
        properties = analyzer.extract_properties(sample_undirected_graph)
        
        # Check if all properties are present
        assert set(properties.keys()) == {'density', 'clustering_coefficient'}
        
        # Check specific property values
        assert properties['density'] == 5 / (4 * 3 / 2)  # 5 edges out of 6 possible edges
        assert 0 <= properties['clustering_coefficient'] <= 1

    def test_extract_properties_numpy_array(self):
        # Create a simple adjacency matrix
        adj_matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        
        analyzer = GraphAnalyzer(['density', 'spectral_radius'])
        properties = analyzer.extract_properties(adj_matrix)
        
        # Check specific property values
        assert properties['density'] == 3 / (3 * 2)  # 3 edges out of 6 possible edges (n*(n-1) for directed)
        assert np.isclose(properties['spectral_radius'], 1.0)  # For this cycle graph

    def test_available_extractors(self):
        extractors = graph_extractors()
        
        # Check if all expected extractors are present
        expected_extractors = {
            'density', 'spectral_radius', 'av_in_degree',
            'av_out_degree', 'clustering_coefficient'
        }
        assert set(extractors.keys()) == expected_extractors
        
        # Check if all extractors are callable
        for extractor in extractors.values():
            assert callable(extractor)

    def test_map_extractor_names(self):
        # Test with valid property names
        prop_names = ['density', 'spectral_radius']
        extractor_dict, extractor_funs = map_graph_extractors(prop_names)
        
        assert set(extractor_dict.keys()) == set(prop_names)
        assert len(extractor_funs) == len(prop_names)
        
        # Test with invalid property name
        prop_names = ['density', 'invalid_property']
        extractor_dict, extractor_funs = map_graph_extractors(prop_names)
        
        assert set(extractor_dict.keys()) == {'density'}
        assert len(extractor_funs) == 1

    def test_list_properties(self):
        analyzer = GraphAnalyzer()
        properties = analyzer.list_properties()
        
        # Check if all expected properties are present
        expected_properties = {
            'density', 'spectral_radius', 'av_in_degree',
            'av_out_degree', 'clustering_coefficient'
        }
        assert set(properties) == expected_properties

class TestRemoveTransients:
    @pytest.fixture
    def sample_data(self):
        # Create sample data for testing
        ResStates = np.random.randn(100, 10)  # 100 timesteps, 10 reservoir states
        X = np.random.randn(5, 100, 3)       # 5 samples, 100 timesteps, 3 input features
        Y = np.random.randn(5, 100, 2)       # 5 samples, 100 timesteps, 2 output features
        return ResStates, X, Y

    def test_remove_transients_res(self, sample_data):
        ResStates, _, _ = sample_data
        Transients = 20
        
        result = RemoveTransients_Res(ResStates, Transients)
        
        # Check shape
        assert result.shape == (80, 10)  # 100 - 20 = 80 timesteps remaining
        
        # Check content
        np.testing.assert_array_equal(result, ResStates[Transients:,:])

    def test_remove_transient_inps(self, sample_data):
        _, X, _ = sample_data
        Transients = 20
        
        result = RemoveTransient_Inps(X, Transients)
        
        # Check shape
        assert result.shape == (5, 80, 3)  # 100 - 20 = 80 timesteps remaining
        
        # Check content
        np.testing.assert_array_equal(result, X[:,Transients:,:])

    def test_remove_transient_outs(self, sample_data):
        _, _, Y = sample_data
        Transients = 20
        
        result = RemoveTransient_Outs(Y, Transients)
        
        # Check shape
        assert result.shape == (5, 80, 2)  # 100 - 20 = 80 timesteps remaining
        
        # Check content
        np.testing.assert_array_equal(result, Y[:,Transients:,:])

    def test_transient_remover_rx(self, sample_data):
        ResStates, X, Y = sample_data
        Transients = 20
        
        result_res, result_x = TransientRemover('RX', ResStates, X, Y, Transients)
        
        # Check shapes
        assert result_res.shape == (80, 10)
        assert result_x.shape == (5, 80, 3)
        
        # Check content
        np.testing.assert_array_equal(result_res, ResStates[Transients:,:])
        np.testing.assert_array_equal(result_x, X[:,Transients:,:])

    def test_transient_remover_rxy(self, sample_data):
        ResStates, X, Y = sample_data
        Transients = 20
        
        result_res, result_x, result_y = TransientRemover('RXY', ResStates, X, Y, Transients)
        
        # Check shapes
        assert result_res.shape == (80, 10)
        assert result_x.shape == (5, 80, 3)
        assert result_y.shape == (5, 80, 2)
        
        # Check content
        np.testing.assert_array_equal(result_res, ResStates[Transients:,:])
        np.testing.assert_array_equal(result_x, X[:,Transients:,:])
        np.testing.assert_array_equal(result_y, Y[:,Transients:,:])

    def test_transient_remover_edge_cases(self, sample_data):
        ResStates, X, Y = sample_data
        
        # Test with zero transients
        result_res, result_x, result_y = TransientRemover('RXY', ResStates, X, Y, 0)
        np.testing.assert_array_equal(result_res, ResStates)
        np.testing.assert_array_equal(result_x, X)
        np.testing.assert_array_equal(result_y, Y)
        
        # Test with transients equal to sequence length
        with pytest.raises(IndexError):
            TransientRemover('RXY', ResStates, X, Y, 100)
        
        # Test with transients greater than sequence length
        with pytest.raises(IndexError):
            TransientRemover('RXY', ResStates, X, Y, 101)

    def test_transient_remover_invalid_mode(self, sample_data):
        ResStates, X, Y = sample_data
        
        with pytest.raises(ValueError):
            TransientRemover('invalid', ResStates, X, Y, 20) 