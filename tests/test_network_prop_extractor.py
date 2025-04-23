import pytest
import numpy as np
from pyreco.core.network_prop_extractor import NetworkQuantifier, NodePropExtractor

class TestNetworkQuantifier:
    @pytest.fixture
    def simple_network(self):
        # Create a simple network with known properties
        return np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0]
        ])

    @pytest.fixture
    def empty_network(self):
        return np.zeros((4, 4))

    @pytest.fixture
    def fully_connected_network(self):
        return np.ones((4, 4)) - np.eye(4)

    def test_initialization(self):
        # Test default initialization
        quantifier = NetworkQuantifier()
        assert set(quantifier.quantities) == {
            'density', 'spectral_radius', 'in_degree_av', 
            'out_degree_av', 'clustering_coefficient'
        }

        # Test custom initialization
        custom_quantities = ['density', 'spectral_radius']
        quantifier = NetworkQuantifier(quantities=custom_quantities)
        assert quantifier.quantities == custom_quantities

    def test_extract_properties_simple_network(self, simple_network):
        quantifier = NetworkQuantifier()
        props = quantifier.extract_properties(simple_network)

        # Test density (5 edges out of possible 16)
        assert props['density'] == 5/16

        # Test in-degree average (should be equal to out-degree average for undirected network)
        assert props['in_degree_av'] == props['out_degree_av']
        assert props['in_degree_av'] == 1.25  # (2+2+3+1)/4

        # Test clustering coefficient (should be between 0 and 1)
        assert 0 <= props['clustering_coefficient'] <= 1

        # Test spectral radius (should be positive)
        assert props['spectral_radius'] > 0

    def test_extract_properties_empty_network(self, empty_network):
        quantifier = NetworkQuantifier()
        props = quantifier.extract_properties(empty_network)

        assert props['density'] == 0
        assert props['in_degree_av'] == 0
        assert props['out_degree_av'] == 0
        assert props['clustering_coefficient'] == 0
        assert props['spectral_radius'] == 0

    def test_extract_properties_fully_connected(self, fully_connected_network):
        quantifier = NetworkQuantifier()
        props = quantifier.extract_properties(fully_connected_network)

        # Test density (should be 1 for fully connected network)
        assert props['density'] == 1

        # Test in-degree and out-degree (should be n-1 for each node)
        assert props['in_degree_av'] == 3
        assert props['out_degree_av'] == 3

        # Test clustering coefficient (should be 1 for fully connected network)
        assert props['clustering_coefficient'] == 1

    def test_invalid_quantity(self, simple_network):
        quantifier = NetworkQuantifier(quantities=['invalid_property'])
        props = quantifier.extract_properties(simple_network)
        assert props == {}  # Should return empty dict for invalid properties

class TestNodePropExtractor:
    @pytest.fixture
    def simple_network(self):
        # Create a simple network with known properties
        return np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0]
        ])

    @pytest.fixture
    def directed_network(self):
        # Create a directed network
        return np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ])

    def test_initialization(self):
        # Test default initialization
        extractor = NodePropExtractor()
        assert set(extractor.properties) == {
            'degree', 'in_degree', 'out_degree', 
            'clustering_coefficient', 'betweenness_centrality', 'pagerank'
        }

        # Test custom initialization
        custom_props = ['degree', 'pagerank']
        extractor = NodePropExtractor(properties=custom_props)
        assert extractor.properties == custom_props

    def test_extract_properties_simple_network(self, simple_network):
        extractor = NodePropExtractor()
        props = extractor.extract_properties(simple_network)

        # Test degree properties
        assert len(props['degree']) == 4
        assert len(props['in_degree']) == 4
        assert len(props['out_degree']) == 4

        # For undirected network, in-degree should equal out-degree
        assert np.array_equal(props['in_degree'], props['out_degree'])

        # Test clustering coefficient
        assert len(props['clustering_coefficient']) == 4
        assert all(0 <= cc <= 1 for cc in props['clustering_coefficient'])

        # Test betweenness centrality
        assert len(props['betweenness_centrality']) == 4
        assert all(0 <= bc <= 1 for bc in props['betweenness_centrality'])

        # Test PageRank
        assert len(props['pagerank']) == 4
        assert np.isclose(sum(props['pagerank']), 1.0)  # PageRank should sum to 1

    def test_extract_properties_directed_network(self, directed_network):
        extractor = NodePropExtractor()
        props = extractor.extract_properties(directed_network)

        # For directed network, in-degree and out-degree may differ
        assert not np.array_equal(props['in_degree'], props['out_degree'])

        # Test that all properties are calculated
        for prop in ['degree', 'in_degree', 'out_degree', 
                    'clustering_coefficient', 'betweenness_centrality', 'pagerank']:
            assert prop in props
            assert len(props[prop]) == 4

    def test_invalid_property(self, simple_network):
        extractor = NodePropExtractor(properties=['invalid_property'])
        props = extractor.extract_properties(simple_network)
        assert props == {}  # Should return empty dict for invalid properties

    def test_extract_properties_with_states(self, simple_network):
        # Test that states parameter is accepted but not used
        states = np.random.rand(4)
        extractor = NodePropExtractor()
        props = extractor.extract_properties(simple_network, states=states)
        
        # Properties should still be calculated correctly
        assert len(props['degree']) == 4
        assert len(props['pagerank']) == 4 