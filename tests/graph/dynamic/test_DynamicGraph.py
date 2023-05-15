import json
from rich import inspect
from seastar.graph.dynamic.DynamicGraph import DynamicGraph
from seastar.graph.dynamic.naive.NaiveGraph import NaiveGraph


def get_edge_list(gra):
    pass


class TestDynamicGraph:
    # Stores the edge-list for each timestamp
    # edge_index[t] has the edge list for timestamp = t
    edge_index = [
        [[0, 1], [1, 2], [2, 3]],
        [[1, 2], [2, 3], [3, 1]],
        [[1, 2], [2, 3], [3, 1], [3, 0]],
        [[2, 3], [3, 1], [3, 0], [0, 1]],
    ]

    # stores the edge-weights for each timestamp
    # edge_weight[t] has the edge weight for timestamp = t
    # corresponding to the edge in edge_index
    edge_weight = [[4, 7, 9], [2, 11, 13], [3, 8, 1, 5], [15, 6, 10, 12]]

    # total timestamps for this dynamic graph dataset
    time_periods = 4

    # node features for each timestamp
    y = [[2, 11, 15, 8], [9, 5, 7, 10], [1, 3, 17, 19], [4, 6, 12, 13]]

    def test_get_num_nodes(self):
        """
            Assert the number of nodes in the graph, then repeat
            this assert step after calling DynamicGraph.get_graph()
            and DynamicGraph.get_backward_graph() in sequential order
        """

        # base graph: t = 0
        naive_graph = NaiveGraph(edge_list=self.edge_index)
        assert naive_graph.get_num_nodes() == 4

        # graph: t = 1
        naive_graph.get_graph(1)
        assert naive_graph.get_num_nodes() == 3

        # graph: t = 2
        naive_graph.get_graph(2)
        assert naive_graph.get_num_nodes() == 4

        # graph: t = 3
        naive_graph.get_graph(3)
        assert naive_graph.get_num_nodes() == 4

        # Now moving the graph in the backward direction
        # graph: t = 2
        naive_graph.get_backward_graph(2)
        assert naive_graph.get_num_nodes() == 4
        
        # graph: t = 1
        naive_graph.get_backward_graph(1)
        assert naive_graph.get_num_nodes() == 3
        
        # graph: t = 1
        naive_graph.get_backward_graph(0)
        assert naive_graph.get_num_nodes() == 4
        
    def test_get_num_edges(self):
        """
            Assert the number of edges in the graph, then repeat
            this assert step after calling DynamicGraph.get_graph()
            and DynamicGraph.get_backward_graph() in sequential order
        """
        
        # base graph: t = 0
        naive_graph = NaiveGraph(edge_list=self.edge_index)
        assert naive_graph.get_num_edges() == 3

        # graph: t = 1
        naive_graph.get_graph(1)
        assert naive_graph.get_num_edges() == 3

        # graph: t = 2
        naive_graph.get_graph(2)
        assert naive_graph.get_num_edges() == 4

        # graph: t = 3
        naive_graph.get_graph(3)
        assert naive_graph.get_num_edges() == 4

        # Now moving the graph in the backward direction
        # graph: t = 2
        naive_graph.get_backward_graph(2)
        assert naive_graph.get_num_edges() == 4
        
        # graph: t = 1
        naive_graph.get_backward_graph(1)
        assert naive_graph.get_num_edges() == 3
        
        # graph: t = 1
        naive_graph.get_backward_graph(0)
        assert naive_graph.get_num_edges() == 3