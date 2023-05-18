import json
from rich import inspect

from seastar.graph.dynamic.naive.NaiveGraph import NaiveGraph
from seastar.graph.static.csr import get_dev_array

class TestDynamicGraphNaive:
    # Stores the edge-list for each timestamp
    # edge_index[t] has the edge list for timestamp = t
    edge_index = [
        [[0, 1], [1, 2], [2, 3]],
        [[1, 2], [2, 3], [3, 1]],
        [[1, 2], [2, 3], [3, 1], [3, 0]],
        [[2, 3], [3, 1], [3, 0], [0, 1]],
    ]

    # sorted based on second-element, then first-element
    sorted_edge_index = [
        [[0, 1], [1, 2], [2, 3]],
        [[3, 1], [1, 2], [2, 3]],
        [[3, 0], [3, 1], [1, 2], [2, 3]],
        [[3, 0], [0, 1], [3, 1], [2, 3]],
    ]

    # stores the edge-weights for each timestamp
    # edge_weight[t] has the edge weight for timestamp = t
    # corresponding to the edge in edge_index
    edge_weight = [[4, 7, 9], [2, 11, 13], [3, 8, 1, 5], [15, 6, 10, 12]]

    # total timestamps for this dynamic graph dataset
    time_periods = 4

    # node features for each timestamp
    y = [[2, 11, 15, 8], [9, 5, 7, 10], [1, 3, 17, 19], [4, 6, 12, 13]]

    def test_get_graph_attr(self):
        naive_graph = NaiveGraph(edge_list=self.sorted_edge_index)
        graph_attr = naive_graph._get_graph_attr(edge_list=self.sorted_edge_index)

        # checking if the size of graph_attr = 4, which is the
        # total number of timestamps present
        assert len(graph_attr) == 4

        # checking the (num_nodes, num_edges) pair for each timestamp
        assert graph_attr["0"] == (4, 3)
        assert graph_attr["1"] == (4, 3)
        assert graph_attr["2"] == (4, 4)
        assert graph_attr["3"] == (4, 4)

    def test_preprocess_graph_structure(self):
        naive_graph = NaiveGraph(edge_list=self.sorted_edge_index)

        # checking graph_updates for t = 0
        graph_updates_0 = naive_graph.graph_updates["0"]
        assert len(graph_updates_0["add"]) == 3
        for edge in [(0, 1), (1, 2), (2, 3)]:
            assert edge in graph_updates_0["add"]

        assert len(graph_updates_0["delete"]) == 0

        assert graph_updates_0["num_nodes"] == 4
        assert graph_updates_0["num_edges"] == 3

        # checking graph_updates for t = 1
        graph_updates_1 = naive_graph.graph_updates["1"]
        assert len(graph_updates_1["add"]) == 1
        assert (3, 1) in graph_updates_1["add"]

        assert len(graph_updates_1["delete"]) == 1
        assert (0, 1) in graph_updates_1["delete"]

        assert graph_updates_1["num_nodes"] == 4
        assert graph_updates_1["num_edges"] == 3

        # checking graph_updates for t = 2
        graph_updates_2 = naive_graph.graph_updates["2"]

        assert len(graph_updates_2["add"]) == 1
        assert (3, 0) in graph_updates_2["add"]

        assert len(graph_updates_2["delete"]) == 0

        assert graph_updates_2["num_nodes"] == 4
        assert graph_updates_2["num_edges"] == 4

        # checking graph_updates for t = 3
        graph_updates_3 = naive_graph.graph_updates["3"]

        assert len(graph_updates_3["add"]) == 1
        assert (0, 1) in graph_updates_3["add"]

        assert len(graph_updates_3["delete"]) == 1
        assert (1, 2) in graph_updates_3["delete"]

        assert graph_updates_3["num_nodes"] == 4
        assert graph_updates_3["num_edges"] == 4

    def test_get_graph(self):
        naive_graph = NaiveGraph(edge_list=self.sorted_edge_index)

        # for time = 0
        row_offset, column_indices, eids = (
            get_dev_array(naive_graph.fwd_row_offset_ptr, 5),
            get_dev_array(naive_graph.fwd_column_indices_ptr, 3),
            get_dev_array(naive_graph.fwd_eids_ptr, 3),
        )

        assert row_offset == [0, 0, 1, 2, 3]
        assert column_indices == [0, 1, 2]
        assert eids == [0, 1, 2]

        # for time = 1
        naive_graph.get_graph(1)
        
        row_offset, column_indices, eids = (
            get_dev_array(naive_graph.fwd_row_offset_ptr, 5),
            get_dev_array(naive_graph.fwd_column_indices_ptr, 3),
            get_dev_array(naive_graph.fwd_eids_ptr, 3),
        )

        assert row_offset == [0, 0, 1, 2, 3]
        assert column_indices == [3, 1, 2]
        assert eids == [0, 1, 2]

        # for time = 2
        naive_graph.get_graph(2)
        
        row_offset, column_indices, eids = (
            get_dev_array(naive_graph.fwd_row_offset_ptr, 5),
            get_dev_array(naive_graph.fwd_column_indices_ptr, 4),
            get_dev_array(naive_graph.fwd_eids_ptr, 4),
        )

        assert row_offset == [0, 1, 2, 3, 4]
        assert column_indices == [3, 3, 1, 2]
        assert eids == [0, 1, 2, 3]

        # for time = 3
        naive_graph.get_graph(3)
        
        row_offset, column_indices, eids = (
            get_dev_array(naive_graph.fwd_row_offset_ptr, 5),
            get_dev_array(naive_graph.fwd_column_indices_ptr, 4),
            get_dev_array(naive_graph.fwd_eids_ptr, 4),
        )

        assert row_offset == [0, 1, 3, 3, 4]
        assert column_indices == [3, 0, 3, 2]
        assert eids == [0, 1, 2, 3]

    def test_get_backward_graph(self):
        
        naive_graph = NaiveGraph(edge_list=self.sorted_edge_index)
        naive_graph.get_graph(3)

        # for time = 3
        naive_graph.get_backward_graph(3)
        
        row_offset, column_indices, eids = (
            get_dev_array(naive_graph.bwd_row_offset_ptr, 5),
            get_dev_array(naive_graph.bwd_column_indices_ptr, 4),
            get_dev_array(naive_graph.bwd_eids_ptr, 4),
        )
        
        assert row_offset == [0,1,1,2,4]
        assert column_indices == [1,3,0,1]
        assert eids == [1,3,0,2]
        
        # for time = 2
        naive_graph.get_backward_graph(2)
        
        row_offset, column_indices, eids = (
            get_dev_array(naive_graph.bwd_row_offset_ptr, 5),
            get_dev_array(naive_graph.bwd_column_indices_ptr, 4),
            get_dev_array(naive_graph.bwd_eids_ptr, 4),
        )
        
        assert row_offset == [0,0,1,2,4]
        assert column_indices == [2,3,0,1]
        assert eids == [2,3,0,1]
        
        # for time = 1
        naive_graph.get_backward_graph(1)
        
        row_offset, column_indices, eids = (
            get_dev_array(naive_graph.bwd_row_offset_ptr, 5),
            get_dev_array(naive_graph.bwd_column_indices_ptr, 3),
            get_dev_array(naive_graph.bwd_eids_ptr, 3),
        )
        
        assert row_offset == [0,0,1,2,3]
        assert column_indices == [2,3,1]
        assert eids == [1,2,0]
        
        # for time = 0
        naive_graph.get_backward_graph(0)
        
        row_offset, column_indices, eids = (
            get_dev_array(naive_graph.bwd_row_offset_ptr, 5),
            get_dev_array(naive_graph.bwd_column_indices_ptr, 3),
            get_dev_array(naive_graph.bwd_eids_ptr, 3),
        )
        
        assert row_offset == [0,1,2,3,3]
        assert column_indices == [1,2,3]
        assert eids == [0,1,2]

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
        assert naive_graph.get_num_nodes() == 4

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
        assert naive_graph.get_num_nodes() == 4

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
