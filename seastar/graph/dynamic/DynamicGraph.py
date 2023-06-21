from seastar.graph.SeastarGraph import SeastarGraph
from abc import abstractmethod

class DynamicGraph(SeastarGraph):
    def __init__(self, edge_list, max_num_nodes):
        super().__init__()
        self.graph_updates = {}
        self.max_num_nodes = max_num_nodes
        self.graph_attr = {str(t): (self.max_num_nodes, len(set(edge_list[t]))) for t in range(len(edge_list))}

        # Indicates whether the graph is currently undergoing backprop
        self._is_backprop_state = False
        self.current_timestamp = 0

        self._preprocess_graph_structure(edge_list)

    def _preprocess_graph_structure(self, edge_list):
        edge_dict = {}
        for i in range(len(edge_list)):
            edge_set = set()
            for j in range(len(edge_list[i])):
                edge_set.add((edge_list[i][j][0], edge_list[i][j][1]))
            edge_dict[str(i)] = edge_set

        self.graph_updates = {}
        self.graph_updates["0"] = {"add": list(edge_dict["0"]), "delete": []}
        for i in range(1, len(edge_list)):
            self.graph_updates[str(i)] = {
                "add": list(edge_dict[str(i)].difference(edge_dict[str(i - 1)])),
                "delete": list(edge_dict[str(i - 1)].difference(edge_dict[str(i)])),
            }

    def get_graph(self, timestamp: int):
        self._is_backprop_state = False

        if timestamp < self.current_timestamp:
            raise Exception(
                "⏰ Invalid timestamp during SeastarGraph.update_graph_forward()"
            )

        while self.current_timestamp < timestamp:
            self._update_graph_forward()
            self.current_timestamp += 1

    def get_backward_graph(self, timestamp: int):
        if not self._is_backprop_state:
            self._is_backprop_state = True
            self._init_reverse_graph()

        if timestamp > self.current_timestamp:
            raise Exception(
                "⏰ Invalid timestamp during SeastarGraph.update_graph_backward()"
            )

        while self.current_timestamp > timestamp:
            self._update_graph_backward()
            self.current_timestamp -= 1

    def get_num_nodes(self):
        return self.graph_attr[str(self.current_timestamp)][0]

    def get_num_edges(self):
        return self.graph_attr[str(self.current_timestamp)][1]
    
    def get_ndata(self, field):
        if str(self.current_timestamp) in self._ndata and field in self._ndata[str(self.current_timestamp)]:
            return self._ndata[str(self.current_timestamp)][field]
        else:
            return None

    def set_ndata(self, field, val):
        if str(self.current_timestamp) in self._ndata:
            self._ndata[str(self.current_timestamp)][field] = val
        else:
            self._ndata[str(self.current_timestamp)] = {field: val}

    @abstractmethod
    def in_degrees(self):
        pass

    @abstractmethod
    def out_degrees(self):
        pass

    @abstractmethod
    def _update_graph_forward(self):
        pass

    @abstractmethod
    def _init_reverse_graph(self):
        pass

    @abstractmethod
    def _update_graph_backward(self):
        pass
