from stgraph.graph.STGraphBase import STGraphBase
from abc import abstractmethod
import time

class DynamicGraph(STGraphBase):
    def __init__(self, edge_list, max_num_nodes, snapshot_edge_list, total_timestamps):
        super().__init__()
        self.graph_updates = {}
        self.max_num_nodes = max_num_nodes
        self.graph_attr = {key_t: (self.max_num_nodes, snapshot_edge_list[key_t]["edge_count"]) for key_t in snapshot_edge_list.keys()}

        # Non-naive variants will hit this
        if edge_list is None:
            for key_t in snapshot_edge_list.keys():
                self.graph_updates[key_t] = {"add": snapshot_edge_list[key_t]["add"], "delete": snapshot_edge_list[key_t]["delete"]}

        # Indicates whether the graph is currently undergoing backprop
        self._is_backprop_state = False
        self.current_timestamp = 0
        self.total_timestamps = total_timestamps

        # Measuring time for operations
        self.get_fwd_graph_time = 0
        self.get_bwd_graph_time = 0
        self.move_to_gpu_time = 0

    # def _preprocess_graph_structure(self, edge_list):
    #     edge_dict = {}
    #     for i in range(len(edge_list)):
    #         edge_set = set()
    #         for j in range(len(edge_list[i])):
    #             edge_set.add((edge_list[i][j][0], edge_list[i][j][1]))
    #         edge_dict[str(i)] = edge_set

    #     self.graph_updates = {}

    #     # Presorting additions and deletions (is a manadatory step for GPMA)
    #     additions = list(edge_dict["0"])
    #     additions.sort(key=lambda x: (x[1], x[0]))
    #     self.graph_updates["0"] = {"add": additions, "delete": []}
    #     for i in range(1, len(edge_list)):
    #         additions = list(edge_dict[str(i)].difference(edge_dict[str(i - 1)]))
    #         additions.sort(key=lambda x: (x[1], x[0]))
    #         deletions = list(edge_dict[str(i - 1)].difference(edge_dict[str(i)]))
    #         deletions.sort(key=lambda x: (x[1], x[0]))
    #         self.graph_updates[str(i)] = {
    #             "add": additions,
    #             "delete": deletions,
    #         }
    
    def reset_graph(self):
        self._get_cached_graph("base")
        self.current_timestamp = 0

        self.get_fwd_graph_time = 0
        self.get_bwd_graph_time = 0
        self.move_to_gpu_time = 0

    def get_graph(self, timestamp: int):
        t0 = time.time()

        self._is_backprop_state = False

        if timestamp < self.current_timestamp:
            raise Exception(
                "⏰ Invalid timestamp during STGraphBase.update_graph_forward()"
            )
        
        if self._get_cached_graph(timestamp - 1):
            self.current_timestamp = timestamp - 1

        while self.current_timestamp < timestamp:
            self._update_graph_forward()
            self.current_timestamp += 1
        
        self.get_fwd_graph_time += time.time() - t0

    def get_backward_graph(self, timestamp: int):
        t0 = time.time()

        if not self._is_backprop_state:
            self._cache_graph()
            self._is_backprop_state = True
            self._init_reverse_graph()

        if timestamp > self.current_timestamp:
            raise Exception(
                "⏰ Invalid timestamp during STGraphBase.update_graph_backward()"
            )

        while self.current_timestamp > timestamp:
            self._update_graph_backward()
            self.current_timestamp -= 1
        
        self.get_bwd_graph_time += time.time() - t0

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
    def _cache_graph(self):
        pass

    @abstractmethod
    def _get_cached_graph(self, timestamp):
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
