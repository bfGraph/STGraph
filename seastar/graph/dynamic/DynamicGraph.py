# TODO: add this import as well
from seastar.graph.SeastarGraph import SeastarGraph

import copy

from abc import ABC, abstractmethod

class DynamicGraph(SeastarGraph):
    def __init__(self, edge_list):
        super().__init__()
        self.graph_updates = {}
        self.max_num_nodes = 0
        self.graph_cache = {}
        self._is_reverse_graph = False
        self.current_timestamp = 0
        
        self._preprocess_graph_structure(edge_list)
        
    def _get_graph_attr(self, edge_list):
        graph_attr = {}
        for time in range(len(edge_list)):
            node_set = set()
            edge_count = 0
            for edge in edge_list[time]:
                src = edge[0]
                dst = edge[1]
                edge_count += 1
                node_set.add(src)
                node_set.add(dst)
            node_count = len(node_set)
            graph_attr[str(time)] = (node_count, edge_count)
        
        return graph_attr
        
    def _preprocess_graph_structure(self, edge_list):
        
        graph_attr = self._get_graph_attr(edge_list)
        
        tmp_set = set()
        for i in range(len(edge_list)):
            tmp_set = set()
            for j in range(len(edge_list[i])):
                tmp_set.add(edge_list[i][j][0])
                tmp_set.add(edge_list[i][j][1])
        self.max_num_nodes = len(tmp_set)

        edge_dict = {}
        for i in range(len(edge_list)):
            edge_set = set()
            for j in range(len(edge_list[i])):
                edge_set.add((edge_list[i][j][0], edge_list[i][j][1]))
            edge_dict[str(i)] = edge_set

        self.graph_updates = {}
        self.graph_updates["0"] = {
            "add": list(edge_dict["0"]),
            "delete": [],
            "num_nodes": graph_attr["0"][0],
            "num_edges": graph_attr["0"][1],
        }
        for i in range(1, len(edge_list)):
            self.graph_updates[str(i)] = {
                "add": list(edge_dict[str(i)].difference(edge_dict[str(i - 1)])),
                "delete": list(edge_dict[str(i - 1)].difference(edge_dict[str(i)])),
                "num_nodes": graph_attr[str(i)][0],
                "num_edges": graph_attr[str(i)][1],
            }
        
    def _update_graph_cache(self, is_reverse=False):
        # saving base graph in cache
        if not is_reverse:
            self.graph_cache['base'] = copy.deepcopy(self.forward_graph)
        else:
        # saving reverse base graph in cache
            self.graph_cache['reverse'] = copy.deepcopy(self.backward_graph)
            
    def _get_cached_graph(self, is_reverse=False):
        if not is_reverse:
            return copy.deepcopy(self.graph_cache['base'])
        else:
            return copy.deepcopy(self.graph_cache['reverse'])
    
    def _revert_to_base_graph(self):
        self.forward_graph = self._get_cached_graph(is_reverse=False)

        self._get_graph_csr_ptrs()
        self._is_reverse_graph = False
        
    def get_graph(self, timestamp: int):

        if self._is_reverse_graph:
            self._revert_to_base_graph()
        
        if timestamp < self.current_timestamp:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")

        while self.current_timestamp < timestamp:
            self._update_graph_forward()

    def get_backward_graph(self, timestamp: int):

        if not self._is_reverse_graph:
            self._init_reverse_graph()
        
        if timestamp > self.current_timestamp:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        while self.current_timestamp > timestamp:
            self._update_graph_backward()
    
    def get_num_nodes(self):
        return self.graph_updates[str(self.current_timestamp)]["num_nodes"]
    
    def get_num_edges(self):
        return self.graph_updates[str(self.current_timestamp)]["num_edges"]
    
    # def _get_graph_csr_ptrs(self):
    #     if not self._is_reverse_graph:
    #         csr_ptrs = self.forward_graph.get_csr_ptrs()
    #     else:
    #         csr_ptrs = self.backward_graph.get_csr_ptrs()
        
    #     self.row_offset_ptr = csr_ptrs[0]
    #     self.column_indices_ptr = csr_ptrs[1]
    #     self.eids_ptr = csr_ptrs[2]
    
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
    
    