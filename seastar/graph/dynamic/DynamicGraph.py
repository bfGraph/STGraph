import copy
import time


from seastar.graph.SeastarGraph import SeastarGraph
from rich import inspect


from abc import ABC, abstractmethod

class DynamicGraph(SeastarGraph):
    def __init__(self, edge_list):
        super().__init__()
        self.graph_updates = {}
        self.max_num_nodes = 0
        self.graph_cache = {}
        
        # Indicates whether the graph is currently undergoing backprop
        self._is_backprop_state = False
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
        
        max_num_nodes = 0
        for i in range(len(edge_list)):
            for j in range(len(edge_list[i])):
                max_num_nodes = max(max_num_nodes,edge_list[i][j][0],edge_list[i][j][1])
        self.max_num_nodes = max_num_nodes + 1
        
        # SINCE THE CONCEPT OF NODE IDS HASNT BEEN IMPLEMENTED
        # WE CANT USE THIS (AS in consider the case where nodes are labelled 1,2,3)
        # What hapens to 0, Our system right now cant map each node to an ID
        # tmp_set = set()
        # for i in range(len(edge_list)):
        #     for j in range(len(edge_list[i])):
        #         tmp_set.add(edge_list[i][j][0])
        #         tmp_set.add(edge_list[i][j][1])
        # self.max_num_nodes = len(tmp_set)

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
        
    def get_graph(self, timestamp: int):
        # print("💄💄💄 Get_graph (forward) called",flush=True)
        self._is_backprop_state = False
        
        if timestamp < self.current_timestamp:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")

        while self.current_timestamp < timestamp:
            self._update_graph_forward()
            self.current_timestamp += 1

    def get_backward_graph(self, timestamp: int):
        # print("📝📝📝 Get_backward_graph (backward) called",flush=True)
        if not self._is_backprop_state:
            # print("💄💄💄 Calling backward init")
            self._is_backprop_state = True
            self._init_reverse_graph()
        
        if timestamp > self.current_timestamp:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        while self.current_timestamp > timestamp:
            # print("🎒🎒🎒 Calling update backward")
            self._update_graph_backward()
            self.current_timestamp -= 1
    
    def get_num_nodes(self) -> int:
        r"""Returns an integer representing the total number of nodes 
            in a dynamic graph at the current timestamp.
        """
        return self.graph_updates[str(self.current_timestamp)]["num_nodes"]
    
    def get_num_edges(self):
        r"""Returns an integer representing the total number of edges 
            in a dynamic graph at the current timestamp.
        """
        return self.graph_updates[str(self.current_timestamp)]["num_edges"]
    
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
    
    