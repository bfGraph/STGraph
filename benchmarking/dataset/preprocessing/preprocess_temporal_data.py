import json
import random
import argparse
import sys
import time
import numpy as np
import pickle

def parse_txt_lines(lines):
	id_to_pid_map = {}   # ID to processed ID
	node_counter = 0

	node_set = set()
	edges = []

	for line in lines:

		parsed_line = line.split(' ')

		src = int(parsed_line[0])
		dst = int(parsed_line[1])

		if src not in id_to_pid_map:
			id_to_pid_map[src] = node_counter
			node_counter += 1

		if dst not in id_to_pid_map:
			id_to_pid_map[dst] = node_counter
			node_counter += 1

		edges.append((id_to_pid_map[src], id_to_pid_map[dst]))
		node_set.add(id_to_pid_map[src])
		node_set.add(id_to_pid_map[dst])

	expected_node_set = set([i for i in range(len(node_set))])
	assert node_set == expected_node_set, "Node labelling is not continuous"
	return edges, len(node_set)


def preprocess_graph(edges, num_nodes, base, add_delta, delete_delta):
	graphs = {}

	base_list = list(set(edges[:base]))
	base_list.sort(key=lambda x: (x[1], x[0]))
	graphs["0"] = {
					"add": base_list,
					"delete": [],
					"neg": [],
					"edge_count": len(base_list)
				}

	add_start_ptr = base
	delete_start_ptr = 0
	time_track = 1

	while (add_start_ptr + add_delta) < len(edges):
		graph_t = set(edges[delete_start_ptr + delete_delta:add_start_ptr + add_delta])
		common_with_prev_graph = graph_t.intersection(set(edges[delete_start_ptr:add_start_ptr]))

		# No need to add an edge thats already there in the potion carried over from previous snapshot
		add_edges = list(set(edges[add_start_ptr:add_start_ptr + add_delta]).difference(common_with_prev_graph))
		add_edges.sort(key=lambda x: (x[1], x[0]))

		# Difference to prevent the deletion of an edge that is present in the snapshot
		del_edges = list(set(edges[delete_start_ptr:delete_start_ptr + delete_delta]).difference(graph_t))
		del_edges.sort(key=lambda x: (x[1], x[0]))

		neg_edge_set_t = set()
		while len(neg_edge_set_t) != len(add_edges):
			candidate = (random.randint(0, num_nodes-1), random.randint(0, num_nodes-1))
			if candidate not in graph_t:
				neg_edge_set_t.add(candidate)

		graphs[str(time_track)] = {
								"add": add_edges,
								"delete": del_edges,
								"neg": list(neg_edge_set_t),
								"edge_count": len(graph_t)
							}
		
		# assert len(set(edges[delete_start_ptr:add_start_ptr])) + len(add_edges) - len(del_edges) == len(graph_t)

		add_start_ptr += add_delta
		delete_start_ptr += delete_delta
		time_track = time_track + 1

	if add_start_ptr < len(edges):
		graph_t = set(edges[delete_start_ptr+delete_delta:])
		common_with_prev_graph = graph_t.intersection(set(edges[delete_start_ptr:add_start_ptr]))

		# No need to add an edge thats already there in the potion carried over from previous snapshot
		add_edges = list(set(edges[add_start_ptr:]).difference(common_with_prev_graph))
		add_edges.sort(key=lambda x: (x[1], x[0]))

		# Difference to prevent the deletion of an edge that is going to be added
		del_edges = list(set(edges[delete_start_ptr:delete_start_ptr + delete_delta]).difference(graph_t))
		del_edges.sort(key=lambda x: (x[1], x[0]))

		neg_edge_set_t = set()
		while len(neg_edge_set_t) != len(add_edges):
			candidate = (random.randint(0, num_nodes-1), random.randint(0, num_nodes-1))
			if candidate not in graph_t:
				neg_edge_set_t.add(candidate)

		graphs[str(time_track)] = {
								"add": add_edges,
								"delete": del_edges,
								"neg": list(neg_edge_set_t),
								"edge_count": len(graph_t)
							}
		
		# assert len(set(edges[delete_start_ptr:add_start_ptr])) + len(add_edges) - len(del_edges) == len(graph_t)

		time_track = time_track+1

	graph_json = graphs

	graph_json2 = {
		"time_periods": time_track,
		"base": base,
		"add_delta": add_delta,
		"delete_delta": delete_delta,
		"max_num_nodes": num_nodes
	}

	return graph_json, graph_json2

def main(args):
	curr_time = time.time()
	file1 = open(f'{args.dataset}.txt', 'r')
	lines = []
	for i, line in enumerate(file1):
		if i == args.cutoff_time:
			break
		else:
			lines.append(line)
	edges, num_nodes = parse_txt_lines(lines)
	print(f"[CHECKPOINT]::FILE_PARSING_COMPLETED in {time.time() - curr_time}s")
	
	curr_time = time.time()
	# Divide by 200 (2*100) because we need the change percentage to be split across additions and deletions
	add_delta = int(args.base * (args.percent_change/200))
	delete_delta = int(args.base * (args.percent_change/200))

	graph_json, graph_json2 = preprocess_graph(edges, num_nodes, args.base, add_delta, delete_delta)
	print(f"[CHECKPOINT]::PREPROCESS_GRAPH in {time.time() - curr_time}s")

	curr_time = time.time()
	out_file = open(f"{args.dataset}-data-{str(args.percent_change)}-split.json", "w")
	json.dump(graph_json, out_file)
	out_file.close()

	out_file2 = open(f"{args.dataset}-data-{str(args.percent_change)}-metadata.json", "w")
	json.dump(graph_json2, out_file2)
	out_file2.close()
	print(f"[CHECKPOINT]::JSON_DUMP in {time.time() - curr_time}s")

	curr_time = time.time()
	np_arr = np.array(edges)
	np.save(f"{args.dataset}-data-{str(args.percent_change)}.npy", np_arr)
	print(f"[CHECKPOINT]::NUMPY_DUMP in {time.time() - curr_time}s")

	curr_time = time.time()
	out_file3 = open(f"{args.dataset}-data-{str(args.percent_change)}.pkl", "wb")
	pickle.dump(edges, out_file3)
	out_file3.close()
	print(f"[CHECKPOINT]::PICKLE_DUMP in {time.time() - curr_time}s")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess Temporal Data')

	parser.add_argument("--dataset", type=str, default="",
			help="Name of the Dataset")
	parser.add_argument("--base", type=int, default=0,
			help="Num of edges in Base Graph")
	parser.add_argument("--percent-change", type=float, default=5,
			help="Percentage of change from base graph, this change is by sliding along the timestamps")
	parser.add_argument("--cutoff-time", type=int, default=sys.maxsize,
			help="Cuttoff time")
	args = parser.parse_args()
	
	print(args)
	main(args)


