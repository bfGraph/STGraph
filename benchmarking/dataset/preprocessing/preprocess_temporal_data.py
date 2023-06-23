import json
import random
import argparse
import sys


def parse_txt_lines(lines, cutoff_time):
	id_to_pid_map = {}   # ID to processed ID
	node_counter = 0

	node_set = set()
	edges = []

	time = 0
	for line in lines:

		if time >= cutoff_time:
			break

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

		time += 1

	expected_node_set = set([i for i in range(len(node_set))])
	assert node_set == expected_node_set, "Node labelling is not continuous"
	# print(node_set.difference(expected_node_set))
	# quit()
	return edges, len(node_set)


def preprocess_graph(edges, num_nodes, base, add_delta, delete_delta):
	# base = 2000000
	# add_delta = 50000
	# delete_delta = 50000

	# base = 100000
	# add_delta = 5000
	# delete_delta = 1000

	graphs = {}

	graphs["0"] = {
					"add": list(set(edges[:base])),
					"delete": [],
					"neg": []
				}

	add_start_ptr = base
	delete_start_ptr = 0
	time = 1

	while (add_start_ptr + add_delta) < len(edges):
		graph_t = set(edges[delete_start_ptr + delete_delta:add_start_ptr + add_delta])
		common_with_prev_graph = graph_t.intersection(set(edges[delete_start_ptr:add_start_ptr]))

		# No need to add an edge thats already there in the potion carried over from previous snapshot
		add_edges = list(set(edges[add_start_ptr:add_start_ptr + add_delta]).difference(common_with_prev_graph))

		# Difference to prevent the deletion of an edge that is present in the snapshot
		del_edges = list(set(edges[delete_start_ptr:delete_start_ptr + delete_delta]).difference(graph_t))

		neg_edge_set_t = set()
		while len(neg_edge_set_t) != len(add_edges):
			candidate = (random.randint(0, num_nodes-1), random.randint(0, num_nodes-1))
			if candidate not in graph_t:
				neg_edge_set_t.add(candidate)

		graphs[str(time)] = {
								"add": add_edges,
								"delete": del_edges,
								"neg": list(neg_edge_set_t)
							}
		
		assert len(set(edges[delete_start_ptr:add_start_ptr])) + len(add_edges) - len(del_edges) == len(graph_t)

		add_start_ptr += add_delta
		delete_start_ptr += delete_delta
		time = time + 1

	if add_start_ptr < len(edges):
		graph_t = set(edges[delete_start_ptr+delete_delta:])
		common_with_prev_graph = graph_t.intersection(set(edges[delete_start_ptr:add_start_ptr]))

		# No need to add an edge thats already there in the potion carried over from previous snapshot
		add_edges = list(set(edges[add_start_ptr:]).difference(common_with_prev_graph))

		# Difference to prevent the deletion of an edge that is going to be added
		del_edges = list(set(edges[delete_start_ptr:delete_start_ptr + delete_delta]).difference(graph_t))

		neg_edge_set_t = set()
		while len(neg_edge_set_t) != len(add_edges):
			candidate = (random.randint(0, num_nodes-1), random.randint(0, num_nodes-1))
			if candidate not in graph_t:
				neg_edge_set_t.add(candidate)

		graphs[str(time)] = {
								"add": add_edges,
								"delete": del_edges,
								"neg": list(neg_edge_set_t)
							}
		
		assert len(set(edges[delete_start_ptr:add_start_ptr])) + len(add_edges) - len(del_edges) == len(graph_t)

		time = time+1

	graph_json = {
		"edge_mapping": {"edge_index": graphs},
		"time_periods": time,
	}

	return graph_json

def main(args):
	file1 = open(f'{args.dataset}.txt', 'r')
	lines = file1.readlines()
	edges, num_nodes = parse_txt_lines(lines, args.cutoff_time)

	add_delta = int(args.base * (args.percent_change/200))
	delete_delta = int(args.base * (args.percent_change/200))
	graph_json = preprocess_graph(edges, num_nodes, args.base, add_delta, delete_delta)
	out_file = open(f"{args.dataset}-data-{str(args.percent_change)}.json", "w")
	json.dump(graph_json, out_file)
	out_file.close()

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


