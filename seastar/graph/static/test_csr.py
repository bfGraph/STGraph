from csr import CSR

edge_list = [
    (7,2),
    (2,6),
    (3,1),
    (3,6),
    (2,8),
    (3,7),
    (2,1),
    (8,4),
    (7,4),
]

c = CSR(edge_list, 9)
c.label_edges()
c.print_graph()

r = CSR(edge_list, 9, is_edge_reverse = True)
r.copy_label_edges(c)
r.print_graph()
r.print_csr_arrays()