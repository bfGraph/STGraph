from prettytable import PrettyTable
from prettytable import SINGLE_BORDER, ALL

from ..utils import ValType
from ..node import CentralNode

def get_val_type_name(val_type: ValType):
    if val_type == ValType.S:
        return "Source"
    if val_type == ValType.D:
        return "Destination"
    if val_type == ValType.E:
        return "Edge"
    if val_type == ValType.P:
        return "Parameter"

def tensor_to_list(tensor):
    tensor_list = []
    for index, row in enumerate(tensor):
        tensor_row = []
        for _, element in enumerate(row):
            tensor_row.append(element)
        tensor_list.append(tensor_row)

    return tensor_list

def pretty_print_GIR(gir, gir_name):

    gir_table = PrettyTable()
    gir_table.set_style(SINGLE_BORDER)
    gir_table.align = "l"
    gir_table.hrules = ALL

    gir_table.field_names = ["op_schema (Schema)", "ret (Var)", "op_name", "args (var)"]
    gir_content_rows = []

    print(f'\n🦋 Pretty Printing GIR - {gir_name}\n')

    for gir_node in gir:
        gir_row = []

        gir_row.append(str(gir_node.op_schema)) # adding schema information
        gir_row.append(str(gir_node.ret))       # adding ret information
        gir_row.append(str(gir_node.op_name))   # adding operation name
        gir_row.append(str(gir_node.print_stmt_args()))

        gir_content_rows.append(gir_row)

    gir_table.add_rows(gir_content_rows)
    print(gir_table)

def pretty_print_Central_Node(central_node: CentralNode, print_tensors=True):
    print("\n🦋 Pretty Printing Central Node ----------------------------------------------------------\n")

    # print(f'⏺️ Central Node : {central_node.get_address()}\n')

    central_node_properties = list(central_node.__dict__.items())
    neighbour_nodes_info = central_node_properties[:2]
    edges_info = central_node_properties[2:4]
    misc_info = central_node_properties[4:]

    central_node_property_names = central_node.__dict__.keys()

    # printing neighbouring nodes information
    for info_name, info_value in neighbour_nodes_info:
        print(f'🔵 {info_name} : {info_value}')

    print("")

    # printing edges information
    for info_name, info_value in edges_info:
        print(f'🟣 {info_name} : {info_value}')

    print("")

    # printing misc information

    val_table = PrettyTable()
    val_table.set_style(SINGLE_BORDER)
    val_table.align = "l"
    val_table.hrules = ALL

    val_table.field_names = ["Val Name", "Val Type", "Backend", "Reduced Dimension", "_t", "ID", "_v","Var"]
    val_table_content_rows = []
    gir_list = []
    tensor_info = {}

    for info_name, info_value in misc_info:
        val_row = []

        tensor_info[info_name] = {}
        tensor_info[info_name]['_t'] = info_value._t
        tensor_info[info_name]['_v'] = info_value._v

        val_row.append(info_name)
        val_row.append(get_val_type_name(info_value._val_type))
        val_row.append(info_value._Bkey)
        val_row.append(info_value._reduce_dim)

        val_row.append(info_value._t if print_tensors else "")

        # _t = info_value._t
        # _v = info_value._v
        val_row.append(info_value._id)
        val_row.append(info_value._v if print_tensors else "")
        val_row.append(info_value.var)
        gir_list.append((info_name, info_value.fprog))

        val_table_content_rows.append(val_row)

    val_table.add_rows(val_table_content_rows)
    print(val_table)

    # print(tensor_info)

    # for info_name, tensor_vals in tensor_info.items():
    #     print(f'🏹 Tensor information for {info_name}\n')
    #     _t, _v = tensor_vals.get("_t", None), tensor_vals.get("_v", None)
    #     print(f'_t = {_t}\n')
    #     print(f'_v = {_v}\n')

    for gir_name, gir in gir_list:
        pretty_print_GIR(gir, gir_name)

    print("\n------------------------------------------------------------------------------------------\n")
    # print("In Neighbouring Nodes:\n")
    # for in_nodes in central_node.innbs:
    #     print(in_nodes)

    # print("\nOut Neighbouring Nodes:\n")
    # for out_nodes in central_node.outnbs:
    #     print(out_nodes)