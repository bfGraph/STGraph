import csv
from rich import inspect

from stgraph.benchmark_tools.table import BenchmarkTable

all_results = []

def is_fixed_backprop(parameters) -> bool:
    return parameters[3] == 'B0'
        
def is_pygt(parameters) -> bool:
    return parameters[0] == 'pygt'

def is_seastar(parameters) -> bool:
    return parameters[0] == 'seastar'
        
def get_time_result(result):
    return round(float(result[1]), 4) if result[1] != 'OOM' else 'OOM'

def get_memory_result(result):
    return round(float(result[2]), 4) if result[2] != 'OOM' else 'OOM'
        
def get_hidden_dimension(parameters) -> int:
    """ Returns the hidden dimension by parsing 'H[hidden_dimension]' """
    return int(parameters[4][1:])

def get_sequence_length(parameters) -> int:
    return int(parameters[3][1:])

def get_dataset_name(parameters):
    return parameters[1]

with open('static-temporal.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    for row in reader:
        if 'Filename' not in row:
            all_results.append(row)
            
dataset_names = ['monte', 'hungarycp', 'pedalme']
            
for dataset in dataset_names:
    
            
    # forming the Table 1: Time measurements for varying feature sizes 
    table_1 = BenchmarkTable(f"Time measurements (s) for varying feature sizes - {dataset}", ["Hidden Dimension", "PyG-T", "STGraph"])
    pygt_results_table_1 = {}
    seastar_results_table_1 = {}

    for result in all_results:
        file_name = result[0]
        parameters = file_name.split("_")
        
        if get_dataset_name(parameters) != dataset:
            continue
        
        if is_fixed_backprop(parameters):
            time_taken = get_time_result(result)
            hidden_dim = get_hidden_dimension(parameters)
            
            if is_pygt(parameters):
                pygt_results_table_1[str(hidden_dim)] = time_taken
            
            if is_seastar(parameters):
                seastar_results_table_1[str(hidden_dim)] = time_taken

    if len(pygt_results_table_1) != 0 and len(seastar_results_table_1):
        for hidden_dim in range(16, 160+1, 16):
            if str(hidden_dim) in pygt_results_table_1 and str(hidden_dim) in seastar_results_table_1:
                table_1.add_row([hidden_dim, pygt_results_table_1[str(hidden_dim)], seastar_results_table_1[str(hidden_dim)]])
            
        table_1.display()

    # forming the Table 2: Memory measurements for varying feature sizes 
    table_2 = BenchmarkTable(f"Memory taken (MB) for varying feature sizes - {dataset}", ["Hidden Dimension", "PyG-T", "STGraph"])
    pygt_results_table_2 = {}
    seastar_results_table_2 = {}

    for result in all_results:
        file_name = result[0]
        parameters = file_name.split("_")
        
        if get_dataset_name(parameters) != dataset:
            continue
        
        if is_fixed_backprop(parameters):
            memory_taken = get_memory_result(result)
            hidden_dim = get_hidden_dimension(parameters)
            
            if is_pygt(parameters):
                pygt_results_table_2[str(hidden_dim)] = memory_taken
            
            if is_seastar(parameters):
                seastar_results_table_2[str(hidden_dim)] = memory_taken
                
    if len(pygt_results_table_2) != 0 and len(seastar_results_table_2) != 0:
        for hidden_dim in range(16, 160+1, 16):
            if str(hidden_dim) in pygt_results_table_2 and str(hidden_dim) in seastar_results_table_2:
                table_2.add_row([hidden_dim, pygt_results_table_2[str(hidden_dim)], seastar_results_table_2[str(hidden_dim)]])
            
        table_2.display()

for dataset in dataset_names:
        
    # forming the Table 1: Time measurements for varying sequence lengths
    table_1 = BenchmarkTable(f"Time measurements (s) for varying sequence lengths - {dataset}", ["Hidden Dimension", "PyG-T", "STGraph"])
    pygt_results_table_1 = {}
    seastar_results_table_1 = {}

    for result in all_results:
        file_name = result[0]
        parameters = file_name.split("_")
        
        if get_dataset_name(parameters) != dataset:
            continue
        
        if not is_fixed_backprop(parameters):
            time_taken = get_time_result(result)
            seq_len = get_sequence_length(parameters)
            
            if is_pygt(parameters):
                pygt_results_table_1[str(seq_len)] = time_taken
            
            if is_seastar(parameters):
                seastar_results_table_1[str(seq_len)] = time_taken

    if len(pygt_results_table_1) != 0 and len(seastar_results_table_1):
        for seq_len in range(100, 700+1, 100):
            if str(seq_len) in pygt_results_table_1 and str(seq_len) in seastar_results_table_1:
                table_1.add_row([seq_len, pygt_results_table_1[str(seq_len)], seastar_results_table_1[str(seq_len)]])
            
        table_1.display()

    # forming the Table 2: Memory measurements for varying sequence lengths
    table_2 = BenchmarkTable(f"Memory taken (MB) for varying sequence lengths- {dataset}", ["Hidden Dimension", "PyG-T", "STGraph"])
    pygt_results_table_2 = {}
    seastar_results_table_2 = {}

    for result in all_results:
        file_name = result[0]
        parameters = file_name.split("_")
        
        if get_dataset_name(parameters) != dataset:
            continue
        
        if not is_fixed_backprop(parameters):
            memory_taken = get_memory_result(result)
            seq_len = get_sequence_length(parameters)
            
            if is_pygt(parameters):
                pygt_results_table_2[str(seq_len)] = memory_taken
            
            if is_seastar(parameters):
                seastar_results_table_2[str(seq_len)] = memory_taken
                
    if len(pygt_results_table_2) != 0 and len(seastar_results_table_2) != 0:
        for seq_len in range(100, 700+1, 100):
            if str(seq_len) in pygt_results_table_2 and str(seq_len) in seastar_results_table_2:
                table_2.add_row([seq_len, pygt_results_table_2[str(seq_len)], seastar_results_table_2[str(seq_len)]])
            
        table_2.display()

