import csv
import json
from rich import inspect

pedalme_json = {}

pedalme_json["edges"] = []
pedalme_json["weights"] = []

# parsing pedalme_edges.csv
with open('pedalme_edges.csv', 'r') as edges_csv_file:
    reader = csv.reader(edges_csv_file)
    
    for row in reader:
        
        # checking if it's the header row
        if "from" in row:
            continue
        
        pedalme_json["edges"].append([int(row[0]), int(row[1])])
        pedalme_json["weights"].append(float(row[2]))
        
# parsing pdealme_features.csv
pedalme_json['time_periods'] = 36

with open('pedalme_features.csv', 'r') as features_csv_file:
    reader = csv.reader(features_csv_file)
    
    for row in reader:
        
        # checking if it's the header row
        if "year" in row:
            continue
        
        time = row[4]
        if time not in pedalme_json:
            pedalme_json[time] = []
        
        pedalme_json[time].append(int(row[5]))
                 
with open('pedalme.json', 'w') as fp:
    json.dump(pedalme_json, fp)
