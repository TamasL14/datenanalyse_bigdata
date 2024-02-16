import h5py
import json
import base64
import numpy as np
import os
from database import upload_file
from itertools import islice
from datetime import datetime


#h5file_path = "/Users/t.lukacs/Downloads/dataset_small"


def convert_dataset_to_list(dataset):
    """Convert a HDF5 `Dataset` object to a list of arrays."""
    data = dataset[:]
    return list(data)

def convert_dataset_bytes_to_json(dataset):
    """Convert a HDF5 `Dataset` object containing bytes to a JSON string."""
    data = dataset[:]
    encoded_data = base64.b64encode(data)
    base64_str = f'data:application/octet-stream;base64,{encoded_data.decode("utf-8")}'
    return base64_str

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        """Custom JSON encoder to handle bytes objects."""
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        return super().default(obj)

def convert_h5_to_json(file_path):
    """Convert a HDF5 file to a JSON file."""
    with h5py.File(file_path) as f:
        data=None
        try:
            data = f['data']
        except:
            try:
                data = f['Daten']
                print(file_path)
            except:
                print("Error missing Daten in dataset: {}".format(file_path))
                return 0
        json_data = {}

        # Convert group attributes to a dictionary
        group_attributes = {}
        for attr_name, attr_value in data.attrs.items():
            if isinstance(attr_value, np.ndarray):
                attr_value = attr_value.tolist()
            group_attributes[attr_name] = attr_value
            if attr_name == "id":
                data_id = attr_value
                data_id = data_id.replace("-", "")

        json_data['attributes'] = group_attributes

        dataset={}
        anz=1000
        # Convert datasets to JSON-compatible format
        for name, obj in data.items():
            name = name.lower()
            name = name.rstrip("_")
            if isinstance(obj, h5py.Dataset) and obj.dtype.char == 'b':
                dataset[name] = convert_dataset_bytes_to_json(obj)
            else:
                if  int(obj.shape[0]) != 1000:
                    anz=int(obj.shape[0])
                    n=name
                dataset[name] = convert_dataset_to_list(obj)
            
        # Create list of dictionaries for JSON representation
        new_json_data = []
        time=0
        j=0
        i=0
        for i, (defect_channel,distance, magnetization,timestamp,velocity,wall_thickness) in \
            enumerate(zip(dataset['defect_channel'], dataset['distance'], dataset['magnetization'], \
                          dataset['timestamp'], dataset['velocity'], dataset['wall_thickness'])):
            try:
                timestamp=timestamp-time
            except:
                timestamp=timestamp.decode()
                try:
                    timestamp=float(timestamp)
                except:
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
                    timestamp=timestamp.timestamp()
                    timestamp=timestamp-time
            try:
                asd=distance*timestamp
            except:
                distance=int.from_bytes(distance)

            if i==0:
                time=timestamp
                timestamp=0 
            try:    
                new_json_data.append({
                    "punkt": int(i),
                    "defect_channel": float(defect_channel),
                    "distance": float(distance),
                    "magnetization": float(magnetization),
                    "timestamp": float(timestamp),
                    "velocity": float(velocity),
                    "wall_thickness": float(wall_thickness),
                })
            except:
                print("Error at: {}".format(data_id))
            j=i+1
            if i==anz-1:
                i+=1
                break

        for (defect_channel,distance, magnetization,timestamp,wall_thickness) in \
        islice(zip(dataset['defect_channel'], dataset['distance'], dataset['magnetization'], dataset['timestamp'], dataset['wall_thickness']),j,None):
            
            try:
                timestamp=timestamp-time
            except:
                timestamp=timestamp.decode()

            try:
                timestamp=float(timestamp)-time
            except:
                timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
                timestamp=timestamp.timestamp()
                timestamp=timestamp-time

            try:
                velocity=(distance*1000)/timestamp
            except:
                distance=int.from_bytes(distance)
                velocity=(distance*1000)/timestamp

            try:    
                new_json_data.append({
                    "punkt": int(i),
                    "defect_channel": float(defect_channel),
                    "distance": float(distance),
                    "magnetization": float(magnetization),
                    "timestamp": float(timestamp),
                    "velocity": float(velocity),
                    "wall_thickness": float(wall_thickness),
                })
            except:
                print("Error at: {}".format(data_id))
            i+=1

        json_data['data'] = new_json_data
        """filename = data_id+ ".json"
        json_file_path = "/Users/t.lukacs/Downloads/data_small/{}".format(filename)
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, cls=CustomEncoder)"""
        return data_id, json_data

def prep_file(file):
        data_id,json_data = convert_h5_to_json(file)
        upload_file(data_id, json_data)

def prep_folder(folder):           
    for file in os.listdir(folder):
        if file.endswith(".h5"):  # Ensure only .h5 files are processed
            full_file_path = os.path.join(folder, file)  # Construct full path
            data_id, json_data = convert_h5_to_json(full_file_path)
            upload_file(data_id, json_data)