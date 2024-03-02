"""
In diesem Modul wird die Funktion prep_folder(folder) definiert. 
Diese Funktion wird in der Datei menu_ui.py aufgerufen, wenn der Benutzer einen Ordner hochladen möchte. 
Die Funktion prep_folder(folder) durchsucht den Ordner nach .h5-Dateien 
und ruft die Funktion convert_h5_to_json(file) auf, um die .h5-Dateien in .json-Dateien umzuwandeln. 
Die .json-Dateien werden dann in der Datenbank hochgeladen.
"""

# Import Bibliotheken
import h5py
import json
import base64
import numpy as np
import os
from itertools import islice
from datetime import datetime

# Import Funktionen aus anderen Modulen
from database import upload_file

# Funktion um die Datensätze in eine Liste umzuwandeln
def convert_dataset_to_list(dataset):
    data = dataset[:]
    return list(data)

# Funktion um die Bytes Datensätze in zu encodierte JSON-Strings umzuwandeln
def convert_dataset_bytes_to_json(dataset):
    data = dataset[:]
    encoded_data = base64.b64encode(data)
    base64_str = f'data:application/octet-stream;base64,{encoded_data.decode("utf-8")}'
    return base64_str

# Funktion um die .h5-Dateien in .json-Dateien umzuwandeln
def convert_h5_to_json(file_path):
    # Öffnen der .h5-Datei
    with h5py.File(file_path) as f:
        data=None

        # Versuch, die Gruppe "data" in der Datei zu öffnen
        try:
            data = f['data']
        except:
            # Versuch, die Gruppe "Daten" in der Datei zu öffnen
            try:
                data = f['Daten']
            except:
                # Fehlermeldung, wenn die Gruppe "data" oder "Daten" nicht in der Datei vorhanden ist
                print("Error missing Daten in dataset: {}".format(file_path))
                return 0
            
        # JSON-Objekt erstellen
        json_data = {}

        # Attribute der Gruppe in ein Dictionary umwandeln
        group_attributes = {}
        for attr_name, attr_value in data.attrs.items(): # Attribute in der Gruppe durchgehen
            if isinstance(attr_value, np.ndarray): # Überprüfen, ob das Attribut ein Numpy Array ist
                attr_value = attr_value.tolist() # Array in eine Liste umwandeln
            group_attributes[attr_name] = attr_value # Attribute in das Dictionary einfügen
            if attr_name == "id": # Wenn das Attribut "id" ist
                data_id = attr_value # data_id speichern
                data_id = data_id.replace("-", "") # data_id in einen String ohne Zusatzzeichen umwandeln

        # Dictionary für die Datensätze erstellen
        dataset={}
        anz=1000
        # Datensätze aus der .h5-Datei in ein Dictionary umwandeln
        for name, obj in data.items():
            # Namen in Kleinbuchstaben umwandeln und überflüssige Unterstriche entfernen
            name = name.lower()
            name = name.rstrip("_")
            # Wenn der Datensatz Bytes enthält, wird die Funktion convert_dataset_bytes_to_json() aufgerufen, um den Datensatz in einen encodierten JSON-String umzuwandeln
            if isinstance(obj, h5py.Dataset) and obj.dtype.char == 'b':
                dataset[name] = convert_dataset_bytes_to_json(obj)
            else:
                # Wenn der Datensatz keine 1000 Punkte enthält, wird die Anzahl der Punkte in der Variable anz gespeichert
                if  int(obj.shape[0]) != 1000:
                    anz=int(obj.shape[0])
                dataset[name] = convert_dataset_to_list(obj)
            
        # Variable für die neuen Datensätze erstellen
        new_json_data = []
        time=0
        j=0
        i=0
        # Datensätze in ein neues Dictionary umwandeln
        for i, (defect_channel,distance, magnetization,timestamp,velocity,wall_thickness) in \
            enumerate(zip(dataset['defect_channel'], dataset['distance'], dataset['magnetization'], \
                          dataset['timestamp'], dataset['velocity'], dataset['wall_thickness'])):
            
            # Wenn der Datensatz Bytes enthält, wird die Funktion convert_dataset_bytes_to_json() aufgerufen, um den Datensatz in einen encodierten JSON-String umzuwandeln
            if type(distance)==bytes:
                distance=int.from_bytes(distance)
            
            # Versuch, den Timestamp zu dekodieren und in einen float umzuwandeln
            try:
                timestamp=timestamp.decode()
            except:
                pass

            # Wenn der Timestamp ein String ist, wird er in ein datetime-Objekt umgewandelt und dann in einen Timestamp
            try:
                timestamp = datetime.strptime(str(timestamp), "%Y-%m-%dT%H:%M:%S")
                timestamp=timestamp.timestamp()
            except:
                pass

            # Wenn der Loop zum ersten Mal durchläuft, wird der Wert von timestamp in der Variable time gespeichert und Timestamp auf 0 gesetzt
            if i==0:
                time=timestamp
                timestamp=0 

            # Versuche Zietpunkt des Datensatzes zu berechnen
            if timestamp!=0:
                    timestamp=float(timestamp)-float(time) # Zeitpunkt des Datensatzes berechnen

            # Datensätze in die Liste einfügen
            new_json_data.append({
                "punkt": i,
                "defect_channel": defect_channel,
                "distance": distance,
                "magnetization": magnetization,
                "timestamp": timestamp,
                "velocity": velocity,
                "wall_thickness": wall_thickness,
            })
            j=i+1

            # Wenn die Anzahl der Datensätze die Anzahl von dem nicht vollständigen Datensatz erreicht, wird der Loop abgebrochen
            if i==anz-1:
                i+=1
                break

        if anz!=1000: # Wenn die Anzahl der Datensätze nicht 1000 beträgt, werden die fehlenden Datensätze ausgerechnet und in die Liste eingefügt
            for (defect_channel,distance, magnetization,timestamp,wall_thickness) in \
            islice(zip(dataset['defect_channel'], dataset['distance'], dataset['magnetization'], dataset['timestamp'], dataset['wall_thickness']),j,None):
                
                # Wenn der Datensatz Bytes enthält, wird die Funktion convert_dataset_bytes_to_json() aufgerufen, um den Datensatz in einen encodierten JSON-String umzuwandeln
                if type(distance)==bytes:
                    distance=int.from_bytes(distance)

                # Versuch, den Timestamp zu dekodieren und in einen float umzuwandeln
                try:
                    timestamp=timestamp.decode()
                except:
                    pass
                
                try:
                    timestamp = datetime.strptime(str(timestamp), "%Y-%m-%dT%H:%M:%S")
                    timestamp=timestamp.timestamp()
                except:
                    pass

                # Wenn der Loop zum ersten Mal durchläuft, wird der Wert von timestamp in der Variable time gespeichert und Timestamp auf 0 gesetzt
                if i==0:
                    time=timestamp  
                    timestamp=0 

                # Versuche Zietpunkt des Datensatzes zu berechnen
                if timestamp!=0:
                        timestamp=float(timestamp)-float(time) # Zeitpunkt des Datensatzes berechnen

                # Fehlende Werte für die Geschwindigkeit berechnen
                if timestamp!=0:
                    try:
                        velocity=(int(distance)*1000)/float(timestamp)
                    except:
                        distance=int.from_bytes(distance)
                        velocity=(int(distance)*1000)/float(timestamp)
                else:
                    velocity=0
            # Datensätze in die Liste einfügen
            new_json_data.append({
                "punkt": i,
                "defect_channel": defect_channel,
                "distance": distance,
                "magnetization": magnetization,
                "timestamp": timestamp,
                "velocity": velocity,
                "wall_thickness": wall_thickness,
            })
            i+=1 # Zahl der punkt erhöhen
            
        group_attributes['datum']=datetime.utcfromtimestamp(float(time)).strftime('%Y-%m-%d') # Datum des Datensatzes als Attribute hinzufügen
        json_data['attributes'] = group_attributes # Attribute in das JSON-Objekt einfügen
        json_data['data'] = new_json_data # Datensätze in das JSON-Objekt einfügen
        return data_id, json_data

def prep_file(file):
        data_id,json_data = convert_h5_to_json(file)      
        upload_file(data_id, json_data) # Die Funktion upload_file() aus der Datei database.py wird aufgerufen und dabei die JSON-Datei in die Datenbank hochgeladen

# Funktion um die .h5-Dateien aus einem Ordner in .json-Dateien umzuwandeln
def prep_folder(folder):
    i=1          
    for file in os.listdir(folder): # Alle Dateien im Ordner durchsuchen
        print(i)
        i+=1
        # Wenn die Datei eine .h5 Datei ist, wird die Funktion convert_h5_to_json() aufgerufen
        if file.endswith(".h5"):            
            full_file_path = os.path.join(folder, file) # Deteiverzeichnis und Dateiname zusammenfügen           
            data_id, json_data = convert_h5_to_json(full_file_path) # Die Funktion convert_h5_to_json() ausführen
            upload_file(data_id, json_data) # Die Funktion upload_file() aus der Datei database.py wird aufgerufen und dabei die JSON-Datei in die Datenbank hochgeladen