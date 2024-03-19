# Dieses Modul stellt die Verbindung zur MongoDB her und ermöglicht das Hochladen und Herunterladen von Daten

# Import Bibliotheken
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv() # Environment Variablen laden

# Datenbank Credentials aus der .env Datei laden
username = os.getenv("DB_USERNAME")
passwort = os.getenv("DB_PASSWORD")
MONGO_URL = "mongodb+srv://{}:{}@rosentestdata.ky0vl7x.mongodb.net/?retryWrites=true&w=majority".format(username, passwort) # Datenbank URL zusammenstellen
client = MongoClient(MONGO_URL) # Verbindung zur Datenbank herstellen
# Datenbank und Collection auswählen
db = client["bigdata"]
collection = db["Sensordaten"]

# Funktion um die Verbindung zur Datenbank herzustellen
def connect_to_db():
    client = MongoClient(MONGO_URL)
    return True

# Funktion um zu überprüfen, ob die Datei bereits in der Datenbank vorhanden ist
def check_for_file(data_id):
    if collection.count_documents({'_id':data_id},limit=1) != 0:
        return True
    else:
        return False

# Funktion um die Daten in die Datenbank hochzuladen
def upload_file(data_id, json_data):
    if check_for_file(data_id)==False: # Überprüfen, ob die Datei bereits in der Datenbank vorhanden ist
        collection.insert_one({'_id':data_id, 'DATASET':json_data})# Wenn die Datei nicht vorhanden ist, wird sie hochgeladen
    else:       
        collection.update_one({'_id':data_id}, {'$set':{'DATASET':json_data}}) # Wenn die Datei bereits vorhanden ist, wird sie aktualisiert

# Funktion um die Daten aus der Datenbank herunterzuladen
def get_data():
    projection = {"_id": 1, "DATASET.attributes.id": 1, "DATASET.attributes.datum": 1, "DATASET.attributes.configuration": 1, "DATASET.attributes.instrument": 1}
    data = list(collection.find({}, projection))
    data_in_collection = []
    for i, document in enumerate(data):
        data_id = document["_id"]
        datum = document["DATASET"]["attributes"].get("datum", "")
        kontinent = document["DATASET"]["attributes"]["configuration"]
        instrument = document["DATASET"]["attributes"]["instrument"]
        data_in_collection.append([i+1, str(data_id), datum, kontinent, instrument])

    return data_in_collection

def get_cluster_center(given_instrument, given_kontinent):
    projection = {
        "_id": 1,
        "DATASET.attributes.Mittelpunkte_Cluster": 1,
        "DATASET.attributes.instrument": 1,
        "DATASET.attributes.configuration": 1
    }
    query = {
        "DATASET.attributes.instrument": given_instrument,
        "DATASET.attributes.configuration": given_kontinent
    }
    data = list(collection.find(query, projection))
    cluster_centers = []

    for document in data:
        if "DATASET" in document and "attributes" in document["DATASET"] and "Mittelpunkte_Cluster" in document["DATASET"]["attributes"]:
            for center in document["DATASET"]["attributes"]["Mittelpunkte_Cluster"]:
                cluster_centers.append(center)

    return cluster_centers

# Verwende die Funktion:
# results = cluster_center(collection, specific_data_id, specific_instrument, specific_kontinent)

def get_cluster_center_from_selectedrows(data_id):
    data = collection.find_one({'_id':data_id}) # Datei mit dem entsprechenden data_id holen   
    data = data.get("DATASET") # Inhalt der Datei holen    
    data = data.get("attributes") # Inhalt der Datei holen
    data = data.get("Mittelpunkte_Cluster") # Inhalt der Datei holen
    return data


# Funktion um die Datensätze aus der Datenbank herunterzuladen
def get_data_property(data_id,property_list):
    punkte=[]   
    data = collection.find_one({'_id':data_id}) # Datei mit dem entsprechenden data_id holen   
    data = data.get("DATASET") # Inhalt der Datei holen    
    data = data.get("data") # Inhalt der Datensaetze holen

    # Ausgewählte Datenpunkte in einer Liste speichern
    for i in property_list:
        temp2=[]
        for temp in data:
            temp2.append(temp[i])
        punkte.append(temp2)

    return punkte # Liste mit den ausgewählten Datenpunkten zurückgeben