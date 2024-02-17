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
    data_in_collection = [str(id) for id in collection.distinct('_id')] # IDs von allen Daten in der Collection holen
    return data_in_collection

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