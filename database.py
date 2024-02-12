from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pymongo
from bson import ObjectId
import json

load_dotenv()

username = os.getenv("DB_USERNAME")
passwort = os.getenv("DB_PASSWORD")
MONGO_URL = "mongodb+srv://{}:{}@rosentestdata.ky0vl7x.mongodb.net/?retryWrites=true&w=majority".format(username, passwort)
client = MongoClient(MONGO_URL)
db = client["bigdata"]
collection = db["Sensordaten"]

def connect_to_db():
    client = MongoClient(MONGO_URL)
    return True

def check_for_file(data_id):
    if collection.count_documents({'_id':data_id},limit=1) != 0:
        return True
    else:
        return False

def upload_file(data_id, json_data):
    if check_for_file(data_id)==False:
        collection.insert_one({'_id':data_id, 'DATASET':json_data})
    else:
        collection.update_one({'_id':data_id}, {'$set':{'DATASET':json_data}})

def get_data():
    data_in_collection = [str(id) for id in collection.distinct('_id')]
    return data_in_collection

def get_data_property(data_id,property_list):
    punkte=[]
    data = collection.find_one({'_id':data_id})
    data = data.get("DATASET")
    data = data.get("data")

    """for i in property_list:
        print(i)
        for temp in data:
            punkte.append(temp[i])"""
    

    for i in property_list:
        temp2=[]
        for temp in data:
            temp2.append(temp[i])
        punkte.append(temp2)
    
    
        
    
    return punkte