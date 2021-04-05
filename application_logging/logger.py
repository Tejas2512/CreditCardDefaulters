from datetime import datetime
from pymongo import MongoClient
import os,yaml

class App_Logger:

    def __init__(self,database = "Training_Logs"):

        with open(os.path.join("configfiles","connection.yaml"),"r") as file:
            self.config = yaml.safe_load(file)
        self.url = self.config["db"]["connection_string"]

        self.database = database

    def log(self, file_object, log_message):
        client = MongoClient(self.url)
        db = client[self.database]
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        collection = db[file_object]
        collection.insert_one({'Date': str(self.date),
                               "time": str(self.current_time),
                               'log': log_message})
