import json,yaml,os
from application_logging.logger import App_Logger
from pymongo import MongoClient
import pandas as pd
from Gcp_methods import Gcp

class dBOperation:
    """
          This class shall be used for handling all the mongodb operations.

          Written By: Tejas Dadhaniya
          Version: 1.0
          Revisions: None

          """

    def __init__(self):
        self.fileName = 'InputFile.csv'

        with open(os.path.join("configfiles","connection.yaml"),"r") as file:
            self.config = yaml.safe_load(file)
        self.url = self.config["db"]["connection_string"]

        self.logger = App_Logger(database="Prediction_Logs")
        self.gcp_log =  "GCPlog"
        self.database_name = "Prediction_Logs"
        self.gcp = Gcp(file_object=self.gcp_log, logger_object=self.logger)

        with open(os.path.join("configfiles", "params.yaml"), "r") as f:
            self.config = yaml.safe_load(f)
        self.fromdb = self.config['prediction']['trainingfilefromdb']
        self.good_raw = self.config['prediction']['good_raw']
        self.bin = self.config['training']['bin']


    def dataBaseConnection(self):

        """
                        Method Name: dataBaseConnection
                        Description: This method creates the database with the given name and if Database already exists then opens the connection to the DB.
                        Output: Connection to the DB
                        On Failure: Raise ConnectionError

                         Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                        """
        try:
            conn = MongoClient(self.url)
            file = "DataBaseConnectionLog"
            self.logger.log(file, "Connection establish")#,database=self.database_name)
        except ConnectionError:
            file = "DataBaseConnectionLog"
            self.logger.log(file, "Error while connecting to database: %s" %ConnectionError)#,database=self.database_name)
            raise ConnectionError
        return conn

    def dropDatabase(self,DatabaseName):

        """
                    Method Name: dropDatabase
                    Description: This method remove the given database from mongodb atlas.
                    Output: None
                    On Failure: Raise Exception

                    Written By: Tejas Dadhaniya
                    Version: 1.0
                    Revisions: None

                                """
        try:
            conn = self.dataBaseConnection()
            conn.drop_database(DatabaseName)
            log_object = "DBlog"
            self.logger.log(log_object,"{} Database delete successfully".format(DatabaseName))#,database=self.database_name)

        except Exception as e:
            log_object = "DBlog"
            self.logger.log(log_object,"Error while deleting {} database : {}".format(DatabaseName,e))#,database=self.database_name)
            raise e

    def dropCollection(self,dataBase,collectionName):

        """
                   Method Name: dropCollection
                   Description: This method remove the given collection from database.
                   Output: None
                   On Failure: Raise Exception

                   Written By: Tejas Dadhaniya
                   Version: 1.0
                   Revisions: None

                                       """

        try:
            conn = self.dataBaseConnection()
            db = conn[dataBase]
            db.drop_collection(collectionName)
            log_object = "DBlog"
            self.logger.log(log_object,"{} collection form {} database delete successfully".format(collectionName,dataBase))#,database=self.database_name)

        except Exception as e:
            log_object = "DBlog"
            self.logger.log(log_object,"Error while deleting {} collection form {} database".format(collectionName, dataBase))#,database=self.database_name)
            return e

    def createDatabase(self,dataBase):


        """
                   Method Name: createDatabase
                   Description: This method create a given database.
                   return: db
                   On Failure: Raise Exception

                    Written By: Tejas Dadhaniya
                   Version: 1.0
                   Revisions: None

                                       """
        try:
            conn = self.dataBaseConnection()
            db = conn[dataBase]
            log_object = "DBlog"
            self.logger.log(log_object,"{} database create successfully".format(dataBase))#,database=self.database_name)

        except Exception as e:
            log_object = "DBlog"
            self.logger.log(log_object,"Error while creating {} database".format(dataBase))#,database=self.database_name)
            raise e

        return db

    def createCollection(self,dataBase,collectionName):


        """
                   Method Name: createCollection
                   Description: This method create a given collection in given database.
                   return: collection
                   On Failure: Raise Exception

                    Written By: Tejas Dadhaniya
                   Version: 1.0
                   Revisions: None

                                       """
        try:
            conn = self.dataBaseConnection()
            db = conn[dataBase]
            collection = db[collectionName]
            log_object = "DBlog"
            self.logger.log(log_object,"{} collection in {} database create successfully".format(collectionName, dataBase))#,database=self.database_name)

        except Exception as e:
            log_object = "DBlog"
            self.logger.log(log_object,"Error while creating {} collection in {} database".format(collectionName, dataBase))#,database=self.database_name)
            raise e

        return collection

    def createTableDb(self,DatabaseName,collectionName):

        """
                    Method Name: createTableDb
                    Description: This method creates a table in the given database which will be used to insert the Good data after raw data validation.
                    Output: None
                    On Failure: Raise Exception

                     Written By: Tejas Dadhaniya
                    Version: 1.0
                    Revisions: None

                                """
        try:
            self.dropDatabase(DatabaseName)
            #directory = self.goodFilePath
            collection = self.createCollection(DatabaseName,collectionName)
            onlyfiles = self.gcp.get_allFiles(self.good_raw)
            log_file = "DbTableCreateLog"
            table_log = "DbInsertLog"

            for file in onlyfiles:
                df = pd.read_csv("gs://{}/{}".format(self.good_raw,file))
                record = json.loads(df.T.to_json()).values()
                collection.insert(record)
                self.logger.log(table_log,"Record insert successfully in collection")#,database=self.database_name)

            self.logger.log(log_file, "Insertion in Table completed!!!")#,database=self.database_name)

        except Exception as e:
            log_file = "DbTableCreateLog"
            self.logger.log(log_file, "Error while creating table: %s " % e)#,database=self.database_name)

            raise e

    def selectingDatafromtableintocsv(self,Database,collectionName):

        """
                               Method Name: selectingDatafromtableintocsv
                               Description: This method exports the data in GoodData table as a CSV file. in a given location.
                               Output: None
                               On Failure: Raise Exception
                               Written By: Tejas Dadhaniya
                               Version: 1.0
                               Revisions: None

        """

        try:
            log_file = "ExportToCsv"
            collection = self.createCollection(dataBase=Database,collectionName=collectionName) #connecting to collection
            df = pd.DataFrame(collection.find())

            if "_id" in df.columns:
                df.drop("_id", axis=1, inplace=True)

            df.rename({"default payment next month": "target"}, axis=1, inplace=True)
            storage_client = self.gcp.connection()
            bucket = storage_client.get_bucket(self.fromdb)
            self.gcp.copy_all_blob(self.fromdb, self.bin)
            self.gcp.delete_blob(self.fromdb, type_="all")
            bucket.blob(self.fileName).upload_from_string(df.to_csv(index=False), 'text/csv')
            self.logger.log(log_file, "File exported successfully!!!")

        except Exception as e:
            log_file = "ExportToCsv"
            self.logger.log(log_file, "File exporting failed. Error : %s" % e)#,database=self.database_name)
            raise e


