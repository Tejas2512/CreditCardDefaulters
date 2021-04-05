import sqlite3
from datetime import datetime
from os import listdir
import os,yaml
import re
import json
import shutil
import pandas as pd
from application_logging.logger import App_Logger
from DataTypeValidation_Insertion_Prediction.DataTypeValidationPrediction import dBOperation
import smtplib
from Gcp_methods import Gcp

class Prediction_Data_validation:
    """
               This class shall be used for handling all the validation done on the Raw Prediction Data!!.

               Written By: iNeuron Intelligence
               Version: 1.0
               Revisions: None

               """

    def __init__(self,path):
        self.db_operation = dBOperation()
        self.Batch_Directory = path
        #self.schema_path = 'schema_prediction.json'
        self.logger = App_Logger(database="Prediction_Logs")
        self.gcp_log = "GCPlog"
        self.gcp = Gcp(file_object=self.gcp_log, logger_object=self.logger)


        with open(os.path.join("configfile", "params.yaml"), "r") as f:
            self.config = yaml.safe_load(f)
        self.predictionfile = self.config["prediction"]["bucket"]
        self.good_raw = self.config["prediction"]["good_raw"]
        self.bad_raw = self.config["prediction"]["bad_raw"]
        self.archive_good_raw = self.config["prediction"]["archive_good_raw"]
        self.archive_bad_raw = self.config["prediction"]["archive_bad_raw"]


    def valuesFromSchema(self):
        """
                                Method Name: valuesFromSchema
                                Description: This method extracts all the relevant information from the pre-defined "Schema" file.
                                Output: LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, Number of Columns
                                On Failure: Raise ValueError,KeyError,Exception

                                 Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                                        """
        file = "valuesfromSchemaValidationLog"
        try:
            client = self.db_operation.dataBaseConnection()
            database = client["schema"]
            schema = database["Prediction_schema"]
            df = pd.DataFrame(schema.find())

            LengthOfDateStampInFile = int(df['LengthOfDateStampInFile'])
            LengthOfTimeStampInFile = int(df['LengthOfTimeStampInFile'])
            NumberofColumns = int(df['NumberofColumns'])
            column_names = df['ColName']

            message ="LengthOfDateStampInFile:: %s" %LengthOfDateStampInFile + "\t" + "LengthOfTimeStampInFile:: %s" % LengthOfTimeStampInFile +"\t " + "NumberofColumns:: %s" % NumberofColumns + "\n"
            self.logger.log(file,message)

        except ValueError:
            self.logger.log(file,"ValueError:Value not found inside schema_training.json")
            raise ValueError

        except KeyError:
            self.logger.log(file, "KeyError:Key value error incorrect key passed")
            raise KeyError

        except Exception as e:
            self.logger.log(file, str(e))
            raise e

        return LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, NumberofColumns

    def manualRegexCreation(self):

        """
                                      Method Name: manualRegexCreation
                                      Description: This method contains a manually defined regex based on the "FileName" given in "Schema" file.
                                                  This Regex is used to validate the filename of the prediction data.
                                      Output: Regex pattern
                                      On Failure: None

                                       Written By: iNeuron Intelligence
                                      Version: 1.0
                                      Revisions: None

                                              """
        regex = "['creditCardFraud']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    def deleteExistingGoodDataTrainingFolder(self):
        """
                                            Method Name: deleteExistingGoodDataTrainingFolder
                                            Description: This method deletes the directory made  to store the Good Data
                                                          after loading the data in the table. Once the good files are
                                                          loaded in the DB,deleting the directory ensures space optimization.
                                            Output: None
                                            On Failure: raise exception

                                             Written By: Tejas Dadhaniya
                                            Version: 1.0
                                            Revisions: None

                                                    """

        try:
            #self.gcp.create_bucket("cs_archive_good_raw")
            self.gcp.copy_all_blob(bucket_name=self.good_raw,destination_bucket_name=self.archive_good_raw)
            self.gcp.delete_blob(self.good_raw, type_="all")
            file = "GeneralLog"
            self.logger.log(file, "All files in GoodRaw bucket deleted successfully!!!")
        except Exception as e:
            file = "GeneralLog"
            self.logger.log(file, "Error while Deleting files from good raw bucket : %s" % e)
            raise e

    def deleteExistingBadDataTrainingFolder(self):

        """
                                            Method Name: deleteExistingBadDataTrainingFolder
                                            Description: This method deletes the directory made to store the bad Data.
                                            Output: None
                                            On Failure: raise exception

                                             Written By: Tejas Dadhaniya
                                            Version: 1.0
                                            Revisions: None

                                                    """

        try:
            # self.gcp.create_bucket("cs_archive_bad_raw")
            self.gcp.copy_all_blob(bucket_name=self.bad_raw, destination_bucket_name=self.archive_bad_raw)
            self.gcp.delete_blob(self.bad_raw, type_="all")
            file = "GeneralLog"
            self.logger.log(file, "All files in BadRaw bucket deleted before starting validation!!!")
        except Exception as e:
            file = "GeneralLog"
            self.logger.log(file, "Error while Deleting files from bad raw : %s" % e)
            raise e

    def notification(self, lst):
        pass

    def moveBadFilesToArchiveBad(self):


        """
                                            Method Name: moveBadFilesToArchiveBad
                                            Description: This method deletes the directory made  to store the Bad Data
                                                          after moving the data in an archive folder. We archive the bad
                                                          files to send them back to the client for invalid data issue.
                                            Output: None
                                            On Failure: OSError

                                             Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None

                                                    """
        try:
            # archiveBucket = "cs_archive_bad_raw"
            # badRaw = "cs_bad_raw"
            # self.gcp.create_bucket(self.archive_bad_raw)
            lst = self.gcp.get_allFiles(self.bad_raw)
            f = "ArchivedBadFile"

            self.notification(lst)
            self.deleteExistingBadDataTrainingFolder()
            self.logger.log(f, "moving bad files to archive and files are :: {}".format(lst))

        except Exception as e:
            file = "GeneralLog"
            self.logger.log(file, "Error while moving bad files to archive:: %s" % e)
            raise e

    def validationFileNameRaw(self,regex,LengthOfDateStampInFile,LengthOfTimeStampInFile):
        """
            Method Name: validationFileNameRaw
            Description: This function validates the name of the prediction csv file as per given name in the schema!
                         Regex pattern is used to do the validation.If name format do not match the file is moved
                         to Bad Raw Data folder else in Good raw data.
            Output: None
            On Failure: Exception

             Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None

        """
        # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        self.deleteExistingBadDataTrainingFolder()
        self.deleteExistingGoodDataTrainingFolder()
        #self.createDirectoryForGoodBadRawData()
        onlyfiles =  self.gcp.get_allFiles(self.predictionfile)
        f = "nameValidationLog"
        try:
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            self.gcp.copy_blob(self.predictionfile, filename, self.good_raw, filename)
                            self.logger.log(f, "Valid File name!! File moved to GoodRaw bucket :: %s" % filename)

                        else:
                            self.gcp.copy_blob(self.predictionfile, filename, self.bad_raw, filename)
                            self.logger.log(f, "Invalid File Name!! File moved to Bad Raw bucket :: %s" % filename)
                    else:
                        self.gcp.copy_blob(self.predictionfile, filename, self.bad_raw, filename)
                        self.logger.log(f, "Invalid File Name!! File moved to Bad Raw bucket :: %s" % filename)
                else:
                    self.gcp.copy_blob(self.predictionfile, filename, self.bad_raw, filename)
                    self.logger.log(f, "Invalid File Name!! File moved to Bad Raw bucket :: %s" % filename)

        except Exception as e:
            self.logger.log(f, "Error occured while validating FileName %s" % e)
            raise e

    def validateColumnLength(self,NumberofColumns):
        """
                    Method Name: validateColumnLength
                    Description: This function validates the number of columns in the csv files.
                                 It is should be same as given in the schema file.
                                 If not same file is not suitable for processing and thus is moved to Bad Raw Data folder.
                                 If the column number matches, file is kept in Good Raw Data for processing.
                                The csv file is missing the first column name, this function changes the missing name to "Wafer".
                    Output: None
                    On Failure: Exception

                     Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None

             """
        f = "columnValidationLog"
        try:
            self.logger.log(f,"Column Length Validation Started!!")
            self.gcp.connection()
            onlyfiles = self.gcp.get_allFiles(self.good_raw)
            for file in onlyfiles:
                csv = pd.read_csv("gs://{}/{}".format(self.good_raw,file))
                if csv.shape[1] == NumberofColumns:
                    pass
                else:
                    self.gcp.copy_blob(self.good_raw, file, self.bad_raw, file)
                    self.gcp.delete_blob(self.good_raw, blob_name=file)
                    self.logger.log(f, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
            self.logger.log(f, "Column Length Validation Completed!!")
        except OSError:
            self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
            raise OSError
        except Exception as e:
            self.logger.log(f, "Error Occured:: %s" % e)
            raise e

    def deletePredictionFile(self):

        if os.path.exists('Prediction_Output_File/Predictions.csv'):
            os.remove('Prediction_Output_File/Predictions.csv')

    def validateMissingValuesInWholeColumn(self):
        """
                                  Method Name: validateMissingValuesInWholeColumn
                                  Description: This function validates if any column in the csv file has all values missing.
                                               If all the values are missing, the file is not suitable for processing.
                                               SUch files are moved to bad raw data.
                                  Output: None
                                  On Failure: Exception

                                   Written By: iNeuron Intelligence
                                  Version: 1.0
                                  Revisions: None

                              """
        f = "missingValuesInColumn"
        try:
            self.logger.log(f, "Missing Values Validation Started!!")
            onlyfiles = self.gcp.get_allFiles(self.good_raw)
            for file in onlyfiles:
                csv = pd.read_csv("gs://{}/{}".format(self.good_raw, file))
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count += 1
                        self.gcp.copy_blob(bucket_name=self.good_raw, blob_name=file,
                                           destination_bucket_name=self.bad_raw, destination_blob_name=file)
                        self.gcp.delete_blob(bucket_name="cs_good_raw", blob_name=file)
                        self.logger.log(f, "Invalid Column for the file!! File moved to Bad Raw Folder :: %s" % file)
                        break
                #if count==0:
                 #   csv.to_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file, index=None, header=True)
        except OSError:
            self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
            raise OSError
        except Exception as e:
            self.logger.log(f, "Error Occured:: %s" % e)
            raise e

