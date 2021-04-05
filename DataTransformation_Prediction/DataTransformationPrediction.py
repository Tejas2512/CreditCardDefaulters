from Gcp_methods import Gcp
import os
import pandas
from application_logging.logger import App_Logger
import yaml

class dataTransformPredict:

     """
                  This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.

                  Written By: iNeuron Intelligence
                  Version: 1.0
                  Revisions: None

                  """

     def __init__(self):
          with open(os.path.join("configfiles","params.yaml"),"r") as f:
               self.config = yaml.safe_load(f)
          self.goodDataPath = self.config["prediction"]["good_raw"]
          self.gcp_log = "GCPlog"
          self.goodDataPath = "Prediction_Raw_Files_Validated/Good_Raw"
          self.logger = App_Logger(database="Prediction_Logs")
          self.gcp = Gcp(file_object=self.gcp_log, logger_object=self.logger)

     def replaceMissingWithNull(self):

          """
                                  Method Name: replaceMissingWithNull
                                  Description: This method replaces the missing values in columns with "NULL" to
                                               store in the table. We are using substring in the first column to
                                               keep only "Integer" data for ease up the loading.
                                               This column is anyways going to be removed during prediction.

                                   Written By: iNeuron Intelligence
                                  Version: 1.0
                                  Revisions: None

                                          """
          log_file ="dataTransformLog"
          try:
               onlyfiles = self.gcp.get_allFiles(self.goodDataPath)
               for file in onlyfiles:
                    storage_client = self.gcp.connection()
                    csv = pandas.read_csv("gs://{}/{}".format(self.goodDataPath,file))
                    csv.fillna('NULL',inplace=True)
                    bucket = storage_client.get_bucket(self.goodDataPath)
                    bucket.blob(file).upload_from_string(csv.to_csv(index=False), 'text/csv')
                    self.logger.log(log_file, " %s: data transform successful!!" % file)
          except Exception as e:
               self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
