from Gcp_methods import Gcp
from application_logging.logger import App_Logger

class Template:

    """
        This class create all require directory through out the project .

        Written By: Tejas Dadhaniya
        Version: 1.0
        Revisions: None

        """

    def __init__(self):
     self.file_object = "templates"
     self.logger_object = App_Logger()
     self.gcp = Gcp(logger_object=self.logger_object,file_object=self.file_object)
     self.dirs = [
         "ccd_archive_bad_raw",
         "ccd_archive_good_raw",
         "ccd_bad_raw",
         "ccd_good_raw",
         "ccd_metadata",
         "ccd_modelsforprediction",
         "ccd_myrecycle_bin",
         "ccd_predictedcsvfiles",
         "ccd_prediction_archive_bad_raw",
         "ccd_prediction_archive_good_raw",
         "ccd_prediction_bad_raw",
         "ccd_prediction_good_raw",
         "ccd_predictionfilefromdb",
         "ccd_predictionfiles",
         "ccd_trainfiles",
         "ccd_trainingfilefromdb"
     ]

    def directory(self):

     try:
         self.logger_object.log(file_object=self.file_object,
                                log_message="Enter in directory method of template class.")

         for dir_ in self.dirs:
            self.gcp.create_bucket(dir_)

         self.logger_object.log(file_object=self.file_object,
                                log_message="Successfully created all directory..")

     except Exception as e:
         self.logger_object.log(file_object=self.file_object,log_message="Exception :: {} occurred template class.".format(e))


