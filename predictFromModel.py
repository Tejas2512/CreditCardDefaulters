import pandas as pd
from datetime import datetime
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
from Gcp_methods import Gcp
import yaml,os

class prediction:

    def __init__(self,path):
        self.file_object = "Prediction_Log"
        self.log_writer = logger.App_Logger(database="Prediction_Logs")
        self.pred_data_val = Prediction_Data_validation(path)

        with open(os.path.join("configfiles", "params.yaml"), "r") as f:
            self.config = yaml.safe_load(f)
        self.predictedfile = self.config["prediction"]["predictedfile"]

        self.gcp_log = "GCPlog"
        self.gcp = Gcp(file_object=self.gcp_log, logger_object=self.log_writer)
        if path is not None:
            self.pred_data_val = Prediction_Data_validation(path)


    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object,'Start of Prediction')

            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()


            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)


            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)

            # if missing values are there, replace them appropriately.
            if (is_null_present):
                data = preprocessor.impute_missing_values(data, cols_with_missing_values)  # missing value imputation

            # Proceeding with more data pre-processing steps
            X = preprocessor.scale_numerical_columns(data)


            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)
            kmeans=file_loader.load_model('KMeans')


            clusters=kmeans.predict(X)#drops the first column for cluster prediction
            X['clusters']=clusters
            clusters=X['clusters'].unique()

            print("no. of clusters::",clusters)

            predictions=[]
            for i in clusters:
                cluster_data= X[X['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                result=(model.predict(cluster_data))

            final = pd.DataFrame(list(zip(result)), columns=['Predictions'])

            now = datetime.now()
            date = now.date()
            current_time = now.strftime("%H:%M:%S")
            filename = "Predictions_{}_{}.csv".format((date), (current_time))

            storage_client = self.gcp.connection()
            #self.gcp.create_bucket("cs_predictedcsvfiles")
            bucket = storage_client.get_bucket(self.predictedfile)
            bucket.blob(filename).upload_from_string(final.to_csv(index=False), 'text/csv')

            # final= pd.DataFrame(list(zip(result)),columns=['Predictions'])
            #path="Prediction_Output_File/Predictions.csv"
            #final.to_csv("Prediction_Output_File/Predictions.csv",header=True,mode='a+') #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')

            return "https://storage.cloud.google.com/ccd_predictedcsvfiles/{}".format(filename)

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        #return path




