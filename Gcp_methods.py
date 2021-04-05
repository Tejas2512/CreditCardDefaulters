from google.cloud import storage
from datetime import datetime
import re

class Gcp:

    """
                This class shall be used to save the model after training
                and load the saved model for prediction from GCP bucket.

                Written By: Tejas Dadhaniya
                Version: 1.0
                Revisions: None

            """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.model_directory='models/'
        self.path = "gcp.json"



    def connection(self):

        try:
            #PATH = os.path.join(os.getcwd(),"gcp.json")
            #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = PATH
            # storage_client = storage.Client(PATH)
            storage_client = storage.Client.from_service_account_json(self.path)
            return storage_client
        except Exception as e:
            self.logger_object.log(self.file_object, "Error: {} while making Connection".format(e))

    def delete_blob(self,bucket_name, blob_name=None, type_ = "deleteSingleFile"):

        try:
            storage_client = self.connection()
            bucket = storage_client.bucket(bucket_name)
            if type_ == "deleteSingleFile":
                blob = bucket.blob(blob_name)
                blob.delete()
                self.logger_object.log(self.file_object,"Blob {} deleted.".format(blob_name))

            else:
                onlyfiles = [x.name for x in list(bucket.list_blobs(prefix=""))]
                [bucket.blob(file).delete() for file in onlyfiles]
                self.logger_object.log(self.file_object, "All Blobs in {} bucket is deleted.".format(bucket_name))

        except Exception as e:
            self.logger_object.log(self.file_object,"Error {} accurred while deleting blob".format(e))

    def delete_bucket(self,bucket_name):

        try:
            storage_client = self.connection()
            bucket = storage_client.get_bucket(bucket_name)
            bucket.delete()
            self.logger_object.log(self.file_object, "Bucket {} deleted".format(bucket.name))

        except Exception as e:
            self.logger_object.log(self.file_object, "Error {} accurred while deleting bucket".format(e))

    def upload_blob(self,bucket_name, source_file_name, destination_blob_name):

        try:
            storage_client = self.connection()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_name)
            self.logger_object.log(self.file_object,  "File {} uploaded to {}.".format(source_file_name, destination_blob_name))

        except Exception as e:
            self.logger_object.log(self.file_object, "Error {} accurred while uploading bucket".format(e))

    def download_blob(self,bucket_name, source_blob_name, destination_file_name):

        try:
            storage_client = self.connection()
            bucket = storage_client.bucket(bucket_name)
            # Construct a client side representation of a blob.
            # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
            # any content from Google Cloud Storage. As we don't need additional data,
            # using `Bucket.blob` is preferred here.
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)
            self.logger_object.log(self.file_object,"Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))

        except Exception as e:
            self.logger_object.log(self.file_object, "Error {} accurred while downloading bucket".format(e))

    def copy_blob(self,bucket_name, blob_name, destination_bucket_name, destination_blob_name):

        try:
            storage_client = self.connection()
            source_bucket = storage_client.bucket(bucket_name)
            source_blob = source_bucket.blob(blob_name)
            destination_bucket = storage_client.bucket(destination_bucket_name)
            blob_copy = source_bucket.copy_blob(source_blob, destination_bucket, destination_blob_name)
            self.logger_object.log(self.file_object,"Blob {} in bucket {} copied to blob {} in bucket {}.".format(source_blob.name,source_bucket.name,blob_copy.name,destination_bucket.name))

        except Exception as e:
            self.logger_object.log(self.file_object, "Error {} accurred while downloading bucket blob {} in bucket {} copied to blob {} in bucket {}".format(e,source_blob.name,source_bucket.name,blob_copy.name,destination_bucket.name))

    def copy_all_blob(self,bucket_name, destination_bucket_name):

        try:
            # self.now = datetime.now()
            # self.date = self.now.date()
            # self.current_time = self.now.strftime("%H:%M:%S")
            storage_client = self.connection()
            source_bucket = storage_client.bucket(bucket_name)
            destination_bucket = storage_client.bucket(destination_bucket_name)
            files = self.get_allFiles(bucket_name=bucket_name)
            for f in files:
                source_blob = source_bucket.blob(f)
                #file_name = f.split(".")[0]+"_{}_{}".format(str(self.date),str(self.current_time))+"."+f.split(".")[1]
                #file_name = f+str(self.date)+str(self.current_time)
                blob_copy = source_bucket.copy_blob(source_blob, destination_bucket, f)
                self.logger_object.log(self.file_object,"Blob {} in bucket {} copied to blob {} in bucket {}.".format(source_blob.name,source_bucket.name,blob_copy.name,destination_bucket.name))

        except Exception as e:
            self.logger_object.log(self.file_object, "Error {} accurred while moving all blobs from {} bucket to {} bucket ".format(e,source_bucket.name,destination_bucket.name))


    def get_allFiles(self,bucket_name):

        try:
            storage_client = self.connection()
            bucket = storage_client.get_bucket(bucket_name)
            onlyfiles = [x.name for x in list(bucket.list_blobs(prefix=""))]
            self.logger_object.log(self.file_object, "Successfully get all files name from {} bucket".format(bucket_name))
            return onlyfiles

        except Exception as e:
            self.logger_object.log(self.file_object, "Error {} Occurred while getting blobs name from {} bucket".format(e,bucket_name))

    def create_bucket(self,bucket_name):

        try:
            storage_client = self.connection()
            buckets = [i.name for i in storage_client.list_buckets()]
            bucket = storage_client.bucket(bucket_name)
            bucket.storage_class = "COLDLINE"
            if bucket_name not in buckets:

                storage_client.create_bucket(bucket, location="us")
                self.logger_object.log(self.file_object,"Successfully create {} bucket".format(bucket_name))
            else:
                self.logger_object.log(self.file_object,"{} bucket already exists".format(bucket_name))

        except Exception as e:
            self.logger_object.log(self.file_object,"Error: {} Occurred while creating {} bucket".format(e,bucket_name))
            raise e