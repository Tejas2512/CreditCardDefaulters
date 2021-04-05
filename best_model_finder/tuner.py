from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics  import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
import os
import yaml

class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.rf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')
        self.lr = LogisticRegression()
        self.kn = KNeighborsClassifier()
        self.svc = SVC()
        self.gnb = GaussianNB()
        self.lrcv = LogisticRegressionCV()
        self.gbt = GradientBoostingClassifier()
        self.dBOperation = dBOperation()
        self.xgb = XGBClassifier(objective='binary:logistic',n_jobs=-1)

        with open(os.path.join("configfile","hyperparameter.yaml"),"r") as f:
            self.para = yaml.safe_load(f)

        self.verbose = self.para['base']['verbose']
        self.cv = self.para['base']['cv']
        self.n_jobs = self.para['base']['n_jobs']

    def scaler(self,x_train,x_test=None,transform=False):
        """
                 Method Name: scaler
                 Description: method use standard scaler to scaling data.
                 Output: scaling data

                 Written By: Tejas Dadhaniya
                 Version: 1.0
                 Revisions: None
                                                 """
        try:
            sc = StandardScaler()
            if not transform:
                data = sc.fit_transform(x_train)
            else:
                x = sc.fit_transform(x_train)
                data = sc.transform(x_test)
            self.logger_object.log(self.file_object, "complete feature scaling")
            return data

        except Exception as e:
            self.logger_object.log(self.file_object, "Exception :: {} occurred in scaler".format(e))

    def matrics(self,model,actual,predicted,cluster):
        """
                 Method Name: matrics
                 Description: calculate accuracy_score,precision_score,recall_score,f1_score and store in mongodb collection.
                 Output: accuracy_score

                 Written By: Tejas Dadhaniya
                 Version: 1.0
                 Revisions: None

                                                 """
        try:
            accuracy = accuracy_score(actual,predicted)
            precision = precision_score(actual,predicted)
            recall = recall_score(actual,predicted)
            f1 = f1_score(actual,predicted)
            #conn = self.dBOperation.createCollection(dataBase="Matrics",collectionName=model)
            self.now = datetime.now()
            self.date = self.now.date()
            self.current_time = self.now.strftime("%H:%M:%S")
            conn = self.dBOperation.createCollection(dataBase="Matrics", collectionName=model)
            dict_ = {"date":str(self.date),
                     "time":str(self.current_time),
                     "cluster":int(cluster),
                     "model_name":str(model),
                     "accuracy_score":float(accuracy),
                     "precision_score":float(precision),
                     "recall_score":float(recall),
                     "f1_score":float(f1)}
            conn.insert_one(dict_)
            file = "matrics"
            self.logger_object.log(file, "Matrices are evaluate successfully..!!\t\t {} ".format(dict_))
            return accuracy

        except Exception as e:
            self.logger_object.log(self.file_object,"Error :: {} occurred in matrics".format(e))

    def support_vector(self, x_train, y_train):

        """
                                            Method Name: support_vector
                                            Description: get the parameters for support vector Algorithm which give the best accuracy.
                                                         Use Hyper Parameter Tuning.
                                            Output: The model with the best parameters
                                            On Failure: Raise Exception

                                            Written By: Tejas Dadhaniya
                                            Version: 1.0
                                            Revisions: None

                                    """
        try:
            self.param_grid_svc = {
                                   "C": self.para['svm']['C'],
                                   "kernel": self.para['svm']['kernel']
                                }
            #x_train = self.scaler(x_train)

            #tuning hyperparameter
            self.grid = GridSearchCV(
                                    estimator=self.svc,
                                    param_grid=self.param_grid_svc,
                                    verbose=self.verbose,
                                    cv=self.cv
                                )

            self.grid.fit(x_train, y_train)

            #extracting the best parameters
            self.c = self.grid.best_params_['C']
            self.kernel = self.grid.best_params_['kernel']

            #creating model with best params
            self.svc = SVC(C= self.c, kernel= self.kernel)
            self.svc.fit(x_train, y_train)

            self.logger_object.log(self.file_object,
                                   'Support vector classifier best params: ' + str(
                                       self.grid.best_params_) + '. Exited the support_vector method of the Model_Finder class')

            return self.svc

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in support_vector method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Support vector Parameter tuning  failed. Exited the support_vector method of the Model_Finder class')
            raise Exception()

    def knn_classifier(self, x_train, y_train):
        """
                  Method Name: knn
                  Description: get the parameters for knn Algorithm which give the best accuracy.
                               Use Hyper Parameter Tuning.
                  Output: The model with the best parameters
                  On Failure: Raise Exception

                  Written By: Tejas Dadhaniya
                  Version: 1.0
                  Revisions: None

                                          """

        try:

            self.pram_grid_knn = {"n_neighbors": self.para['knn']['n_neighbors'],
                                  "algorithm": self.para['knn']['algorithm'],
                                  "p": self.para['knn']['p']}
            #x_train = self.scaler(x_train)

            self.grid = GridSearchCV(
                                        self.kn,
                                        param_grid=self.pram_grid_knn,
                                        verbose=self.verbose,
                                        cv=self.cv
                                    )

            self.grid.fit(x_train, y_train)

            self.n_neighbors = self.grid.best_params_["n_neighbors"]
            self.algorithm = self.grid.best_params_["algorithm"]
            self.p = self.grid.best_params_["p"]

            self.knn = KNeighborsClassifier(
                                            n_neighbors=self.n_neighbors,
                                            algorithm=self.algorithm,
                                            p=self.p
                                        )

            self.knn.fit(x_train, y_train)

            self.logger_object.log(self.file_object,
                                   'Support vector classifier best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.knn

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in knn method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'KNN Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()

    def bagging_cls(self, x_train, y_train):

        """
                 Method Name: bagging_cls
                 Description: get the parameters for bagging_cls Algorithm which give the best accuracy.
                              bagging classifier use logistic regression as a base model.
                              Use Hyper Parameter Tuning.
                 Output: The model with the best parameters
                 On Failure: Raise Exception

                 Written By: Tejas Dadhaniya
                 Version: 1.0
                 Revisions: None

                                                 """
        try:
            #self.scaler(x_train)
            self.bgc = BaggingClassifier(
                                         base_estimator=self.lr,
                                         n_estimators=self.para['BaggingClassifier']['n_estimators'],
                                         bootstrap=self.para['BaggingClassifier']['bootstrap'],
                                         oob_score=self.para['BaggingClassifier']['oob_score']
                                         )
            self.bgc.fit(x_train, y_train)
            self.logger_object.log(self.file_object,'Successfully execute bagging_cls method of Model_Finder class.')

            return self.bgc

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_bagging_cls method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Bagging_cls Parameter tuning failed. Exited the bagging_cls method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_random_forest(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {
                               "n_estimators": self.para['RandomForest']['n_estimators'],
                               "criterion": self.para['RandomForest']['criterion'],
                               "max_depth": self.para['RandomForest']['max_depth'],
                               "max_features": self.para['RandomForest']['max_features']
            }

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(
                                    estimator=self.rf,
                                    param_grid=self.param_grid,
                                    cv=self.cv,
                                    verbose=self.verbose
                                )
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(
                                            n_estimators=self.n_estimators,
                                            criterion=self.criterion,
                                            max_depth=self.max_depth,
                                            max_features=self.max_features
                                        )

            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_naive_bayes(self,train_x,train_y):
        """
        Method Name: get_best_params_for_naive_bayes
        Description: get the parameters for the Naive Bayes's Algorithm which give the best accuracy.
                     Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_naive_bayes method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {
                            "var_smoothing": self.para['NB']['var_smoothing']}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(
                                    estimator=self.gnb,
                                    param_grid=self.param_grid,
                                    cv=self.cv,
                                    verbose=self.verbose
                                )

            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.var_smoothing = self.grid.best_params_['var_smoothing']


            #creating a new model with the best parameters
            self.gnb = GaussianNB(var_smoothing=self.var_smoothing)
            # training the mew model
            self.gnb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Naive Bayes best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')

            return self.gnb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_naive_bayes method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Naive Bayes Parameter tuning  failed. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters

            self.param_grid_xgboost = {
                            "n_estimators": self.para['xgboost']['n_estimators'],
                            "max_depth": self.para['xgboost']['max_depth']
                           }

            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(
                                    XGBClassifier(objective='binary:logistic'),
                                    self.param_grid_xgboost,
                                    verbose=self.verbose,
                                    cv=self.cv,
                                    n_jobs=-1
                               )

            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(
                                    random_state=self.para['xgboost']['random_state'],
                                    max_depth=self.max_depth,
                                    n_estimators= self.n_estimators,
                                    n_jobs=self.n_jobs
                                  )
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def stacking(self,x_train, y_train):

        try:
            self.stc = StackingClassifier(
                            estimators = [( "lr",self.lrcv),( "svm" ,self.svc ),(  "knn", self.rf )],
                            final_estimator = KNeighborsClassifier(),
                            cv = self.cv
                        )

            self.stc.fit(x_train, y_train)
            self.logger_object.log(self.file_object,"Successfully execute stacking method of Model_Finder class.")
            return self.stc

        except Exception as e:
            self.logger_object.log(self.file_object,"Exception:: {} occurred in stacking method of Model_finder class.".format(e))

    def gradient_boosting(self,x_train, y_train):
        try:
            param_grid_gbt = {
                            "n_estimators": self.para['GradientBoosting']['n_estimators'],
                            "learning_rate": self.para['GradientBoosting']['learning_rate'],
                            "min_samples_split": self.para['GradientBoosting']['min_samples_split']
                      }
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(
                                    self.gbt,
                                    param_grid = param_grid_gbt,
                                    verbose=self.verbose,
                                    cv=self.cv,
                                    n_jobs=self.n_jobs
                                )
            # finding the best parameters
            self.grid.fit(x_train, y_train)

            # extracting the best parameters
            n_estimators = self.grid.best_params_['n_estimators']
            learning_rate = self.grid.best_params_['learning_rate']
            min_samples_split = self.grid.best_params_['min_samples_split']

            # creating a new model with the best parameters
            self.gbt = XGBClassifier(
                                     learning_rate=learning_rate,
                                     min_samples_split=min_samples_split,
                                     n_estimators=n_estimators,
                                     n_jobs=self.n_jobs
                                )
            # training the mew model
            self.gbt.fit(x_train, y_train)

            self.logger_object.log(self.file_object,"Successfully execute gradient_boosting method of Model_Finder class.")
            return self.gbt

        except Exception as e:
            self.logger_object.log(self.file_object,"Exception:: {} occurred in gradient_boosting method of Model_finder class.".format(e))

    def get_best_model(self, x_train, y_train, x_test, y_test,cluster):
        """
                Method Name: get_best_model
                Description: Find out the Model which has the best accuracy score.
                Output: The best model name and the model object
                On Failure: Raise Exception

                Written By: Tejas Dadhaniya
                Version: 1.0
                Revisions: None

                                        """


        self.logger_object.log(self.file_object,'Entered the get_best_model method of the Model_Finder class')

        try:

            #x_test_sc = self.scaler(x_train, x_test=x_test, transform=True)

            self.xgboostc = self.get_best_params_for_xgboost(x_train, y_train)
            self.xgb_pred = self.xgboostc.predict(x_test)
            self.xgb_accuracy_score = self.matrics("XGboost",y_test,self.xgb_pred,cluster=cluster)

            #self.xgb_accuracy_score = accuracy_score(y_test, self.xgb_pred)

            self.rfc = self.get_best_params_for_random_forest(x_train, y_train)
            self.rf_pred = self.rfc.predict(x_test)
            self.rf_accuracy_score = self.matrics("Randomforest",y_test,self.rf_pred,cluster=cluster)

            self.bgc = self.bagging_cls(x_train, y_train)
            self.bgc_pred = self.bgc.predict(x_test)
            self.bgc_accuracy_score = self.matrics("BaggingClassifier",y_test,self.bgc_pred,cluster=cluster)

            self.svmc = self.support_vector(x_train, y_train)
            self.svm_pred = self.svmc.predict(x_test)
            self.svm_accuracy_score = self.matrics("SupportVector",y_test,self.svm_pred,cluster=cluster)

            self.knnc = self.knn_classifier(x_train, y_train)
            self.knn_pred = self.knnc.predict(x_test)
            self.knn_accuracy_score = self.matrics("KNearestNeighbour",y_test,self.knn_pred,cluster=cluster)

            self.GNBC = self.get_best_params_for_naive_bayes(x_train,y_train)
            self.GNB_prediction = self.GNBC.predict(x_test)
            self.GNB_accuracy_score = self.matrics("Gaussian NB",y_test,self.GNB_prediction,cluster=cluster)

            self.gbtc = self.gradient_boosting(x_train,y_train)
            self.gbt_prediction = self.gbtc.predict(x_test)
            self.gbt_accuracy_score = self.matrics("GradientBoosting",y_test,self.gbt_prediction,cluster=cluster)

            self.stcc = self.stacking(x_train, y_train)
            self.stc_prediction = self.stcc.predict(x_test)
            self.stc_accuracy_score = self.matrics("stacking", y_test, self.gbt_prediction,cluster=cluster)

            name = ["Xgboost", "Randomforest", "BaggingClassifier", "KNN","Gaussian NB","SVC","GradientBoosting","stacking"]
            obj = [self.xgboostc, self.rfc, self.bgc, self.knnc,self.GNBC,self.svmc,self.gbtc,self.stcc]
            score = [self.xgb_accuracy_score, self.rf_accuracy_score, self.bgc_accuracy_score
                     , self.knn_accuracy_score,self.GNB_accuracy_score,self.svm_accuracy_score,self.gbt_accuracy_score,self.stc_accuracy_score]

            model = name[np.array(score).argmax()]
            model_obj = obj[np.array(score).argmax()]

            self.logger_object.log(self.file_object, "Best model :: {}".format(model))
            return  model, model_obj

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

