from sklearn.preprocessing import OneHotEncoder,StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj, load_object
import os
import sys
import pandas as pd
import numpy as np

@dataclass
class DataTransformationConfig:
    data_transformation_config_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self,df,train_path,test_path,target):
        self.DataTransformationConfig = DataTransformationConfig()
        self.df = df
        self.train_data_path = train_path
        self.test_data_path = test_path
        self.target = target
    
    def get_data_transformation_obj(self):
        try:
            num_cols = []
            cat_cols = []
            for cols in self.df.columns:
                if self.df[cols].dtype == "int64" or self.df[cols].dtype == "float64":
                    num_cols.append(cols)
                elif self.df[cols].dtype == "object":
                    cat_cols.append(cols)

            if self.target in num_cols:
                num_cols.remove(self.target)
            elif self.target in cat_cols:
                cat_cols.remove(self.target)

            num_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                # ('standardscaler',StandardScaler())
            ])        

            cat_pipeline = Pipeline(steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder',OneHotEncoder(handle_unknown='ignore')),
                    # ('standardscaler',StandardScaler(with_mean=False))
                ])
            

            preprocessor_obj = ColumnTransformer(transformers=[
                ('num_pipeline',num_pipeline,num_cols),
                ('cat_pipeline',cat_pipeline,cat_cols),
            ])

            return preprocessor_obj
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def feature_selection(self,X,y,percent=0.5):
        df = pd.DataFrame(X)
        df['target'] = y
        corr = df.corr()['target'].drop('target')
        selected_features_ = corr.abs().sort_values(ascending=False)
        n_features = int(len(selected_features_) * percent)
        top_features = selected_features_.index[:n_features]
        return X.loc[:, top_features]

    def initiate_data_tranformation(self):
        try:
            logging.info("Entered into initiate_data_tranformation()")

            train_data = pd.read_csv(self.train_data_path)
            test_data = pd.read_csv(self.test_data_path)
            logging.info("read the train and test files")

            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformation_obj()
            target_column_name = self.target

            X_train = train_data.drop(columns=[target_column_name])
            X_test = test_data.drop(columns=[target_column_name])
            
            y_train = train_data[target_column_name]
            y_test = test_data[target_column_name]
            logging.info("initialized xtrain, xtest, ytrain, ytest")

            X_train_preprocessed = preprocessor_obj.fit_transform(X_train)
            X_test_preprocessed  = preprocessor_obj.transform(X_test)
            logging.info("fitted the preprocessing object on to the data")
 
            X_train_selected = self.feature_selection(pd.DataFrame(X_train_preprocessed.toarray()), y_train)
            X_test_preprocessed = pd.DataFrame(X_test_preprocessed.toarray())
            X_test_selected = X_test_preprocessed.loc[:, X_train_selected.columns]

            train_data_preprocessed = pd.concat([X_train_selected, y_train],axis=1)
            test_data_preprocessed = pd.concat([X_test_selected, y_test],axis=1)

            save_obj(file_path=self.DataTransformationConfig.data_transformation_config_path,obj=preprocessor_obj)
            logging.info("saved the preprocessor object in the directory")

            return (
                train_data_preprocessed,
                test_data_preprocessed,
                self.DataTransformationConfig.data_transformation_config_path
            )

        except Exception as e:
                raise CustomException(e,sys)
