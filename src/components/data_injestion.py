import pandas as pd
import sys
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
# from src.components.data_transformation import DataTransformation
# from src.components.model_training import ModelTraining

@dataclass
class DataInjestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')

class DataInjestion:
    def __init__(self,input_data) -> None:
        self.DataInjestionConfig = DataInjestionConfig()
        self.input_data = input_data

    def initiate_data_injestion(self):
        try:
            df = self.input_data
            df.head()
            logging.info("Entered into data injestion")

            os.makedirs(os.path.dirname(self.DataInjestionConfig.raw_data_path),exist_ok=True)
            logging.info("artifacts directory created")

            df.to_csv(self.DataInjestionConfig.raw_data_path,index=False,header=True)
            logging.info("raw data saved into the artifacts folder")

            train,test = train_test_split(df,test_size=0.3,random_state=42)
            logging.info("dataset splitted into train and test")

            train.to_csv(self.DataInjestionConfig.train_data_path,index=False,header=True)
            logging.info("training data saved to artifacts")
            
            test.to_csv(self.DataInjestionConfig.test_data_path,index=False,header=True)
            logging.info("test data saved to artifacts")

            return (
                self.DataInjestionConfig.train_data_path,
                self.DataInjestionConfig.test_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)

# if __name__ == '__main__':
#     di_obj = DataInjestion(input)
#     train_data_path, train_data_path = di_obj.initiate_data_injestion()
    # dt = DataTransformation()
    # train_data_preprocessed,test_data_preprocessed,_ = dt.initiate_data_tranformation(train_data_path, train_data_path)
    # mt = ModelTraining()
    # mt.initiate_model_training(train_data_preprocessed,test_data_preprocessed)