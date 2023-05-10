import os 
import sys 
import pandas as pd 
import numpy as np 
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging 
from src.utils  import save_object
from sklearn.linear_model import LogisticRegression
from src.utils import evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self,train_array,test_array):
        try:

            logging.info('Splitting the dependent and independent variable from rain and test data')
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model = LogisticRegression()
            model.fit(x_train,y_train)

            model_report:dict = evaluate_model(x_train,y_train,x_test,y_test,model)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = model
            )

            return model_report
        
        except Exception as e:
            logging.info('Error has occured in initiating model traininng')
            raise CustomException(e,sys)
    

        

