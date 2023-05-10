import os
import sys 
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path :str= os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transfromation_config = DataTransformationConfig()

    def get_data_transformation_object(self):

        logging.info('getting the data transformation object')
        try:
            

            numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss','hours-per-week']
            categorical_cols = ['workClass', 'marital-status', 'occupation','relationship', 'race', 'native-country','sex']

            logging.info('Pipeline has initiated')

            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy=('most_frequent'))),
                    ('ordinalencoder',OrdinalEncoder()),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor =ColumnTransformer([
                ('num_pipeline',num_pipeline,numeric_cols),
                ('cat_pipelinine',cat_pipeline,categorical_cols)
            ])

            logging.info('pipeline has compled')
            
            return preprocessor


        except Exception as e:
            logging.info('Error has occured in getting the data transformation object')
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):

        preprocessor_obj = self.get_data_transformation_object()
        logging.info('Data transformation has started')

        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = 'income'
            drop_column = [target_column,'fnlwgt','education']

            input_feature_train_df = train_df.drop(columns=drop_column,axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns =drop_column,axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing datasets.")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr =np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path = self.data_transfromation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info('preprocessor pickel file saved')


            return(
                train_arr,
                test_arr,
                self.data_transfromation_config.preprocessor_obj_file_path
            )



        except Exception as e :
            logging.info('Error has occured in data transformation')
            raise CustomException(e,sys)

