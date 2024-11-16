# importing libraries
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
            This function is responsible for data Transformation based on different data types.
        """
        try:
            numerical_features = ['writing score', 'reading score']
            cat_features = [
                'gender',
                'race/ethnicity',
                'parental level of education', 
                'lunch',
                'test preparation course',
            ]
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())

                ]
            )
            logging.info('Numeric columns transformation completed')
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one hot encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info(f'Categorical Columns: {cat_features}')
            logging.info(f'Numeric Columns: {numerical_features}')
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )
            logging.info('Columns Transformation Completed') 
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
            
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('read train and test data completed')

            logging.info('obtaining preprocessor object')
        
            preprocessing_object = self.get_data_transformer_object()
            target_columns_name = 'math score'

            numerical_features = ['writing score', 'reading score']

            input_feature_train_df = train_df.drop(columns=[target_columns_name], axis=1)
            target_feature_train_df = train_df[target_columns_name]

            input_feature_test_df = test_df.drop(columns=[target_columns_name], axis=1)
            target_feature_test_df = test_df[target_columns_name]
            logging.info(f'Appling preprocessing object on training dataframe and test dataframe.')

            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.fit_transform(input_feature_test_df) 

            train_arr = np.c_[input_feature_train_arr, np.array(input_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('scaved preprocessing object.')
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_object
            )

            return (
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)