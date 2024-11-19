import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        logging.info('model report creation started')
        report = {}
        for i in range(len(list(models.keys()))):
            
            model=(list(models.values()))[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_predict = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_predict)
            report[model] = test_model_score
        logging.info('Models report creation completed')
        return report
    
    except Exception as e:
        raise CustomException(e, sys)


