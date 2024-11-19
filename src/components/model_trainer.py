import os
from dataclasses import dataclass
import sys

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models


from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('model training started--')
            X_train, y_train, X_test, y_test = (train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:,-1],)
            models = { 
                'Decision Tree':DecisionTreeRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K Neighbors Classifier': KNeighborsRegressor(),
                'XGBClassifier': XGBRegressor(),
                'AdaBoost Classifier': AdaBoostRegressor(),
                'Ridge':Ridge(),
                'Lasso':Lasso()
            }
            
            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, 
                                            X_test=X_test, y_test=y_test, models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = best_model_name
            if best_model_score < 0.6:
                raise CustomException('No Best Model Found!!!')
            


            logging.info('Best model found on traning and testing datasets.')


            save_object(
                file_path = self.model_trainer_config.trained_model_file_path, obj=best_model
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return round(r2_square, 4)

        except Exception as e:
            raise CustomException(e, sys)