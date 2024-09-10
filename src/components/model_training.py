# import os
# import sys
# from dataclasses import dataclass
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# from sklearn.metrics import accuracy_score,r2_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import Ridge
# from src.exception import CustomException
# from src.logger import logging
# import warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning)

# from src.utils import save_obj

# @dataclass
# class ModelTrainingConfig:
#     model_training_path = os.path.join('artifacts','model.pkl')

# class ModelTraining:
#     def __init__(self):
#         self.ModelTrainingConfig = ModelTrainingConfig()
    
#     def model(self,prob,X_train,y_train,X_test,y_test):
#         models = {}
#         max_model = {}
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
        
#         regression = {
#                 "ridge_regression": {
#                     "algorithm": Ridge(),
#                     "scale": 1,
#                     "param_grid": {
#                         'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
#                         'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
#                         'max_iter': [100, 200, 500, 1000, 3000, 5000, 10000],
#                         'tol': [1e-4, 1e-3, 1e-2],
#                         'fit_intercept': [True, False],
#                     }
#                 },
#                 "lasso_regression": {
#                     "algorithm": Lasso(),
#                     "scale": 1,
#                     "param_grid": {
#                         'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
#                         'max_iter': [100, 200, 500, 1000, 3000, 5000, 10000],
#                         'tol': [1e-4, 1e-3, 1e-2],
#                         'fit_intercept': [True, False],
#                         'selection': ['cyclic', 'random'],
#                     }
#                 },
#                 "elasticnet_regression": {
#                     "algorithm": ElasticNet(),
#                     "scale": 1,
#                     "param_grid": {
#                         'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
#                         'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
#                         'max_iter': [100, 200, 500, 1000, 3000, 5000, 10000],
#                         'tol': [1e-4, 1e-3, 1e-2],
#                         'fit_intercept': [True, False],
#                     }
#                 },
#                 "random_forest_regression": {
#                     "algorithm": RandomForestRegressor(),
#                     "scale": 0,  # Feature scaling might not be required
#                     "param_grid": {
#                         'n_estimators': [10, 50, 100, 200, 500],
#                         'max_depth': [None, 10, 20, 30, 50, 100],
#                         'min_samples_split': [2, 5, 10],
#                         'min_samples_leaf': [1, 2, 4],
#                         'bootstrap': [True, False],
#                     }
#                 },
#                 "svr": {
#                     "algorithm": SVR(),
#                     "scale": 1,
#                     "param_grid": {
#                         'C': [0.001, 0.01, 0.1, 1, 10, 100],
#                         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#                         'degree': [2, 3, 4],
#                         'gamma': ['scale', 'auto'],
#                         'epsilon': [0.1, 0.2, 0.5],
#                     }
#                 },
#                 "gradient_boosting_regression": {
#                     "algorithm": GradientBoostingRegressor(),
#                     "scale": 0,  # Feature scaling might not be required
#                     "param_grid": {
#                         'n_estimators': [100, 200, 500],
#                         'learning_rate': [0.001, 0.01, 0.1, 0.2],
#                         'max_depth': [3, 5, 7, 10],
#                         'min_samples_split': [2, 5, 10],
#                         'min_samples_leaf': [1, 2, 4],
#                         'subsample': [0.5, 0.7, 1.0],
#                     }
#                 }
#             }
        

#         classifier = {
#                 "logistic_regression": {
#                     "algorithm": LogisticRegression(),
#                     "scale": 1,
#                     "param_grid": {
#                         'C': [0.001, 0.01, 0.1, 1, 10, 100],
#                         'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
#                         'max_iter': [1000, 3000, 5000, 10000],
#                         'class_weight': [None, 'balanced'],
#                         'tol': [1e-4, 1e-3, 1e-2],
#                     }
#                 },
#                 "svc": {
#                     "algorithm": SVC(),
#                     "scale": 1,
#                     "param_grid": {
#                         'C': [0.1, 1, 10, 100, 1000],
#                         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#                         'gamma': ['scale', 'auto'],
#                         'degree': [2, 3, 4, 5],
#                         'class_weight': [None, 'balanced'],
#                     }
#                 },
#                 "random_forest_classifier": {
#                     "algorithm": RandomForestClassifier(),
#                     "scale": 0,
#                     "param_grid": {
#                         'n_estimators': [50, 100, 200],
#                         'max_depth': [None, 10, 20, 30, 50],
#                         'min_samples_split': [2, 5, 10],
#                         'min_samples_leaf': [1, 2, 4],
#                         'bootstrap': [True, False],
#                         'criterion': ['gini', 'entropy', 'log_loss'],
#                     }
#                 },
#                 "knn": {
#                     "algorithm": KNeighborsClassifier(),
#                     "scale": 1,
#                     "param_grid": {
#                         'n_neighbors': [2, 3, 5, 7, 9, 11],
#                         'weights': ['uniform', 'distance'],
#                         'metric': ['euclidean', 'manhattan', 'minkowski'],
#                     }
#                 }
#             }

#         if prob == 0:
#             algorithm = regression
#             metrics = 'r2'
#         else:
#             algorithm = classifier
#             metrics = 'accuracy'
#         max_score = 0
#         for model_name in algorithm.keys():
#             grid_search = GridSearchCV(algorithm[model_name]["algorithm"], algorithm[model_name]["param_grid"], cv=5, scoring=metrics, n_jobs=-1)
            
#             if algorithm[model_name]["scale"] == 1:
#                 grid_search.fit(X_train_scaled, y_train)
#             else:
#                 grid_search.fit(X_train, y_train)
            
#             # Best model and parameters
#             best_model = grid_search.best_estimator_
#             best_params = grid_search.best_params_
            
#             if grid_search.best_score_ > max_score:
#                 max_score = grid_search.best_score_
#                 max_model['metrics'] = metrics
#                 max_model["max_score"] = max_score
#                 max_model["max_algo"] = model_name
#                 max_model["best_model"] = best_model
#                 max_model["best_params"] = best_params
            
#             models[model_name] = {
#                 "metrics":metrics,
#                 "model_name": model_name,
#                 "model_details": best_model,
#                 "model_params": best_params,
#                 metrics: grid_search.best_score_
#             }
        
#         best_model = max_model['best_model']
#         if algorithm[max_model["max_algo"]]["scale"] == 1:
#             best_model.fit(X_train_scaled, y_train)
#             y_pred = best_model.predict(X_test_scaled)
#         else:
#             best_model.fit(X_train, y_train)
#             y_pred = best_model.predict(X_test)
        
#         testing_score = 0
#         if max_model["metrics"] == 'r2':
#             testing_score = r2_score(y_test, y_pred)
#         else:
#             testing_score = accuracy_score(y_test, y_pred)
        
#         max_model["testing_score"] = testing_score

#         # return [models["best_model"],models["max_algo"],models['max_score'],testing_score,metrics]
#         return models,max_model


#     def initiate_model_training(self,train,test,prob):
#         try:

#             X_train,y_train,X_test,y_test = (
#                 train.iloc[:,:-1],
#                 train.iloc[:,-1],
#                 test.iloc[:,:-1],
#                 test.iloc[:,-1]
#             )

#             models = self.model(prob,X_train,y_train,X_test,y_test)
#             return models



#         except Exception as e:
#             raise CustomException(e,sys)



import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import parallel_backend
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Custom exception and logger imports
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class ModelTrainingConfig:
    model_training_path = os.path.join('artifacts', 'model.pkl')

class ModelTraining:
    def __init__(self):
        self.ModelTrainingConfig = ModelTrainingConfig()

    def model(self, prob, X_train, y_train, X_test, y_test):
        models = {}
        max_model = {}
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Regression Algorithms
        regression = {
            "ridge_regression": {
                "algorithm": Ridge(),
                "scale": 1,
                "param_grid": {
                    'alpha': [0.001, 0.01, 0.1, 1, 10],
                    'solver': ['auto', 'svd', 'lsqr'],
                    'max_iter': [100, 500, 1000],
                    'tol': [1e-4, 1e-3],
                    'fit_intercept': [True, False],
                }
            },
            "lasso_regression": {
                "algorithm": Lasso(),
                "scale": 1,
                "param_grid": {
                    'alpha': [0.001, 0.01, 0.1, 1, 10],
                    'max_iter': [100, 500, 1000],
                    'tol': [1e-4, 1e-3],
                    'fit_intercept': [True, False],
                    'selection': ['cyclic', 'random'],
                }
            },
            "elasticnet_regression": {
                "algorithm": ElasticNet(),
                "scale": 1,
                "param_grid": {
                    'alpha': [0.001, 0.01, 0.1, 1, 10],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'max_iter': [100, 500, 1000],
                    'tol': [1e-4, 1e-3],
                    'fit_intercept': [True, False],
                }
            },
            "linear_regression": {
                "algorithm": LinearRegression(),
                "scale": 1,
                "param_grid": {
                    'fit_intercept': [True, False],
                    'normalize': [True, False],
                    'n_jobs': [-1]
                }
            },
            "random_forest_regression": {
                "algorithm": RandomForestRegressor(),
                "scale": 0,
                "param_grid": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 30],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'bootstrap': [True, False],
                }
            },
            "svr": {
                "algorithm": SVR(),
                "scale": 1,
                "param_grid": {
                    'C': [0.01, 0.1, 1, 10],
                    'kernel': ['linear', 'poly', 'rbf'],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.1, 0.2, 0.5],
                }
            },
            "gradient_boosting_regression": {
                "algorithm": GradientBoostingRegressor(),
                "scale": 0,
                "param_grid": {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'subsample': [0.5, 0.7, 1.0],
                }
            }
        }

        # Classifier Algorithms
        classifier = {
            "logistic_regression": {
                "algorithm": LogisticRegression(),
                "scale": 1,
                "param_grid": {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['lbfgs', 'sag'],
                    'max_iter': [1000, 3000],
                    'class_weight': [None, 'balanced'],
                    'tol': [1e-4, 1e-3],
                }
            },
            "random_forest_classifier": {
                "algorithm": RandomForestClassifier(),
                "scale": 0,
                "param_grid": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 30],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'bootstrap': [True, False],
                    'criterion': ['gini', 'entropy'],
                }
            },
            "svc": {
                "algorithm": SVC(),
                "scale": 1,
                "param_grid": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'poly', 'rbf'],
                    'gamma': ['scale', 'auto'],
                    'degree': [2, 3, 4],
                    'class_weight': [None, 'balanced'],
                }
            },
            "knn": {
                "algorithm": KNeighborsClassifier(),
                "scale": 1,
                "param_grid": {
                    'n_neighbors': [2, 3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan'],
                }
            },
            "decision_tree_classifier": {
                "algorithm": DecisionTreeClassifier(),
                "scale": 0,
                "param_grid": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'class_weight': [None, 'balanced'],
                }
            },
            "gradient_boosting_classifier": {
                "algorithm": GradientBoostingRegressor(),
                "scale": 0,
                "param_grid": {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'subsample': [0.5, 0.7, 1.0],
                }
            }
        }

        if prob == 0:
            algorithm = regression
            metrics = 'r2'
        else:
            algorithm = classifier
            metrics = 'accuracy'

        max_score = 0
        with parallel_backend('loky', n_jobs=-1):  # Parallelism across all models
            for model_name in algorithm.keys():
                grid_search = GridSearchCV(
                    algorithm[model_name]["algorithm"], 
                    algorithm[model_name]["param_grid"], 
                    cv=5, 
                    scoring=metrics, 
                    n_jobs=-1  # Parallelism in GridSearchCV
                )
                
                if algorithm[model_name]["scale"] == 1:
                    grid_search.fit(X_train_scaled, y_train)
                else:
                    grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                if grid_search.best_score_ > max_score:
                    max_score = grid_search.best_score_
                    max_model['metrics'] = metrics
                    max_model["max_score"] = max_score
                    max_model["max_algo"] = model_name
                    max_model["best_model"] = best_model
                    max_model["best_params"] = best_params

                models[model_name] = {
                    "metrics": metrics,
                    "model_name": model_name,
                    "model_details": best_model,
                    "model_params": best_params,
                    metrics: grid_search.best_score_
                }

        # Final Model Prediction and Testing
        best_model = max_model['best_model']
        if algorithm[max_model["max_algo"]]["scale"] == 1:
            best_model.fit(X_train_scaled, y_train)
            y_pred = best_model.predict(X_test_scaled)
        else:
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

        testing_score = r2_score(y_test, y_pred) if max_model["metrics"] == 'r2' else accuracy_score(y_test, y_pred)
        max_model["testing_score"] = testing_score

        return models, max_model

    def initiate_model_training(self, train, test, prob):
        try:
            X_train, y_train, X_test, y_test = (
                train.iloc[:, :-1],
                train.iloc[:, -1],
                test.iloc[:, :-1],
                test.iloc[:, -1]
            )

            models = self.model(prob, X_train, y_train, X_test, y_test)
            return models

        except Exception as e:
            raise CustomException(e, sys)
