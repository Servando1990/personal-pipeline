
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgbm
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score




class ModelFinder:


    def nested_cv(self, features:DataFrame): # TODO Refactor

        X, y = features.iloc[:, 1:].values, features.iloc[:, 0].values

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1)
        model1 = RandomForestRegressor()
        model2 = lgbm.LGBMRegressor()
        model3 = xgb.XGBRegressor()

        print('Initializating', model1, model2, model3)

        parameter_grid1 = [ { # 1st param grid, corresponding to RandomForestRegressor
                                'max_depth': [3, None],
                                'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
                                'max_features' : [50,100,150,200]}]

        parameter_grid2 = [ { # 2rd param grid, corresponding to LGBMRegressor
                                'learning_rate': [0.05],
                                'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
                                'reg_alpha' : (1,1.2),
                                'reg_lambda' : (1,1.2,1.4)
                        }]

        parameter_grid3 = [ { # 3rd param grid, corresponding to XGBRegressor
                                'learning_rate': [0.05],
                                'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
                                'reg_alpha' : (1,1.2),
                                'colsample_bytree': np.linspace(0.3, 0.5),
                                'reg_lambda' : (1,1.2,1.4)
                        }]
        
        # Setting up RandomizedSearchCV objects
        grid_cvs = {}
        inner_cv = KFold(n_splits=2, shuffle=True, random_state=1)

        for param_grid, model, name in zip((parameter_grid1,
                                parameter_grid2,
                                parameter_grid3),
                                (model1, model2, model3),
                                ('RandomForestRegressor', 'LGBM', 'XGBOOST')):


            gcv = RandomizedSearchCV(estimator=model,
                        param_distributions=param_grid,
                        scoring='neg_mean_squared_error',
                        n_jobs=1, 
                        cv=inner_cv,
                        verbose=0,
                        refit=True)
            grid_cvs[name] = gcv

        outer_cv = KFold(n_splits=3, shuffle=True, random_state=1)

        print('Initializing Nested Cross Validation for Model Selection')

        for name, gs_est in sorted(grid_cvs.items()):
            nested_score = cross_val_score(gs_est, 
                                    X=X_train, 
                                    y=y_train, 
                                    cv=outer_cv,
                                    scoring='neg_mean_squared_error',
                                    n_jobs=1) 
            
            

        return print(f'{name:<7} | outer MSR {nested_score.mean()} ')


