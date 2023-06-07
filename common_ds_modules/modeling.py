from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.neighbors import KNeighborsRegressor

MAX_MODELS = 100

def evaluate(train, y_train, model):
    scores = cross_val_score(model, train, y_train, scoring='neg_root_mean_squared_error', cv=5)
    print(f'mean RMSLE = {-np.mean(scores)}, std RMSLE = {np.std(scores)}')

def get_random_forest_regressor_results(train, y_train, param_grid, test=None, y_test=None):
    if len(param_grid) > 0:
        model = RandomizedSearchCV(RandomForestRegressor(), param_grid, scoring='neg_root_mean_squared_error',
                                   n_iter=MAX_MODELS)
    else:
        print(f'Training default model')
    model = model.fit(train, y_train)
    return model

def get_decision_tree_model(train, y_train, param_grid, test=None, y_test=None):
    model = RandomizedSearchCV(DecisionTreeRegressor(), param_grid, scoring='neg_root_mean_squared_error',
                               n_iter=MAX_MODELS)
    model = model.fit(train, y_train)
    return model

def get_xgb_regressor_model(train, y_train, param_grid, test=None, y_test=None):
    model = RandomizedSearchCV(XGBRegressor(), param_grid, scoring='neg_root_mean_squared_error',
                               n_iter=MAX_MODELS)
    model = model.fit(train, y_train)
    return model

def get_linear_regressor_model(train, y_train, test=None, y_test=None):
    model = LinearRegression().fit(train, y_train)
    return model

def get_lasso_regression_model(train_model_data_final, y_train, lasso_param_grid):
    lasso_gs = GridSearchCV(Lasso(), lasso_param_grid, cv=5, scoring='neg_root_mean_squared_error')
    lasso_gs = lasso_gs.fit(StandardScaler().fit_transform(train_model_data_final), y_train)
    return lasso_gs

def get_ridge_regressor_model(train_model_data_final, y_train, ridge_param_grid):
    ridge_gs = GridSearchCV(Ridge(), ridge_param_grid, cv=5, scoring='neg_root_mean_squared_error')
    ridge_gs = ridge_gs.fit(StandardScaler().fit_transform(train_model_data_final), y_train)
    return ridge_gs

def get_elastic_net_regressor_model(train_model_data_final, y_train, elastic_net_param_grid):
    elastic_net_gs = GridSearchCV(ElasticNet(), elastic_net_param_grid, cv=5, scoring='neg_root_mean_squared_error')
    elastic_net_gs = elastic_net_gs.fit(StandardScaler().fit_transform(train_model_data_final), y_train)
    return elastic_net_gs

def get_knn_regressor_model(train_model_data_final, y_train, knn_param_grid):
    knn_grid_search = RandomizedSearchCV(KNeighborsRegressor(), knn_param_grid, cv=5, verbose=0, n_iter=MAX_MODELS)
    knn_grid_search = knn_grid_search.fit(train_model_data_final, y_train)
    return knn_grid_search

def get_stacking_regressor(train_model_data_final, y_train, estimators, final_estimator):
    return StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
    )