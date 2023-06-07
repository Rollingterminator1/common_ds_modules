import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

def get_variable_missing_values(df):
    percent_missing = df.isnull().sum() * 100 / df.shape[0]
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
    return missing_value_df

def print_missing_values_info(df):
    df = df[df['percent_missing'] > 0]
    for variable, percent_missing in zip(df['column_name'],df['percent_missing']):
        print(f'Variable: {variable}, percent missing: {percent_missing}')
"""
This function finds out which variables to remove based on high missing value count
@df: dataframe to find missing value count
@threshold: if missing value ratio is above this threshold, remove variable from dataframe
"""
def get_high_missing_value_columns(df, threshold):
    #missing_value_df = get_variable_missing_values(df)
    #print_missing_values(missing_value_df)
    percent_missing = df.isnull().sum() * 100 / df.shape[0]
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
    drop_vars = missing_value_df[missing_value_df['percent_missing'] > threshold]['column_name'].values
    return list(drop_vars)

def get_null_columns(df):
    for column in df.columns:
        if df.isna().sum()[column] > 0:
            print(f'Column: {column} has {df.isna().sum()[column]} missing values')


def fill_missing_values(train, test, ignore_variables=[]):
    for column in train.columns:
        if column not in ignore_variables:
            if train[column].dtype == 'O':
                imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                train[column] = imp_mode.fit_transform(np.array(train[column]).reshape(-1, 1))
                # categorical_variables.append(column)
            else:
                if 'Id' != column:
                    # dealing with discrete numerical variables
                    if len(train[column].unique()) <= 15:
                        imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                        train[column] = imp_mode.fit_transform(np.array(train[column]).reshape(-1, 1))
                        # discrete_numerical_variables.append(column)
                    else:
                        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
                        train[column] = imp_mean.fit_transform(np.array(train[column]).reshape(-1, 1))
                        # continuous_numerical_variables.append(column)

    for column in test.columns:
        if column not in ignore_variables:
            if test[column].dtype == 'O':
                imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                test[column] = imp_mode.fit_transform(np.array(test[column]).reshape(-1, 1))
            else:
                if 'Id' != column:
                    if len(test[column].unique()) <= 15:
                        imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                        # print(f'Filling missing values in {column} with mode')
                        test[column] = imp_mode.fit_transform(np.array(test[column]).reshape(-1, 1))
                    else:
                        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
                        test[column] = imp_mean.fit_transform(np.array(test[column]).reshape(-1, 1))

    return train, test