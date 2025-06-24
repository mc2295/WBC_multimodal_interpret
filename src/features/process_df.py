import pandas as pd
import numpy as np
import random

def split_and_convert(value_str, column):
    # Split the string into individual values
    if pd.isna(value_str):
        return [np.nan] * len(column.split('_')[-1])
    values = value_str.split(',')
    
    # Convert the values to floats
    return list(map(float, values))
    
def split_columns(df):
    df.drop('dataset', axis=1, inplace=True)
    df.drop('dataset_y', axis=1, inplace=True)
    df.drop('label', axis=1, inplace=True)
    df.drop('label_y', axis=1, inplace=True)
    df.rename(columns={'dataset_x': 'dataset', 'label_x': 'label'}, inplace=True)
    df = df.reset_index()

    list_column = df.columns.tolist()
    column_to_process = [i for i in list_column if isinstance(df[i][0], str) and i not in ['label', 'dataset', 'cell', 'is_valid']]
    for column in df.columns[1:]:
    # Create new column names based on the original column
        if column in column_to_process and column.endswith('_RGB'):
            new_columns = [column.replace('_RGB', f'_{channel}') for channel in ['R', 'G', 'B']]
        elif column in column_to_process and column.endswith('_CMYK'):
            new_columns = [column.replace('_CMYK', f'_{channel}') for channel in ['C', 'M', 'Y', 'K']]
        elif column in column_to_process and column.endswith('_HSV'):
            new_columns = [column.replace('_HSV', f'_{channel}') for channel in ['H', 'S', 'V']]
        else:
            continue  # Skip columns that don't match the pattern
        df[new_columns] = df.apply(lambda row: split_and_convert(row[column], column), axis=1).apply(pd.Series)
    df.drop(columns=column_to_process, inplace=True)
    return df
    
def split_train_valid(df):
    indices = [i for i in range(len(df))]
    random.Random(4).shuffle(indices)
    splits = [indices[:len(indices) - len(indices)//10], indices[len(indices) - len(indices)//10:]]
    df = df.assign(is_valid = False)
    df.loc[df['index'].isin(splits[1]), 'is_valid'] = True
    return df

def remove_nan(df, remove = False):
    nan_rows_cytoonly = df[df['cytoonly_area'].isnull()].index
    cytoonly_columns = [i for i in df.columns if 'cytoonly' in i]
    nan_rows_label = df[df['label'].isnull()].index
    df.loc[nan_rows_label, ['label']] = 'artefact'
    df.loc[nan_rows_cytoonly , cytoonly_columns] = -1
    df = df.fillna(-1)

    return df
    
def normalize_df(df1):
    normalized_df=(df1-df1.mean())/df1.std()
    columns_tabular = [i for i in df1.columns if i not in ['cell', 'label', 'dataset', 'Unnamed: 0', 'is_valid', 'index']]
    df1.loc[:,columns_tabular] = normalized_df.loc[:,columns_tabular]
    return df1