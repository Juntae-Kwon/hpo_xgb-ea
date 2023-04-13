import pandas as pd
import numpy as np

def preprocessing(df):

    df_numeric = df.select_dtypes(include='float')
    df_str = df.select_dtypes([np.object])
    df_str = df_str.stack().str.decode("utf-8").unstack()
    df = pd.concat([df_numeric, df_str], axis=1)
            
    df = df.replace("?", np.nan)

    # Remove ID column
    id_col_to_drop = [col for col in df.columns if col.lower() == "id"]
    df = df.drop(id_col_to_drop, axis=1)

    # Remove features that have only a single kind of value
    single_value_cols = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            single_value_cols.append(col)
    df = df.drop(single_value_cols, axis=1)

    # Remove features that have more than 50% NaN values
    nan_percentages = df.isna().mean()
    df = df.loc[:, nan_percentages < 0.5]

    # Drop Na
    df = df.dropna()

    return df 

def one_hot_encoding(X):
    if len(X.select_dtypes(include=['object']).columns) == 0: # meaning there is only numeric one
        return X
    else:
        # Separate X into numeric and categorical features
        X_num = X.select_dtypes(include="float")
        X_cate = pd.get_dummies(X.select_dtypes(include=["object"])) # one-hot encoding

        # concatenate
        X_res = pd.concat([X_num, X_cate], axis=1)

        return X_res

def converting_y(y):
    val_count = y.value_counts()
    c1 = val_count.index.tolist()[0]
    c2 = val_count.index.tolist()[1]
    y_res = y.map({c1: 0, c2: 1})
    return y_res