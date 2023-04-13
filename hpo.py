from sklearn.model_selection import StratifiedKFold
from helpers import fit_fun_xgb
from ea_algorithm import ea_xgb
import xgboost as xgb
import pandas as pd
import numpy as np


def do_opt_xgb(out_k, in_k, popsize, num_offspring, n_evals, X_numpy, y_numpy):
    
    skf1 = StratifiedKFold(n_splits=out_k, shuffle=True)
    k = 0
    perf_outer = []
    res_config = pd.DataFrame(columns = ["num_boost_round", "eta", "max_depth", "max_delta_step", "max_leaves", "subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode", "reg_lambda", "reg_alpha", "focal_gamma", "loss", "eval"])
    for train_idx, test_idx in skf1.split(X_numpy, y_numpy):
        X_train, X_test = X_numpy[train_idx], X_numpy[test_idx]
        y_train, y_test = y_numpy[train_idx], y_numpy[test_idx]

        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dtest = xgb.DMatrix(data=X_test, label=y_test)

        skf2 = StratifiedKFold(n_splits=in_k, shuffle=True)

        perf_inner = []
        config_inner = pd.DataFrame()
        for train_inner_idx, val_inner_idx in skf2.split(X_train, y_train):

            X_inner_train, X_inner_val = X_train[train_inner_idx], X_train[val_inner_idx]
            y_inner_train, y_inner_val = y_train[train_inner_idx], y_train[val_inner_idx]

            dtrain_inner = xgb.DMatrix(data=X_inner_train, label=y_inner_train)
            dtrain_val = xgb.DMatrix(data=X_inner_val, label=y_inner_val)

            best_config = ea_xgb(popsize=popsize, num_offspring=num_offspring, n_evals=n_evals, df=dtrain_inner)
            config_inner = pd.concat([config_inner, best_config], axis=0)
            best_config = best_config.to_dict('records')[0]
            res_inner = fit_fun_xgb(best_config, dtrain_inner, dtrain_val)
            perf_inner.append(res_inner)
        
        best_idx = np.argmax(perf_inner)
        final_config = config_inner.iloc[best_idx].to_dict()
        res_outer = fit_fun_xgb(final_config, dtrain, dtest)
        perf_outer.append(res_outer)
        res_config.loc[k] = list(final_config.values())
        k += 1

    res_config["perf"] = perf_outer
    return res_config
