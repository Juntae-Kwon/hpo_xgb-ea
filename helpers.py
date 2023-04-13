import numpy as np
from sklearn.metrics import balanced_accuracy_score
from loss import focal_binary_Loss, weight_binary_Cross_Entropy
import xgboost as xgb

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def balanced_acc(pred, dtrain):
    y_true = dtrain.get_label()
    y_pred = (sigmoid(pred) > 0.5).astype(float)
    val = balanced_accuracy_score(y_true, y_pred)
    return val

def fit_fun_xgb(config, dtrain, dval):
    
    params = dict([(key, val) for key, val in config.items() if key not in ['num_boost_round', 'focal_gamma', 'imbalance_alpha', 'loss', 'eval']])

    if config["loss"] == "focal":
        loss = focal_binary_Loss(gamma=config["focal_gamma"])
        xgb_fit = xgb.train(
            params=params,
            dtrain=dtrain,
            obj = loss.focal_binary_object,
            num_boost_round = config["num_boost_round"]
            )
    else:
        loss = weight_binary_Cross_Entropy(imbalance_alpha=config["imbalance_alpha"])
        xgb_fit = xgb.train(
            params=params,
            dtrain=dtrain,
            obj = loss.weighted_binary_cross_entropy,
            num_boost_round = config["num_boost_round"]
            )
        
    pred = xgb_fit.predict(dval)
    bal_acc = balanced_acc(pred, dval)
    
    return bal_acc