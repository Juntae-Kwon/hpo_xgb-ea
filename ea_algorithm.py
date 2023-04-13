from loss import focal_binary_Loss, weight_binary_Cross_Entropy
from helpers import balanced_acc
import numpy as np
import xgboost as xgb
import pandas as pd

# Generating population
def generate_pop(x_id, x, bounds, popsize):
    
    if x.dtype == int:
        x = np.random.randint(low=bounds[x_id][0], high=bounds[x_id][1], size=popsize)
    elif x.dtype == float:
        x = np.random.uniform(low=bounds[x_id][0], high=bounds[x_id][1], size=popsize)
    elif x.dtype == object:
        x = np.random.choice(bounds[x_id], size=popsize, replace=True)
    elif x.dtype == bool:
        x = np.random.choice(bounds[x_id], size=popsize, replace=True)
    
    return x

# Uniform crossover
def unif_crossover(parent, partner, bounds):
    res = {}
    keys = list(bounds.keys())
    for i in range(len(parent)):
        res[keys[i]] = np.random.choice([parent[i], partner[i]], size=1)
    return res

def evaluation_xgb(config, dtrain):
    
    res = []

    exclude = ["num_boost_round", "focal_gamma", "imbalance_alpha", "loss"]
    params = {k: v for k, v in config.items() if k not in exclude}
    
    for i in range(len(config[exclude[0]])):
        
        if config["loss"][i] == "focal":
            loss = focal_binary_Loss(gamma=config["focal_gamma"][i])
            fit = xgb.train(
                params={k: v[i] for k, v in params.items()},
                dtrain=dtrain,
                obj = loss.focal_binary_object,
                num_boost_round = config["num_boost_round"][i]
                )
        else:
            loss = weight_binary_Cross_Entropy(imbalance_alpha=config["imbalance_alpha"][i])
            fit = xgb.train(
                params={k: v[i] for k, v in params.items()},
                dtrain=dtrain,
                obj = loss.weighted_binary_cross_entropy,
                num_boost_round = config["num_boost_round"][i]
                )
        
        pred = fit.predict(dtrain)
        bal_acc = balanced_acc(pred, dtrain)
        res.append(bal_acc)

    return np.array(res)


# Evolutionary Alrogithm to tune hyperparameters of XGBoost
# Parent Selection: Proportional to fitness values
# Variation:
#       Uniform Crossover with p=0.5
# Survival Selection: Plus selection (popsize + num_of_offspring)

def ea_xgb(popsize, num_offspring, n_evals, df):

    # 1.
    # Define hyperparameters to be tuned and their search spaces
    num_boost_round = np.empty((popsize,), dtype=int)
    eta = np.empty((popsize,), dtype=float)
    max_depth = np.empty((popsize,), dtype=int)
    max_delta_step = np.empty((popsize,), dtype=int)
    max_leaves = np.empty((popsize,), dtype=int)
    subsample = np.empty((popsize,), dtype=float)
    colsample_bytree = np.empty((popsize,), dtype=float)
    colsample_bylevel = np.empty((popsize,), dtype=float)
    colsample_bynode = np.empty((popsize,), dtype=float)
    reg_lambda = np.empty((popsize,), dtype=float)
    reg_alpha = np.empty((popsize,), dtype=float)
    focal_gamma = np.empty((popsize,), dtype=float)
    imbalance_alpha = np.empty((popsize,), dtype=float)
    loss = np.empty((popsize,), dtype=object)

    bounds = {
        "num_boost_round": [400, 500],
        "eta": [0.05, 0.1],
        "max_depth": [5, 9],
        "max_delta_step": [5, 9],
        "max_leaves": [200, 300],
        "subsample": [0.5, 0.8],
        "colsample_bytree": [0.5, 0.8],
        "colsample_bylevel": [0.5, 0.8],
        "colsample_bynode": [0.5, 0.8],
        "reg_lambda": [0.05, 0.08],
        "reg_alpha": [0.05, 0.08],
        "focal_gamma": [0.3, 1.0],
        "imbalance_alpha": [0.3, 1.],
        "loss": ["focal", "weighted"]
    }

    # 2.
    # Generate initial populations
    num_boost_round = generate_pop("num_boost_round", num_boost_round, bounds, popsize)
    eta = generate_pop("eta", eta, bounds, popsize)
    max_depth = generate_pop("max_depth", max_depth, bounds, popsize)
    max_delta_step = generate_pop("max_delta_step", max_delta_step, bounds, popsize)
    max_leaves = generate_pop("max_leaves", max_leaves, bounds, popsize)
    subsample = generate_pop("subsample", subsample, bounds, popsize)
    colsample_bytree = generate_pop("colsample_bytree", colsample_bytree, bounds, popsize)
    colsample_bylevel = generate_pop("colsample_bylevel", colsample_bylevel, bounds, popsize)
    colsample_bynode = generate_pop("colsample_bynode", colsample_bynode, bounds, popsize)
    reg_lambda = generate_pop("reg_lambda", reg_lambda, bounds, popsize)
    reg_alpha = generate_pop("reg_alpha", reg_alpha, bounds, popsize)
    focal_gamma = generate_pop("focal_gamma", focal_gamma, bounds, popsize)
    imbalance_alpha = generate_pop("imbalance_alpha", imbalance_alpha, bounds, popsize)
    loss = generate_pop("loss", loss, bounds, popsize)

    population = {
        "num_boost_round": num_boost_round,
        "eta": eta,
        "max_depth": max_depth,
        "max_delta_step": max_delta_step,
        "max_leaves": max_leaves,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "colsample_bylevel": colsample_bylevel,
        "colsample_bynode": colsample_bynode,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "focal_gamma": focal_gamma,
        "imbalance_alpha": imbalance_alpha,
        "loss": loss
    }

    # 3. 
    # Evaluate their fitness
    eval = evaluation_xgb(population, df)
    pop_df = pd.DataFrame({**population, "eval": eval})

    # 4.
    # Generate offspring with uniform crossover
    traj = pop_df # nrow: popsize 
    while(len(traj) < n_evals):

        # Parent Selection
        # by fitness-proportional
        prob = np.array(pop_df['eval'] / sum(pop_df['eval']))
        idx = np.random.choice(np.arange(popsize), size=num_offspring, p=prob)
        selected_parents = pop_df.iloc[idx]

        
        # Variation
        # Uniform crossover
        children = pd.DataFrame()
        input = selected_parents.drop('eval', axis=1)
        for j in range(len(input)):
            idx_partner = int(np.random.choice(range(len(input)), size=1))
            uo = pd.DataFrame(unif_crossover(parent=input.iloc[j], partner=input.iloc[idx_partner], bounds=bounds))
            children = pd.concat([children, uo], axis=0)
        
        children = children.to_dict('list')
        for k, v in children.items():
            children[k] = np.array(v)
        
        eval_children = evaluation_xgb(children, df)
        children = pd.DataFrame({**children, "eval": eval_children})
        pop_df = pd.concat([pop_df, children], axis=0)
        pop_df = pop_df.sort_values("eval", ascending=False).head(popsize)
        traj = pd.concat([traj, pop_df])

    
    return traj.nlargest(1, "eval")
    
