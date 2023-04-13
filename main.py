from data_preprocessing import preprocessing, one_hot_encoding, converting_y
from hpo import do_opt_xgb
from scipy.io import arff
import pandas as pd

# Load arff datasets

df_jv = arff.loadarff(r"automldata/kdd_JapaneseVowels.arff")
df_jv = pd.DataFrame(df_jv[0])

df_opt = arff.loadarff(r"automldata/optdigits.arff")
df_opt = pd.DataFrame(df_opt[0])

df_la_98 = arff.loadarff(r"automldata/kdd_ipums_la_98-small.arff")
df_la_98 = pd.DataFrame(df_la_98[0])

df_la_00 = arff.loadarff(r"automldata/kdd_ipums_la_99-small.arff")
df_la_00 = pd.DataFrame(df_la_00[0])

df_pen = arff.loadarff(r"automldata/pendigits.arff")
df_pen = pd.DataFrame(df_pen[0])

df_page = arff.loadarff(r"automldata/page-blocks.arff")
df_page = pd.DataFrame(df_page[0])

df_sylva = arff.loadarff(r"automldata/sylva_prior.arff")
df_sylva = pd.DataFrame(df_sylva[0])

df_jm1 = arff.loadarff(r"automldata/jm1.arff")
df_jm1 = pd.DataFrame(df_jm1[0])

df_musk = arff.loadarff(r"automldata/musk.arff")
df_musk = pd.DataFrame(df_musk[0])

df_rl = arff.loadarff(r"automldata/file7b533afc80a.arff")
df_rl = pd.DataFrame(df_rl[0])

df_jv = preprocessing(df_jv)
df_opt = preprocessing(df_opt)
df_la_98 = preprocessing(df_la_98)
df_la_00 = preprocessing(df_la_00)
df_pen = preprocessing(df_pen)
df_page = preprocessing(df_page)
df_sylva = preprocessing(df_sylva)
df_jm1 = preprocessing(df_jm1)
df_musk = preprocessing(df_musk)
df_rl = preprocessing(df_rl)

# Define the label vector respectively
y_jv = df_jv["binaryClass"]
y_opt = df_opt["binaryClass"]
y_la_98 = df_la_98["binaryClass"]
y_la_00 = df_la_00["binaryClass"]
y_pen = df_pen["binaryClass"]
y_page = df_page["binaryClass"]
y_sylva = df_sylva["label"]
y_jm1 = df_jm1["defects"]
y_musk = df_musk["class"]
y_rl = df_rl["class"]

# Convert the label to live in {0, 1}
y_jv = converting_y(y_jv)
y_opt = converting_y(y_opt)
y_la_98 = converting_y(y_la_98)
y_la_00 = converting_y(y_la_00)
y_pen = converting_y(y_pen)
y_page = converting_y(y_page)
y_sylva = converting_y(y_sylva)
y_jm1 = converting_y(y_jm1)
y_musk = converting_y(y_musk)
y_rl = converting_y(y_rl)

# Do one-hot encoding for categorical features
X_jv = df_jv.drop("binaryClass", axis=1)
X_jv = one_hot_encoding(X_jv)

X_opt = df_opt.drop("binaryClass", axis=1)
X_opt = one_hot_encoding(X_opt)

X_la_98 = df_la_98.drop("binaryClass", axis=1)
X_la_98 = one_hot_encoding(X_la_98)

X_la_00 = df_la_00.drop("binaryClass", axis=1)
X_la_00 = one_hot_encoding(X_la_00)

X_pen = df_pen.drop("binaryClass", axis=1)
X_pen = one_hot_encoding(X_pen)

X_page = df_page.drop("binaryClass", axis=1)
X_page = one_hot_encoding(X_page)

X_sylva = df_sylva.drop("label", axis=1)
X_sylva = one_hot_encoding(X_sylva)

X_jm1 = df_jm1.drop("defects", axis=1)
X_jm1 = one_hot_encoding(X_jm1)

X_musk = df_musk.drop("class", axis=1)
X_musk = one_hot_encoding(X_musk)

X_rl = df_rl.drop("class", axis=1)
X_rl = one_hot_encoding(X_rl)

# Define X and y as numpy array
X_jv_np = X_jv.values
y_jv_np = y_jv.values

X_opt_np = X_opt.values
y_opt_np = y_opt.values

X_la_98_np = X_la_98.values
y_la_98_np = y_la_98.values

X_la_00_np = X_la_00.values
y_la_00_np = y_la_00.values

X_pen_np = X_pen.values
y_pen_np = y_pen.values

X_page_np = X_page.values
y_page_np = y_page.values

X_sylva_np = X_sylva.values
y_sylva_np = y_sylva.values

X_jm1_np = X_jm1.values
y_jm1_np = y_jm1.values

X_musk_np = X_musk.values
y_musk_np = y_musk.values

X_rl_np = X_rl.values
y_rl_np = y_rl.values


do_opt_xgb(out_k=3, in_k=3, popsize=5, num_offspring=5, n_evals=10, X_numpy=X_jv_np, y_numpy=y_jv_np)
do_opt_xgb(out_k=3, in_k=3, popsize=5, num_offspring=5, n_evals=10, X_numpy=X_opt_np, y_numpy=y_opt_np)
do_opt_xgb(out_k=3, in_k=3, popsize=5, num_offspring=5, n_evals=10, X_numpy=X_la_98_np, y_numpy=y_la_98_np)
do_opt_xgb(out_k=3, in_k=3, popsize=5, num_offspring=5, n_evals=10, X_numpy=X_la_00_np, y_numpy=y_la_00_np)
do_opt_xgb(out_k=3, in_k=3, popsize=5, num_offspring=5, n_evals=10, X_numpy=X_pen_np, y_numpy=y_pen_np)
do_opt_xgb(out_k=3, in_k=3, popsize=5, num_offspring=5, n_evals=10, X_numpy=X_page_np, y_numpy=y_page_np)
do_opt_xgb(out_k=3, in_k=3, popsize=5, num_offspring=5, n_evals=10, X_numpy=X_sylva_np, y_numpy=y_sylva_np)
do_opt_xgb(out_k=3, in_k=3, popsize=5, num_offspring=5, n_evals=10, X_numpy=X_jm1_np, y_numpy=y_jm1_np)
do_opt_xgb(out_k=3, in_k=3, popsize=2, num_offspring=2, n_evals=2, X_numpy=X_musk_np, y_numpy=y_musk_np)
do_opt_xgb(out_k=3, in_k=3, popsize=2, num_offspring=2, n_evals=2, X_numpy=X_rl_np, y_numpy=y_rl_np)