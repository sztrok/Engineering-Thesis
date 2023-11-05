import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
import seaborn
import numpy
import matplotlib.pyplot as plot
from scipy.sparse import csr_matrix
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVDpp, accuracy, SVD
from surprise.model_selection import cross_validate, KFold, GridSearchCV
from collections import defaultdict
import time

# data = Dataset.load_builtin("ml-100k")
# algo = SVDpp()

# kf = KFold(n_splits=5)
# for trainset, testset in kf.split(data):
#     predictions = algo.fit(trainset).test(testset)
#     accuracy.rmse(predictions)

# trainset, testset = train_test_split(data, test_size=0.2)
# algo.fit(trainset)
# predictions = algo.test(testset)
# accuracy.rmse(predictions)

# cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

#
# param_grid = {"n_epochs": [5, 10], "lr_all": [0.002, 0.005], "reg_all": [0.4, 0.6]}
# gs = GridSearchCV(SVDpp, param_grid, measures=["rmse"], cv=3)
# gs.fit(data)
# print(gs.best_score)
# print(gs.best_params)

"""
algorytm SVD
Argumenty:
    n_factors: number of factors, def=100
    n_epochs: number of epochs, def=20
    biased(bool): whether to use biases, def=False
    init_mean: the mean of the normal distribution for factor vectors initialization, def=0
    init_std_dev: standard deviation of the normal distribution for factor vectors initialization, def=0.1
    lr_all: learning rate for all parameters, def=0.005
    reg_all: regularization term for all parameters, def=0.02
    
    lr_bu
    lr_bi
    lr_pu
    lr_qi
    reg_bu
    reg_bi
    reg_pu
    reg_qi
"""


def svd_algorithm_test_1(data, trainset, testset):
    # algo = SVD()
    # finding the best parameters for algorithm
    param_grid = {"n_factors": [20, 30, 40, 50, 60, 70, 80, 90, 100],
                  "n_epochs": [5, 10, 15, 20, 25],
                  "lr_all": [0.002, 0.003, 0.004, 0.005],
                  "reg_all": [0.2, 0.3, 0.4, 0.5, 0.6]}
    gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=5)
    gs.fit(data)
    print(gs.best_score["rmse"])
    print(gs.best_params["rmse"])
    best_algo = gs.best_estimator["rmse"]
    predictions = best_algo.fit(trainset).test(testset)
    accuracy.rmse(predictions)


"""
algorytm SVD++
bierze pod uwagę niejawne oceny, w tym przypdaku jest to fakt wystawienia oceny, niezależnie od tego, ile ona wyniosła
Argumenty:
    n_factors: number of factors, def=100
    n_epochs: number of epochs, def=20
    cache_ratings(bool): whether to cache ratings during 'fit()', speeds-up the training but has higher memory footprint
    init_mean: the mean of the normal distribution for factor vectors initialization, def=0
    init_std_dev: standard deviation of the normal distribution for factor vectors initialization, def=0.1
    lr_all: learning rate for all parameters
    reg_all: regularization term for all parameters
    
    lr_bu
    lr_bi
    lr_pu
    lr_qi
    reg_bu
    reg_bi
    reg_pu
    reg_qi
"""


def svdpp_algorithm_test_1(data, trainset, testset):
    param_grid = {"n_factors": [30]}
    gs = GridSearchCV(SVDpp, param_grid, measures=["rmse"], cv=2, n_jobs=-1)
    gs.fit(data)
    print(gs.best_score["rmse"])
    print(gs.best_params["rmse"])
    best_algo = gs.best_estimator["rmse"]
    predictions = best_algo.fit(trainset).test(testset)
    accuracy.rmse(predictions)


def get_gs_for_given_algo(data, algo, param_grid):
    gs = GridSearchCV(algo, param_grid, measures=["rmse"], cv=3, n_jobs=15)
    gs.fit(data)
    return gs


def get_param_grid_for_svd(n_factors=None, n_epochs=None, biased=None, init_mean=None, init_std_dev=None, lr_all=None,
                           reg_all=None):
    if reg_all is None:
        reg_all = [0.02]
    if lr_all is None:
        lr_all = [0.005]
    if init_std_dev is None:
        init_std_dev = [0.1]
    if init_mean is None:
        init_mean = [0]
    if biased is None:
        biased = [False]
    if n_epochs is None:
        n_epochs = [20]
    if n_factors is None:
        n_factors = [100]
    return {"n_factors": n_factors,
            "n_epochs": n_epochs,
            "biased": biased,
            "init_mean": init_mean,
            "init_std_dev": init_std_dev,
            "lr_all": lr_all,
            "reg_all": reg_all
            }


def get_param_grid_for_svdpp(n_factors=None, n_epochs=None, cache_ratings=None, init_mean=None, init_std_dev=None,
                             lr_all=None,
                             reg_all=None):
    if reg_all is None:
        reg_all = [0.02]
    if lr_all is None:
        lr_all = [0.005]
    if init_std_dev is None:
        init_std_dev = [0.1]
    if init_mean is None:
        init_mean = [0]
    if cache_ratings is None:
        cache_ratings = [True]
    if n_epochs is None:
        n_epochs = [20]
    if n_factors is None:
        n_factors = [100]
    return {"n_factors": n_factors,
            "n_epochs": n_epochs,
            "cache_ratings": cache_ratings,
            "init_mean": init_mean,
            "init_std_dev": init_std_dev,
            "lr_all": lr_all,
            "reg_all": reg_all
            }


def run_all_tests_for_svd_type_algo(data, algo, params, filename, trainset, testset):
    start = time.time()
    gs = get_gs_for_given_algo(data, algo, params)
    print(filename + " TEST RESULTS: ")
    print(gs.best_params["rmse"])
    print(gs.best_score["rmse"])
    rec = gs.best_estimator["rmse"]
    accuracy.rmse(rec.fit(trainset).test(testset))
    end = time.time()
    filename += '.csv'
    pd.DataFrame.from_dict(gs.cv_results).to_csv(filename)
    print("TIME ELAPSED: " + str(end - start))
    print("---------------------------------------")


if __name__ == '__main__':
    data = Dataset.load_builtin("ml-100k")
    trainset, testset = train_test_split(data, test_size=0.25)

    factors_test_values = [25, 50, 75, 100, 125, 150, 175, 200]
    epochs_test_values = [10, 20, 30, 40, 50, 60, 70, 80]
    biased_test_values = [True, False]
    lr_test_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
    reg_test_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]

    factors_test_param_grid = get_param_grid_for_svd(n_factors=factors_test_values)
    epochs_test_param_grid = get_param_grid_for_svd(n_epochs=epochs_test_values)
    biased_test_param_grid = get_param_grid_for_svd(biased=biased_test_values)
    lr_test_param_grid = get_param_grid_for_svd(lr_all=lr_test_values)
    reg_test_param_grid = get_param_grid_for_svd(reg_all=reg_test_values)
    factors_and_epochs_test_param_grid = get_param_grid_for_svd(n_factors=factors_test_values,
                                                                n_epochs=epochs_test_values)
    factors_epochs_biased_test_param_grid = get_param_grid_for_svd(n_factors=factors_test_values,
                                                                   n_epochs=epochs_test_values,
                                                                   biased=biased_test_values)
    factors_epochs_lr_test_param_grid = get_param_grid_for_svd(n_factors=factors_test_values,
                                                               n_epochs=epochs_test_values,
                                                               biased=[True],
                                                               lr_all=lr_test_values)
    factors_epochs_reg_test_param_grid = get_param_grid_for_svd(n_factors=factors_test_values,
                                                                n_epochs=epochs_test_values,
                                                                biased=[True],
                                                                reg_all=reg_test_values)
    factors_epochs_lr_reg_test_param_grid = get_param_grid_for_svd(n_factors=factors_test_values,
                                                                   n_epochs=epochs_test_values,
                                                                   biased=[True],
                                                                   lr_all=lr_test_values,
                                                                   reg_all=reg_test_values)

    factors_test_param_grid_svdpp = get_param_grid_for_svdpp(n_factors=factors_test_values)
    epochs_test_param_grid_svdpp = get_param_grid_for_svdpp(n_epochs=epochs_test_values)
    lr_test_param_grid_svdpp = get_param_grid_for_svdpp(lr_all=lr_test_values)
    reg_test_param_grid_svdpp = get_param_grid_for_svdpp(reg_all=reg_test_values)
    factors_and_epochs_test_param_grid_svdpp = get_param_grid_for_svdpp(n_factors=factors_test_values,
                                                                        n_epochs=epochs_test_values)
    factors_epochs_lr_test_param_grid_svdpp = get_param_grid_for_svdpp(n_factors=factors_test_values,
                                                                       n_epochs=epochs_test_values,
                                                                       lr_all=lr_test_values)
    factors_epochs_reg_test_param_grid_svdpp = get_param_grid_for_svdpp(n_factors=factors_test_values,
                                                                        n_epochs=epochs_test_values,
                                                                        reg_all=reg_test_values)
    factors_epochs_lr_reg_test_param_grid_svdpp = get_param_grid_for_svdpp(n_factors=factors_test_values,
                                                                           n_epochs=epochs_test_values,
                                                                           lr_all=lr_test_values,
                                                                           reg_all=reg_test_values)

    svd_test_data = [factors_test_param_grid, epochs_test_param_grid, biased_test_param_grid,
                     lr_test_param_grid, reg_test_param_grid, factors_and_epochs_test_param_grid,
                     factors_epochs_biased_test_param_grid, factors_epochs_lr_test_param_grid,
                     factors_epochs_reg_test_param_grid, factors_epochs_lr_reg_test_param_grid]

    svd_file_names = ["factors_test_svd", "epochs_test_svd", "biased_test_svd",
                      "lr_test_svd", "reg_test_svd", "factors_and_epochs_test_svd",
                      "factors_epochs_biased_test_svd", "factors_epochs_lr_test_svd",
                      "factors_epochs_reg_test_svd", "factors_epochs_lr_reg_test_svd"]

    svdpp_test_data = [factors_test_param_grid_svdpp, epochs_test_param_grid_svdpp,
                       lr_test_param_grid_svdpp, reg_test_param_grid_svdpp, factors_and_epochs_test_param_grid_svdpp,
                       factors_epochs_lr_test_param_grid_svdpp,
                       factors_epochs_reg_test_param_grid_svdpp, factors_epochs_lr_reg_test_param_grid_svdpp]

    svdpp_file_names = ["factors_test_svdpp", "epochs_test_svdpp",
                        "lr_test_svdpp", "reg_test_svdpp", "factors_and_epochs_test_svdpp",
                        "factors_epochs_lr_test_svdpp",
                        "factors_epochs_reg_test_svdpp", "factors_epochs_lr_reg_test_svdpp"]

    # for i in range(len(svd_test_data)):
    #     run_all_tests_for_svd_type_algo(data, SVD, svd_test_data[i], svd_file_names[i], trainset, testset)

    # for i in range(len(svdpp_test_data)):
    #     run_all_tests_for_svd_type_algo(data, SVDpp, svdpp_test_data[i], svdpp_file_names[i], trainset, testset)

    svdpp_algorithm_test_1(data, trainset, testset)
