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
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise.model_selection import cross_validate, KFold, GridSearchCV
from collections import defaultdict
import time


def baseline_test(data, algo, filename, param_grid):
    start = time.time()
    gs = get_gs_for_given_algo(data, algo, param_grid)
    gs.fit(data)
    print(filename + " TEST RESULTS: ")
    print(gs.best_params["rmse"])
    print(gs.best_score["rmse"])
    # rec = gs.best_estimator["rmse"]
    # accuracy.rmse(rec.fit(trainset).test(testset))
    end = time.time()
    filename += '.csv'
    pd.DataFrame.from_dict(gs.cv_results).to_csv(filename)
    print("TIME ELAPSED: " + str(end - start))
    print("---------------------------------------")


def test_for_different_neighborhood_size(data, algo, filename, param_grid):
    # param_grid = {"k": [20, 30, 40, 50, 60, 70]}

    start = time.time()
    gs = get_gs_for_given_algo(data, algo, param_grid)
    gs.fit(data)
    print(filename + " TEST RESULTS: ")
    print(gs.best_params["rmse"])
    print(gs.best_score["rmse"])
    # rec = gs.best_estimator["rmse"]
    # accuracy.rmse(rec.fit(trainset).test(testset))
    end = time.time()
    filename += '.csv'
    pd.DataFrame.from_dict(gs.cv_results).to_csv(filename)
    print("TIME ELAPSED: " + str(end - start))
    print("---------------------------------------")


def get_gs_for_given_algo(data, algo, param_grid):
    gs = GridSearchCV(algo, param_grid, measures=["rmse"], cv=5, n_jobs=-1)
    gs.fit(data)
    return gs


def get_sim_options(similarity_name=None, user_based=None):
    if similarity_name is None:
        similarity_name = "msd"
    if user_based is None:
        user_based = True
    return {"name": similarity_name,
            "user_based": user_based
            }


def get_param_grid(k=None, similarity_measures=None, user_based=None):
    if k is None:
        k = [40]
    if similarity_measures is None:
        similarity_measures = ["msd"]
    if user_based is None:
        user_based = [True]
    return {"k": k,
            "sim_options": {
                "name": similarity_measures,
                "user_based": user_based
            }}


if __name__ == '__main__':
    data = Dataset.load_builtin("ml-100k")
    trainset, testset = train_test_split(data, test_size=0.25)

    similarity_values = ['msd', 'cosine', 'pearson']
    user_based_values = [True, False]
    neighborhood_size_values = [20, 30, 40, 50, 60, 70]
    algos = [KNNBasic, KNNWithMeans, KNNWithZScore]

    fixed_neighborhood_size_param_grid = get_param_grid()
    similarity_values_param_grid = get_param_grid(similarity_measures=similarity_values)
    different_neighborhood_size_param_grid = get_param_grid(neighborhood_size_values)

    fixed_neighborhood_size_file_names = ["KNNBasic_fixed_size", "KNNWithMeans_fixed_size", "KNNWithZScore_fixed_size"]
    for i in range(len(algos)):
        baseline_test(data, algos[i], fixed_neighborhood_size_file_names[i], fixed_neighborhood_size_param_grid)
