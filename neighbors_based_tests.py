import pandas as pd
from surprise import Dataset
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise.model_selection import GridSearchCV
import time


def baseline_test(data, algo, filename, param_grid):
    start = time.time()
    gs = get_gs_for_given_algo(data, algo, param_grid)
    gs.fit(data)
    print(filename + " TEST RESULTS: ")
    print(gs.best_params["rmse"])
    print(gs.best_score["rmse"])
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

    similarity_values = ['msd', 'cosine', 'pearson']
    user_based_values = [True, False]
    neighborhood_size_values = [20, 30, 40, 50, 60, 70, 80]
    algos = [KNNBasic, KNNWithMeans, KNNWithZScore]

    fixed_neighborhood_size_param_grid = get_param_grid()
    similarity_values_param_grid = get_param_grid(similarity_measures=similarity_values)
    different_neighborhood_size_param_grid = get_param_grid(neighborhood_size_values)
    user_vs_item_based_param_grid = get_param_grid(user_based=user_based_values)
    user_vs_item_with_sim_meas_param_grid = get_param_grid(similarity_measures=similarity_values, user_based=user_based_values)

    fixed_neighborhood_size_file_names = ["KNNBasic_fixed_size", "KNNWithMeans_fixed_size",
                                          "KNNWithZScore_fixed_size"]
    for i in range(len(algos)):
        baseline_test(data, algos[i], fixed_neighborhood_size_file_names[i], fixed_neighborhood_size_param_grid)

    file_names = ["KNNBasic_different_size", "KNNWithMeans_different_size", "KNNWithZScore_different_size"]
    for i in range(len(algos)):
        baseline_test(data, algos[i], file_names[i], different_neighborhood_size_param_grid)

    file_names = ["KNNBasic_different_sim_measures", "KNNWithMeans_different_sim_measures",
                  "KNNWithZScore_different_sim_measures"]
    for i in range(len(algos)):
        baseline_test(data, algos[i], file_names[i], similarity_values_param_grid)

    file_names = ["KNNBasic_item_vs_user_based", "KNNWithMeans_item_vs_user_based",
                  "KNNWithZScore_item_vs_user_based"]
    for i in range(len(algos)):
        baseline_test(data, algos[i], file_names[i], user_vs_item_based_param_grid)

    file_names = ["KNNBasic_item_vs_user_with_different_sim_measures",
                  "KNNWithMeans_item_vs_user_with_different_sim_measures",
                  "KNNWithZScore_item_vs_user_with_different_sim_measures"]
    for i in range(len(algos)):
        baseline_test(data, algos[i], file_names[i], user_vs_item_with_sim_meas_param_grid)
