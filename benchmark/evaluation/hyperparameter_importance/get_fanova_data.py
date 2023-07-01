# Run `cv_analysis.py` first to generate the results we need here.

import pandas as pd
import json
import numpy as np
import optuna
from optuna_fast_fanova import FanovaImportanceEvaluator

name_file_list = [
    ("Binary Balanced Accuracy",
     "performance_per_dataset_for Task Type Binary Classification and Metric balanced_accuracy [Test Score].csv"),
    ("Binary ROC AUC",
     "performance_per_dataset_for Task Type Binary Classification and Metric roc_auc [Test Score].csv"),
    ("Multi-class Balanced Accuracy",
     "performance_per_dataset_for Task Type Multi-class Classification and Metric balanced_accuracy [Test Score].csv"),
    ("Multi-class ROC AUC",
     "performance_per_dataset_for Task Type Multi-class Classification and Metric roc_auc [Test Score].csv")
]


def data_for_map(method_name, name_grid_mapping):
    prefix, x = method_name.split(".", 1)
    method, config_id = x.rsplit("_", 1)
    config = name_grid_mapping[f"config_{config_id}"].copy()
    config["preprocessing"] = prefix
    config["method"] = method
    return config


def specific_config_to_row(config):
    return [config["preprocessing"], config["batch_size"], config["max_elites"],
            config["emitter_initialization_method"], config["elite_selection_method"], config["crossover"]]


data_col_names = ["preprocessing", "batch_size", "archive_size", "init_method", "sampling_method", "crossover_method"]

optuna.logging.set_verbosity(optuna.logging.ERROR)


def run():
    with open("../results/name_grid_mapping.json", "r") as f:
        name_grid_mapping = json.load(f)

    overall_importance = []

    for name, file in name_file_list:
        print("#", name)
        df = pd.read_csv(f"../out/{file}")

        col_map = {c: data_for_map(c, name_grid_mapping) for c in list(df)[1:]}
        qdo_mask = np.array([True] + [c["method"] == "QDO-ES" for c in col_map.values()])
        qo_mask = np.array([True] + [c["method"] == "QO-ES" for c in col_map.values()])
        mask_importance = []

        for m_n, mask in [("QDO", qdo_mask), ("QO", qo_mask)]:
            print("##", m_n)
            tmp_df = df.loc[:, mask]

            importance_per_dataset = []

            for r_i, row in tmp_df.iterrows():
                dataset_name, *scores = list(row)
                print("###", dataset_name)
                configs = [col_map[n] for n in list(tmp_df)[1:]]

                identifier = [hash(tuple([hash(val) for val in (specific_config_to_row(c))])) for c in configs]
                y = np.array(scores)
                tmp = []

                def objective(trial, tmp_y, tmp_identifier, tmp):
                    trial.suggest_categorical("preprocessing", ["SiloTopN", "TopN"])
                    trial.suggest_categorical("batch_size", [20, 40])
                    trial.suggest_categorical("archive_size", [16, 49])
                    trial.suggest_categorical("init_method", ["AllL1", "RandomL2Combinations", "L2ofSingleBest"])
                    trial.suggest_categorical("sampling_method", ["deterministic", "tournament", "combined_dynamic"])
                    trial.suggest_categorical("crossover_method", ["two_point_crossover", "average", "no_crossover"])
                    config = trial.params
                    id_list = [config["preprocessing"], config["batch_size"], config["archive_size"],
                               config["init_method"], config["sampling_method"], config["crossover_method"]]
                    id_for_y = hash(tuple([hash(val) for val in id_list]))

                    tmp.append(id_for_y)

                    return tmp_y[tmp_identifier.index(id_for_y)]

                study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler(seed=42))
                func = lambda t: objective(t, y, identifier, tmp)
                study.optimize(func, n_trials=len(identifier))

                importance = optuna.importance.get_param_importances(
                    study, evaluator=FanovaImportanceEvaluator()
                )
                assert set(tmp) == set(identifier)
                importance_per_dataset.append(importance)

            mask_importance.append((m_n, importance_per_dataset))

        overall_importance.append((name, mask_importance))

    ## Print results
    print("\n\n\n ########### RESULTS:")
    for task_type, data in overall_importance:
        print(" --- Task Type:", task_type)

        for method, importance_per_dataset in data:
            mean_importance = pd.DataFrame(importance_per_dataset).mean()
            print("-- Mean Importance:", method)
            print(mean_importance)


if __name__ == "__main__":
    run()
