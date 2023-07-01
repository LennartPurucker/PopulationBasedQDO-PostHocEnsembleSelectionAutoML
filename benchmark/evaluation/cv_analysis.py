import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "../.."))

import pandas as pd
import numpy as np
from benchmark.evaluation import analysis_facets
from benchmark.evaluation import analysis_utils
import json
from collections import Counter
from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'


def _preprocess_ensemble_names(performance_per_fold_per_dataset):
    idea_map_archives = {"sliding": "QDO-ES", "quality": "QO-ES"}
    idea_map_methods = {"SingleBest": "SingleBest", "EnsembleSelection": "GES"}
    fallback_name = "Other"
    baseline_map = "config_0"

    with open(f"./results/name_grid_mapping.json") as f:
        name_grid_mapping = json.load(f)

    def name_mapping(row):
        name = row[0]
        prefix = name.split(".", 1)[0] + "."
        config_name = name.split(".", 1)[1]
        config_id = "_" + name.rsplit("_", 1)[-1]

        if config_name == baseline_map:
            return prefix + "SingleBest" + config_id, "SingleBest"

        config = name_grid_mapping[config_name]

        if config["method"] == "EnsembleSelection":
            return prefix + idea_map_methods["EnsembleSelection"] + config_id, idea_map_methods["EnsembleSelection"]

        if config["method"] == "QDOEnsembleSelection":
            a_n = idea_map_archives[config["archive_type"]]
            return prefix + a_n + config_id, a_n

        # Fall back case for unknown method
        return prefix + fallback_name + config_id, fallback_name

    performance_per_fold_per_dataset[["Ensemble Technique", "Method"]] = performance_per_fold_per_dataset[
        ["Ensemble Technique", "Fold"]].apply(name_mapping, axis=1, result_type="expand")

    return performance_per_fold_per_dataset


def _handle_nan_values(performance_per_fold_per_dataset, baseline_algorithm):
    # Fill crashed runs with bad value
    print(f"Found {performance_per_fold_per_dataset.isna().sum()} nan values")
    performance_per_fold_per_dataset[['validation_loss', 'ensemble_size']] = performance_per_fold_per_dataset[
        ['validation_loss', 'ensemble_size']].fillna(value=-1)
    # Set SingleBest Size to 1
    performance_per_fold_per_dataset.loc[
        performance_per_fold_per_dataset["Ensemble Technique"] == baseline_algorithm, "ensemble_size"] = 1
    # Fill remaining nan values with 0
    performance_per_fold_per_dataset = performance_per_fold_per_dataset.fillna(value=0)
    return performance_per_fold_per_dataset


def _filter_ens(performance_per_fold_per_dataset, methods_to_drop, ens_eval_config, baseline_algorithm,
                allowed_preselection):
    performance_per_fold_per_dataset = performance_per_fold_per_dataset[~performance_per_fold_per_dataset[
        "Ensemble Technique"].isin(methods_to_drop)]
    if ens_eval_config is not None:
        methods_to_keep = [k + "." + n for n in ens_eval_config for k in allowed_preselection] + [baseline_algorithm]
        performance_per_fold_per_dataset = performance_per_fold_per_dataset[
            performance_per_fold_per_dataset["Ensemble Technique"].isin(methods_to_keep)]

    print("Methods in Evaluation:", set(performance_per_fold_per_dataset["Ensemble Technique"].tolist()))
    return performance_per_fold_per_dataset


def _get_metric_name(benchmark_name):
    with open(f"../setup_data/{benchmark_name}_data.json") as f:
        metric_name = json.load(f)["framework_extras"]["metric"]

    return metric_name


def _get_results_per_task(performance_per_fold_per_dataset):
    binary = performance_per_fold_per_dataset[performance_per_fold_per_dataset["n_classes"] == 2]
    multi = performance_per_fold_per_dataset[performance_per_fold_per_dataset["n_classes"] != 2]

    results_per_task = [
        ("Binary Classification", binary),
        ("Multi-class Classification", multi),
    ]

    return results_per_task


def _select_for_data(in_ppf, baseline_algorithm, maximize_metric):
    best_list = [("SingleBest", baseline_algorithm)]
    methods = list(set([n.split(".", 1)[1].split("_")[0] for n in list(in_ppf)]) - {"SingleBest"})

    for method in methods:
        # Median Normalized Performance
        tmp_ppf = in_ppf[[n for n in list(in_ppf) if n.split(".", 1)[1].split("_")[0] == method] + [baseline_algorithm]]
        tmp_np = analysis_utils.normalize_performance(tmp_ppf, baseline_algorithm, maximize_metric).iloc[:, :-1]
        filter_p = tmp_np.median(axis=0)

        # Filter (max because normalized performance is to be maximized)
        best = filter_p.index[filter_p.argmax()]
        best_list.append((method, best))

    return best_list


def _run_for_fold(fold_i, performance_per_dataset, baseline_algorithm, maximize_metric):
    print(f"Processing Fold {fold_i}")

    # -- Split benchmark into selection and test data
    test_performance_per_dataset = performance_per_dataset.iloc[fold_i]
    selection_performance_per_dataset = pd.concat([performance_per_dataset.iloc[:fold_i],
                                                   performance_per_dataset.iloc[fold_i + 1:]])
    assert len(selection_performance_per_dataset) == len(performance_per_dataset) - 1

    best_list = _select_for_data(selection_performance_per_dataset, baseline_algorithm, maximize_metric)

    return [(m, n, test_performance_per_dataset[n]) for m, n in best_list]


def plot_res(plot_data, baseline_algorithm, maximize_metric, plot_postfix, eff_pd):
    # -- Different data views
    normalized_ppd = analysis_utils.normalize_performance(plot_data, baseline_algorithm, maximize_metric)

    # -- Different plotting options
    better_order, meanrank = analysis_facets.performance_analysis(plot_data, normalized_ppd,
                                                                  maximize_metric, baseline_algorithm, plot_postfix)
    print(meanrank)

    # -- Basic Efficiency numbers
    eff_pd["Relative Fit Time"] = eff_pd["fit_time"] / eff_pd["mean_fit_time_base_models"]
    eff_pd = eff_pd.drop(columns=["TaskID", "mean_distinct_algorithms_count", "mean_fit_time_base_models",
                                  "mean_predict_time_base_models", "mean_base_models_count"])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(eff_pd.groupby("Ensemble Technique").mean())


def _filter_ppf(config_dict, datasets, ppf):
    filtered_ppf = None
    for config_names in list(config_dict.values()):
        for n, d in zip(config_names, datasets):
            if filtered_ppf is None:
                filtered_ppf = ppf[(ppf["Ensemble Technique"] == n) & (ppf["dataset_name"] == d)]
            else:
                filtered_ppf = pd.concat([filtered_ppf,
                                          ppf[(ppf["Ensemble Technique"] == n) & (ppf["dataset_name"] == d)]])

    filtered_ppf["Ensemble Technique"] = filtered_ppf["Ensemble Technique"].apply(
        lambda x: x.split(".", 1)[1].rsplit("_", 1)[0])
    return filtered_ppf


def run_for_task_type(task_type, ppf, score_type, metric_name, col_metric_name, baseline_algorithm, cv=True):
    print("Number of Dataset for {}: {}".format(task_type, len(set(ppf["dataset_name"].tolist()))))

    plot_postfix = "for Task Type {} and Metric {} [{}]".format(task_type, metric_name, score_type)
    maximize_metric = False if col_metric_name in ["validation_loss"] else True

    performance_per_dataset = analysis_utils.transpose_means(
        analysis_utils.get_mean_over_cross_validation(ppf, col_metric_name))

    # Sanity Check
    assert not performance_per_dataset.isnull().values.any()
    performance_per_dataset.to_csv("./out/performance_per_dataset_{}.csv".format(plot_postfix))

    if cv:
        # Use cross validation to select performance of methods
        fold_results_list = []
        for i in range(len(performance_per_dataset)):
            fold_results = _run_for_fold(i, performance_per_dataset, baseline_algorithm, maximize_metric)
            fold_results_list.append(fold_results)

        # Set up results
        res_dict = {}
        config_dict = {}
        for res in fold_results_list:
            for method, config, score in res:
                # Store score
                if method not in res_dict:
                    res_dict[method] = []

                res_dict[method].append(score)

                # Store config
                if method not in config_dict:
                    config_dict[method] = []

                config_dict[method].append(config)
    else:
        raise NotImplementedError

    # Performance data
    filtered_res = pd.DataFrame(res_dict, index=performance_per_dataset.index)
    filtered_res.columns.name = "Ensemble Technique"

    # Efficiency data
    eff_pd = analysis_utils.get_efficiency_data(ppf)

    # Filter
    filtered_ppf = _filter_ppf(config_dict, performance_per_dataset.index, eff_pd)
    res_ppf = _filter_ppf(config_dict, performance_per_dataset.index, ppf)

    # Save results to file
    if score_type == "Val Score":
        res_ppf["val_score"] = res_ppf["validation_loss"].apply(lambda x: 1 - x)

    std_pd = analysis_utils.transpose_means(
        analysis_utils.get_std_over_cross_validation(res_ppf,
                                                     "val_score" if score_type == "Val Score" else col_metric_name))
    mean_pd = analysis_utils.transpose_means(
        analysis_utils.get_mean_over_cross_validation(res_ppf,
                                                      "val_score" if score_type == "Val Score" else col_metric_name))

    overview_pd = mean_pd.round(4).astype(str) + " (± " + std_pd.round(4).astype(str) + ")"
    for index_name in mean_pd.index:
        sel_mask = np.isclose(mean_pd.loc[index_name][mean_pd.loc[index_name].idxmax()], mean_pd.loc[index_name])
        overview_pd.loc[index_name, sel_mask] = r"\textbf{" + overview_pd.loc[index_name, sel_mask] + "}"
    overview_pd.to_csv(f"./out/performance_overview_{metric_name}_{task_type}_{score_type}.csv")

    print({k: Counter(v) for k, v in config_dict.items()})
    plot_res(filtered_res, "SingleBest", maximize_metric, plot_postfix, filtered_ppf)

    return res_ppf


def run(benchmark_name, eval_name, allowed_preselection=("SiloTopN", "TopN"),
        ens_eval_config=None, use_val_loss=False, method_subsets=None):
    performance_per_fold_per_dataset = pd.read_csv(f"./results/{benchmark_name}_{eval_name}_fold_results.csv")

    # -- Preprocess the names
    baseline_name = "SingleBest_0"
    performance_per_fold_per_dataset = _preprocess_ensemble_names(performance_per_fold_per_dataset)
    if method_subsets is not None:
        performance_per_fold_per_dataset = performance_per_fold_per_dataset[
            performance_per_fold_per_dataset["Method"].isin(method_subsets)]

    # -- Define what algorithms to evaluate
    performance_per_fold_per_dataset = performance_per_fold_per_dataset[
        performance_per_fold_per_dataset["Setting"].isin(allowed_preselection)]
    first_ap = "SiloTopN" if "SiloTopN" in allowed_preselection else "TopN"
    second_ap = "TopN" if "SiloTopN" in allowed_preselection else "SiloTopN"
    baseline_algorithm = f"{first_ap}.{baseline_name}"
    methods_to_drop = [f"{second_ap}.{baseline_name}"]

    # -- Preprocess the data
    performance_per_fold_per_dataset = _filter_ens(performance_per_fold_per_dataset, methods_to_drop, ens_eval_config,
                                                   baseline_algorithm, allowed_preselection)
    performance_per_fold_per_dataset = _handle_nan_values(performance_per_fold_per_dataset, baseline_algorithm)
    print("Time in years for predicting and fitting all configurations:",
          (performance_per_fold_per_dataset["fit_time"].sum()
           + performance_per_fold_per_dataset["predict_time"].sum()) / 60 / 60 / 24 / 365)

    # - Add task metadata
    performance_per_fold_per_dataset = performance_per_fold_per_dataset.merge(
        pd.read_csv(f"./results/{benchmark_name}_metatask_analysis_results.csv"),
        left_on=["TaskID", "Setting"], right_on=["TaskID", "Setting"],
        how="inner", validate="many_to_one"
    )

    # - Define evaluation metric
    col_metric_name = _get_metric_name(benchmark_name)
    metric_name = col_metric_name

    if use_val_loss:
        col_metric_name = "validation_loss"
        performance_per_fold_per_dataset = performance_per_fold_per_dataset[
            performance_per_fold_per_dataset["validation_loss"] != -1]

    # - Split data per task type
    results_per_task = _get_results_per_task(performance_per_fold_per_dataset)

    for task_type, ppf in results_per_task:
        score_type = "Val Score" if use_val_loss else "Test Score"
        res_ppf = run_for_task_type(task_type, ppf, score_type, metric_name, col_metric_name, baseline_algorithm)


def _run():
    Path("./out").mkdir(exist_ok=True)

    to_anal_data = [
        ("bacc_benchmark", "ensemble_evaluations_qdo",
         dict(use_val_loss=False,
              method_subsets=["SingleBest", "GES", "QDO-ES", "QO-ES"])),
        ("roc_benchmark", "ensemble_evaluations_qdo",
         dict(use_val_loss=False,
              method_subsets=["SingleBest", "GES", "QDO-ES", "QO-ES"])),
        ("bacc_benchmark", "ensemble_evaluations_qdo",
         dict(use_val_loss=True,
              method_subsets=["SingleBest", "GES", "QDO-ES", "QO-ES"])),
        ("roc_benchmark", "ensemble_evaluations_qdo",
         dict(use_val_loss=True,
              method_subsets=["SingleBest", "GES", "QDO-ES", "QO-ES"]))
    ]

    for bm_name, eval_name, args in to_anal_data:
        run(bm_name, eval_name, **args)


if __name__ == "__main__":
    _run()
