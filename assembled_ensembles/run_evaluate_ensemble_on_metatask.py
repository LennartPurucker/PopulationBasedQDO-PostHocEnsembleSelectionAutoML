import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))  # have to do this because of singularity. I hate it

from pathlib import Path

from assembled.metatask import MetaTask
from assembled.ensemble_evaluation import evaluate_ensemble_on_metatask

from assembled_ensembles.util.config_mgmt import get_ensemble_switch_case_config
from assembled_ensembles.default_configurations.supported_metrics import msc

from assembled_ensembles.configspaces.evaluation_parameters_grid import get_config_space, get_name_grid_mapping
from ConfigSpace import Configuration

if __name__ == "__main__":
    # -- Get Input Parameter
    openml_task_id = sys.argv[1]
    pruner = sys.argv[2]  # "TopN", "SiloTopN"
    ensemble_method_name = sys.argv[3]
    metric_name = sys.argv[4]
    benchmark_name = sys.argv[5]
    evaluation_name = sys.argv[6]
    isolate_execution = sys.argv[7] == "yes"
    load_method = sys.argv[8]
    folds_to_run_on = sys.argv[9]
    config_space_name = sys.argv[10]
    ens_save_name = sys.argv[11]

    if folds_to_run_on == "-1":
        folds_to_run_on = None
        state_ending = ""
    else:
        folds_to_run_on = [int(folds_to_run_on)]
        state_ending = f"_{folds_to_run_on}"

    # Default to -1
    n_jobs = -1

    delayed_evaluation_load = True if load_method == "delayed" else False

    # -- Build Paths
    file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    tmp_input_dir = file_path.parent / "benchmark" / "input" / benchmark_name / pruner
    print("Path to Metatask: {}".format(tmp_input_dir))

    out_path = file_path.parent / "benchmark" / "output" / benchmark_name / "task_{}/{}/{}".format(openml_task_id,
                                                                                                   evaluation_name,
                                                                                                   pruner)
    out_path.mkdir(parents=True, exist_ok=True)

    s_path = file_path.parent / "benchmark/state/{}/task_{}/{}".format(benchmark_name, openml_task_id, evaluation_name)
    s_path.mkdir(parents=True, exist_ok=True)
    s_path = s_path / "{}_{}{}.done".format(pruner, ens_save_name, state_ending)
    print("Path to State: {}".format(s_path))

    # -- Rebuild The Metatask
    print("Load Metatask")
    mt = MetaTask()
    mt.read_metatask_from_files(tmp_input_dir, openml_task_id, delayed_evaluation_load=delayed_evaluation_load)

    # -- Setup Evaluation variables
    # Get the metric(s)
    is_binary = len(mt.class_labels) == 2
    # If the ensemble requires the metric, we assume the labels to be encoded
    ens_metric = msc(metric_name, is_binary, list(range(mt.n_classes)))
    # For the final score, we need the original labels
    score_metric = msc(metric_name, is_binary, mt.class_labels)
    predict_method = "predict_proba" if ens_metric.requires_confidences else "predict"

    # -- Handle Config Input
    cs = get_config_space(config_space_name)
    name_grid_mapping = get_name_grid_mapping(config_space_name)
    rng_seed = cs.meta["rng_seed"] if folds_to_run_on is None \
        else cs.meta["seed_function_individual_fold"](cs.meta["rng_seed"], folds_to_run_on[0])

    config = Configuration(cs, name_grid_mapping[ensemble_method_name])
    cs.check_configuration(config)
    technique_run_args = get_ensemble_switch_case_config(config,
                                                         rng_seed=rng_seed, metric=ens_metric, n_jobs=n_jobs,
                                                         is_binary=is_binary, labels=list(range(mt.n_classes)))
    print("Run for Config:", config)

    # -- Run Evaluation
    print("#### Process Task {} for Dataset {} with Ensemble Technique {} ####".format(mt.openml_task_id,
                                                                                       mt.dataset_name,
                                                                                       ensemble_method_name))

    scores = evaluate_ensemble_on_metatask(mt, technique_name=ens_save_name, **technique_run_args,
                                           output_dir_path=out_path, store_results="parallel",
                                           save_evaluation_metadata=True,
                                           return_scores=score_metric, folds_to_run=folds_to_run_on,
                                           use_validation_data_to_train_ensemble_techniques=True,
                                           verbose=True, isolate_ensemble_execution=isolate_execution,
                                           predict_method=predict_method,
                                           store_metadata_in_fake_base_model=True)
    print(scores)
    print("K-Fold Average Performance:", sum(scores) / len(scores))

    print("Storing State")
    s_path.touch()
    print("Done")
