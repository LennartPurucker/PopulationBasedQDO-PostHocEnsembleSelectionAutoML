import sys
import os
import glob
import logging
import pathlib
import json
import pickle
import time
import warnings
import hashlib
import numpy as np
import pynisher
import math
from heapq import heappush, heappop
from shutil import rmtree
from joblib import cpu_count

import autosklearn.classification
from autosklearn.metrics import Scorer, _PredictScorer

from sklearn.model_selection import PredefinedSplit, StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder

from typing import Union, Optional, List
from tables import NaturalNameWarning

from assembled.metatask import MetaTask

from assembled_ask.util.splitter import ask_holdout_split

# -- Logging
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger("AskAssembler")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# -- Ignore Ask and Sklearn Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=NaturalNameWarning)


class AskAssembler:
    """ Assembler to build Metatasks from Auto-Sklearn Data

    Parameters
    ----------
    metatask: MetaTask
        Metatask for which we want to collect predictor data
    tmp_output_dir: str, Path
        Path to the dir where ask's output shall be stored
    folds_to_run: List[int], default=None
        Which outer folds of the metatak's split to run
    resampling_strategy: str in ["cv", "holdout"]
        Define the re-sampling strategy.
            If "holdout", we do holdout validation with a 66:33 split.
            If "cv", we do 5 fold cross-validation.
        For classification, we do splits in a stratified fashion.
    save_disc_space: bool, default=True
        If true, we delete already processed data from auto-sklearn as much as possible.
    """

    def __init__(self, metatask: MetaTask, tmp_output_dir: Union[str, pathlib.Path],
                 folds_to_run: Optional[List[int]] = None, resampling_strategy: str = "holdout",
                 save_disc_space: bool = True):

        self.metatask = metatask
        self.tmp_output_dir = pathlib.Path(tmp_output_dir)
        self.folds_to_run = folds_to_run if folds_to_run is not None else [i for i in range(self.metatask.max_fold + 1)]
        self.resampling_strategy = resampling_strategy
        self.save_disc_space = save_disc_space

        self.classification = True
        self.ask_model_choices = 16
        self.config_key_for_model_type = "classifier:__choice__"

    def get_resampling_strategy(self, X_train, y_train):
        """Get the resampling strategy appropriate for our framework and auto-sklearn."""

        fold_indicator = np.full(len(X_train), -1)

        if self.resampling_strategy == "holdout":
            cv = ask_holdout_split(y_train, 0.67)
        elif self.resampling_strategy == "cv":
            cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        else:
            raise ValueError("Unknown resampling_strategy. Got: {}".format(self.resampling_strategy))

        for fold_idx, (_, val_i) in enumerate(cv.split(np.zeros(len(y_train)), y_train)):
            fold_indicator[val_i] = fold_idx

        return PredefinedSplit(fold_indicator)

    def run_ask(self, metric_to_optimize: Scorer, time_limit: int, memory_limit: int,
                max_models_on_disc: Optional[int] = None):
        """Run ASK on a Metatask and store the search results in a specific way

        Parameters
        ----------
        metric_to_optimize: Auto-sklearn Scorer
            Metric that is optimized
        time_limit: int
            Time in hours for the search
        memory_limit: int
            Memory in GB for the search
        max_models_on_disc: int, default=None
            The number of models to keep on disc (to save disc space). If None, all models are kept.
        """
        logger.info("Run Ask on Data from Metatask")
        self._verify_run_environment()

        for iter_idx, (fold_idx, X_train, X_test, y_train, y_test) in enumerate(
                self.metatask._exp_yield_data_for_base_model_across_folds(self.folds_to_run), 1):
            logger.info("### Processing Fold {} | {}/{} ###".format(fold_idx, iter_idx, len(self.folds_to_run)))
            resampling_strategy = self.get_resampling_strategy(X_train, y_train)
            tmp_folder = self.tmp_output_dir.joinpath("fold_{}".format(fold_idx))

            memory = max(memory_limit * 1024 / cpu_count(), 3072)
            automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=int(time_limit * 60 * 60),
                                                                      memory_limit=memory,
                                                                      n_jobs=-1,
                                                                      metric=metric_to_optimize,
                                                                      tmp_folder=tmp_folder,
                                                                      delete_tmp_folder_after_terminate=False,
                                                                      ensemble_size=0,
                                                                      max_models_on_disc=max_models_on_disc,
                                                                      load_models=False,
                                                                      resampling_strategy=resampling_strategy
                                                                      )
            logger.info("Start Search")
            automl.fit(X_train, y_train, dataset_name=self.metatask.dataset_name)

            logger.info("Search finished")
            self._fix_logger_after_ask()
            logger.info(automl.sprint_statistics())

            # --- Fold Specific Info about prediction data
            ensemble_val_indices = np.array([ele for sublist in
                                             [val_i for train_i, val_i in list(resampling_strategy.split())]
                                             for ele in sublist])
            os.mkdir(tmp_folder.joinpath(".ask_assembler"))
            np.save(tmp_folder.joinpath(".ask_assembler/ensemble_val_indices"), ensemble_val_indices)

        # --- Add Default Selection Constraints
        logger.info("Finished Run Ask on Data from Metatask")

    def set_constraints(self, metric_name, time_limit, memory_limit, max_models_on_disc, ensembling, refit):
        self.metatask.selection_constraints["manual"] = True
        self.metatask.selection_constraints["autosklearn"] = {
            "metric": metric_name,
            "time_limit": time_limit,
            "memory_limit": memory_limit,
            "ensembling": ensembling,
            "max_models_on_disc": max_models_on_disc,
            "refit": refit
        }

    def collect_predictor_data_from_ask_data(self, refit=False):
        """Collect data on predictors from the search results of ask

        Includes getting validation data and re-fitting/fitting the evaluated models if enabled.
        """
        logger.info("Start collecting predictor data from ask data...")
        self._verify_collect_predictor_environment()
        ask_errors_ = {fold_idx: {"refit_and_predict_errors": {}} for fold_idx in self.folds_to_run}

        for iter_idx, (fold_idx, X_train, X_test, y_train, y_test) in enumerate(
                self.metatask._exp_yield_data_for_base_model_across_folds(self.folds_to_run), 1):
            logger.info("### Processing Fold {} | {}/{} ###".format(fold_idx, iter_idx, len(self.folds_to_run)))

            # --- Parse File Structure
            base_dir = self.tmp_output_dir.joinpath("fold_{}".format(fold_idx))
            ask_dir = base_dir.joinpath(".auto-sklearn")
            bm_dir_names = [os.path.basename(run_folder) for run_folder in
                            glob.glob(str(ask_dir.joinpath("runs/1_*")))]

            # --- Get Relevant Data from Files
            ensemble_val_indices = np.load(base_dir.joinpath(".ask_assembler/ensemble_val_indices.npy"),
                                           allow_pickle=True)
            ensemble_y = np.load(ask_dir.joinpath("true_targets_ensemble.npy"), allow_pickle=True)
            val_indices = np.array(X_train.index[ensemble_val_indices])
            with open(base_dir.joinpath("smac3-output/run_1/runhistory.json"), "r") as f:
                run_history = json.load(f)["data"]

            # --- Parse y_test
            classes_, y_train_int = np.unique(y_train, return_inverse=True)
            # Save classes for the current fold
            np.save(base_dir.joinpath(".ask_assembler/classes_.npy"), classes_)

            # --- Parse X_train for auto-sklearn
            # Fix Problem with dtypes for columns that only contain none (must be cat columns)
            only_na_cols = X_train.columns[X_train.isna().apply(lambda x: sum(x) == X_train.shape[0])]
            X_train.loc[:, only_na_cols] = X_train[only_na_cols].apply(lambda x: x.astype("category"))

            # --- Some Verification
            self._verify_resampling_strategy(y_train_int[ensemble_val_indices], ensemble_y)

            # --- Get Predictor Data
            n_base_models = len(bm_dir_names)
            for bm_idx, bm_identifier in enumerate(bm_dir_names, 1):
                # -- Handle IDs
                ask_run_id = bm_identifier.split("_")[1]
                logger.info("# Processing Base Model {} | {}/{} #".format(ask_run_id, bm_idx, n_base_models))

                # -- Skippy Dummy Model
                if ask_run_id == "1":
                    continue
                smac_run_id = int(ask_run_id) - 1

                # -- Get data and model from files
                model_dir = ask_dir.joinpath("runs/{}".format(bm_identifier))
                path_to_y_pred = model_dir.joinpath("predictions_ensemble_{}.npy".format(bm_identifier))
                val_y_pred = np.load(path_to_y_pred, allow_pickle=True)
                bm_model = np.load(model_dir.joinpath("{}.model".format(bm_identifier.replace("_", "."))),
                                   allow_pickle=True)
                bm_config = bm_model.config._values
                model_evaluated_time = path_to_y_pred.stat().st_mtime

                # -- Re-Fit (if enabled) and Predict for Test Data
                val_model_time = run_history[smac_run_id - 1][1][1]
                try:
                    if refit:
                        logger.info("Start Re-Fit")
                        limited_fit = pynisher.enforce_limits(wall_time_in_s=math.ceil(val_model_time) * 2)(
                            _fit_wrapper)
                        st = time.time()
                        bm_model = limited_fit(bm_model, X_train, y_train_int)
                        fit_time = time.time() - st

                        if bm_model is None:
                            raise ValueError("Fit timeout!")

                    else:
                        logger.info("Skip Re-Fit")
                        fit_time = val_model_time

                    logger.info("Start Predict")
                    st = time.time()
                    if self.classification:
                        test_y_pred = bm_model.predict_proba(X_test)
                    else:
                        test_y_pred = bm_model.predict(X_test)
                    predict_time = time.time() - st

                except Exception as e:
                    # Something about (auto-)sklearn is broken and re-fit does not work
                    # save for later  and skip this base model as a result of the bug
                    logger.info(str(e))
                    ask_errors_[fold_idx]["refit_and_predict_errors"][ask_run_id] = (str(e), bm_config)
                    continue

                # -- Store Data
                logger.info("Store Predictor Data")
                self._store_fold_predictors(fold_idx, ask_run_id, bm_config, val_y_pred, val_indices, test_y_pred,
                                            fit_time, predict_time, model_evaluated_time)

            # -- Store fold errors
            logger.info("Found the following Errors with Auto-sklearn: {}".format(ask_errors_[fold_idx]))
            with open(base_dir.joinpath(".ask_assembler/ask_errors.json"), 'w') as f:
                json.dump(ask_errors_[fold_idx], f)

            logger.info("Finished collecting predictor data from ask data.")

            if self.save_disc_space:
                logger.info("Start cleaning ASK disc space.")
                rmtree(ask_dir)
                logger.info("Finished cleaning ASK disc space.")

    def build_metatask_from_predictor_data(self, pruner=None, metric=None):
        """Build a metatask from stored prediction data

        Parameters
        ----------
        pruner: str in ["TopN", "SiloTopN"] or None, default=None
            Filters the predictors before returning the metatask.
        metric: Callable
            Metric used to determine performance for filter predicotrs.

        Returns
        -------
        metatask
        """

        logger.info("Start Building Metatask from predictor data...")
        self._verify_collect_predictor_environment()

        # ----- Determine which models to load into metatask
        if pruner is not None:
            # Input validation
            if pruner not in ["TopN", "SiloTopN"]:
                raise ValueError("filter_predictors is a wrong value. Got: {}".format(pruner))
            if metric is None:
                raise ValueError("Require a metric to filter predictors!")

            logger.info("Pre-Filter Predictors")

            eval_results = self.evaluate_existing_predictors(metric)

            to_load_predictors_per_fold = {f: self.filter_predictors(f_e_d, pruner)
                                           for f, f_e_d in eval_results.items()}

            # Report / Handle Removal Results
            for f, ask_ids_to_keep in to_load_predictors_per_fold.items():
                n_all = len(eval_results[f])
                if not ask_ids_to_keep:
                    # Not enough base models, break and stop here.
                    logger.info(
                        "No enough predictors exist. Fold: {}, N_Predictor: {}".format(f, n_all))
                    logger.info("Stopping Metatask Builder. No need to build a Metatask for this dataset.")
                    return None

                logger.info("Removed {} predictors due to Pruner Settings for fold {}".format(
                    n_all - len(ask_ids_to_keep), f))
        else:
            # --- Get all IDs that exist
            logger.info("Get all Ask Run IDs from Disc.")
            to_load_predictors_per_fold = {}
            for iter_idx, fold_idx in enumerate(self.folds_to_run, 1):
                assembler_dir = self.tmp_output_dir.joinpath("fold_{}/.ask_assembler".format(fold_idx))
                if not assembler_dir.exists():
                    raise ValueError("No Predictor Data Exists for Fold {}!".format(fold_idx))

                fold_ask_ids = [os.path.basename(run_folder).split("_")[1].split(".")[0] for run_folder in
                                glob.glob(str(assembler_dir.joinpath("prediction_data/model_*.pkl")))]

                to_load_predictors_per_fold[fold_idx] = fold_ask_ids

        # ----- Given the IDs from before, load data and build metatask
        for iter_idx, (fold_idx, ask_ids_to_load) in enumerate(to_load_predictors_per_fold.items(), 1):
            logger.info("### Processing Fold {} | {}/{} ###".format(fold_idx, iter_idx, len(self.folds_to_run)))

            # --- Parse File Structure
            assembler_dir = self.tmp_output_dir.joinpath("fold_{}/.ask_assembler".format(fold_idx))
            n_to_load = len(ask_ids_to_load)

            if self.classification:
                classes_ = np.load(assembler_dir.joinpath("classes_.npy"), allow_pickle=True)
            else:
                classes_ = None

            # --- Iterate over base models
            for inner_iter_idx, ask_run_id in enumerate(ask_ids_to_load, 1):
                logger.info("# Processing Prediction Data for ASK Run {} | {}/{} #".format(ask_run_id, inner_iter_idx,
                                                                                           n_to_load))

                # --- Load Prediction Data
                predictor_name, predictor_description, test_y_pred, test_confs, validation_data \
                    = self._load_predictor_data_for_metatask(fold_idx, ask_run_id, classes_=classes_)

                # -- Add data to metatask
                self.metatask.add_predictor(predictor_name, test_y_pred, confidences=test_confs,
                                            conf_class_labels=list(classes_),
                                            predictor_description=predictor_description,
                                            validation_data=validation_data,
                                            fold_predictor=True, fold_predictor_idx=fold_idx)

            # --- Add fold-specific configuration to metatask
            h_k = "config_space_for_fold"
            smac_dir = self.tmp_output_dir.joinpath("fold_{}/smac3-output/run_1/configspace.json".format(fold_idx))
            # Set variable if new
            if h_k not in self.metatask._custom_meta_data_container:
                self.metatask._custom_meta_data_container[h_k] = dict()

            # Read and store config space
            with open(smac_dir) as f:
                self.metatask._custom_meta_data_container[h_k][str(fold_idx)] = json.load(f)

        logger.info("Finished Building Metatask from predictor data.")
        return self.metatask

    def evaluate_existing_predictors(self, metric):
        """ Get the score and other values needed to filter predictors.
        """
        eval_results = {f: [] for f in self.folds_to_run}
        # Iterate over all predictors and compute score
        for iter_idx, fold_idx in enumerate(self.folds_to_run, 1):
            logger.info("### Pre-Processing Fold {} | {}/{} ###".format(fold_idx, iter_idx, len(self.folds_to_run)))

            # --- Parse File Structure
            base_dir = self.tmp_output_dir.joinpath("fold_{}".format(fold_idx))
            assembler_dir = base_dir.joinpath(".ask_assembler")
            if not assembler_dir.exists():
                raise ValueError("No Predictor Data Exists!")

            pred_data_dir_names = [os.path.basename(run_folder) for run_folder in
                                   glob.glob(str(assembler_dir.joinpath("prediction_data/model_*.pkl")))]
            n_pred_data = len(pred_data_dir_names)

            if self.classification:
                classes_ = np.load(assembler_dir.joinpath("classes_.npy"), allow_pickle=True)
                le_ = LabelEncoder().fit(classes_)

            else:
                classes_ = None
                le_ = None

            # --- Iterate over base models
            for pred_d_idx, pred_d_identifier in enumerate(pred_data_dir_names, 1):
                # -- Handle IDs
                ask_run_id = pred_d_identifier.split("_")[1].split(".")[0]
                logger.info("# Evaluating Prediction Data for ASK Run {} | {}/{} #".format(ask_run_id, pred_d_idx,
                                                                                           n_pred_data))

                # --- Load Prediction Data
                predictor_name, predictor_description, test_y_pred, test_confs, validation_data \
                    = self._load_predictor_data_for_metatask(fold_idx, ask_run_id, classes_=classes_)

                # --- Get data used to filter predictors
                y_ture_val = self.metatask.ground_truth.iloc[validation_data[0][-1]]

                if isinstance(metric, _PredictScorer):
                    loss = abs(metric._optimum - metric(y_ture_val, validation_data[0][1]))
                else:
                    loss = abs(metric._optimum - metric(le_.transform(y_ture_val), validation_data[0][2]))

                model_type = predictor_description["config"][self.config_key_for_model_type]
                eval_time = predictor_description["model_evaluated_time"]
                filter_data = (loss, model_type, eval_time)

                # --- Save
                eval_results[fold_idx].append((ask_run_id, filter_data))

        return eval_results

    def filter_predictors(self, fold_eval_data, pruner, min_n_predictor=10, top_n=50):
        """

        Parameters
        ----------
        fold_eval_data: (ask_run_id, (filter_data))
            filter_data = (loss, model_type, eval_time)
        """

        if len(fold_eval_data) < min_n_predictor:
            return []

        # Return immediately if less than top n models exit
        if len(fold_eval_data) <= top_n:
            return [ask_id for ask_id, _ in fold_eval_data]

        # start_time = min(predictor_timings.values())
        # stop_time = max(predictor_timings.values())
        predictor_over_time = sorted(fold_eval_data, key=lambda x: x[-1][2])

        # -- Filter Predictors
        if pruner == "TopN":
            top_n_heap = []

            for ask_run_id, (loss, _, _) in predictor_over_time:
                # Negative loss because we want to make sure that greater is always better
                _add_to_top_n(top_n_heap, top_n, (-loss, ask_run_id))

            preds_to_keep = [ask_run_id for _, ask_run_id in top_n_heap]

        elif pruner == "SiloTopN":
            # We would require a dynamic silo filling algorithm that adjusts the possible number of base models per silo
            # depending on the number of different model types seen so far. We leave this for future work to design
            # such an algorithm and only take the "oracle-like" perspective here.
            #   TODO: implement such an algorithm
            #       Concept: while n_models smaller equal top_n, just collect all
            #                once larger than top_n build silos from current state,
            #       Problem: handle case with more silos than top_n; how to balance number of items in silos?

            # Get silos
            cat_to_model = {}
            for ask_run_id, (loss, model_type, _) in predictor_over_time:
                if model_type not in cat_to_model:
                    cat_to_model[model_type] = []

                # Negative loss because we want to make sure that greater is always better
                cat_to_model[model_type].append((-loss, ask_run_id, model_type))

            # Sort silos by performance
            for model_type in cat_to_model.keys():
                cat_to_model[model_type] = sorted(cat_to_model[model_type], key=lambda x: x[0])

            # Get silo min value
            min_silo_val = max(math.floor(top_n / len(cat_to_model.keys())), 1)

            while sum(len(val) for val in cat_to_model.values()) > top_n:
                # Find silos with too many values
                too_large_silos = [k for k, val in cat_to_model.items() if len(val) > min_silo_val]
                if not too_large_silos:
                    break

                # For all these, remove the value with the smallest performance
                #   This wont remove silos entirely, because silos with at least 1 element wont be too large
                to_remove = sorted([vals for k in too_large_silos for vals in cat_to_model[k]], key=lambda x: x[0])[0]
                cat_to_model[to_remove[-1]].remove(to_remove)

            if sum(len(val) for val in cat_to_model.values()) > top_n:
                # In this case, we have more silos than top_n (cat_to_model.keys() > top_n)
                # Moreover, at this point, all silos will only have 1 element in it.
                # To resolve this, we can simply return the top_n models over these silos
                # (other fallbacks like random for more diversity would work as well but we think top is best)
                sort_rest = sorted([val for val_list in cat_to_model.values() for val in val_list], key=lambda x: x[0])
                preds_to_keep = [ask_run_id for _, ask_run_id, _ in sort_rest[-top_n:]]
            else:
                preds_to_keep = [ask_run_id for vals in cat_to_model.values() for _, ask_run_id, _ in vals]

        else:
            raise ValueError("Unknown Pruner: {}".format(pruner))

        return preds_to_keep

    # -- OS Utils
    def _store_fold_predictors(self, fold_idx, ask_run_id, bm_config, val_y_pred, val_indices, test_y_pred,
                               fit_time, predict_time, model_evaluated_time):
        store_dir = self.tmp_output_dir.joinpath("fold_{}/.ask_assembler".format(fold_idx))
        predictor_dir = store_dir.joinpath("prediction_data")
        if not predictor_dir.exists():
            os.mkdir(predictor_dir)

        predictor_data = {
            "bm_config": bm_config,
            "val_y_pred": val_y_pred,
            "val_indices": val_indices,
            "test_y_pred": test_y_pred,
            "fit_time": fit_time,
            "predict_time": predict_time,
            "model_evaluated_time": model_evaluated_time
        }
        with open(predictor_dir.joinpath("model_{}.pkl".format(ask_run_id)), "wb") as f:
            pickle.dump(predictor_data, f)

    def _load_predictor_data_for_metatask(self, fold_idx, ask_run_id, classes_=None):
        store_dir = self.tmp_output_dir.joinpath("fold_{}/.ask_assembler".format(fold_idx)).joinpath(
            "prediction_data")

        with open(store_dir.joinpath("model_{}.pkl".format(ask_run_id)), "rb") as f:
            predictor_data = pickle.load(f)

        config = predictor_data["bm_config"]
        check_sum_for_name = hashlib.md5(str(config).encode('utf-8')).hexdigest()
        predictor_name = config["classifier:__choice__"] + "({})".format(str(check_sum_for_name))
        predictor_description = {
            "auto-sklearn-model": True,
            "config": config,
            "fit_time": predictor_data["fit_time"],
            "predict_time": predictor_data["predict_time"],
            "model_evaluated_time": predictor_data["model_evaluated_time"]
        }

        # Get Predictions
        if classes_ is not None:
            # Classification
            test_confs = predictor_data["test_y_pred"]
            test_y_pred = classes_.take(np.argmax(test_confs, axis=1), axis=0)
            val_confs = predictor_data["val_y_pred"]
            val_y_pred = classes_.take(np.argmax(val_confs, axis=1), axis=0)
            validation_data = [(fold_idx, val_y_pred, val_confs, predictor_data["val_indices"])]
        else:
            # Regression case
            raise NotImplementedError

        # --- Check Prediction Data
        if not np.isfinite(test_confs).all():
            # Try to fix this problem for known problem cases
            if predictor_description["config"][self.config_key_for_model_type] in ["sgd", "passive_aggressive"]:
                logger.info("Fixing Corrupted Confidence Values")
                # Fix by setting confidence for correct class to 1
                bad_rows = list(np.where(~np.isfinite(test_confs).all(axis=1))[0])
                for row in bad_rows:
                    index_correct_class = np.where(classes_ == test_y_pred[row])[0][0]
                    test_confs[row] = 0.
                    test_confs[row, index_correct_class] = 1.
            else:
                raise ValueError("Test Confidences contain non-finite values.")

        if not np.isfinite(validation_data[0][2]).all():
            raise ValueError("Validation Confidences contain non-finite values.")

        return predictor_name, predictor_description, test_y_pred, test_confs, validation_data

    def _verify_run_environment(self):
        # Verify clean env environment
        fold_dir_exists = all([self.tmp_output_dir.joinpath("fold_{}".format(f_idx)).exists()
                               for f_idx in self.folds_to_run])
        if self.tmp_output_dir.exists() and fold_dir_exists:
            raise ValueError("tmp_output_path {} and folds {} already exist. ".format(self.tmp_output_dir,
                                                                                      self.folds_to_run)
                             + "We wont delete it. Make sure to delete it yourself.")

    def _verify_collect_predictor_environment(self):
        if not self.tmp_output_dir.exists():
            raise ValueError("No Auto-Sklearn Output Data exists.")

    def _fix_logger_after_ask(self):
        logging.getLogger().setLevel(logging.DEBUG)  # have to reset after ask code

    def _verify_resampling_strategy(self, sane_ensemble_y, ensemble_y):
        # --- Sanity Check that we know the correct used resampling strategy
        if not np.array_equal(sane_ensemble_y, ensemble_y):
            raise ValueError("We were not able to reproduce ensemble_y. The resampling strategy might be mixed up.")


def _fit_wrapper(bm, X, y):
    return bm.fit(X, y)


def _add_to_top_n(heap, n, heap_item, key=lambda x: x[0]):
    if len(heap) < n:
        heappush(heap, heap_item)
    elif key(heap[0]) < key(heap_item):
        heappop(heap)
        heappush(heap, heap_item)
