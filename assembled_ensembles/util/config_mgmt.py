from assembled_ensembles.default_configurations import ens_ensemble_selection, ens_other


def get_ensemble_switch_case_config(config, rng_seed=None, metric=None, n_jobs=None,
                                    is_binary=None, labels=None):
    method = config["method"]

    if method == "SingleBest":
        return ens_other.customSingleBest(metric=metric)
    elif method == "EnsembleSelection":
        return ens_ensemble_selection._factory_es(rng_seed, metric, n_jobs, config["use_best"])
    elif method == "QDOEnsembleSelection":
        # QDO
        return ens_ensemble_selection._factory_qdo(
            rng_seed, metric, is_binary, labels, n_jobs, config["archive_type"], config["behavior_space"],
            config["max_elites"], config["emitter_initialization_method"], config["starting_step_size"],
            config["elite_selection_method"], config["crossover"], config["crossover_probability"],
            config["crossover_probability_dynamic"], config["mutation_probability_after_crossover"],
            config["mutation_probability_after_crossover_dynamic"], config["negative_steps"],
            config["weight_random_elite_selection"], config["weight_random_step_selection"],
            config["buffer_ratio"], config["dynamic_updates_consider_rejections"], config["batch_size"]
        )
    else:
        raise ValueError(f"Unknown method! Got: {method}")
