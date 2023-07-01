def get_qdo_config_space(seed_function_individual_fold):
    from ConfigSpace import ConfigurationSpace, Categorical, EqualsCondition, InCondition, OrConjunction

    cs = ConfigurationSpace(
        name="evaluation_parameters_qdo_benchmark",
        seed=42,
        meta={"seed_function_individual_fold": seed_function_individual_fold, "rng_seed": 315185350},
    )

    # -- Basic Space
    hp_methods = Categorical("method", ["SingleBest", "EnsembleSelection", "QDOEnsembleSelection"])

    # -- EnsembleSelection Parameters
    hp_use_best = Categorical("use_best", [True, False])
    cond_1 = EqualsCondition(hp_use_best, hp_methods, "EnsembleSelection")

    # -- QDOEnsembleSelection Parameters
    hp_batch_size = Categorical("batch_size", [20, 40])
    cond_batch_size = EqualsCondition(hp_batch_size, hp_methods, "QDOEnsembleSelection")

    hp_archive_type = Categorical("archive_type", ["sliding", "quality"])
    cond_3 = EqualsCondition(hp_archive_type, hp_methods, "QDOEnsembleSelection")

    # - Diversity Space (only valid for quality diversity search method)
    hp_behavior_space = Categorical("behavior_space", ["bs_configspace_similarity_and_loss_correlation"])
    cond_4 = InCondition(hp_behavior_space, hp_archive_type, ["sliding"])

    hp_buffer_ratio = Categorical("buffer_ratio", [1.])
    cond_17 = InCondition(hp_buffer_ratio, hp_archive_type, ["sliding"])

    # Must be squared numbers to guarantee equal contribution between the two dimensions.
    hp_max_elites = Categorical("max_elites", [16, 49])
    cond_5 = EqualsCondition(hp_max_elites, hp_methods, "QDOEnsembleSelection")

    hp_emitter_init = Categorical("emitter_initialization_method", ["AllL1", "RandomL2Combinations", "L2ofSingleBest"])
    cond_6 = EqualsCondition(hp_emitter_init, hp_methods, "QDOEnsembleSelection")

    hp_emitter_starting_step_size = Categorical("starting_step_size", [1])
    cond_7 = EqualsCondition(hp_emitter_starting_step_size, hp_methods, "QDOEnsembleSelection")

    hp_emitter_elite_selection_method = Categorical("elite_selection_method",
                                                    ["deterministic", "tournament", "combined_dynamic"]
                                                    )
    cond_8 = EqualsCondition(hp_emitter_elite_selection_method, hp_methods, "QDOEnsembleSelection")

    # -- Crossover Stuff
    crossover_choices = ["two_point_crossover", "average", "no_crossover"]
    hp_emitter_crossover = Categorical("crossover", crossover_choices)
    cond_9 = EqualsCondition(hp_emitter_crossover, hp_methods, "QDOEnsembleSelection")

    hp_crossover_probability = Categorical("crossover_probability", [0.5])
    cond_10 = InCondition(hp_crossover_probability, hp_emitter_crossover, crossover_choices)

    hp_crossover_probability_dynamic = Categorical("crossover_probability_dynamic", [True])
    cond_11 = EqualsCondition(hp_crossover_probability_dynamic, hp_crossover_probability, 0.5)

    hp_mutation_probability_after_crossover = Categorical("mutation_probability_after_crossover", [0.5])
    cond_12 = InCondition(hp_mutation_probability_after_crossover, hp_emitter_crossover, crossover_choices)

    hp_mutation_probability_after_crossover_dynamic = Categorical("mutation_probability_after_crossover_dynamic",
                                                                  [True])
    cond_13 = EqualsCondition(hp_mutation_probability_after_crossover_dynamic, hp_mutation_probability_after_crossover,
                              0.5)

    # -- Construct ConfigSpace
    cs.add_hyperparameters([
        hp_methods, hp_use_best, hp_archive_type, hp_behavior_space, hp_emitter_init,
        hp_emitter_elite_selection_method, hp_max_elites, hp_emitter_starting_step_size, hp_buffer_ratio,
        hp_batch_size,

        # Crossover
        hp_emitter_crossover, hp_crossover_probability, hp_crossover_probability_dynamic,
        hp_mutation_probability_after_crossover, hp_mutation_probability_after_crossover_dynamic,

    ])
    cs.add_conditions([
        cond_1, cond_3, cond_4, cond_5, cond_6, cond_7, cond_8, cond_batch_size,

        # Crossover
        cond_9, cond_10, cond_11, cond_12, cond_13, cond_17
    ])

    return cs


def get_config_space(benchmark_name, return_grid=False):
    from ConfigSpace.util import generate_grid

    def seed_function_individual_fold(seed, fold_str):
        return seed + int(fold_str)

    if benchmark_name == "QDO":
        cs = get_qdo_config_space(seed_function_individual_fold)
    else:
        raise ValueError(f"Unknown Config Space. Got: {benchmark_name}")

    if return_grid:
        return cs, generate_grid(cs)

    return cs


def get_name_grid_mapping(benchmark_name):
    import json
    import os
    from pathlib import Path

    file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    with open(file_path / f"name_grid_mapping_{benchmark_name}.json", "r") as f:
        name_grid_mapping = json.load(f)

    return name_grid_mapping


# -- utils
def _config_to_unique_name(config):
    ks = sorted(list(config.keys()))
    ks.remove("method")
    filtered_ks = [k for k in ks if ((not isinstance(config[k], bool)) or (config[k]))]
    string = str(config["method"])
    for k in filtered_ks:
        if isinstance(config[k], bool):
            string += f"|{k}"
        else:
            string += f"|{k}:{config[k]}"

    return string


def _create_name_grid_map(b_n, grid_cs):
    import json

    name_map = {_config_to_unique_name(conf): dict(conf) for conf in grid_cs}

    with open(f"name_grid_mapping_{b_n}.json", "w") as outfile:
        outfile.write(json.dumps(name_map))


def _run_local():
    b_n = "QDO"

    out_cs, grid_cs = get_config_space(b_n, return_grid=True)
    print(out_cs)
    print(len(grid_cs))
    _create_name_grid_map(b_n, grid_cs)


if __name__ == "__main__":
    _run_local()
