# Run `cv_analysis.py` first to generate the results we need here.

import pandas as pd
import json
import numpy as np
import optuna
from benchmark.evaluation.analysis_utils import normalize_performance
from benchmark.evaluation.evaluations import normalized_improvement_boxplot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.cbook import boxplot_stats
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
name_file_list = [
    ("Binary Balanced Accuracy",
     "performance_per_dataset_for Task Type Binary Classification and Metric balanced_accuracy [Test Score].csv"),
    ("Multi-class Balanced Accuracy",
     "performance_per_dataset_for Task Type Multi-class Classification and Metric balanced_accuracy [Test Score].csv"),
    ("Binary ROC AUC",
     "performance_per_dataset_for Task Type Binary Classification and Metric roc_auc [Test Score].csv"),
    ("Multi-class ROC AUC",
     "performance_per_dataset_for Task Type Multi-class Classification and Metric roc_auc [Test Score].csv")
]

HP_VALUES = {
    "preprocessing": ["SiloTopN", "TopN"],
    "batch_size": [20, 40],
    "max_elites": [16, 49],
    "emitter_initialization_method": ["AllL1", "L2ofSingleBest", "RandomL2Combinations"],
    "elite_selection_method": ["deterministic", "tournament", "combined_dynamic"],
    "crossover": ["two_point_crossover", "average", "no_crossover"]
}


def data_for_map(method_name, name_grid_mapping):
    prefix, x = method_name.split(".", 1)
    method, config_id = x.rsplit("_", 1)
    config = name_grid_mapping[f"config_{config_id}"].copy()
    config["preprocessing"] = prefix
    config["method"] = method
    return config


def _plot(normalized_ppd, plot_postfix, xlim=-2):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_df = pd.melt(normalized_ppd)
    plot_df.columns = ["Ensemble Technique", "value"]
    outlier = plot_df.groupby("Ensemble Technique").apply(
        lambda x: sum(boxplot_stats(x).pop(0)['fliers'] < xlim)).to_dict()
    sns.boxplot(data=plot_df, y="Ensemble Technique", x="value",
                showfliers=False)
    sns.stripplot(data=plot_df, y="Ensemble Technique", x="value", color="black")

    ax.axvline(x=-1, c="red")
    plt.xlabel("Normalised Improvement")
    yticks = [item.get_text() for item in ax.get_yticklabels()]
    new_yticks = [ytick + f" [{outlier[ytick]}]" for ytick in yticks]
    ax.set_yticklabels(new_yticks)
    plt.xlim(xlim, 0)
    plt.legend(handles=[Line2D([0], [0], label="SingleBest", color="r")])
    plt.ylabel("Hyperparameter Value")
    plt.tight_layout()

    Path("../out/ablation_study").mkdir(exist_ok=True)
    plt.savefig("../out/ablation_study/hp_box_plot_{}.pdf".format(plot_postfix))


data_col_names = ["preprocessing", "batch_size", "archive_size", "init_method", "sampling_method", "crossover_method"]

optuna.logging.set_verbosity(optuna.logging.ERROR)


def run():
    with open("../results/name_grid_mapping.json", "r") as f:
        name_grid_mapping = json.load(f)

    # Compute average normalized performance per parameter value
    for hp, values in HP_VALUES.items():

        hp_df_qdo = []
        hp_df_qo = []
        col_n = []

        for name, file in name_file_list:
            print("####", name)
            df = pd.read_csv(f"../out/{file}")

            col_map = {c: data_for_map(c, name_grid_mapping) for c in list(df)[1:]}
            qdo_mask = np.array([True] + [c["method"] == "QDO-ES" for c in col_map.values()])
            qo_mask = np.array([True] + [c["method"] == "QO-ES" for c in col_map.values()])

            for m_n, mask in [("QDO", qdo_mask)]:
                tmp_df = df.loc[:, mask]
                tmp_df["SiloTopN.SingleBest_0"] = df["SiloTopN.SingleBest_0"]

                norm_df = normalize_performance(tmp_df.iloc[:, 1:], "SiloTopN.SingleBest_0", True).drop(
                    columns=["SiloTopN.SingleBest_0"])

                # Compute average normalized performance per parameter value
                for hp_value in values:
                    # Find all configs with this values
                    hp_v_confs = [c for c in list(norm_df) if col_map[c][hp] == hp_value]
                    hp_df_qdo.append(norm_df[hp_v_confs].mean(axis=1).median())
                    col_n.append(f"{hp_value}-{name}")

            for m_n, mask in [("QO", qo_mask)]:
                tmp_df = df.loc[:, mask]
                tmp_df["SiloTopN.SingleBest_0"] = df["SiloTopN.SingleBest_0"]

                norm_df = normalize_performance(tmp_df.iloc[:, 1:], "SiloTopN.SingleBest_0", True).drop(
                    columns=["SiloTopN.SingleBest_0"])

                # Compute average normalized performance per parameter value
                for hp_value in values:
                    # Find all configs with this values
                    hp_v_confs = [c for c in list(norm_df) if col_map[c][hp] == hp_value]
                    hp_df_qo.append(norm_df[hp_v_confs].mean(axis=1).median())

        print(values)
        print(col_n)
        print("QDO", [round(x, 2) for x in hp_df_qdo])
        print("QO", [round(x, 2) for x in  hp_df_qo])


if __name__ == "__main__":
    run()
