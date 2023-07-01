import pandas as pd
import json

bacc_df = pd.read_csv("../evaluation/results/bacc_benchmark_metatask_analysis_results.csv")
roc_df = pd.read_csv("../evaluation/results/roc_benchmark_metatask_analysis_results.csv")

with open("../setup_data/task_data.json") as f:
    task_json = json.load(f)

with open("../setup_data/roc_benchmark_data.json") as f:
    bm_json = json.load(f)

task_ids = bacc_df["TaskID"].unique()

cols = ["Dataset Name", "OpenML Task ID", "#instances", "#features", "#classes", "Memory (GB)",
        "B - TopN - avg. # base models", "B - SiloTopN - avg. # base models", "R - TopN - avg. # base models",
        "R - SiloTopN - avg. # base models",
        "B - TopN - avg. # distinct algorithms", "B - SiloTopN - avg. # distinct algorithms",
        "R - TopN - avg. # distinct algorithms", "R - SiloTopN - avg. # distinct algorithms",
        ]
rows = []
for task_id in task_ids:
    row = [task_json[str(task_id)]["dataset_name"], task_id,
           task_json[str(task_id)]["n_instances"], task_json[str(task_id)]["n_features"],
           task_json[str(task_id)]["n_classes"]]

    # Same for roc and bacc
    row.append(bm_json["framework_memory_gbs"]
               + bm_json["special_cases"]["framework_memory_gbs"].get(str(task_id), 0))

    tmp_b = bacc_df[(bacc_df["TaskID"] == task_id) & (bacc_df["Setting"] == "TopN")][
        ["mean_base_models_count", "mean_distinct_algorithms_count"]]
    tmp_b_s = bacc_df[(bacc_df["TaskID"] == task_id) & (bacc_df["Setting"] == "SiloTopN")][
        ["mean_base_models_count", "mean_distinct_algorithms_count"]]

    tmp_r = roc_df[(roc_df["TaskID"] == task_id) & (roc_df["Setting"] == "TopN")][
        ["mean_base_models_count", "mean_distinct_algorithms_count"]]
    tmp_r_s = roc_df[(roc_df["TaskID"] == task_id) & (roc_df["Setting"] == "SiloTopN")][
        ["mean_base_models_count", "mean_distinct_algorithms_count"]]

    row.extend([
        tmp_b.iloc[0][0], tmp_b_s.iloc[0][0], tmp_r.iloc[0][0], tmp_r_s.iloc[0][0],
        tmp_b.iloc[0][1], tmp_b_s.iloc[0][1], tmp_r.iloc[0][1], tmp_r_s.iloc[0][1]
    ])
    rows.append(row)

res = pd.DataFrame(rows, columns=cols)
res.to_csv("../evaluation/out/data_overview.csv", index=False)
