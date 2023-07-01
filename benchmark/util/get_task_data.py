import openml
import json
import os
import sys
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], "../.."))
from assembled_ask.util.metatask_base import get_metatask

capture_task_data = False
benchmark_name = "benchmark_0"


def get_openml_benchmark_suite_task_ids():
    return openml.study.get_suite(271).tasks
    # [2073, 3945, 7593, 10090, 146818, 146820, 167120, 168350, 168757, 168784, 168868,
    # 168909, 168910, 168911, 189354, 189355, 189356, 189922, 190137, 190146, 190392,
    # 190410, 190411, 190412, 211979, 211986, 359953, 359954, 359955, 359956, 359957,
    # 359958, 359959, 359960, 359961, 359962, 359963, 359964, 359965, 359966, 359967,
    # 359968, 359969, 359970, 359971, 359972, 359973, 359974, 359975, 359976, 359977,
    # 359979, 359980, 359981, 359982, 359983, 359984, 359985, 359986, 359987, 359988,
    # 359989, 359990, 359991, 359992, 359993, 359994, 360112, 360113, 360114, 360975]


task_ids = get_openml_benchmark_suite_task_ids()
print("Number of Tasks:", len(task_ids))

task_dict = {}
for idx, task_id in enumerate(task_ids, 1):
    print(idx, "/", len(task_ids))
    print(task_id)
    openml_task = openml.tasks.get_task(task_id)
    ds = openml_task.get_dataset()
    dataset, _, cat_indicator, feature_names = ds.get_data()
    openml_task.split = openml_task.download_split()
    mt = get_metatask(task_id)

    qual = ds.qualities
    res_dict = {
        "n_instances": qual["NumberOfInstances"],
        "n_classes": qual["NumberOfClasses"],
        "n_features": qual["NumberOfFeatures"],
        "n_cat_features": qual["NumberOfSymbolicFeatures"],
        "dataset_name": ds.name,
        "class_labels": mt.class_labels,
    }
    task_dict[task_id] = res_dict

if capture_task_data:
    print(task_dict)

    file_path = Path(os.path.dirname(os.path.abspath(__file__))).parent / "output" / benchmark_name / "task_data.json"

    with open(file_path, 'w') as f:
        json.dump(task_dict, f)
