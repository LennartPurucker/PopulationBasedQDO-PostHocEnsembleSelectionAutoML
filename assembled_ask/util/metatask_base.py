import numpy as np
from sklearn.model_selection import StratifiedKFold
from assembled.metatask import MetaTask
from assembledopenml.openml_assembler import init_dataset_from_task


def get_example_manual_metatask_for_ask() -> MetaTask:
    print("Get Toy Metatask")
    from sklearn.datasets import load_breast_cancer
    metatask_id = -1
    task_data = load_breast_cancer(as_frame=True)
    target_name = task_data.target.name
    dataset_frame = task_data.frame
    class_labels = np.array([str(x) for x in task_data.target_names])  # cast labels to string
    feature_names = task_data.feature_names
    cat_feature_names = []
    dataset_frame[target_name] = class_labels[dataset_frame[target_name].to_numpy()]
    metatask = MetaTask()
    fold_indicators = np.empty(len(dataset_frame))
    cv_spliter = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for fold_idx, (train_index, test_index) in enumerate(
            cv_spliter.split(dataset_frame[feature_names], dataset_frame[target_name])):
        # This indicates the test subset for fold with number fold_idx
        fold_indicators[test_index] = fold_idx

    metatask.init_dataset_information(dataset_frame, target_name=target_name, class_labels=class_labels,
                                      feature_names=feature_names, cat_feature_names=cat_feature_names,
                                      task_type="classification", openml_task_id=metatask_id,
                                      dataset_name="breast_cancer.csv", folds_indicator=fold_indicators)
    return metatask


def get_openml_metatask_for_ask(mt_id) -> MetaTask:
    print("Get OpenML Metatask")

    metatask = MetaTask()
    init_dataset_from_task(metatask, mt_id)
    metatask.read_randomness("OpenML", 0)

    return metatask


def get_metatask(openml_task_id):
    if openml_task_id == "-1":
        mt = get_example_manual_metatask_for_ask()
    else:
        mt = get_openml_metatask_for_ask(openml_task_id)

    return mt
