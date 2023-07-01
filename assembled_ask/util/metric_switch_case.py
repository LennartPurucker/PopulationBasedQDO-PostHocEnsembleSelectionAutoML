from functools import partial

from autosklearn.metrics import balanced_accuracy, make_scorer

from sklearn.metrics import f1_score, log_loss
from assembled_ask.util.custom_metrics.roc_auc import roc_auc_score


def msc(metric_name, is_binary, labels):
    if metric_name == "balanced_accuracy":
        return balanced_accuracy
    elif metric_name == "f1":
        return make_scorer("f1", partial(f1_score, zero_division=0, average="macro"))
    elif metric_name == "roc_auc":
        if is_binary:
            return make_scorer("roc_auc", roc_auc_score,
                               needs_threshold=True)
        else:
            return make_scorer("roc_auc", partial(roc_auc_score, average="macro",
                                                  multi_class="ovr", labels=labels),
                               needs_proba=True)
    elif metric_name == "log_loss":
        return make_scorer('log_loss', log_loss, optimum=0,
                           worst_possible_result=2 ** 31 - 1,
                           greater_is_better=False,
                           needs_proba=True)
    else:
        raise ValueError("Unknown metric name: {}".format(metric_name))
