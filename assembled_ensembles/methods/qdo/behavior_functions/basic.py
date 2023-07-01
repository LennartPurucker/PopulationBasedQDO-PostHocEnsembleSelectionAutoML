from assembled_ensembles.methods.qdo.behavior_space import BehaviorFunction
from assembled_ensembles.util.diversity_metrics import LossCorrelation
from functools import partial

# -- Make Behavior Functions
# - Diversity Metrics
LossCorrelationMeasure = BehaviorFunction(partial(LossCorrelation, checks=False), ["y_true", "Y_pred_base_models"],
                                          (0, 1), "proba", name=LossCorrelation.name + "(Lower is more Diverse)")
