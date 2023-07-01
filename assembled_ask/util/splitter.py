from sklearn.model_selection import StratifiedShuffleSplit
from autosklearn.evaluation.splitter import CustomStratifiedShuffleSplit
import copy


def ask_holdout_split(y, train_size):
    """ Split the data like auto-sklearn would split the data but keep track of it.

    Returns fold indicator array.
    """

    test_size = float("%.4f" % (1 - train_size))

    try:
        cv = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=0,
        )
        test_cv = copy.deepcopy(cv)
        next(test_cv.split(y, y))
    except ValueError as e:
        if "The least populated class in y has only" in e.args[0]:
            cv = CustomStratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=0,
            )
        else:
            raise e

    return cv
