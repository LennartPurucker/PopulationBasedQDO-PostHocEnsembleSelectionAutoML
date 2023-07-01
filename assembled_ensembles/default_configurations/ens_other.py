def customSingleBest(metric=None, **kwargs):
    # "custom.SingleBest"
    from assembled_ensembles.methods.other.baselines import SingleBest

    return {
        "technique": SingleBest,
        "technique_args": {"metric": metric,
                           "predict_method": "predict_proba"},
        "pre_fit_base_models": True
    }