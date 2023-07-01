from benchmark.evaluation import evaluations


def performance_analysis(plot_ppd, normalized_ppd, maximize_metric,
                         baseline_algorithm, plot_postfix):
    # -- Start Evaluations
    cd_results = evaluations.cd_evaluation(plot_ppd, maximize_metric, plot_postfix, plot=True)

    # -- hard coded order
    plot_order = ["QO-ES", "QDO-ES", "GES", "SingleBest"]  # list(cd_results.rankdf.index)
    meanrank = cd_results.rankdf["meanrank"]
    normalized_ppd = normalized_ppd[plot_order]

    # Normalized Plot
    evaluations.normalized_improvement_boxplot(normalized_ppd, baseline_algorithm, plot_postfix)

    return None, meanrank
