import numpy as np
import pandas as pd

def read_results_10_runs(basefolder):

    # instantiate collector
    coverage_error_all, empirical_coverage_all = pd.DataFrame(), pd.DataFrame()
    RMSE_all, MAPE_all, APL_all, AACE_all = [], [], [], []

    # for the 10 training (with different random seed)
    for i in range(1, 11):

        # read results from training stats
        train_stats, test_stats, _, _ = read_results(basefolder + "/OutFiles" + str(i) + "/training_stats.csv")

        # compute the coverage error
        coverage_error_i = 100*test_stats.coverage - test_stats.coverage.index.values
        empirical_coverage_all["Run"+str(i)] = 100*test_stats.coverage
        coverage_error_all["Run"+str(i)] = coverage_error_i

        # all stats
        RMSE_all.append(test_stats.RMSE)
        MAPE_all.append(100*test_stats.MAPE)
        APL_all.append(test_stats.APL)
        AACE_all.append(np.mean(np.abs(coverage_error_i)))

    # compute means and se of relevant quantities
    se = lambda l: np.std(l)/np.sqrt(len(l))
    df_results = np.array((np.mean(MAPE_all), se(MAPE_all), np.mean(RMSE_all), se(RMSE_all),
                           np.mean(APL_all),  se(APL_all), np.mean(AACE_all), se(AACE_all),
                           np.mean(empirical_coverage_all.loc[90]), se(empirical_coverage_all.loc[90]),
                           np.mean(empirical_coverage_all.loc[95]), se(empirical_coverage_all.loc[95])))
    coverage_results = pd.concat((coverage_error_all.mean(axis=1), coverage_error_all.std(axis=1)/np.sqrt(10)), axis=1)
    coverage_results.columns = ["mean", "se"]

    return df_results, coverage_results


def read_results(filepath):

    # instantiate ad-hoc classes to collect the results
    train = ResultsClass()
    test = ResultsClass()

    training_stats = pd.read_csv(filepath)

    # Number of Epochs and TrainingTime
    numEpochs = training_stats.iloc[-31].values[0].split()[0]
    trainingTime = int(training_stats.iloc[-30].values[0].split()[4])

    # statistics & coverage (training-set)
    stats = training_stats.iloc[-27:-24].values.ravel()
    backtest = training_stats.iloc[-24:-14].values.ravel()

    train.RMSE, train.MAPE = float(stats[0].split()[1]), float(stats[0].split()[4])
    train.APL, train.AWS = float(stats[1].split()[1]), float(stats[2].split()[1])
    train.set_coverage([float(b.split()[3]) for b in backtest])

    # statistics & coverage (test-set)
    stats = training_stats.iloc[-13:-10].values.ravel()
    backtest = training_stats.iloc[-10:].values.ravel()

    test.RMSE, test.MAPE = float(stats[0].split()[1]), float(stats[0].split()[4])
    test.APL, test.AWS = float(stats[1].split()[1]), float(stats[2].split()[1])
    test.set_coverage([float(b.split()[3]) for b in backtest])

    return train, test, numEpochs, trainingTime


class ResultsClass:

    def __init__(self):
        self.coverage = pd.Series(None, index=np.arange(90, 100), dtype=float)
        self.MAPE = None
        self.RMSE = None
        self.APL = None

    def set_coverage(self, vec):
        for i in range(len(vec)):
            self.coverage.iloc[i] = vec[i]
