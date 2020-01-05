import math

import numpy as np


class Stats:

    def __init__(self, num_measures, confidence):
        self._num_measures = num_measures

        self._runs = []

        self._confidence = confidence
        self._means = np.zeros((num_measures,))
        self._std_devs = np.zeros((num_measures,))
        self._errors = np.zeros((num_measures,))

    def update(self, run):
        self._runs.append(run)
        num_runs = len(self._runs)
        runs = np.array(self._runs)
        for measure in range(self._num_measures):
            column = runs[:, measure]
            self._means[measure] = column.mean()
            self._std_devs[measure] = column.std()
            self._errors[measure] = Stats.z_table(self._confidence) * (self._std_devs[measure] / math.sqrt(num_runs))

    @property
    def means(self):
        return self._means

    @property
    def errors(self):
        return self._errors

    @property
    def num_runs(self):
        num_runs = len(self._runs)
        return num_runs

    @staticmethod
    def z_table(confidence):
        return {
            0.99: 2.576,
            0.95: 1.96,
            0.90: 1.645
        }[confidence]
