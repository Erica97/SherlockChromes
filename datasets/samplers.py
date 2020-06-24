import numpy as np
import os
import pandas as pd
import random

from collections import defaultdict
from itertools import chain

class StratifiedSampler(object):
    """Sample through class stratified strategy."""

    def __call__(self, data, test_batch_proportion=0.1):
        labels = data.labels

        train_idx, val_idx, test_idx = [], [], []

        for i in np.unique(labels):
            idx = list(np.where(labels == i)[0])
            random.shuffle(idx)

            n = len(idx)
            n_test = int(n * test_batch_proportion)
            n_train = n - 2 * n_test

            train_idx+= idx[:n_train]
            val_idx+= idx[n_train:(n_train + n_test)]
            test_idx+= idx[(n_train + n_test):]

        random.shuffle(train_idx)
        random.shuffle(val_idx)
        random.shuffle(test_idx)

        return train_idx, val_idx, test_idx

class StratifiedSubSampler(object):
    """Subsample in stratified manner such that there are equal amounts of
    positive and negative labelled chromatograms."""

    def __call__(self, data, test_batch_proportion=0.1):
        labels = data.labels

        train_idx, val_idx, test_idx = [], [], []

        n_test_positive, n_train_positive = 0, 0

        for i in [np.int64(1), np.int64(0)]:
            idx = list(np.where(labels == i)[0])
            random.shuffle(idx)

            if i == np.int64(1):
                n = len(idx)
                n_test = int(n * test_batch_proportion)
                n_train = n - 2 * n_test
                n_test_positive = n_test
                n_train_positive = n_train
            else:
                n_test = n_test_positive
                n_train = n_train_positive

            train_idx+= idx[:n_train]
            val_idx+= idx[n_train:(n_train + n_test)]
            test_idx+= idx[(n_train + n_test):]

        random.shuffle(train_idx)
        random.shuffle(val_idx)
        random.shuffle(test_idx)

        return train_idx, val_idx, test_idx

class LoadingSampler(object):
    """Load pre-existing idx numpy txt files."""

    def __init__(self, **kwargs):
        self.dt = kwargs['dt'] if 'dt' in kwargs else 'float'
        self.filenames = [
            os.path.join(kwargs['root_path'], filename)
            for filename in kwargs['filenames']]
        self.shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else False

    def __call__(self, data=None, test_batch_proportion=None):
        if self.dt == 'float':
            self.dt = np.float32
        elif self.dt == 'int':
            self.dt = np.int32
            
        idxs = []
        for filename in self.filenames:
            idxs.append(list(np.loadtxt(filename, dtype=self.dt)))

        if self.shuffle:
            for idx in idxs:
                random.shuffle(idx)

        return idxs

class GroupBySequenceSampler(object):
    """Sample by sequence groups. 
    Naked sequences are sequences without modifications."""

    def __init__(self, **kwargs):
        self.naked = kwargs['naked']

    def __call__(self, data, test_batch_proportion=0.1):
        seq_to_idx = defaultdict(list)

        for i in range(len(data)):
            seq = data.chromatograms.iloc[i, 1].split('_')[-2]

            if self.naked:
                mod_start = seq.rfind('(')
                mod_end = seq.find(')')

                if mod_start != -1 and mod_end != -1:
                    seq = seq[:mod_start] + seq[mod_end + 1:]

            seq_to_idx[seq].append(i)

        grouped_idx = [seq_to_idx[seq] for seq in seq_to_idx]

        random.shuffle(grouped_idx)

        n = len(grouped_idx)
        n_test = int(n * test_batch_proportion)
        n_train = n - 2 * n_test

        train_idx = list(chain.from_iterable(grouped_idx[:n_train]))
        val_idx = list(
            chain.from_iterable(grouped_idx[n_train:(n_train + n_test)]))
        test_idx = list(chain.from_iterable(grouped_idx[(n_train + n_test):]))

        random.shuffle(train_idx)
        random.shuffle(val_idx)
        random.shuffle(test_idx)

        return train_idx, val_idx, test_idx

class GroupByRunSampler(object):
    """Sample by mass spec runs."""

    def __init__(self, **kwargs):
        self.total_runs = kwargs['total_runs']

    def __call__(self, data, test_batch_proportion=0.1):
        run_to_idx = defaultdict(list)

        for i in range(len(data)):
            run = data.chromatograms.iloc[i, 1].split('_')[0:-3]

            run_to_idx[run].append(i)

        assert len(run_to_idx) == self.total_runs

        grouped_idx = [run_to_idx[run] for run in run_to_idx]

        random.shuffle(grouped_idx)

        n = len(grouped_idx)
        n_test = int(n * test_batch_proportion)
        n_train = n - 2 * n_test

        train_idx = list(chain.from_iterable(grouped_idx[:n_train]))
        val_idx = list(
            chain.from_iterable(grouped_idx[n_train:(n_train + n_test)]))
        test_idx = list(chain.from_iterable(grouped_idx[(n_train + n_test):]))

        random.shuffle(train_idx)
        random.shuffle(val_idx)
        random.shuffle(test_idx)

        return train_idx, val_idx, test_idx
