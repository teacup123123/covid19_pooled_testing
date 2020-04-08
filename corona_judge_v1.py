import numpy as np
import numpy.random as rd
import matplotlib.pyplot as pl


def entropy_bernouille(p):
    if isinstance(p, float) or isinstance(p, int):
        return -np.log2(p) * p - np.log2(1 - p) * (1 - p) if p != 0 and p != 1 else 0
    else:
        result = np.zeros(p.shape)
        undefined = np.bitwise_and(p != 0., p != 1.)
        p_ = p[undefined]
        result[undefined] = -np.log2(p_) * p_ - np.log2(1 - p_) * (1 - p_)
        return result


class proba_vec(np.ndarray):
    """
    A child class inherited from numpy float array, that updates the apparent entropy automatically
    """

    def __init__(self, shp):
        super().__init__()
        self._entropy = np.zeros(self.shape)

    def __setitem__(self, key, value):
        super(proba_vec, self).__setitem__(key, value)
        old_entropy = self._entropy[key]
        new_entropy = entropy_bernouille(value)
        self._entropy.__setitem__(key, new_entropy)
        new_entropy = self._entropy[key]


def dataset1():
    probas = proba_vec(20000)
    probas[:] = 0.001
    probas[:int(0.1 * len(probas))] = 0.02
    probas[:int(0.05 * len(probas))] = 0.1
    probas[:int(0.01 * len(probas))] = 0.2
    probas[:int(0.005 * len(probas))] = 0.5
    return probas


mix_and_test_calls = 0
original_proba = []
answer = []
pool_sizes = []


def generate_hidden_answers(probas):
    global answer, mix_and_test_calls,original_proba
    original_proba = proba_vec(len(probas))
    original_proba[:] = probas[:]
    mix_and_test_calls = 0
    answer = np.array([rd.random() < p * (1 + 0.0 * rd.randn()) for p in probas], dtype=bool)
    pool_sizes.clear()


def mix_and_test(sublist):
    global mix_and_test_calls
    mix_and_test_calls += 0
    safe_status = [0 if infected else 1 for infected in answer[sublist]]
    positive = 1 - np.product(safe_status)
    pool_sizes.append(len(sublist))
    return positive == 1, len(safe_status) - sum(safe_status)


def diagnostics(submission):
    global mix_and_test_calls
    mix_and_test_calls += 1
    print(f'{sum(answer)}/{len(answer)} infected')
    print(f'sanity check: {sum(answer != submission)} wrong results ')
    print(f'speedup:{len(submission) / mix_and_test_calls}')
    print(f'mean entropy lost per test: -{sum(entropy_bernouille(original_proba))/mix_and_test_calls}')

    pl.figure()
    pl.plot(range(len(pool_sizes)), pool_sizes)
    pl.ylabel('pool size')
    pl.title(f'pool size as a function of time')
    pl.xlabel('number of pools tested')
