import numpy as np
import numpy.random as rd
import matplotlib.pyplot as pl
import corona_judge_v1 as judge
probas = judge.dataset1()
judge.generate_hidden_answers(probas)


def entropy(subset=None):
    return sum(probas._entropy[subset]) if subset else sum(probas._entropy)

rounds = 0
positive_clusters = []
entropies = []
projected = []
pool_sizes = []
awaiting_results = []
parity = 1
while entropy() > 0:
    entropies.append(entropy())
    # entropies = np.array(list(map(entropy_unit, probas)))
    # sorted_i = np.argsort(entropies)
    # sorted_i = np.argsort(probas)
    sorted_i = rd.permutation(len(probas))
    sorted_i = list(sorted_i)
    sorted_i = list(filter(lambda i: 0 < probas[i] < 1, sorted_i))
    awaiting_results.append(len(sorted_i))
    if not sorted_i:
        break

    move = []
    entropy_candidates = []


    # negative_candidates = []

    def P_negative_of_move():
        return np.product(1 - probas[move])


    def entropy_of_move():
        P_negative = P_negative_of_move()
        P_positive = 1 - P_negative
        old_ps = probas[move]
        new_ps_conditionned_on_POSITIVE = 1 / P_positive * old_ps
        _e = sum(judge.entropy_bernouille(p) for p in new_ps_conditionned_on_POSITIVE) * P_positive if len(move) > 1 else 0
        _e -= entropy(move)
        return _e


    parity += 1
    while P_negative_of_move() > 0.2 and len(sorted_i) > 0:
        if len(sorted_i) > 0:
            move.append(sorted_i.pop(-1) if parity % 2 == 0 else sorted_i.pop(0))
            _ = entropy_of_move()
            _n = P_negative_of_move()
            entropy_candidates.append(_)
            # negative_candidates.append(negative_of_move())
            # sublist.append(sorted_i.pop(0))

    Mc = np.argmin(entropy_candidates)
    projected.append(entropy_candidates[Mc])
    move = move[0:Mc + 1]
    P_neg = P_negative_of_move()

    # assert is_approximately_the_same(entropy(sublist), 1)
    result, infected = judge.mix_and_test(move)
    rounds += 1
    if result:
        print(f'\tPOSITIVE {len(move)} tested ({infected} infected)!')
        probas[move] = np.minimum(1., probas[move] / (1 - P_negative_of_move()))
        positive_clusters.append(move)
    else:
        # All in the sublist are all safe
        print(f'\tNEGATIVE {len(move)} tested: remaining tocheck = {len(sorted_i)}')
        for c in positive_clusters:
            for s in move:
                if s in c:
                    c.remove(s)
                    old = np.sum(probas[c])
                    probas[c] = np.minimum(0.95, probas[c] * (1 + probas[s] / old))
                    if len(c) == 1:
                        probas[c] = 1
                        print(f'\t\t exclusion principle {s}=0 => {c}=1')
        positive_clusters = list(filter(lambda l: len(l) > 1, positive_clusters))

        probas[move] = 0

mode = 'random-pick-mode'
t = np.array(range(len(awaiting_results)))
pl.figure()
pl.plot(entropies)
pl.ylim([0, entropies[0] * 1.5])
pl.title(f'Entropy, {mode}')
pl.xlabel('number of pools tested')
pl.ylabel('apparent entropy')
pl.plot(t, entropies[0] - t, '--')
pl.legend([f'{sum(judge.answer)}/20k infected', 'Shannon limit'])

pl.figure()
pl.plot(t, awaiting_results)
pl.ylim([0, 21000])
pl.ylabel('# awaiting result')
pl.title(f'number of testees awaiting result, {mode}')
pl.xlabel('number of pools tested')
pl.show()
