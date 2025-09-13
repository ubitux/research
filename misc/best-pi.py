"""
Find the best approximation of π with one integer division
"""

from math import pi

err = lambda a: (a - pi) ** 2

best = (1, 1)
best_err = err(best[0] / best[1])

digits_num, digits_den = (3, 3)

for i in range(1, 10**digits_num):
    for j in range(1, 10**digits_den):
        e = err(i / j)
        if e < best_err:
            best = i, j
            print(best, i / j)
            best_err = e
