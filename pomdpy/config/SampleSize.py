import operator as op
from functools import reduce
import math
from collections import namedtuple

def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


# borrowed from: https://github.com/veekaybee/data/blob/master/samplesize.py
confidence_level_constant = {50: 0.67, 68: 0.99, 90: 1.64, 95: 1.96, 99: 2.57}


# CALCULATE THE SAMPLE SIZE
def sample_size(population_size, confidence_level, confidence_interval):
    p = 0.5
    e = confidence_interval / 100.0
    N = population_size

    # LOOP THROUGH SUPPORTED CONFIDENCE LEVELS AND FIND THE NUM STD
    # DEVIATIONS FOR THAT CONFIDENCE LEVEL
    Z = confidence_level_constant.get(confidence_level, -1)
    assert Z > 0, "SUPPORTED CONFIDENCE LEVELS: 50%, 68%, 90%, 95%, and 99%"
    # CALC SAMPLE SIZE
    n_0 = ((Z ** 2) * p * (1 - p)) / (e ** 2)

    # ADJUST SAMPLE SIZE FOR FINITE POPULATION
    n = n_0 / (1 + ((n_0 - 1) / float(N)))

    return int(math.ceil(n))  # THE SAMPLE SIZE


def MapStat(grid,confidence_interval = 1.0):
    rows = grid.size[1]
    cols = grid.size[0]
    num_obs = grid.obstacle
    population_sz = ncr(rows * cols, num_obs) * rows * cols
    confidence_level = 95


    sample_sz = sample_size(population_sz, confidence_level, confidence_interval)

    return {"num_states":population_sz, "particles":sample_sz}
    # print("POPULATION SIZE %d SAMPLE SIZE: %d" % (population_sz, sample_sz))
if __name__ == '__main__':

    Grid = namedtuple("Grid","name obstacle size dir")
    Grid1 = Grid(name='ral-3-12-1', obstacle=1, size=[4, 13], dir="./pomdpy/config/ral-3-12-1.txt")
    Grid2 = Grid(name='ral-3-12-2', obstacle=2, size=[4, 13], dir="./pomdpy/config/ral-3-12-2.txt")
    Grid3 = Grid(name='ral-3-12-3', obstacle=3, size=[4, 13], dir="./pomdpy/config/ral-3-12-3.txt")
    Grid6 = Grid(name='map-14-4-6', obstacle=6, size=[4, 14], dir="./pomdpy/config/map-14-4-6.txt")
    Grid7 = Grid(name='map-14-4-7', obstacle=7, size=[4, 14], dir="./pomdpy/config/map-14-4-7.txt")

    res =MapStat(Grid7)
    print("POPULATION SIZE %d SAMPLE SIZE: %d" % (res["num_states"], res["particles"]))

