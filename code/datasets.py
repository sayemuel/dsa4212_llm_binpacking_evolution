"""Fixed, seeded instance sets. Search sees only TRAIN; VALID picks the final
program from a run's top candidates; TEST sets are never seen during search."""
from binpack import weibull_instance, uniform_instance, triplet_instance

# Seeds are disjoint across splits.
TRAIN = [weibull_instance(1000, s) for s in range(0, 5)]
VALID = [weibull_instance(1000, s) for s in range(100, 105)]

TEST = {
    # in-distribution, larger instances (FunSearch's Weibull 5k setting)
    "weibull_5k": [weibull_instance(5000, s) for s in range(1000, 1005)],
    "weibull_10k": [weibull_instance(10000, s) for s in range(2000, 2003)],
    # smaller instances, same distribution
    "weibull_200": [weibull_instance(200, s) for s in range(3000, 3020)],
    # distribution shift: OR-Library binpack1-4 style U{20..100}, C=150
    "uniform_500": [uniform_instance(500, s) for s in range(4000, 4010)],
    # distribution shift: Weibull sizes but larger bins (more items per bin)
    "weibull_c150": [weibull_instance(1000, s, capacity=150) for s in range(5000, 5005)],
    # adversarial structure: perfect triplets, OPT = n/3 exactly
    "triplet_501": [triplet_instance(167, s) for s in range(6000, 6010)],
}
