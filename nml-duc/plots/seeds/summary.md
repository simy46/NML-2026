# Multi-seed aggregate summary (n_seeds = 5)

- seeds: [0, 1, 2, 3, 4]
- epochs aggregated: 50

## final-epoch loss
- train: 3.1985  (95% CI [3.1861, 3.2108])
- val:   3.2525  (95% CI [3.0977, 3.4072])

## best val loss
- per seed: [3.2276, 3.2638, 3.3146, 3.3826, 3.0485]
- aggregate: 3.2474  (95% CI [3.0915, 3.4033])

## per-horizon ADE (val, end of training)
- step  1: 0.476  [95% CI 0.357, 0.594]
- step  2: 0.350  [95% CI 0.324, 0.376]
- step  3: 0.516  [95% CI 0.506, 0.525]
- step  4: 0.752  [95% CI 0.740, 0.764]
- step  5: 1.037  [95% CI 1.018, 1.057]
- step  6: 1.339  [95% CI 1.315, 1.362]
- step  7: 1.655  [95% CI 1.629, 1.681]
- step  8: 1.992  [95% CI 1.965, 2.019]
- step  9: 2.346  [95% CI 2.321, 2.370]
- step 10: 2.714  [95% CI 2.691, 2.737]
- step 11: 3.097  [95% CI 3.073, 3.120]
- step 12: 3.492  [95% CI 3.464, 3.519]

## per-entity ADE (val, end of training)
- Team_A: 1.4447  (95% CI [1.4313, 1.4582])
- Ball: 3.7089  (95% CI [3.6184, 3.7994])
- Team_B: 1.4369  (95% CI [1.4217, 1.4521])