"""CORL-style reward scale (return_reward_range: close an episode at a terminal or at 1000 steps,
over the file's transition order) vs. the mixture's own reward_scale. Read-only."""

import sys

import numpy as np

for path in sys.argv[1:]:
    z = np.load(path)
    rew, term = z["rew"].astype(np.float64), z["done"].astype(bool)
    rets, r, n = [], 0.0, 0
    for i in range(len(rew)):
        r += rew[i]
        n += 1
        if term[i] or n == 1000:
            rets.append(r)
            r, n = 0.0, 0
    rets.append(r)
    corl = 1000 / (max(rets) - min(rets))
    ours = float(z["reward_scale"])
    print(f"{path.split('/')[-1]}: ours {ours:.5f}, corl-style {corl:.5f}, ratio {corl / ours:.4f}")
