# Vendored code

## `corl_iql.py`: CORL's IQL, the Phase 5 reference

- **Upstream:** [CORL](https://github.com/corl-team/CORL) `algorithms/offline/iql.py` at commit
  `23f8b2e46abf5aef73db268cd1c5a42eb8ef45dc` (2023-08-16), file sha256
  `7cb46b5497877905e60e5458a3325dfa6c7bbe394bafd34251b8895e8c35c880`.
- **License:** Apache-2.0, copied to `LICENSE` in this folder. Upstream has no `NOTICE` file.
- **Content:** selected lines, each block unmodified. Nothing that imports `d4rl`, `gym`, `pyrallis`
  or `wandb` (the `@pyrallis.wrap()` entry point would run at import).

| Upstream lines | Content |
|---|---|
| 1, 2, 3, 9, 13, 15, 16, 17, 19, 20 | source comment and the imports the blocks below need |
| 22 | `TensorBatch` |
| 25 to 27 | `EXP_ADV_MAX`, `LOG_STD_MIN`, `LOG_STD_MAX` |
| 88 to 90 | `soft_update` |
| 255 to 256 | `asymmetric_l2_loss` |
| 259 to 265 | `Squeeze` |
| 268 to 300 | `MLP` |
| 303 to 333 | `GaussianPolicy` |
| 368 to 384 | `TwinQ` |
| 387 to 394 | `ValueFunction` |
| 397 to 537 | `ImplicitQLearning` |

**Assembly:** line 1 of `corl_iql.py` is our notice. The rest is the header lines joined by `\n`, one
blank line, then the blocks in the order above joined by two blank lines, ending in `\n`. Its
sha256, checked by `tests/test_corl.py` (I12):

`679a0ce093bcb7fab578c0a4ee6b000457742105512aa9d173c16769111b7299`

To reproduce from an upstream checkout:

```python
lines = open("iql.py", newline="").read().split("\n")
header = "\n".join(lines[n - 1] for n in (1, 2, 3, 9, 13, 15, 16, 17, 19, 20))
blocks = [(22, 22), (25, 27), (88, 90), (255, 256), (259, 265), (268, 300), (303, 333),
          (368, 384), (387, 394), (397, 537)]
body = header + "\n\n" + "\n\n\n".join("\n".join(lines[a - 1:b]) for a, b in blocks) + "\n"
```

`offrl/algos/corl.py` (ours) adapts `ImplicitQLearning` to the `Trainer` protocol.
