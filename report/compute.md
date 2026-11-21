# Compute log (Explorer HPC)

Usage log required by the responsible-use rules (ROADMAP 4.2.1, rule 7). Source: `sacct -X -u $USER -S 2026-11-16`; earlier jobs are listed under the table. GPU-hours are elapsed time × GPUs allocated.

## Jobs

| Job | Phase | Partition | GPU | CPUs, memory | State | Elapsed | GPU-hours | Notes |
|---|---|---|---|---|---|---|---|---|
| 11916105 | 4 | gpu-short | T4 | 4, 16G | Cancelled while pending | 0:00:00 | 0.00 | superseded by review fixes before it started |
| 11916731 | 4 | gpu-short | T4 | 4, 16G | Cancelled while pending | 0:00:00 | 0.00 | resubmitted after clearing stale runs |
| 11916966 | 4 | gpu-short | T4 | 4, 16G | Failed | 0:44:50 | 0.75 | the 4 runs finished; the resume demo failed (checkpoint loaded onto the GPU; fixed in `6d1c2b6`, with a CUDA resume test) |
| 11921436 | 4 | gpu-short | T4 | 4, 16G | Completed | 0:54:10 | 0.90 | official sanity runs, all 15 checks pass (`report/phase4/bc_sanity.json`) |
| 11958505 | 5 | short | none | 2, 8G | Failed | 0:01:21 | 0.00 | p = 0.2 dev mixtures built, 49/49 builder checks; failed by an over-strict manifest byte check (data hashes matched; check fixed) |
| 11958506 | 5 | gpu-short | A100 | 8, 32G | Cancelled (dependency) | 0:00:00 | 0.00 | smoke test, cancelled because 11958505 failed |
| 11958589 | 5 | gpu-short | A100 | 8, 32G | Completed | 0:04:50 | 0.08 | smoke test: BC 6.39, CORL 10.83, IQL 12.05 ms/update alone; two IQL runs packed 14.1 each |
| 11981886 | 5 | gpu | A100 | 8, 32G | Completed (7 tasks) | 0:36:59 to 1:25:21 | 7.09 | validation, 14 runs at 300k, two per A100 |
| 11981887 | 5 | short | none | 2, 16G | Failed (exit 1) | 0:00:52 | 0.00 | validation analysis: 37/38 checks, V1 failed (IQL 85.3 vs. BC 105.1 at p = 1) |
| 12014197 | 5 | short | none | 2, 8G | Failed (copy step) | 0:06:39 | 0.00 | p = 1 dev mixtures s102-s104, 27/27 builder checks; outputs removed by 12014258's cleanup before its own copy (submitted by an SSH command thought killed) |
| 12014258 | 5 | short | none | 2, 8G | Failed (refused) | 0:00:29 | 0.00 | duplicate submission; refused on 12014197's untracked manifests, but its copy holds 12014197's complete outputs |
| 12014512 | 5 | short | none | 1, 4G | Completed | 0:00:33 | 0.00 | check: copied manifests and audit match the s102-s104 mixtures in scratch (content and data hashes) |
| 12014822 | 5 | gpu | A100 | 8, 32G | Cancelled (3 tasks) | 1:07:50 to 1:56:27 | 4.82 | V1 follow-up, seeds 102-104; storage stalls left the GPUs near 0% (Research Computing idle-GPU notice for task 1), cancelled by the user; no result used |
| 12031087 | 5 | gpu-short | T4 | 4, 8G | Failed (tests) | 0:02:59 | 0.05 | test gate at `228a7b9`: 238 passed, 1 failed (a new test assumed a CPU run records no GPU; on a GPU node it records the node's GPU; test fixed) |
| 12031088 | 5 | gpu-short | T4 | 8, 8G | Cancelled (dependency) | 0:00:00 | 0.00 | T4 timing test for the follow-up, cancelled because 12031087 failed |
| 12031926 | 5 | gpu-short | T4 | 4, 8G | Completed | 0:02:28 | 0.04 | test gate at `22af6c9`: 240 passed (CUDA tests included) |
| 12031927 | 5 | gpu-short | T4 | 8, 8G | Completed | 0:03:21 | 0.06 | T4 timing test, ours and CORL packed, 10k steps: ours 16.26, CORL 12.97 ms/update; eval 1.41 and 1.67 ms per env step; sizes `iql_seeds.sbatch` at 2:40 |

Earlier: Phase 0 smoke job 11863650 (Tesla T4, 2026-11-13), a few minutes; not itemized.

**Total from 2026-11-16: 1.80 GPU-hours (T4) and 11.99 GPU-hours (A100).**

The A100 (user's choice for Phase 5) is not faster per update than Phase 4's T4 for these small networks (BC 6.39 vs. 2.58 ms): the step is bound by kernel launches and CPU overhead, not GPU compute. Packing helps: two IQL runs on one A100 each slow by about 17%.

## Measured throughput (Phase 6 input)

From `report/phase4/bc_sanity.json`, Tesla T4, one run at a time, 4 CPUs:

| Quantity | Value |
|---|---|
| BC update (batch 256, 2x256 MLP) | 2.58 ms |
| Evaluation, 10 episodes | 7.4 to 9.6 s |
| Evaluation, 50 episodes | 46.6 to 48.1 s |
| One 100k-step BC run (15 evaluations of 10 episodes, 5 of 50, logging, checkpoints) | about 10 minutes |
| Whole job (tests, mixture build, 4 runs, resume demo of about 1.1 runs, analysis) | 54 minutes |

Per 100k steps, updates take about 4.3 minutes and evaluations about 6 minutes (15 × 10 episodes plus 5 × 50), so evaluation on the CPU, not the GPU, dominates the run time. The GPU is mostly idle at one run per job; packing several runs per GPU (ROADMAP 4.1, Phase 6) is the main lever.
