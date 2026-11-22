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
| 12034720 | 5 | gpu-short | T4 | 4, 8G | Completed | 0:01:59 | 0.03 | test gate at `db9c4c5`: 240 passed |
| 12034721 | 5 | gpu | T4 | 8, 8G | Completed (3 tasks) | 1:10:19 to 1:10:51 | 3.53 | V1 follow-up, seeds 102-104, ours and CORL packed per task, 300k steps; all six runs complete |
| 12048550 | 5 | short | none | 1, 4G | Completed | 0:00:23 | 0.00 | seed follow-up scoring at `a389b60`: proceed (B false, 1 of 3 seeds) |
| 12053724 | 5 | short | none | 2, 16G | Failed (exit 1, V1 miss, as expected) | 0:00:40 | 0.00 | validation reanalysis at `1939b0d` with the checks added in review: same 38 outcomes, no numeric difference from 11981887 |
| 12055647 | 6 | gpu-short | T4 | 4, 8G | Completed | 0:02:16 | 0.04 | test gate at `d037a0d`: 264 passed (CUDA tests included) |
| 12055648 | 6 | short | none | 2, 16G | Completed | 0:05:01 | 0.00 | mixtures: seeds 3, 4 (all p) and dev p 0.05, 0.5 at s100, s101; procedure check (s0 p 0.05, 0.5) and the six existing dev files match their committed hashes; 14 new files installed, existing untouched; 79/79 builder checks for each grid build (17/17 for the s0 procedure check); the six existing s100/s101 manifests stay as committed (the rebuild's copies record other file bytes and build commit, same data) |
| 12055649 | 6 | gpu-short | T4 | 4, 8G | Completed | 0:01:58 | 0.03 | benchmark G1: IQL k=1, 8.97 ms/update, 0.967 ms per eval step |
| 12055650 | 6 | gpu-short | T4 | 8, 12G | Completed | 0:02:10 | 0.04 | benchmark G2: IQL k=2, 9.70 ms/update, 1.152 ms per eval step |
| 12055651 | 6 | gpu-short | T4 | 16, 20G | Completed | 0:02:43 | 0.05 | benchmark G4: IQL k=4, 12.25 ms/update, 1.470 ms per eval step |
| 12055652 | 6 | gpu-short | T4 | 16, 20G | Completed | 0:01:29 | 0.02 | benchmark GB: BC k=4, 4.58 ms/update, 1.330 ms per eval step |
| 12055653 | 6 | gpu-short | V100-PCIE | 8, 12G | Failed | 0:00:47 | 0.01 | benchmark GV: "no kernel image is available" (PyTorch 2.11+cu128 has no V100 kernels); V100 ineligible |
| 12055654 | 6 | gpu-short | V100-PCIE | 16, 20G | Failed | 0:01:17 | 0.02 | benchmark GVB: same failure as GV |
| 12055655 | 6 | short | none | 4, 8G | Completed | 0:02:23 | 0.00 | benchmark C1: IQL k=1 on cascadelake (Xeon Platinum 8276), 10.99 ms/update, 0.366 ms per eval step |
| 12055656 | 6 | short | none | 4, 8G | Completed | 0:01:01 | 0.00 | benchmark CB: BC k=1 on cascadelake (Xeon Platinum 8276), 1.88 ms/update, 0.267 ms per eval step |

Earlier: Phase 0 smoke job 11863650 (Tesla T4, 2026-11-13), a few minutes; not itemized.

**Total from 2026-11-16: 5.54 GPU-hours (T4), 11.99 GPU-hours (A100) and 0.03 GPU-hours (V100).**

The A100 (user's choice for Phase 5) is not faster per update than Phase 4's T4 for these small networks (BC 6.39 vs. 2.58 ms): the step is bound by kernel launches and CPU overhead, not GPU compute. Packing helps: two IQL runs on one A100 each slow by about 17%.

## Phase 6 benchmark and hardware choice (D1, approved by the user 2026-11-21)

Benchmark jobs 12055649 to 12055656 at `d037a0d` (spec Section 5; logs in `report/phase6/benchmark/`): 10,000 steps per run on the development mixture `mix_p1_s100`, development evaluation seeds, `log_every` 1000. Hours per 300k-step run `h = (300,000 u + 800 x 1,000 e) / 3,600`, `e` at the 1,000-step upper bound; `u` and `e` the largest among a job's packed runs. Computed by `scripts/phase6_cost_model.py`; its output is `report/phase6/benchmark/cost_model_output.txt`.

| Config | Hardware | Algorithm, runs per job | CPUs, memory | u (ms/update) | e (ms/eval step) | h per run | Peak memory (MiB) |
|---|---|---|---|---|---|---|---|
| G1 | T4 | IQL, 1 | 4, 8G | 8.97 | 0.967 | 0.96 | 1030 |
| G2 | T4 | IQL, 2 | 8, 12G | 9.70 | 1.152 | 1.06 | 2064 |
| G4 | T4 | IQL, 4 | 16, 20G | 12.25 | 1.470 | 1.35 | 4227 |
| GB | T4 | BC, 4 | 16, 20G | 4.58 | 1.330 | 0.68 | 3753 |
| GV, GVB | V100-PCIE | IQL 2, BC 4 | | failed: the installed PyTorch 2.11.0+cu128 has no kernels for compute capability 7.0 (V100); V100 ineligible in this environment | | | |
| C1 | CPU, cascadelake | IQL, 1 | 4, 8G | 10.99 | 0.366 | 1.00 | 1242 |
| CB | CPU, cascadelake | BC, 1 | 4, 8G | 1.88 | 0.267 | 0.22 | 1226 |

**Validation (within 20% required):** G2's `u` and `e` predict the three archived IQL runs of array 12034721 (packed with CORL on a T4): training -3%, -7%, -4%; evaluation -6%, -7%, -6%, using each run's actual episode lengths (negative = prediction below the measured time; a CORL partner slows a run). Pass. The validation covers the T4 model only: the CPU figures have no archived counterpart and are checked by the dry run.

**Concurrency (`J_eff`):** T4: one node of 28 cores and 4 GPUs, so 4 jobs at 4 cores, 3 at 8, 1 at 16; `gpu` takes 8 submitted jobs, so the 19 to 52 tasks go in 3 to 7 submission waves. CPU: the lowest free-core reading on the 24 cascadelake nodes, 90 idle of 1,344 cores (recorded by the CPU benchmark jobs, 2026-11-21 23:34; 98 at 22:30), gives 22 jobs of 4 cores by total idle cores (20 when packed per node, about 16 when each node's free memory is also counted; the wave counts, 2 for BC and 3 for IQL, are the same for any `J_eff` of 15 or more) under the D3 throttle of 25; `short` shows no submit limit; measured queue wait 50 s from eligible to start (the jobs were held on their dependency until 23:32:55).

**Wall clock for the 70 grid runs (BC 25, IQL 45), BC and IQL counted one after the other:** CPU cascadelake 3.5 h, 0 GPU-hours; T4 with IQL at k = 2 13.3 h (29.2 GPU-hours), k = 1 16.3 h (48.0), k = 4 20.9 h (20.9); V100 ineligible; the T4 figures exclude the human latency per submission wave, which would only lengthen them. **Choice by the pre-fixed rule: CPU on `short`, `--constraint=cascadelake`**, 1 run per job, 4 CPUs, 8G (2G per CPU; measured peak 1.2 GiB), `--time` 1:30 for IQL and 0:20 for BC (h x 1.5, rounded up to 10 minutes), required CPU model `Intel(R) Xeon(R) Platinum 8276 CPU @ 2.20GHz` (both CPU jobs, on nodes d0014 and d0036, recorded it; all 24 cascadelake nodes have the same 56-core layout; a task on any other model is refused by the sweep).

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
