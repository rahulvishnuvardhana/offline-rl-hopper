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
| 12064513 | 6 | gpu-short | T4 | 4, 8G | Timeout | 0:30:09 | 0.50 | test gate at `88cdccc`: the Slurm-lib tests inherited the gate job's own Slurm variables, the sweep test grid's `auto` device picked CUDA, and S7 left out the grid's config files, so its children trained 300k steps; fixed in `2373a24` |
| 12065803 | 6 | gpu-short | T4 | 4, 8G | Failed (tests) | 0:02:58 | 0.05 | test gate at `67c9d94`: 324 passed, 1 failed (a provenance test assumed a CPU run records no GPU; fixed in `f026524` and `a11dc28`) |
| 12067471 | 6 | gpu-short | T4 | 4, 8G | Completed | 0:03:10 | 0.05 | test gate at `a11dc28`: 325 passed; dry-run grids initialised |
| 12068556 | 6 | short | none | 4, 8G | Completed (9 tasks) | 0:01:38 to 0:03:23 | 0.00 | dry run, main IQL bundles 0 to 8 (10k steps) |
| 12068558 | 6 | short | none | 4, 8G | Completed (8 tasks) | 0:01:31 to 0:03:14 | 0.00 | dry run, ablation IQL bundles 0 to 7 |
| 12068562 | 6 | short | none | 4, 8G | Completed (10 tasks) | 0:00:43 to 0:00:59 | 0.00 | dry run, main BC bundles 0 to 9 |
| 12068612 | 6 | short | none | 4, 8G | Cancelled by the user's `scancel` | 0:00:54 | 0.00 | dry run, cancel test (main IQL bundle 9, after its step-2000 checkpoint) |
| 12068733 | 6 | short | none | 4, 8G | Completed | 0:01:18 | 0.00 | dry run, resume of bundle 9 from step 2000 |
| 12068758 | 6 | short | none | 4, 8G | Completed (2 tasks) | 0:00:07 | 0.00 | dry run, skip-only resubmission of bundles 0 and 9 |
| 12068768, 12068769 | 6 | short | none | 4, 8G | Completed; Failed (exit 4, as expected) | 0:02:26; 0:00:12 | 0.00 | dry run, lock test: the second job found the run locked and exited 4 without training |
| 12068835 | 6 | short | none | 2, 4G | Completed | 0:00:40 | 0.00 | dry run, `sweep --check` and `--list` on both grids |
| 12072515 | 6 | short | none | 2, 4G | Completed | 0:00:24 | 0.00 | dry-run evidence collected to `~/archive/` (`report/phase6/dryrun/`), with `sweep --check` rerun |
| 12072591 | 6 | short | none | 2, 4G | Completed | 0:01:27 | 0.00 | spec Section 4 archive: the 25 final-grid and 4 new development mixtures copied to `~/archive/phase6_mixtures` (2.9 GB, `SHA256SUMS` match the source); `check_mixture_files.py` 29/29 against the committed manifests |
| 12072875 | 6 | gpu-short | T4 | 4, 8G | Completed | 0:03:08 | 0.05 | test gate at the merge commit `ea11e14` (d1025): 325 passed; initialised both grids' `sweep.json`, then checked against D1 and D3 (17/17) |
| 12092405 | 7, 8 | short | none | 2, 4G | Completed | 0:00:39 | 0.00 | grid submission from inside an `srun` job (deviation, spec Section 9, 2026-11-23): `--submit-plan` one wave, then the three arrays below |
| 12092428 | 7 | short | none | 4, 8G | Completed (25 tasks) | 0:46:25 to 1:06:31 | 0.00 | main grid, IQL `tau = 0.7`, 25 runs, `%10`, `--time` 2:00, 11 cascadelake nodes |
| 12092429 | 8 | short | none | 4, 8G | Completed (20 tasks) | 0:41:43 to 1:08:53 | 0.00 | ablation, IQL `tau` 0.5 and 0.9, 20 runs, `%8`, `--time` 2:00, 7 nodes |
| 12092441 | 7 | short | none | 4, 8G | Completed (25 tasks) | 0:15:34 to 0:27:22 | 0.00 | main grid, BC, 25 runs, `%7`, `--time` 0:30, 9 nodes |
| 12092466 | 7, 8 | short | none | 2, 4G | Cancelled while pending | 0:00:00 | 0.00 | `--summary` in `srun`; its SSH session dropped while it was queued, so it was cancelled and replaced by batch jobs |
| 12092848, 12093238 | 7, 8 | short | none | 2, 4G | Completed | 0:00:12; 0:00:22 | 0.00 | progress, `sweep --summary` only (score-blind) |
| 12107680 | 7, 8 | short | none | 2, 4G | Completed | 0:00:54 | 0.00 | `--summary` (done 50 and done 20), then `sweep --check`: both grids "done and consistent", 12 and 19 runs flagged |
| 12114018 | 7, 8 | short | none | 2, 4G | Completed | 0:00:38 | 0.00 | both grids' runs, `sweep.json`, Slurm logs, `--check` output and operator logs copied to `~/archive/phase6_grid_ea11e14d4ab7` (857 files, 210 MB, `SHA256SUMS` match the source); second copy on the laptop, every hash checked |

Earlier: Phase 0 smoke job 11863650 (Tesla T4, 2026-11-13), a few minutes; not itemized.

**Total from 2026-11-16: 6.19 GPU-hours (T4), 11.99 GPU-hours (A100) and 0.03 GPU-hours (V100).**

**Deleted (approved by the user, 2026-11-24):** after the archive and its hash check passed (job 12114018, laptop copy 857/857), the checkout `/scratch/$USER/src/ea11e14...`, by `archive_runs.sh --remove-checkout` ("removed checkout", no leftovers); the clone's worktree list then held only the clone. The run directories and logs stay on /scratch until the purge; their copies are in `~/archive/phase6_grid_ea11e14d4ab7`.

**Grids at `ea11e14` (2026-11-23):** 70 runs, all completed on the first submission with no time-out, refusal or rerun; wall clock 3:04 (first task 18:55:43, last 21:59:48 EST), 194.8 CPU-hours (4 CPUs per task), 0 GPU-hours. Every task ran on an `Intel(R) Xeon(R) Platinum 8276 CPU @ 2.20GHz` cascadelake node (all 70 `meta.json` files record that model). The arrays' accounting is in the archive's `ops/sacct.txt`; the helper jobs' (12092466, 12092848, 12093238, 12107680, 12114018) in `report/phase7/sacct_helpers.txt`.

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

**Wall clock for the 70 grid runs (BC 25, IQL 45), BC and IQL counted one after the other:** CPU cascadelake 3.5 h, 0 GPU-hours; T4 with IQL at k = 2 13.3 h (29.2 GPU-hours), k = 1 16.3 h (48.0), k = 4 20.9 h (20.9); V100 ineligible; the T4 figures exclude the human latency per submission wave, which would only lengthen them. **Choice by the pre-fixed rule: CPU on `short`, `--constraint=cascadelake`**, 1 run per job, 4 CPUs, 8G (2G per CPU; measured peak 1.2 GiB), `--time` 1:30 for IQL and 0:20 for BC (h x 1.5, rounded up to 10 minutes; raised by the user on 2026-11-22 to 2:00 for IQL and 0:30 for BC, since each `h` comes from one benchmark job on shared nodes), required CPU model `Intel(R) Xeon(R) Platinum 8276 CPU @ 2.20GHz` (both CPU jobs, on nodes d0014 and d0036, recorded it; all 24 cascadelake nodes have the same 56-core layout; a task on any other model is refused by the sweep).

**Throttle split (D3, 25 in total; comparison requested by the user 2026-11-22):** `scripts/phase6_throttle_sim.py` (output `report/phase6/throttle_sim_output.txt`) simulates the three arrays (main IQL 25 runs, ablation IQL 20, main BC 25; IQL 1.00 h, BC 0.22 h per run from C1 and CB) with Slurm filling a free slot from the earliest-submitted array under its throttle, for a cluster cap `C` of 25, 22, 20 and 16 concurrent jobs (22 and 16 are the idle-core and memory readings above). Makespan in hours, range over the 6 submission orders:

| Split (main IQL / ablation IQL / BC) | C = 25 | C = 22 | C = 20 | C = 16 |
|---|---|---|---|---|
| **10/8/7 (approved)** | **3.00** | **3.00** | **3.00 to 3.66** | **3.44 to 4.00** |
| 12/10/3 (proportional to work) | 3.00 | 3.00 to 3.98 | 3.00 to 3.98 | 4.00 |
| 13/10/2 | 2.86 | 3.00 to 4.86 | 3.00 to 4.86 | 4.00 to 4.86 |
| 11/9/5 | 3.00 | 3.00 | 3.10 | 3.54 to 4.10 |
| 9/9/7 | 3.00 | 3.00 to 3.66 | 3.00 to 3.66 | 3.44 to 4.00 |

At its best submission order 10/8/7 ties or beats every split except 13/10/2 at `C` = 25 (2.86 h, 8 minutes faster); 13/10/2 loses up to 1.9 h at any smaller `C`. Only 10/8/7 and 11/9/5 have no spread at `C` of 22 or more, and 10/8/7 is faster than 11/9/5 at `C` = 20 (3.00 against 3.10) and at `C` = 16 (3.44 against 3.54). Submitting main IQL, then ablation IQL, then BC gives 3.00, 3.00, 3.00 and 3.44 h. The split is kept; the arrays go in that order. Durations are the benchmark means with no spread, so these are planning figures, not guarantees.

## Phase 6 dry run (development data, commit `a11dc28`, 2026-11-22)

Spec Section 9, on the D1 hardware (CPU `short`, cascadelake, every task on `Intel(R) Xeon(R) Platinum 8276 CPU @ 2.20GHz`, checked in each run's `meta.json` by `sweep --check`), grids `dryrun_main` (20 runs) and `dryrun_ablation` (8 runs) at 10k steps. Evidence, copied to `~/archive/phase6_dryrun_a11dc28a285a/` on Explorer and to `report/phase6/dryrun/` (with `SHA256SUMS`):

| Step | Result | Evidence |
|---|---|---|
| (0) `KillWait`, `ProctrackType`, the guard's `squeue` | 600 s, `proctrack/cgroup`; the command exits 0 with empty output | terminal |
| (1) test gate | 12067471, 325 passed | table above |
| (2) arrays, main IQL, ablation IQL, BC | 27 tasks, all exit 0 | `sacct.txt` |
| (3) cancel and lock release | plain `scancel` on 12068612_9 at 54 s, after its step-2000 checkpoint; no lock left (terminal); the sweep's line `signal 15: runs stopped, releasing the bundle's locks, exit 143`; `--list` showed `partial` (terminal) | `logs/*12068612*`, `cancel_watch.log` |
| (4) resume | only bundle 9 unfinished (terminal); 12068733 resumed from step 2000 with the grid's commit and CPU model; `metrics.csv` steps 1000 to 10000 once each, `eval.csv` and `eval_episodes.csv` steps 2000 to 10000 once each; `final.json`, `done`, no lock | `resume_check.txt` (all PASS), `resumed_iql_p1_s101_meta.json`, `logs/*12068733*` |
| (5) submit guard and skip-only resubmission | `--sbatch-args` exited 3 naming the bundle while it was pending and while it was running (terminal); 12068758 (`--array=0,9`) printed `DONE`, exit 0, no rows added | `logs/*12068758*` |
| (6) lock test | 12068769 printed `RUNNING` and exited 4 after 12 s; 12068768 trained and finished | `logs/*12068768*`, `logs/*12068769*`, `locktest_iql_p0_s100_meta.json` |
| `sweep --check` | 20 and 8 runs, all done and consistent, exit 0 (the high-fall flags are expected at 10k steps; development data, and no score was used for any decision) | `sweep_check.txt` (rerun in 12072515, same result as 12068835) |

"terminal" marks what was observed interactively at the time and is not in the archive.

**Timing against the cost model.** The benchmark predicts at most about 128 s of training and evaluation for a 10k-step IQL run (10,000 x 10.99 ms + at most 50,000 eval steps x 0.366 ms; episodes end early on falls) and at most 32 s for BC. Job elapsed times, startup included, were 1:31 to 3:23 for IQL (at most 1.6 times the prediction; the slowest were two tasks sharing a node) and 0:43 to 0:59 for BC (at most 1.8 times), consistent with the `--time` factors of 2.0 and 2.3, though startup weighs more at 10k steps than at 300k, and with fewer eval steps the true ratios are higher.

**Deleted (approved by the user, 2026-11-22):** after the evidence copy, the run directories `/scratch/$USER/runs_grid/dryrun_main/a11dc28a285a` and `dryrun_ablation/a11dc28a285a` (78 MB), their logs under `/scratch/$USER/logs/`, and the checkout `/scratch/$USER/src/a11dc28...`. `archive_runs.sh --remove-checkout` deregistered the checkout but stopped on "Directory not empty" with 200 files left; they were checked against the commit (132 unchanged tracked files and 68 `__pycache__` files) and removed by hand.

**Observations.**
- **Login node:** several `python -m offrl.sweep` calls on `explorer-01` were killed (exit 137) with their peak memory at 32 to 515 MB, under the 6 GiB per-user limit and with no out-of-memory event (terminal). The cause is not confirmed. Submissions run with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` were not killed, and inspection commands ran in a small `srun` job on `short` (spec Section 9).
- **Clocks:** the login node's clock ran about 1.5 minutes ahead of Slurm's and the compute node's, which agree with each other (`scancel` logged at 17:17:03 by Slurm, 17:18:30 by the watcher on the login node). The login node was also seen (terminal) to list a new file about 20 s after it was written.

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
