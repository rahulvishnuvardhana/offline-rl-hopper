"""Phase 6 spec S1 and S8: the committed grid files expand as pre-registered, every run resolves to
its golden pin, and every planned config (grids, dry run, benchmark) passes the config part of
pre-flight against the committed manifests."""

from pathlib import Path

import pytest
import yaml

from offrl import sweep, train
from offrl.config import load_checked

CONFIGS = train.REPO_ROOT / "configs"
GRIDS = {"grid_main": 50, "grid_ablation": 20, "grid_dryrun_main": 20, "grid_dryrun_ablation": 8}
SETS = ["data.mixture_dir=/scratch/mixtures"]


def runs_of(name):
    return sweep.expand(sweep.load_grid(CONFIGS / f"{name}.yaml"), Path("/scratch/out"), SETS)


@pytest.mark.parametrize(("name", "n"), GRIDS.items())
def test_grid_sizes_and_unique_names(name, n):
    runs = runs_of(name)
    assert len(runs) == n and len({r.name for r in runs}) == n


def test_main_grid_order_and_cells():
    runs = runs_of("grid_main")
    assert [r.name for r in runs[:6]] == [
        "bc_p0_s0",
        "bc_p0_s1",
        "bc_p0_s2",
        "bc_p0_s3",
        "bc_p0_s4",
        "bc_p0.05_s0",
    ]
    cells = {(r.algo, r.cfg["data"]["p_expert"]) for r in runs}
    assert cells == {(a, p) for a in ("bc", "iql") for p in (0.0, 0.05, 0.2, 0.5, 1.0)}
    assert sorted({r.cfg["seed"] for r in runs}) == [0, 1, 2, 3, 4]


def test_ablation_pairs_tau_with_its_suffix():
    for r in runs_of("grid_ablation"):
        assert r.cfg["train"]["run_suffix"] == f"_tau{r.cfg['algo']['expectile']:g}"
        assert r.cfg["data"]["p_expert"] in (0.05, 0.5) and r.cfg["seed"] in range(5)


@pytest.mark.parametrize("name", GRIDS)
def test_every_run_resolves_to_its_golden_pin(name):
    grid = sweep.load_grid(CONFIGS / f"{name}.yaml")
    for r in sweep.expand(grid, Path("/scratch/out"), SETS):
        assert sweep.pin_view(r.cfg) == sweep.pin_for(grid, r), r.name


@pytest.mark.parametrize("name", ["grid_main", "grid_ablation"])
def test_real_grids_pin_the_preregistered_values(name):
    """Phase 6 spec Section 2 and the pre-registration: 300k steps, 60 evaluations, the last 5 at
    50 episodes on the final evaluation seeds, final-grid mixtures 0 to 4."""
    for r in runs_of(name):
        tr, ev, a = r.cfg["train"], r.cfg["eval"], r.cfg["algo"]
        assert (tr["steps"], tr["eval_every"], tr["checkpoint_every"], tr["log_every"]) == (
            300000,
            5000,
            25000,
            1000,
        )
        assert (ev["episodes"], ev["final_episodes"], ev["final_score_last_n"], ev["seed_offset"]) == (
            10,
            50,
            5,
            10000,
        )
        assert (tr["lr"], tr["batch_size"], tr["actor_cosine_lr"]) == (3e-4, 256, True)
        assert r.cfg["seed"] in range(5) and r.cfg["seed"] < train.DEV_SEED_MIN
        d, n, s = r.cfg["data"], r.cfg["net"], r.cfg["score"]
        assert n["hidden"] == [256, 256] and d["n_transitions"] == 900000
        assert d["normalize_states"] is True and d["state_std_eps"] == 1e-3
        assert d["scale_rewards"] is (r.algo == "iql")
        assert (round(s["r_random"], 2), round(s["r_expert"], 2)) == (17.03, 3857.80)
        if r.algo == "iql":
            assert a["beta"] == 3.0
            assert (a["discount"], a["target_update_rate"], a["exp_adv_max"]) == (0.99, 0.005, 100.0)
            assert a["expectile"] == 0.7 if name == "grid_main" else a["expectile"] in (0.5, 0.9)
        else:
            assert a["beta"] == 0.0


def test_bc_and_iql_differ_only_in_algo_and_reward_scaling():
    """ROADMAP fairness requirement: same network, data handling, budget and evaluation."""
    runs = runs_of("grid_main")
    bc_run = next(r for r in runs if r.algo == "bc")
    iql_run = next(r for r in runs if r.algo == "iql")
    bc, iql = (sweep.pin_for({"pin": "main"}, r) for r in (bc_run, iql_run))
    for pin in (bc, iql):
        pin.pop("algo")
        pin["data"].pop("scale_rewards")
    assert bc == iql


def test_ablation_matches_main_iql_except_expectile():
    """H3 compares the ablation with the main grid's tau = 0.7 cells: nothing else may differ."""
    main = yaml.safe_load((sweep.PIN_DIR / "main_iql_tau0.7.yaml").read_text(encoding="utf-8"))
    main["algo"].pop("expectile")
    for t in ("0.5", "0.9"):
        abl = yaml.safe_load((sweep.PIN_DIR / f"ablation_iql_tau{t}.yaml").read_text(encoding="utf-8"))
        abl["algo"].pop("expectile")
        assert abl == main
    grids = [sweep.load_grid(CONFIGS / f"{g}.yaml") for g in ("grid_main", "grid_ablation")]
    assert grids[0]["extra_configs"] == grids[1]["extra_configs"]


@pytest.mark.parametrize("name", ["grid_dryrun_main", "grid_dryrun_ablation"])
def test_dry_run_uses_development_data_only(name):
    """User decision D4: no final-grid mixture or final evaluation seed before the real grid."""
    for r in runs_of(name):
        assert r.cfg["seed"] >= train.DEV_SEED_MIN
        assert r.cfg["eval"]["seed_offset"] not in train.FINAL_EVAL_RESERVED


def benchmark_configs():
    """slurm/benchmark.sbatch's resolved configs (spec Section 5): keep in step with its
    ``common=(...)`` overrides."""
    overrides = [
        "seed=100", "data.p_expert=1", "train.steps=10000", "train.log_every=1000",
        "train.eval_every=2000", "train.checkpoint_every=2000", "eval.episodes=10",
        "eval.final_episodes=10", "eval.final_score_last_n=1",
    ]  # fmt: skip
    return [
        load_checked(train.BASE_CONFIG, CONFIGS / "algo" / f"{a}.yaml", overrides=overrides)
        for a in ("bc", "iql")
    ]


def test_every_planned_config_passes_preflight():
    """S8: config-only pre-flight on every grid run and the benchmark, against the committed
    manifests (this would have caught v1's two blockers)."""
    configs = [r.cfg for name in GRIDS for r in runs_of(name)] + benchmark_configs()
    failures = {train.run_name(c): e for c in configs if (e := train.preflight_config(c, train.MANIFEST_DIR))}
    assert failures == {}
