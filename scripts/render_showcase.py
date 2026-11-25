"""Render the showcase demo: one clean video per policy plus their step-by-step trajectories, for the
project's story page. Visual only; it never trains, evaluates or writes a result.

    python scripts/render_showcase.py --run <run_dir> --norm <npz> --name bc \
        --run <run_dir> --norm <npz> --name iql --ffmpeg <ffmpeg.exe> --out-dir <dir> [--seed 10000]

Each policy is first replayed on the stock Hopper-v5 model and must reproduce its run's logged final
evaluation for the seed (``record_demo.logged_episode``). It is then rendered every simulation step on
``report/demo/hopper_showcase.xml``, a restyled copy of the stock model whose physics arrays are checked
equal to the stock model's here before use (every numeric model and option field except a short list
of visual ones), and the rendered episode must match the replay too. Videos
run at real time (125 frames per second, one per 0.008 s step). A policy that falls early holds its
last frame so both videos have the same length and can be played in sync. ``trajectories.json`` holds
each step's torso x, height, pitch and running return for the page's readouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_trajectories import eval_threads  # noqa: E402
from offrl.evaluate import make_env  # noqa: E402
from record_demo import load_policy, logged_episode, replay, same, score  # noqa: E402

W, H = 800, 600
FPS = 125  # Hopper-v5 steps 0.008 s
SHOWCASE = Path(__file__).resolve().parents[1] / "report" / "demo" / "hopper_showcase.xml"
CAMERA = {"distance": 3.4, "elevation": -8.0, "azimuth": 90.0, "lookat": np.array([0.0, 0.0, 0.85])}
# Model fields that only affect appearance; every other numeric field of the model and of its options
# must match the stock model exactly. The floor plane's geom_size is drawn size only (planes collide as
# infinite planes) and is checked separately.
VISUAL = ("geom_rgba", "geom_matid", "mat_", "tex_", "light_", "site_", "name", "nlight", "nmat", "nsite",
          "ntex", "nnames", "nbuffer", "ntext", "signature")  # fmt: skip


def _numeric_fields(obj) -> dict:
    out = {}
    for f in dir(obj):
        if f.startswith("_"):
            continue
        try:
            v = getattr(obj, f)
        except Exception:  # noqa: BLE001, S112 - some bindings raise on unsupported fields
            continue
        if isinstance(v, (int, float, np.ndarray)):
            out[f] = np.asarray(v)
    return out


def check_physics() -> None:
    """The restyled model must differ from the stock one in appearance only."""
    import gymnasium

    stock_path = Path(gymnasium.__file__).parent / "envs" / "mujoco" / "assets" / "hopper.xml"
    stock, show = mujoco.MjModel.from_xml_path(str(stock_path)), mujoco.MjModel.from_xml_path(str(SHOWCASE))
    bad = []
    for label, a, b in (("", stock, show), ("opt.", stock.opt, show.opt)):
        fa, fb = _numeric_fields(a), _numeric_fields(b)
        for f in sorted(fa.keys() | fb.keys()):
            if f.startswith(VISUAL) or f == "geom_size":
                continue
            if f not in fa or f not in fb or not np.array_equal(fa[f], fb[f]):
                bad.append(label + f)
    if not np.array_equal(stock.geom_size[1:], show.geom_size[1:]):  # geom 0 is the floor plane
        bad.append("geom_size")
    if bad:
        raise SystemExit(f"{SHOWCASE.name} changes physics: {bad}")


def render(cfg: dict, policy, seed: int):
    """Yield (frame, step, x, z, pitch, return) for every step, then the episode summary."""
    import minari

    env = minari.load_dataset(cfg["data"]["expert_id"]).recover_environment(
        render_mode="rgb_array", width=W, height=H, xml_file=str(SHOWCASE), default_camera_config=dict(CAMERA)
    )
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    data = env.unwrapped.data
    viewer = env.unwrapped.mujoco_renderer._get_viewer(render_mode="rgb_array")
    total, steps, terminated, truncated = 0.0, 0, False, False
    while True:
        viewer.cam.lookat[0] = data.qpos[0]  # follow the torso along the track
        yield (
            "frame",
            env.render(),
            steps,
            float(data.qpos[0]),
            float(data.qpos[1]),
            float(data.qpos[2]),
            total,
        )
        if terminated or truncated:
            break
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += float(reward)
        steps += 1
    yield ("end", total, steps, bool(terminated))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--norm", type=Path, action="append", required=True)
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--crf", type=int, default=24)
    args = ap.parse_args()
    if not len(args.run) == len(args.norm) == len(args.name):
        raise SystemExit("give --run, --norm and --name once per policy")
    if len(set(args.name)) != len(args.name):
        raise SystemExit("--name has duplicates")
    check_physics()
    print(f"{SHOWCASE.name}: physics identical to the stock model")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "trajectories.json").unlink(missing_ok=True)  # never leave one from an older render
    sides = []
    for run, norm, name in zip(args.run, args.norm, args.name, strict=True):
        cfg, policy = load_policy(run, norm)
        torch.set_num_threads(eval_threads(run))  # as in evaluation: rounding depends on it
        got = replay(make_env(cfg["data"]["expert_id"]), policy, args.seed)
        want = logged_episode(run, cfg["train"]["steps"], args.seed)
        if not same(got, want):
            raise SystemExit(f"{run.name}: replay (return, steps, fell) {got}, logged {want}")
        print(
            f"{name}: {run.name} seed {args.seed}: {got[0]:.1f}, {got[1]} steps, fell {got[2]}; matches log"
        )
        sides.append((run, cfg, policy, name, got))

    length = max(got[1] for *_, got in sides) + 1  # frames: the start pose plus one per step
    traj = {"seed": args.seed, "fps": FPS, "frames": length, "runs": {}}
    for run, cfg, policy, name, got in sides:
        out = args.out_dir / f"{name}.mp4"
        raw = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
        enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf), "-preset", "slow"]
        cmd = [args.ffmpeg, "-y", "-loglevel", "error", *raw, *enc, "-movflags", "+faststart", str(out)]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        rows, last = [], None
        try:
            for kind, *rest in render(cfg, policy, args.seed):
                if kind == "end":
                    if not same((rest[0], rest[1], rest[2]), got):
                        raise SystemExit(
                            f"{name}: rendered episode {tuple(rest)} differs from the replay {got}"
                        )
                    break
                frame, step, x, z, pitch, ret = rest
                ff.stdin.write(np.ascontiguousarray(frame).tobytes())
                rows.append([step, round(x, 4), round(z, 4), round(pitch, 4), round(ret, 2)])
                last = frame
            for _ in range(length - len(rows)):  # hold the final pose after an early fall
                ff.stdin.write(np.ascontiguousarray(last).tobytes())
        except BaseException:
            ff.kill()
            ff.wait()
            out.unlink(missing_ok=True)
            raise
        ff.stdin.close()
        if ff.wait() != 0:
            raise SystemExit("ffmpeg failed")
        traj["runs"][name] = {
            "run": run.name, "return": round(got[0], 2), "steps": got[1], "fell": got[2],
            "score": round(score(got[0], cfg), 1),
            "columns": ["step", "x", "z", "pitch", "return"], "rows": rows,
        }  # fmt: skip
        print(f"wrote {out} ({os.path.getsize(out) // 1024} KB, {len(rows)} live frames of {length})")
    (args.out_dir / "trajectories.json").write_text(json.dumps(traj, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.out_dir / 'trajectories.json'}")


if __name__ == "__main__":
    main()
