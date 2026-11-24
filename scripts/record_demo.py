"""Record a side-by-side demo video of two archived grid runs on one final evaluation seed.

    python scripts/record_demo.py --left <run_dir> --right <run_dir> --left-norm <npz> --right-norm <npz>
        --ffmpeg <ffmpeg.exe> --out <video.mp4> [--seed 10000]

Each run's final policy (``ckpt/latest.pt``, step 300000) acts deterministically (the tanh mean, as
in evaluation) on the observations normalized with its own mixture's normalizer. Before any frame
is written, the episode is replayed without rendering and its return and fall must equal the run's
own logged final evaluation for that seed (``eval_episodes.csv`` at step 300000), so the video
shows exactly the episode the scores were computed from; the rendered pass must end the same way.
Frames are streamed to ffmpeg.

``--*-norm`` holds a mixture's ``state_mean``, ``state_std`` and ``state_std_eps`` arrays, copied
unchanged from the archived mixture file (``~/archive/phase6_mixtures/mix_p<p>_s<seed>.npz`` on
Explorer, content hash checked against its manifest), so the 2 GB mixture need not be on the
laptop. A wrong normalizer would fail the replay check."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from offrl.data.normalize import StateNormalizer  # noqa: E402
from offrl.evaluate import deterministic_policy, make_env  # noqa: E402
from offrl.train import TRAINERS, load_checkpoint  # noqa: E402

W, H = 480, 480
CAMERA = {
    "distance": 4.0,
    "elevation": -12.0,
    "azimuth": 90.0,
    "lookat": np.array([0.0, 0.0, 1.0]),
}  # side view
STRIDE = 2  # render every 2nd step; Hopper-v5 steps 0.008 s, so 62.5 fps is real time
FPS = 62.5


def load_policy(run_dir: Path, norm_path: Path):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    state = load_checkpoint(run_dir / "ckpt" / "latest.pt")
    if int(state["step"]) != cfg["train"]["steps"]:
        raise SystemExit(f"{run_dir.name}: checkpoint at step {state['step']}, not the final one")
    trainer = TRAINERS[cfg["algo"]["name"]](cfg, 11, 3, torch.device("cpu"))
    trainer.load_state_dict(state["trainer"])
    z = np.load(norm_path)
    mean, std = z["state_mean"].astype(np.float64), z["state_std"].astype(np.float64)
    norm = StateNormalizer(mean, std, float(z["state_std_eps"]))
    return cfg, deterministic_policy(trainer, norm, torch.device("cpu"))


def logged_episode(run_dir: Path, step: int, seed: int) -> tuple[float, int, bool]:
    with (run_dir / "eval_episodes.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["step"]) == step and int(r["seed"]) == seed:
                return float(r["return"]), int(r["length"]), r["fell"] == "True"
    raise SystemExit(f"{run_dir.name}: no logged final episode for seed {seed}")


def same(a: tuple[float, int, bool], b: tuple[float, int, bool]) -> bool:
    return math.isclose(a[0], b[0], rel_tol=1e-6) and a[1:] == b[1:]


def frame(env) -> np.ndarray:
    """Render with the free camera moved to the torso's x first (the tracking setting is not
    applied by gymnasium's offscreen renderer), so the robot stays in view as it hops forward."""
    viewer = env.unwrapped.mujoco_renderer._get_viewer(render_mode="rgb_array")
    viewer.cam.lookat[0] = env.unwrapped.data.qpos[0]
    return env.render()


def episode(env, policy, seed: int, render: bool):
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    total, steps, terminated, truncated = 0.0, 0, False, False
    while not (terminated or truncated):
        if render and steps % STRIDE == 0:
            yield ("frame", frame(env), steps, total)
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += float(reward)
        steps += 1
    if render:
        yield ("frame", frame(env), steps, total)
    yield ("end", total, steps, bool(terminated))


def replay(env, policy, seed: int) -> tuple[float, int, bool]:
    for kind, *rest in episode(env, policy, seed, render=False):
        if kind == "end":
            return rest[0], rest[1], rest[2]
    raise AssertionError


def score(ret: float, cfg: dict) -> float:
    return 100 * (ret - cfg["score"]["r_random"]) / (cfg["score"]["r_expert"] - cfg["score"]["r_random"])


def panel(frame: np.ndarray, title: str, line2: str, font, small) -> Image.Image:
    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 54], fill=(252, 252, 251))
    d.text((12, 6), title, fill=(31, 31, 30), font=font)
    d.text((12, 32), line2, fill=(107, 106, 100), font=small)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    for side in ("left", "right"):
        ap.add_argument(f"--{side}", type=Path, required=True)
        ap.add_argument(f"--{side}-norm", type=Path, required=True)
        ap.add_argument(f"--{side}-label", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--frame-at", type=float, default=None, help="also save the frame at this time (s)")
    args = ap.parse_args()

    sides = []
    for side in ("left", "right"):
        run = getattr(args, side)
        cfg, policy = load_policy(run, getattr(args, f"{side}_norm"))
        env = make_env(cfg["data"]["expert_id"])
        got = replay(env, policy, args.seed)
        want = logged_episode(run, cfg["train"]["steps"], args.seed)
        if not same(got, want):
            raise SystemExit(f"{run.name}: replay (return, steps, fell) {got}, logged {want}")
        print(f"{run.name}: seed {args.seed}: {got[0]:.1f}, {got[1]} steps, fell {got[2]}; matches log")
        sides.append((cfg, policy, getattr(args, f"{side}_label"), got))

    import minari

    font = ImageFont.truetype("arialbd.ttf", 20)
    small = ImageFont.truetype("arial.ttf", 15)
    gens, last = [], []
    for cfg, policy, label, got in sides:
        dataset = minari.load_dataset(cfg["data"]["expert_id"])
        env = dataset.recover_environment(
            render_mode="rgb_array", width=W, height=H, default_camera_config=dict(CAMERA)
        )
        # Draw the floor plane larger, so a policy that hops far does not run past its visible edge.
        # Planes collide as infinite planes whatever their size; the end check below guards that.
        model = env.unwrapped.model
        model.geom("floor").size[:2] = 200.0
        model.mat_texrepeat[model.material("MatPlane").id] *= 10  # keep the checker size
        gens.append((episode(env, policy, args.seed, render=True), cfg, label, got))
        last.append(None)
    ff = subprocess.Popen(
        [
            args.ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{2 * W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(args.out),
        ],
        stdin=subprocess.PIPE,
    )
    done = [False, False]
    n_frames, mid = 0, None
    try:
        while not all(done):
            imgs = []
            for i, (gen, cfg, label, got) in enumerate(gens):
                if not done[i]:
                    kind, *rest = next(gen)
                    if kind == "end":
                        if not same((rest[0], rest[1], rest[2]), got):
                            raise SystemExit(
                                f"{label}: rendered episode {tuple(rest)} differs from the replay {got}"
                            )
                        done[i] = True
                        status = "fell" if rest[2] else "time limit"
                        last[i] = (
                            last[i][0],
                            f"step {rest[1]}  episode score {score(rest[0], cfg):.1f}  ({status})",
                        )
                    else:
                        frame, steps, total = rest
                        last[i] = (frame, f"step {steps}  return {total:.0f}")
                imgs.append(panel(last[i][0], label, last[i][1], font, small))
            canvas = Image.new("RGB", (2 * W, H))
            canvas.paste(imgs[0], (0, 0))
            canvas.paste(imgs[1], (W, 0))
            ff.stdin.write(np.asarray(canvas).tobytes())
            if args.frame_at is not None and n_frames == round(args.frame_at * FPS):
                mid = canvas
            n_frames += 1
        for _ in range(int(FPS * 2)):  # hold the final frame for 2 s
            ff.stdin.write(np.asarray(canvas).tobytes())
    except BaseException:
        ff.kill()
        ff.wait()
        args.out.unlink(missing_ok=True)
        raise
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("ffmpeg failed")
    canvas.save(args.out.with_suffix(".png"))
    if mid is not None:
        mid.save(args.out.with_name(args.out.stem + f"_t{args.frame_at:g}s.png"))
    print(f"wrote {args.out} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
