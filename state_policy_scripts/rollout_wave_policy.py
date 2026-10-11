#!/usr/bin/env python3
"""Roll out a proprioception-only LeRobot Diffusion Policy on the FFW SG2 in MuJoCo.

Mirrors the cyclo_intelligence policy-container runtime: one observation per policy
request, a 16-step action chunk at 15 Hz, pushed through cyclo's ActionChunkProcessor
(l2 alignment + interpolation to 100 Hz), then applied to MuJoCo position actuators.
Initial states are sampled from a per-joint Gaussian fitted to the demonstrations'
first frames (std floored so every rollout starts from a genuinely new state).

usage: rollout_wave_policy.py --task wave-right --n 10 --out /path/to/out
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import pyarrow.parquet as pq
import torch

CYCLO = Path("/home/local/ASURITE/opatil3/src/cyclo_intelligence")
DATA = Path("/mount/scratch4/omkar/datasets/wave-traj")
HERE = Path(__file__).resolve().parent.parent  # repo root
SCENE = HERE / "asset/robotis_ffw/scene_ffw_sg2_wave.xml"

sys.path.insert(0, str(CYCLO / "cyclo_brain/sdk/action_chunk_processing"))
from action_chunk_processing import ActionChunkProcessor  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "diffusion_compat", CYCLO / "cyclo_brain/policy/lerobot/lerobot_engine/diffusion_compat.py")
diffusion_compat = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(diffusion_compat)
diffusion_compat.allow_state_only_diffusion()
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402

INFERENCE_HZ, CONTROL_HZ = 15.0, 100.0
MOBILE_DIMS = 3  # linear_x, linear_y, angular_z (zero in sim: base is not driven)


# ----------------------------------------------------------------------------- policy
class StateOnlyPolicy:
    def __init__(self, model_path: Path, device: str):
        self.device = device
        self.policy = get_policy_class("diffusion").from_pretrained(str(model_path)).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=str(model_path),
            preprocessor_overrides={"device_processor": {"device": device}})
        self.state_dim = int(self.policy.config.input_features["observation.state"].shape[0])
        self.n_action_steps = self.policy.config.n_action_steps

    @torch.inference_mode()
    def chunk(self, state: np.ndarray) -> np.ndarray:
        obs = {"observation.state": torch.from_numpy(state.astype(np.float32)).unsqueeze(0), "task": [""]}
        batch = diffusion_compat.expand_obs_time_dim(self.pre(obs), self.policy.config.n_obs_steps)
        action = self.post(self.policy.predict_action_chunk(batch))
        return action[0].detach().cpu().numpy().astype(np.float64)  # (T, D)


# ------------------------------------------------------------------------------- sim
class FFWSim:
    def __init__(self, joint_names: list[str], render_size=(360, 480), cameras=("front", "threequarter")):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.joint_names = joint_names
        self.qpos_idx = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in joint_names])
        self.qvel_idx = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in joint_names])
        self.act_idx = np.array([self.model.actuator(n).id for n in joint_names])
        self.ctrl_lo = self.model.actuator_ctrlrange[self.act_idx, 0]
        self.ctrl_hi = self.model.actuator_ctrlrange[self.act_idx, 1]
        self.dt = self.model.opt.timestep
        self.steps_per_tick = int(round(1.0 / CONTROL_HZ / self.dt))
        self.renderer = mujoco.Renderer(self.model, *render_size)
        self.cameras = cameras

    @property
    def joint_limits(self):
        jid = [self.model.joint(n).id for n in self.joint_names]
        return self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1]

    def reset(self, q0: np.ndarray, settle_s: float = 1.0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_idx] = q0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.act_idx] = q0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(settle_s / self.dt)):
            mujoco.mj_step(self.model, self.data)

    def q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def qd(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].copy()

    def base_xy(self) -> np.ndarray:
        return self.data.qpos[:2].copy()  # freejoint x, y

    def tick(self, target_q: np.ndarray):
        self.data.ctrl[self.act_idx] = np.clip(target_q, self.ctrl_lo, self.ctrl_hi)
        mujoco.mj_step(self.model, self.data, nstep=self.steps_per_tick)

    def render(self) -> np.ndarray:
        frames = []
        for cam in self.cameras:
            self.renderer.update_scene(self.data, camera=cam)
            frames.append(self.renderer.render().copy())
        return np.concatenate(frames, axis=1)


# ------------------------------------------------------------------- initial states
def load_demo_states(task: str):
    root = DATA / f"lerobot/{task}-state-15hz"
    names = json.load(open(root / "meta/info.json"))["features"]["observation.state"]["names"]
    tb = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pandas()
    episodes = [np.stack(g["observation.state"].values) for _, g in tb.groupby("episode_index")]
    return names, episodes


class InitialStateSampler:
    """Per-joint Gaussian fitted to the demos' first frames, std floored, clipped to the demo range."""

    def __init__(self, episodes, n_joints, std_floor=0.02, gripper_std_floor=0.01, margin=0.05, rng=None):
        first = np.stack([ep[0, :n_joints] for ep in episodes])
        allf = np.concatenate([ep[:, :n_joints] for ep in episodes])
        self.mean, std = first.mean(0), first.std(0)
        floor = np.full(n_joints, std_floor); floor[[7, 15]] = gripper_std_floor  # gripper dims
        self.std = np.maximum(std, floor)
        self.lo, self.hi = allf.min(0) - margin, allf.max(0) + margin
        self.rng = rng or np.random.default_rng()
        self.demo_first = first

    def sample(self, jlo, jhi):
        q = self.mean + self.std * self.rng.standard_normal(self.mean.shape)
        return np.clip(q, np.maximum(self.lo, jlo), np.minimum(self.hi, jhi))


# ----------------------------------------------------------------------------- rollout
def rollout(sim: FFWSim, policy: StateOnlyPolicy, q0: np.ndarray, seconds: float, record_video=True, align="l2"):
    n_joints = len(sim.joint_names)
    sim.reset(q0)
    proc = ActionChunkProcessor(inference_hz=INFERENCE_HZ, control_hz=CONTROL_HZ, alignment_mode=align)
    ticks = int(seconds * CONTROL_HZ)
    record_every = int(round(CONTROL_HZ / INFERENCE_HZ))  # 15 Hz logging / video
    states, targets, times, frames, chunks, infer_ms = [], [], [], [], [], []
    q_cmd = q0.copy()
    for t in range(ticks):
        if proc.buffer_size == 0:  # sync request mode: refill when the buffer is empty
            obs = np.concatenate([sim.q(), np.zeros(MOBILE_DIMS)])
            t0 = time.perf_counter()
            chunk = policy.chunk(obs)
            infer_ms.append((time.perf_counter() - t0) * 1e3)
            chunks.append(chunk)
            proc.push_chunk(chunk)
        action = proc.pop_action()
        if action is not None:
            q_cmd = action[:n_joints]
        sim.tick(q_cmd)
        if t % record_every == 0:
            states.append(sim.q()); targets.append(q_cmd.copy()); times.append(t / CONTROL_HZ)
            if record_video:
                frames.append(sim.render())
    states, targets = np.array(states), np.array(targets)
    return dict(states=states, targets=targets, times=np.array(times), frames=frames,
                chunks=np.array(chunks), infer_ms=np.array(infer_ms), q0=q0,
                base_drift=float(np.linalg.norm(sim.base_xy() - np.array([-0.8, 0.0]))))


def summarize(res, names, demo_episodes):
    n = len(names)
    s = res["states"]
    ptp = s.max(0) - s.min(0)
    demo_ptp = np.mean([ep[:, :n].max(0) - ep[:, :n].min(0) for ep in demo_episodes], axis=0)
    track_err = np.abs(res["states"] - res["targets"]).mean()
    return {
        "finite": bool(np.isfinite(s).all()),
        "duration_s": float(res["times"][-1]),
        "policy_calls": int(len(res["chunks"])),
        "inference_ms_median": float(np.median(res["infer_ms"])),
        "tracking_mae_rad": float(track_err),
        "base_drift_m": res["base_drift"],
        "ptp_left_arm_rad": float(ptp[0:7].max()),
        "ptp_right_arm_rad": float(ptp[8:15].max()),
        "demo_ptp_left_arm_rad": float(demo_ptp[0:7].max()),
        "demo_ptp_right_arm_rad": float(demo_ptp[8:15].max()),
        "ptp_per_joint": dict(zip(names, ptp.round(4).tolist())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["wave-right", "wave-left", "wave-both"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--step", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DATA / "rollouts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--align", default="l2", choices=["l2", "none", "rtc"], help="ActionChunkProcessor alignment mode")
    args = ap.parse_args()

    ckpt = DATA / f"checkpoints/diffusion_state_{args.task}/checkpoints/{args.step:06d}/pretrained_model"
    policy = StateOnlyPolicy(ckpt, args.device)
    names, demo_episodes = load_demo_states(args.task)
    joint_names = names[: len(names) - MOBILE_DIMS]
    assert policy.state_dim == len(names), f"policy expects {policy.state_dim}-D state, dataset has {len(names)}"
    sim = FFWSim(joint_names)
    jlo, jhi = sim.joint_limits
    sampler = InitialStateSampler(demo_episodes, len(joint_names), rng=np.random.default_rng(args.seed))

    out = args.out / args.task
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    summaries = []
    print(f"[{args.task}] checkpoint {ckpt}\n  init-state std (rad): {np.round(sampler.std, 3).tolist()}")
    for i in range(args.n):
        q0 = sampler.sample(jlo, jhi)
        res = rollout(sim, policy, q0, args.seconds, record_video=not args.no_video, align=args.align)
        summ = summarize(res, joint_names, demo_episodes)
        summ["rollout"] = i
        summ["q0"] = q0.round(4).tolist()
        summ["q0_dist_to_nearest_demo_rad"] = float(np.abs(sampler.demo_first - q0).max(1).min())
        summaries.append(summ)
        np.savez_compressed(out / f"rollout_{i:02d}.npz", states=res["states"], targets=res["targets"],
                            times=res["times"], chunks=res["chunks"], q0=q0, joint_names=np.array(joint_names))
        if res["frames"]:
            imageio.mimwrite(out / f"rollout_{i:02d}.mp4", res["frames"], fps=int(INFERENCE_HZ),
                             codec="libx264", quality=7, macro_block_size=1)
            imageio.imwrite(out / f"rollout_{i:02d}_first.png", res["frames"][0])
        print(f"  rollout {i:02d}: finite={summ['finite']} calls={summ['policy_calls']} "
              f"infer={summ['inference_ms_median']:.0f}ms track_mae={summ['tracking_mae_rad']:.3f} "
              f"ptp L={summ['ptp_left_arm_rad']:.2f} R={summ['ptp_right_arm_rad']:.2f} "
              f"(demo L={summ['demo_ptp_left_arm_rad']:.2f} R={summ['demo_ptp_right_arm_rad']:.2f}) "
              f"q0->nearest demo {summ['q0_dist_to_nearest_demo_rad']:.3f} rad")
    json.dump({"task": args.task, "checkpoint": str(ckpt), "seed": args.seed, "seconds": args.seconds, "align": args.align,
               "init_state_mean": sampler.mean.round(4).tolist(), "init_state_std": sampler.std.round(4).tolist(),
               "joint_names": joint_names, "rollouts": summaries}, open(out / "summary.json", "w"), indent=2)
    print(f"saved {args.n} rollouts to {out}")


if __name__ == "__main__":
    main()
