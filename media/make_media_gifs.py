# Regenerate media/*.gif: run from the repo root under xvfb-run, e.g.
#   xvfb-run -a -s "-screen 0 1280x960x24" python media/make_media_gifs.py --mode act --out /tmp/frames
#   ffmpeg -framerate 20 -i /tmp/frames/f%04d.png -vf "scale=640:480" media/rollout.gif
"""Capture viewer frames for the FFW tutorial media.
modes: data   - replay a collected episode (as 2.visualize_data.ipynb)
       teleop - same replay, rendered with the teleop overlays (as 1.collect_data.ipynb)
       act    - roll out the pretrained ACT checkpoint (as 4.deploy.ipynb)
"""
import os, sys, subprocess, argparse
import numpy as np, torch, torchvision
from PIL import Image
sys.path.insert(0, ".")
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["data", "teleop", "act"], required=True)
ap.add_argument("--out", required=True); ap.add_argument("--episode", type=int, default=0)
ap.add_argument("--max_steps", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cam", type=float, nargs=6, default=[170, 1.3, -25, -0.40, 0.0, 0.80], help="azimuth distance elevation lookat_xyz")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from mujoco_env.y_env import SimpleEnv

env = SimpleEnv("./asset/example_scene_y.xml", action_type="joint_angle", seed=a.seed)
env.reset(seed=a.seed)
env.env.set_viewer(azimuth=a.cam[0], distance=a.cam[1], elevation=a.cam[2], lookat=np.array(a.cam[3:6]))

def shot(i): subprocess.run(["import", "-window", "root", f"{a.out}/f{i:04d}.png"], check=True)

if a.mode in ("data", "teleop"):
    ds = LeRobotDataset("ffw", root="./ffw_one_arm_data")
    f, t = ds.episode_data_index["from"][a.episode].item(), ds.episode_data_index["to"][a.episode].item()
    step = 0
    while env.env.is_viewer_alive() and step < (t - f):
        env.step_env()
        if env.env.loop_every(HZ=20):
            d = ds[f + step]
            if step == 0: env.set_obj_pose(d["obj_init"][:3], d["obj_init"][3:])
            env.step(d["action"].numpy())
            for k, attr in [("agentview", "rgb_agent"), ("topview", "rgb_top"), ("sideview", "rgb_side")]:
                setattr(env, attr, (d[f"observation.images.{k}"].numpy() * 255).astype(np.uint8).transpose(1, 2, 0))
            env.render(teleop=(a.mode == "teleop"))
            shot(step); step += 1
    print("captured", step, "frames (episode replay)")
else:
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.policies.act.configuration_act import ACTConfig
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import FeatureType
    device = "cuda"
    meta = LeRobotDatasetMetadata("ffw", root="./ffw_one_arm_data")
    feats = dataset_to_policy_features(meta.features)
    out = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    inp = {k: v for k, v in feats.items() if k not in out}
    inp.pop("observation.images.topview"); inp.pop("observation.images.sideview")
    cfg = ACTConfig(input_features=inp, output_features=out, chunk_size=10, n_action_steps=1, temporal_ensemble_coeff=0.9)
    policy = ACTPolicy.from_pretrained("./ckpt/act_y", config=cfg, dataset_stats=meta.stats).to(device)
    policy.reset(); policy.eval()
    tt = torchvision.transforms.ToTensor()
    step, success = 0, False
    while env.env.is_viewer_alive() and step < a.max_steps:
        env.step_env()
        if env.env.loop_every(HZ=20):
            state = env.get_joint_state()
            image, _, _ = env.grab_image()
            image = tt(Image.fromarray(image).resize((256, 256)))
            data = {"observation.state": torch.tensor([state]).to(device),
                    "observation.images.agentview": image.unsqueeze(0).to(device),
                    "task": ["Put mug cup on the plate"], "timestamp": torch.tensor([step / 20]).to(device)}
            with torch.no_grad(): action = policy.select_action(data)[0].cpu().numpy()
            env.step(action); env.render(); shot(step); step += 1
            if env.check_success(): success = True; print("Success at step", step); break
    print("captured", step, "frames (ACT rollout), success =", success)
env.env.close_viewer()
