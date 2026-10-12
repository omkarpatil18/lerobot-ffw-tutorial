"""inspect-robots embodiment for the bimanual ROBOTIS FFW SG2 in the tutorial's MuJoCo scene.

Action contract: 14-D ``eef_delta_pose`` in the world frame, per arm
``[dx, dy, dz, droll, dpitch, dyaw, dgripper]``. Position deltas are metres, rotation
deltas are radians applied in the tool frame, and the gripper dimension is a
displacement of the gripper command on ``[0, 1]`` (``+1`` closes, ``-1`` opens).
The scene is simulated headlessly through EGL; no display is required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspect_robots.embodiment import (  # noqa: E402
    AUTO_RESET,
    PRIVILEGED_SUCCESS,
    RENDERABLE,
    RESETTABLE,
    SEEDABLE,
    EmbodimentBase,
    EmbodimentInfo,
)
from inspect_robots.scene import Scene  # noqa: E402
from inspect_robots.spaces import (  # noqa: E402
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, Observation, StepResult  # noqa: E402

SCENE_XML = REPO_ROOT / "asset" / "example_scene_llm.xml"  # tutorial scene with a cube instead of the mug
OBJECT_BODY, TARGET_BODY = "body_obj_cube", "body_obj_plate_11"
OBJECT_REST_Z, TARGET_REST_Z = 0.825, 0.808
CONTROL_HZ = 20.0
ARMS = ("left", "right")
DIM_LABELS = tuple(f"{arm}_{d}" for arm in ARMS for d in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"))
POS_STEP, ROT_STEP, GRIP_STEP = 0.01, 0.05, 1.0
ALL_CAMERAS = ("agentview", "topview", "sideview")
# Table-top manipulation start pose captured from the real FFW SG2 (ai_worker
# ffw_sg2_rev1_follower/tabletop_manip_position.yaml, 2026-08-27): grippers point straight down.
TABLETOP_HOME_Q = np.array(
    [-1.2994, 0.4105, -2.3000, -0.7982, 2.0193, 0.7225, 0.1107,
     -1.2994, -0.4105, 2.3000, -0.7982, -2.0193, 0.7225, -0.1107]
)
TUTORIAL_HOME_Q = np.array([-0.5, 0, 0, -0.5, 0, 0, 0] * 2)
ORIENTATION_WEIGHT = 0.5  # IK: keep the gripper vertical softly; position wins near the reach limit
# Reachable box for the kinematic targets (metres, world frame); keeps runaway targets off.
WORKSPACE_LOW = np.array([-0.60, -0.50, 0.80])
WORKSPACE_HIGH = np.array([-0.22, 0.50, 1.20])

DOCS = """\
Robot: bimanual ROBOTIS FFW SG2 standing at one end of a wooden table, torso at x=-0.8, in the
real robot's table-top start pose.
World frame: +x points from the robot across the table (forward), +y points to the robot's
left, +z is up. The table top is at z=0.80. Both grippers start pointing straight down with
the tool centre point (tcp) at about (-0.30, +0.31, 0.91) for the left arm and (-0.30, -0.31,
0.91) for the right arm. Reach (measured): each arm reaches its own half of the table and the
centre line (y=0) at any height, but not the far half at table height. Both objects are on the
left (+y) half, so use the left arm. Reachable x is about -0.55 to -0.25 (move in -x, toward
the robot, to reach the objects); z from 0.80 to 1.15.
Objects: a red cube 0.05 m on a side (centre reported 0.025 m above the table) near the
centre line of the table, and a flat white plate about 0.2 m across on the robot's left (+y)
side; both are within the left arm's reach, at x about -0.38. Positions are given in the
state when privileged state is enabled; otherwise locate them in the camera images.
Actions: move_by displacements in metres (x, y, z) and radians (roll, pitch, yaw, applied in the
tool frame; with the gripper pointing down, yaw turns the fingers about the vertical axis).
The arms are position-controlled and settle within about 0.5 s after a move ends; check the
reported tcp position after each move and correct any residual before grasping.
Gripper: `<arm>_gripper` is a displacement of the gripper command on [0, 1]; +1 closes fully,
-1 opens fully. The state reports the measured closure: 0 open, about 0.9 closed on nothing,
about 0.5 when closed on the cube. Closing takes about 1.5 s, so after closing wait (a
zero-motion move of a few steps) before lifting.
Gripper geometry: the two finger pads open along the tool's y axis, 0.057 m either side of the
tcp (0.114 m opening), and reach 0.04 m below the tcp; the wrist body sits 0.02 m below the tcp
between the pads. Top-down grasp of the cube:
  1. move over the cube centre at z of about 1.0 (leave roll, pitch and yaw at 0);
  2. descend until the tcp is 0.05 m above the cube centre (0.025 m above its top face), so the
     pads overlap the cube's upper half and the wrist stays just above it;
  3. close, wait about 1.5 s, then lift to z of about 1.05 and carry it slowly;
  4. place with the tcp 0.09 m above the plate centre, open, wait, lift the tcp back to 1.05.
State: `eef_pose` is [x, y, z, roll, pitch, yaw, gripper] for the left arm then the right arm,
in the same order as the action dimensions. `joint_pos` is the 14 arm joint angles.
Success: the cube rests on the plate (within 0.08 m of its centre), both grippers are open, and
neither tcp is within 0.12 m of the cube.
"""


class FFWMujocoEmbodiment(EmbodimentBase):
    """The FFW tabletop pick-and-place scene as an inspect-robots embodiment."""

    def __init__(
        self,
        *,
        privileged_state: bool = False,
        cameras: str = "agentview,topview,sideview",
        image_height: int = 300,
        image_width: int = 400,
        physics_substeps: int = 25,
        ik_iters: int = 100,
        gravity_comp: bool = True,
        home: str = "tabletop",
        seed: int = 0,
    ) -> None:
        self.privileged_state = bool(privileged_state)
        self.cameras = tuple(c.strip() for c in cameras.split(",") if c.strip())
        unknown = set(self.cameras) - set(ALL_CAMERAS)
        if unknown:
            raise ValueError(f"unknown cameras {sorted(unknown)}; choose from {ALL_CAMERAS}")
        self.image_hw = (int(image_height), int(image_width))
        self.physics_substeps = int(physics_substeps)
        self.ik_iters = int(ik_iters)
        self.gravity_comp = bool(gravity_comp)
        if home not in ("tabletop", "tutorial"):
            raise ValueError("home must be 'tabletop' (real-robot start pose) or 'tutorial'")
        self.home_q = TABLETOP_HOME_Q if home == "tabletop" else TUTORIAL_HOME_Q
        self._arm_dofs: np.ndarray | None = None
        self._seed = int(seed)
        self._sim: Any = None
        self._grip_cmd = np.zeros(2)
        self.num_steps = 0

        low = np.array([-POS_STEP] * 3 + [-ROT_STEP] * 3 + [-GRIP_STEP]) 
        low = np.concatenate([low, low])
        fields = [
            StateField(key="eef_pose", shape=(14,), unit="m,rad,normalized"),
            StateField(key="joint_pos", shape=(14,), unit="rad"),
        ]
        if self.privileged_state:
            fields += [StateField(key="cube_pos", shape=(3,), unit="m"), StateField(key="plate_pos", shape=(3,), unit="m")]
        self.info = EmbodimentInfo(
            name="ffw_mujoco",
            action_space=Box(
                shape=(14,),
                low=low,
                high=-low,
                semantics=ActionSemantics(
                    control_mode="eef_delta_pose",
                    rotation_repr="none",
                    gripper="binary",
                    frame="world",
                    dim_labels=DIM_LABELS,
                ),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(name=c, height=self.image_hw[0], width=self.image_hw[1], channels=3) for c in self.cameras),
                state=StateSpec(fields=tuple(fields)),
            ),
            control_hz=CONTROL_HZ,
            is_simulated=True,
            capabilities=frozenset({SEEDABLE, RESETTABLE, AUTO_RESET, PRIVILEGED_SUCCESS, RENDERABLE}),
            docs=DOCS + ("Privileged state: `cube_pos` and `plate_pos` are the objects' world positions.\n" if self.privileged_state else ""),
            environment_id="lerobot-mujoco-tutorial/asset/example_scene_llm.xml",
        )

    # ------------------------------------------------------------------ lifecycle
    def _ensure_sim(self) -> Any:
        if self._sim is None:
            from mujoco_env.y_env import SimpleEnv

            cwd = os.getcwd()
            os.chdir(REPO_ROOT)  # the scene XML references assets relative to the repo root
            try:
                self._sim = SimpleEnv.__new__(SimpleEnv)
                self._sim.home_q = self.home_q
                self._sim.obj_names = (OBJECT_BODY, TARGET_BODY)
                SimpleEnv.__init__(self._sim, str(SCENE_XML.relative_to(REPO_ROOT)), action_type="eef_pose", state_type="joint_angle", seed=self._seed, headless=True)
            finally:
                os.chdir(cwd)
        return self._sim

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Reset the robot to its home pose, place the cube and plate from the scene, and observe."""
        sim = self._ensure_sim()
        sim.reset(seed=self._seed if seed is None else int(seed))
        meta = scene.metadata or {}
        if "object_xy" in meta and "plate_xy" in meta:
            # Spawn at resting height; spawning inside the table launches the object.
            env = sim.env
            env.set_p_base_body(body_name=OBJECT_BODY, p=np.array([*meta["object_xy"], OBJECT_REST_Z + 0.003]))
            env.set_p_base_body(body_name=TARGET_BODY, p=np.array([*meta["plate_xy"], TARGET_REST_Z + 0.003]))
            env.data.qvel[:] = 0.0
            env.forward(increase_tick=False)
            for _ in range(100):
                sim.step_env()
        # Kinematic targets start at the forward kinematics of the commanded home pose
        # (sim.reset computes p0/R0 before the arm settles under gravity).
        self._p0 = {"l": sim.p0_l.copy(), "r": sim.p0_r.copy()}
        self._R0 = {"l": sim.R0_l.copy(), "r": sim.R0_r.copy()}
        self._q_cmd = {"l": sim.q[:7].copy(), "r": sim.q[7:14].copy()}
        self._grip_cmd[:] = 0.0
        self.num_steps = 0
        return self._observe(scene.instruction)

    def _solve_arm(self, arm: str, delta: np.ndarray) -> None:
        """Advance one arm's kinematic target by ``delta`` and re-solve its IK (warm started).

        Damped least squares on the stacked position and orientation errors, with the
        orientation rows down-weighted so the gripper stays near its commanded orientation
        where possible but position is still reached at the edge of the workspace.
        """
        from mujoco_env.transforms import rpy2r

        sim = self._sim
        env = sim.env
        self._p0[arm] = np.clip(self._p0[arm] + delta[:3], WORKSPACE_LOW, WORKSPACE_HIGH)
        self._R0[arm] = self._R0[arm] @ rpy2r(delta[3:6])
        joints = sim.joint_names_l if arm == "l" else sim.joint_names_r
        body = f"tcp_link_{arm}"
        idx_fwd = env.get_idxs_fwd(joint_names=joints)
        idx_jac = env.get_idxs_jac(joint_names=joints)
        lo = env.joint_ranges[[env.joint_names.index(j) for j in joints], 0]
        hi = env.joint_ranges[[env.joint_names.index(j) for j in joints], 1]
        q = self._q_cmd[arm].copy()
        env.store_state()
        try:
            env.forward(q=q, joint_idxs=idx_fwd, increase_tick=False)
            for _ in range(self.ik_iters):
                J, err = env.get_ik_ingredients(body_name=body, p_trgt=self._p0[arm], R_trgt=self._R0[arm], IK_P=True, IK_R=True)
                J = J[:, idx_jac]
                w = np.array([1.0] * 3 + [ORIENTATION_WEIGHT] * 3)
                Jw, ew = J * w[:, None], err * w
                dq = np.linalg.solve(Jw.T @ Jw + 1e-2 * np.eye(len(q)), Jw.T @ ew)
                step = np.linalg.norm(dq)
                if step > 0.1:  # at most ~6 degrees per iteration keeps the solve from oscillating
                    dq *= 0.1 / step
                q = np.clip(q + dq, lo, hi)
                env.forward(q=q, joint_idxs=idx_fwd, increase_tick=False)
                if np.linalg.norm(err[:3]) < 1e-3 and step < 1e-4:
                    break
            # Second stage: position only, so xyz is exact even where the orientation cannot be held.
            for _ in range(40):
                J, err = env.get_ik_ingredients(body_name=body, p_trgt=self._p0[arm], R_trgt=None, IK_P=True, IK_R=False)
                J = J[:, idx_jac]
                dq = np.linalg.solve(J.T @ J + 1e-3 * np.eye(len(q)), J.T @ err)
                step = np.linalg.norm(dq)
                if step > 0.05:
                    dq *= 0.05 / step
                q = np.clip(q + dq, lo, hi)
                env.forward(q=q, joint_idxs=idx_fwd, increase_tick=False)
                if np.linalg.norm(err) < 5e-4:
                    break
        finally:
            env.restore_state()
        self._q_cmd[arm] = q

    def _fk(self, joints: list[str], q: np.ndarray, body: str) -> np.ndarray:
        env = self._sim.env
        env.store_state()
        env.forward(q=q, joint_idxs=env.get_idxs_fwd(joint_names=joints), increase_tick=False)
        p = np.asarray(env.get_p_body(body), dtype=np.float64).copy()
        env.restore_state()
        return p

    def step(self, action: Action) -> StepResult:
        """Apply one 14-D displacement for one 20 Hz control tick and report privileged success."""
        sim = self._sim
        a = np.clip(np.asarray(action.data, dtype=np.float64), self.info.action_space.low, self.info.action_space.high)
        for arm, o in (("l", 0), ("r", 7)):
            if np.any(a[o : o + 6] != 0.0):
                self._solve_arm(arm, a[o : o + 6])
        self._grip_cmd = np.clip(self._grip_cmd + np.array([a[6], a[13]]), 0.0, 1.0)
        sim.q = np.concatenate([self._q_cmd["l"], self._q_cmd["r"], (self._grip_cmd > 0.5).astype(np.float64)])
        for _ in range(self.physics_substeps):
            if self.gravity_comp:
                self._apply_gravity_comp()
            sim.step_env()
        self.num_steps += 1
        success, dist = self._success()
        return StepResult(
            observation=self._observe(None),
            reward=-dist,
            terminated=success,
            termination_reason="success" if success else None,
            info={"success": success, "distance": dist},
        )

    def _apply_gravity_comp(self) -> None:
        """Cancel gravity and Coriolis torques on the arm joints so position control does not sag."""
        import mujoco

        sim = self._sim
        if self._arm_dofs is None:
            m = sim.env.model
            self._arm_dofs = np.array(
                [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in sim.joint_names]
            )
        d = sim.env.data
        d.qfrc_applied[self._arm_dofs] = d.qfrc_bias[self._arm_dofs]

    def close(self) -> None:
        """Free the offscreen GL context."""
        if self._sim is not None:
            self._sim.env.close_offscreen()
            self._sim = None

    # ------------------------------------------------------------------ helpers
    def _poses(self) -> dict[str, np.ndarray]:
        sim = self._sim
        from mujoco_env.transforms import r2rpy

        out: dict[str, np.ndarray] = {}
        for arm, body, joint in (("l", "tcp_link_l", "gripper_l_joint1"), ("r", "tcp_link_r", "gripper_r_joint1")):
            p, R = sim.env.get_pR_body(body_name=body)
            grip = float(np.clip(sim.env.get_qpos_joint(joint)[0] / 1.1, 0.0, 1.0))  # measured closure
            out[arm] = np.concatenate([p, r2rpy(R), [grip]]).astype(np.float64)
        out["cube"] = np.asarray(sim.env.get_p_body(OBJECT_BODY), dtype=np.float64)
        out["plate"] = np.asarray(sim.env.get_p_body(TARGET_BODY), dtype=np.float64)
        return out

    def _success(self) -> tuple[bool, float]:
        p = self._poses()
        dxy = float(np.linalg.norm(p["cube"][:2] - p["plate"][:2]))
        on_plate = dxy < 0.08 and 0.0 < (p["cube"][2] - p["plate"][2]) < 0.15
        grippers_open = p["l"][6] < 0.1 and p["r"][6] < 0.1
        clear = all(np.linalg.norm(p[arm][:3] - p["cube"]) > 0.12 for arm in ("l", "r"))
        return bool(on_plate and grippers_open and clear), dxy

    def _observe(self, instruction: str | None) -> Observation:
        import cv2

        sim = self._sim
        p = self._poses()
        images = {}
        for cam in self.cameras:
            rgb = sim.env.get_fixed_cam_rgb(cam_name=cam)
            images[cam] = np.ascontiguousarray(cv2.resize(rgb, (self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_AREA))
        state = {
            "eef_pose": np.concatenate([p["l"], p["r"]]),
            "joint_pos": np.asarray(sim.env.get_qpos_joints(joint_names=sim.joint_names), dtype=np.float64),
        }
        if self.privileged_state:
            state["cube_pos"], state["plate_pos"] = p["cube"], p["plate"]
        t = float(self.num_steps / CONTROL_HZ)
        return Observation(images=images, state=state, instruction=instruction, image_times={c: t for c in images}, state_time=t)


def ffw_mujoco_embodiment(**kwargs: Any) -> FFWMujocoEmbodiment:
    """Registry factory for ``--embodiment ffw_mujoco`` (kwargs come from ``-E key=value``)."""
    return FFWMujocoEmbodiment(**kwargs)
