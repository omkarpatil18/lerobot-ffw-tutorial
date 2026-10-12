"""A privileged scripted pick-and-place policy for the FFW cube-on-plate scene.

It exists to validate the embodiment (IK, gripper, success check) end to end through
``eval()``; it reads the objects' world positions, so it needs
``FFWMujocoEmbodiment(privileged_state=True)``.
"""

from __future__ import annotations

import numpy as np

from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace
from inspect_robots.types import Action, ActionChunk, Observation

from ffw_inspect.embodiment import DIM_LABELS, GRIP_STEP, POS_STEP, ROT_STEP

GRASP_DZ = 0.05   # tcp height above the cube centre: pads overlap its upper half, wrist clears the top
PLACE_DZ = 0.09   # tcp height above the plate centre at release

_LOW = np.concatenate([[-POS_STEP] * 3 + [-ROT_STEP] * 3 + [-GRIP_STEP]] * 2)


class FFWOraclePolicy:
    """Scripted left-arm top-down cube grasp: hover, descend, close, lift, carry, lower, open, lift."""

    def __init__(self, *, settle_steps: int = 15, max_tries: int = 6):
        self.settle_steps = int(settle_steps)
        self.max_tries = int(max_tries)
        self.tries = 0
        self.phase = 0
        self.dwell = 0
        self._latched: dict[int, tuple[int, str, np.ndarray | float]] = {}
        self.info = PolicyInfo(
            name="ffw_oracle",
            action_space=Box(shape=(14,), low=_LOW, high=-_LOW, semantics=ActionSemantics(control_mode="eef_delta_pose", rotation_repr="none", gripper="binary", frame="world", dim_labels=DIM_LABELS)),
            observation_space=ObservationSpace(state_keys=frozenset({"eef_pose", "cube_pos", "plate_pos"})),
            control_hz=20.0,
        )
        self.config = PolicyConfig(action_horizon=64)

    def reset(self, scene: Scene) -> None:
        """Start again from the first phase."""
        self.phase, self.dwell, self.tries = 0, 0, 0
        self._latched = {}

    def _targets(self, obs: Observation) -> list[tuple[int, str, np.ndarray | float]]:
        cube = np.asarray(obs.state["cube_pos"], dtype=np.float64)
        plate = np.asarray(obs.state["plate_pos"], dtype=np.float64)
        L = 0
        return [
            (L, "move", np.array([cube[0], cube[1], 1.0])),
            (L, "fine", np.array([cube[0], cube[1], cube[2] + 0.12])),      # re-centre above the cube
            (L, "fine", np.array([cube[0], cube[1], cube[2] + GRASP_DZ])),  # pads around the upper half
            (L, "grip", 1.0),
            (L, "move", np.array([cube[0], cube[1], 1.05])),
            (L, "move", np.array([plate[0], plate[1], 1.05])),
            (L, "move", np.array([plate[0], plate[1], plate[2] + PLACE_DZ])),
            (L, "grip", -1.0),
            (L, "move", np.array([plate[0], plate[1], 1.05])),
        ]

    def act(self, observation: Observation) -> ActionChunk:
        """Emit the remaining displacement to the current phase target, then a short settle."""
        actions: list[Action] = []
        n_phases = len(self._targets(observation))
        while self.phase < n_phases and not actions:
            if self.phase not in self._latched:  # targets are fixed when a phase starts
                self._latched[self.phase] = self._targets(observation)[self.phase]
            o, kind, target = self._latched[self.phase]
            eef = np.asarray(observation.state["eef_pose"], dtype=np.float64)[o : o + 3]
            if kind == "yaw":
                n = int(np.ceil(abs(float(target)) / ROT_STEP))
                for _ in range(n):
                    a = np.zeros(14)
                    a[o + 5] = float(target) / n
                    actions.append(Action(data=a))
                actions += [Action(data=np.zeros(14)) for _ in range(self.settle_steps)]
                self.phase, self.tries = self.phase + 1, 0
            elif kind == "grip":
                a = np.zeros(14)
                a[o + 6] = float(target)
                actions.append(Action(data=a))
                actions += [Action(data=np.zeros(14)) for _ in range(30)]  # 1.5 s for the fingers
                self.phase, self.tries = self.phase + 1, 0
            else:
                delta = np.asarray(target) - eef
                dist = float(np.linalg.norm(delta))
                tol = 0.004 if kind == "fine" else 0.01   # the pads clear the rim by only 1 cm
                if dist < tol or self.tries >= self.max_tries:
                    self.phase, self.tries = self.phase + 1, 0
                    continue
                self.tries += 1
                steps = max(1, int(np.ceil(dist / POS_STEP)))
                per = delta / steps
                for _ in range(steps):
                    a = np.zeros(14)
                    a[o : o + 3] = per
                    actions.append(Action(data=a))
                actions += [Action(data=np.zeros(14)) for _ in range(self.settle_steps)]
        if not actions:
            actions.append(Action(data=np.zeros(14)))
        return ActionChunk(actions=actions, control_hz=20.0)


def ffw_oracle_policy(**kwargs: object) -> FFWOraclePolicy:
    """Registry factory for ``--policy ffw_oracle``."""
    return FFWOraclePolicy(**kwargs)  # type: ignore[arg-type]
