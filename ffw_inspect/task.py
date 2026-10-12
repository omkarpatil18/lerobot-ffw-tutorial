"""The ``ffw-cube-on-plate`` benchmark: seeded tabletop layouts of the tutorial scene."""

from __future__ import annotations

from inspect_robots.scene import Scene
from inspect_robots.scorer import episode_length, success_at_end
from inspect_robots.task import Task

INSTRUCTION = "Pick up the red cube and place it on the white plate, then release it and lift the gripper away."


# Both objects on the left arm's side of the table (the real-robot start pose keeps the grippers
# vertical, and from there one arm cannot reach both halves of the table at table height).
CUBE_XY = (-0.38, 0.06)
PLATE_XY = (-0.36, 0.26)


def ffw_cube_on_plate(n_scenes: int = 1, max_steps: int = 1200, first_seed: int = 0, jitter: float = 0.02) -> Task:
    """Build the task: ``n_scenes`` seeded layouts, ``max_steps`` control steps each (20 Hz).

    Scene ``layout-0`` uses the nominal layout; later seeds jitter both objects by up to
    ``jitter`` metres. Positions travel in ``Scene.metadata`` and the embodiment applies them.
    """
    import numpy as np

    scenes = []
    for seed in range(first_seed, first_seed + int(n_scenes)):
        rng = np.random.RandomState(seed)
        d = rng.uniform(-jitter, jitter, size=4) if seed != 0 else np.zeros(4)
        scenes.append(
            Scene(
                id=f"layout-{seed}",
                instruction=INSTRUCTION,
                init_seed=seed,
                metadata={
                    "object_xy": [CUBE_XY[0] + d[0], CUBE_XY[1] + d[1]],
                    "plate_xy": [PLATE_XY[0] + d[2], PLATE_XY[1] + d[3]],
                },
            )
        )
    return Task(
        name="ffw-cube-on-plate",
        scenes=scenes,
        scorer=[success_at_end(), episode_length()],
        max_steps=int(max_steps),
    )
