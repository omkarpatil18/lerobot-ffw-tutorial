"""inspect-robots components for the FFW mug-on-plate MuJoCo scene.

Registered through entry points (see pyproject.toml):

- embodiment ``ffw_mujoco``: the bimanual FFW SG2 in the tutorial's tabletop scene
- task ``ffw-cube-on-plate``: pick the red cube and place it on the plate
- policy ``ffw_oracle``: a privileged scripted policy used to validate the embodiment
"""

from ffw_inspect.embodiment import FFWMujocoEmbodiment, ffw_mujoco_embodiment
from ffw_inspect.oracle import FFWOraclePolicy, ffw_oracle_policy
from ffw_inspect.task import ffw_cube_on_plate

__all__ = [
    "FFWMujocoEmbodiment",
    "FFWOraclePolicy",
    "ffw_cube_on_plate",
    "ffw_mujoco_embodiment",
    "ffw_oracle_policy",
]
