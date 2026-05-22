from __future__ import annotations
from pathlib import Path
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


class HumanoidHoleEnv_v2(HumanoidEnv):
    """
    HumanoidHole-v0 환경

    - Gymnasium 기본 Humanoid-v5 환경 로직을 그대로 사용하면서..!
    - MuJoCo 모델(xml_file)만 커스텀 humanoid_hole.xml로 교체
    """

    def __init__(self, **kwargs):
        xml_path = Path(__file__).resolve().parent / "humanoid_hole_v2.xml"
        super().__init__(
            xml_file=str(xml_path),
            **kwargs,
        )
