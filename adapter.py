"""Humanoid-GPT 264: the author's non-privileged 136D deployment input."""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation as R

from motion_challenge.adapters.onnx import ONNXController
from reference import ReferenceFeatures, quat_heading


class HumanoidGPT(ONNXController):
    action_output = 'continuous_actions'
    raw_action_history = True

    def __init__(self, root, runtime):
        super().__init__(root, runtime)
        self.reference = ReferenceFeatures(root, runtime, self.names)

    def reset(self, state, reference, seed):
        super().reset(state, reference, seed)
        self.reference.reset(self.order)

    def observations(self, state, reference):
        q, qd, omega, gravity, _ = self.proprio(state)
        current, future = self.reference.update(reference)
        actual_heading = quat_heading(np.asarray(state.root_orientation_xyzw)[[3, 0, 1, 2]])
        position_error = R.from_euler('z', -float(actual_heading)).apply(
            current['qpos'][:3] - np.asarray(state.root_position_world_m))[:2]
        heading_error = quat_heading(current['qpos'][3:7]) - actual_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        command = np.r_[future['qpos'][7:] - self.default.astype(np.float32), future['qpos'][2],
                        future['gravity'], future['cvel'],
                        np.cos(heading_error), np.sin(heading_error), position_error]
        return {'obs': np.r_[omega, gravity, q - self.default, qd, self.last, command][None]}

    def act(self, state, reference):
        command = super().act(state, reference)
        return replace(command, joint_position_target_rad=tuple(
            float(np.float32(x)) for x in command.joint_position_target_rad))


def create(root, runtime):
    return HumanoidGPT(root, runtime)
