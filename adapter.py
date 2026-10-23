"""Humanoid-GPT PNS216: author's 136D deployment input and PD action map."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from motion_challenge.adapters.onnx import ONNXController, yaw
from motion_challenge.resources import resource_root


class HumanoidGPT(ONNXController):
    action_output = 'continuous_actions'
    raw_action_history = True

    def __init__(self, root, runtime):
        super().__init__(root, runtime)
        # Kinematics only; all episode physics remains in the benchmark kernel.
        self.model = mujoco.MjModel.from_xml_path(str(resource_root()/'robots/g1-29dof-cpu-v0.xml'))
        self.data = mujoco.MjData(self.model)
        ids = [self.model.joint(n).id for n in self.names]
        self.qadr = self.model.jnt_qposadr[ids]
        self.vadr = self.model.jnt_dofadr[ids]
        self.pelvis = self.model.body('pelvis').id

    def reset(self, state, reference, seed):
        super().reset(state, reference, seed)
        self.actual_origin = np.asarray(state.root_position_world_m)
        self.reference_origin = np.asarray(reference[0].root_position_world_m)
        delta = yaw(R.from_quat(state.root_orientation_xyzw))-yaw(R.from_quat(reference[0].root_orientation_xyzw))
        self.alignment = R.from_rotvec([0, 0, float(delta)])

    def observations(self, state, reference):
        q, qd, omega, gravity, actual = self.proprio(state)
        current, future = reference
        r0, r1 = R.from_quat(current.root_orientation_xyzw), R.from_quat(future.root_orientation_xyzw)
        dt = self.runtime['control_period_s']
        self.data.qpos[:3] = future.root_position_world_m
        self.data.qpos[3:7] = np.asarray(future.root_orientation_xyzw)[[3, 0, 1, 2]]
        self.data.qpos[self.qadr] = np.asarray(future.joint_position_rad)[self.order]
        self.data.qvel[:3] = (np.asarray(future.root_position_world_m)-current.root_position_world_m)/dt
        # Preserve the author's reference qvel convention before mj_forward.
        self.data.qvel[3:6] = r0.apply((r0.inv()*r1).as_rotvec()/dt)
        self.data.qvel[self.vadr] = (np.asarray(future.joint_position_rad)-current.joint_position_rad)[self.order]/dt
        mujoco.mj_forward(self.model, self.data)
        heading_inverse = R.from_rotvec([0, 0, -float(yaw(r1))])
        cvel = self.data.cvel[self.pelvis].copy()
        cvel = np.r_[heading_inverse.apply(cvel[:3]), heading_inverse.apply(cvel[3:])]
        target = self.actual_origin+self.alignment.apply(np.asarray(current.root_position_world_m)-self.reference_origin)
        delta_xy = R.from_rotvec([0, 0, -float(yaw(actual))]).apply(target-np.asarray(state.root_position_world_m))[:2]
        delta_yaw = yaw(self.alignment*r0)-yaw(actual)
        command = np.r_[np.asarray(future.joint_position_rad)[self.order]-self.default,
                        future.root_position_world_m[2], r1.inv().apply([0, 0, -1]), cvel,
                        np.cos(delta_yaw), np.sin(delta_yaw), delta_xy]
        return {'obs': np.r_[omega, gravity, q-self.default, qd, self.last, command][None]}


def create(root, runtime):
    return HumanoidGPT(root, runtime)
