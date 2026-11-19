"""CPU diagnostic only: paired actuator integration from raw motion frame zero."""
import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import mujoco
import numpy as np
import torch

from motion_challenge.resources import resource_root
from motion_challenge.sdk import DeployableStatePacket, ReferenceFrame
from motion_challenge.simulation.physics import World, command_values
from motion_challenge.simulation.tensor_reference import TensorMotionBank, TensorReference
from motion_challenge.validation import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    root = args.base / 'MotionChallenge-humanoid-gpt'
    sys.path.insert(0, str(root))
    sys.modules['onnxruntime'] = types.ModuleType('onnxruntime')
    from adapter import HumanoidGPT
    from reference import ReferenceFeatures
    from torch_graph import TensorGraph
    from motion_challenge.adapters.onnx import vector
    torch.set_num_threads(1)
    runtime = json.loads((root / 'runtime.mjwarp.json').read_text())
    cfg = json.loads((root / 'policy.json').read_text())
    protocol_path = resource_root() / 'suites/lafan/protocol.mjwarp311-no-linear-velocity-v1.json'
    protocol = json.loads(protocol_path.read_text())
    profile = resource_root() / protocol['robot_profile']
    meta = json.loads((args.prepared / 'dataset.json').read_text())
    bank = TensorMotionBank(np.load(args.prepared / 'poses.npy'), meta['starts'], meta['lengths'],
                            np.full(len(meta['names']), meta['fps']), meta['names'], 'cpu')
    network = TensorGraph(args.base / 'converted-weights/humanoid-gpt', 'cpu', runtime['weights_sha256'])
    selected = list(range(len(meta['names']))) if args.all else [meta['names'].index(n) for n in
        ('g1/dance2_subject4.csv', 'g1/fallAndGetUp1_subject1.csv')]
    offsets = torch.tensor(runtime['reference_offsets_s'], dtype=torch.float64)
    report = {'status': 'running', 'scope': 'CPU diagnostic, no production changes',
              'source_sha256': runtime['weights_sha256'], 'script_sha256': sha256_file(Path(__file__)),
              'mujoco_version': mujoco.__version__, 'profile_sha256': sha256_file(profile),
              'initialization': 'Current TensorReference.initial raw motion frame zero, including finite-difference velocities',
              'observations': 'Original author adapter observations, private EMA/FK; same converted tensor network on CPU',
              'timing': '20ms control; four 5ms substeps, exact final endpoint',
              'variants': ['external_pd', 'builtin_affine_pd', 'zero_torque'], 'cases': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for clip in selected:
        for mode in report['variants']:
            started = time.monotonic()
            world = World(profile)
            model, data = world.model, world.data
            ref = TensorReference(world, bank, len(offsets), 'cpu')
            ids = torch.tensor([clip])
            qpos, qvel = ref.initial(ids)
            data.qpos[:] = qpos[0].numpy()
            data.qvel[:] = qvel[0].numpy()
            mujoco.mj_forward(model, data)
            controller = HumanoidGPT.__new__(HumanoidGPT)
            controller.cfg, controller.runtime = cfg, runtime
            controller.names = cfg['policy_joint_names']
            controller.default = vector(cfg['default_joint_pos'], controller.names)
            controller.scale = vector(cfg['action_scale'], controller.names, 1.)
            controller.kp = vector(cfg['joint_kp'], controller.names)
            controller.kd = vector(cfg['joint_kd'], controller.names)
            controller.reference = ReferenceFeatures(root, runtime, controller.names)
            def infer(self, inputs):
                with torch.inference_mode():
                    return {k: v.numpy() for k, v in network(**{k: torch.as_tensor(np.asarray(v, np.float32))
                                                               for k, v in inputs.items()}).items()}
            controller.infer = types.MethodType(infer, controller)
            if mode == 'builtin_affine_pd':
                model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
                model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
                model.actuator_ctrllimited[:] = False
                model.actuator_forcelimited[:] = True
                model.actuator_forcerange[:, 0] = -world.limits
                model.actuator_forcerange[:, 1] = world.limits
            duration = float(bank.duration[clip])
            nsteps = math.ceil(duration / .02 - 1e-10)
            outcome = {'clip': meta['names'][clip], 'variant': mode, 'duration_s': duration,
                       'status': 'complete', 'initial_qvel_max_abs': float(abs(data.qvel).max()),
                       'peak_qvel': float(abs(data.qvel).max()), 'peak_target': 0., 'samples': []}
            for step in range(nsteps):
                clock = step * .02
                requested = torch.tensor([[clock]], dtype=torch.float64) + offsets
                batch = ref.motion(ids, requested)
                frames = tuple(ReferenceFrame(float(batch.requested_time_s[0, i]), float(batch.source_time_s[0, i]),
                    tuple(batch.joint_position_rad[0, i].numpy()), tuple(batch.root_position_world_m[0, i].numpy()),
                    tuple(batch.root_orientation_xyzw[0, i].numpy()), tuple(batch.joint_velocity_rad_s[0, i].numpy()),
                    tuple(batch.root_linear_velocity_world_m_s[0, i].numpy()), tuple(batch.root_angular_velocity_world_rad_s[0, i].numpy()))
                    for i in range(len(offsets)))
                full = world.state()
                state = DeployableStatePacket(clock, world.joint_names, full.joint_position_rad,
                    full.joint_velocity_rad_s, full.root_orientation_xyzw, full.root_angular_velocity_world_rad_s)
                if step == 0:
                    controller.reset(state, frames, 0)
                try:
                    q, v, kp, kd, ff = command_values(controller.act(state, frames), 29)
                except (ValueError, OverflowError) as error:
                    outcome.update(status='invalid_policy_command', error=str(error), failure_time_s=clock)
                    break
                outcome['peak_target'] = max(outcome['peak_target'], float(abs(q).max()))
                if mode == 'builtin_affine_pd':
                    model.actuator_gainprm[:, 0] = kp
                    model.actuator_biasprm[:, 0] = kd * v + ff
                    model.actuator_biasprm[:, 1] = -kp
                    model.actuator_biasprm[:, 2] = -kd
                    data.ctrl[:] = q
                control_dt = min(.02, duration - clock)
                model.opt.timestep = control_dt / 4
                for substep in range(4):
                    if mode == 'external_pd':
                        torque = kp * (q - data.qpos[world.qadr]) + kd * (v - data.qvel[world.vadr]) + ff
                        data.ctrl[:] = np.clip(torque, -world.limits, world.limits)
                    elif mode == 'zero_torque':
                        data.ctrl[:] = 0
                    before = float(data.time)
                    mujoco.mj_step(model, data)
                    warnings = [{'type': mujoco.mjtWarning(i).name, 'number': int(w.number), 'lastinfo': int(w.lastinfo)}
                                for i, w in enumerate(data.warning) if w.number]
                    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
                    if finite:
                        outcome['peak_qvel'] = max(outcome['peak_qvel'], float(abs(data.qvel).max()))
                    if warnings or not finite or abs(float(data.time) - before - model.opt.timestep) > 1e-9:
                        outcome.update(status='physics_failure', failure_time_s=before, control_step=step,
                                       substep=substep+1, warnings=warnings, finite=finite)
                        break
                if outcome['status'] != 'complete':
                    break
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                mujoco.mj_comVel(model, data)
                if step % 250 == 0 or step + 1 == nsteps:
                    outcome['samples'].append({'time_s': float(data.time), 'qvel_max_abs': float(abs(data.qvel).max()),
                                                'root_z': float(data.qpos[2]), 'target_max_abs': float(abs(q).max())})
            outcome['wall_s'] = time.monotonic() - started
            outcome['final_time_s'] = float(data.time)
            report['cases'].append(outcome)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(json.dumps({k:v for k,v in outcome.items() if k != 'samples'}), flush=True)
    report['status'] = 'complete'
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
