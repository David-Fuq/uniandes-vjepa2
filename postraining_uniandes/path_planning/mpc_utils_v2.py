# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# --- FIX 1: Update Import to match your project structure ---
from postraining_uniandes.logger_helper import get_logger
# ------------------------------------------------------------

logger = get_logger() # Removed __name__ arg to match your logger helper signature if needed


def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)


def round_small_elements(tensor, threshold):
    mask = torch.abs(tensor) < threshold
    new_tensor = tensor.clone()
    new_tensor[mask] = 0
    return new_tensor


def cem(
    context_frame,
    context_pose,
    goal_frame,
    world_model,
    rollout=1,
    cem_steps=100,
    momentum_mean=0.25,
    momentum_std=0.95,
    momentum_mean_gripper=0.15,
    momentum_std_gripper=0.15,
    samples=100,
    topk=10,
    verbose=False,
    maxnorm=0.05,
    axis={},
    objective=l1,
    close_gripper=None,
):
    """
    :param context_frame: [B=1, T=1, HW, D]
    :param goal_frame: [B=1, T=1, HW, D]
    :param world_model: f(context_frame, action) -> next_frame [B, 1, HW, D]
    :return: [B=1, rollout, 7] an action trajectory over rollout horizon

    Cross-Entropy Method
    -----------------------
    1. for rollout horizon:
    1.1. sample several actions
    1.2. compute next states using WM
    3. compute similarity of final states to goal_frames
    4. select topk samples and update mean and std using topk action trajs
    5. choose final action to be mean of distribution
    """
    # Reshape inputs to batch size of 'samples'
    context_frame = context_frame.repeat(samples, 1, 1, 1)  # [S, 1, HW, D]
    goal_frame = goal_frame.repeat(samples, 1, 1, 1)  # [S, 1, HW, D]
    context_pose = context_pose.repeat(samples, 1, 1)  # [S, 1, 7]

    # Current estimate of the mean/std of distribution over action trajectories
    mean = torch.cat(
        [
            torch.zeros((rollout, 3), device=context_frame.device),
            torch.zeros((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    std = torch.cat(
        [
            torch.ones((rollout, 3), device=context_frame.device) * maxnorm,
            torch.ones((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    for ax in axis.keys():
        mean[:, ax] = axis[ax]

    def sample_action_traj():
        """Sample several action trajectories"""
        action_traj, frame_traj, pose_traj = None, context_frame, context_pose

        for h in range(rollout):
            # -- sample new action
            action_samples = torch.randn(samples, mean.size(1), device=mean.device) * std[h] + mean[h]
            action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-maxnorm, max=maxnorm)
            action_samples[:, -1:] = torch.clip(action_samples[:, -1:], min=-0.75, max=0.75)
            
            for ax in axis.keys():
                action_samples[:, ax] = axis[ax]
            
            # Construct 7D action (dx, dy, dz, dr, dp, dy, gripper)
            action_samples = torch.cat(
                [
                    action_samples[:, :3],
                    torch.zeros((len(action_samples), 3), device=mean.device),
                    action_samples[:, -1:],
                ],
                dim=-1,
            )[:, None] # [S, 1, 7]

            if close_gripper is not None and h >= close_gripper:
                action_samples[:, :, -1] = 1.0

            action_traj = (
                torch.cat([action_traj, action_samples], dim=1) if action_traj is not None else action_samples
            )

            # -- compute next state
            # Note: We pass the *accumulated* trajectories. The Wrapper must handle picking the last step.
            next_frame, next_pose = world_model(frame_traj, action_traj, pose_traj)
            
            frame_traj = torch.cat([frame_traj, next_frame], dim=1)
            pose_traj = torch.cat([pose_traj, next_pose], dim=1)

        return action_traj, frame_traj

    def select_topk_action_traj(final_state, goal_state, actions):
        """Get the topk action trajectories that bring us closest to goal"""
        sims = objective(final_state.flatten(1), goal_state.flatten(1))
        indices = sims.topk(topk, largest=False).indices
        selected_actions = actions[indices]
        return selected_actions

    # Main Optimization Loop
    for step in tqdm(range(cem_steps), disable=not verbose):
        action_traj, frame_traj = sample_action_traj()
        
        selected_actions = select_topk_action_traj(
            final_state=frame_traj[:, -1], goal_state=goal_frame, actions=action_traj
        )
        mean_selected_actions = selected_actions.mean(dim=0)
        std_selected_actions = selected_actions.std(dim=0)

        # -- Update new sampling mean and std based on the top-k samples
        mean = torch.cat(
            [
                mean_selected_actions[..., :3] * (1.0 - momentum_mean) + mean[..., :3] * momentum_mean,
                mean_selected_actions[..., -1:] * (1.0 - momentum_mean_gripper)
                + mean[..., -1:] * momentum_mean_gripper,
            ],
            dim=-1,
        )
        std = torch.cat(
            [
                std_selected_actions[..., :3] * (1.0 - momentum_std) + std[..., :3] * momentum_std,
                std_selected_actions[..., -1:] * (1.0 - momentum_std_gripper) + std[..., -1:] * momentum_std_gripper,
            ],
            dim=-1,
        )

        if verbose:
            logger.info(f"Step {step} | Mean: {mean.sum(dim=0).cpu().numpy()} | Std: {std.sum(dim=0).cpu().numpy()}")

    new_action = torch.cat(
        [
            mean[..., :3],
            torch.zeros((rollout, 3), device=mean.device),
            round_small_elements(mean[..., -1:], 0.25),
        ],
        dim=-1,
    )[None, :]

    return new_action


def compute_new_pose(pose, action):
    """
    Computes kinematics. 
    Can accept sequences [B, T, 7] or single steps [B, 1, 7].
    ALWAYS uses the *last* element in the sequence to compute the update.
    """
    device, dtype = pose.device, pose.dtype
    
    # --- FIX 2: Safer Slicing ---
    # Convert to numpy and take the LAST element ([:, -1])
    # This ensures that if we pass a history of poses, we update from the most recent one.
    pose_np = pose[:, -1].detach().cpu().numpy()
    action_np = action[:, -1].detach().cpu().numpy()
    # ----------------------------

    # -- compute delta xyz
    new_xyz = pose_np[:, :3] + action_np[:, :3]
    
    # -- compute delta theta
    thetas = pose_np[:, 3:6]
    delta_thetas = action_np[:, 3:6]
    
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas]
    
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 3]
    
    # -- compute delta gripper
    new_closedness = pose_np[:, -1:] + action_np[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)
    
    # -- new pose
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)
    
    # Return as [B, 1, 7] to append to trajectory
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]


def poses_to_diff(start, end):
    """
    :param start: [7]
    :param end: [7]
    """
    try:
        start = start.cpu().numpy()
        end = end.cpu().numpy()
    except Exception:
        pass

    # --

    s_xyz = start[:3]
    e_xyz = end[:3]
    xyz_diff = e_xyz - s_xyz

    # --

    s_thetas = start[3:6]
    e_thetas = end[3:6]
    s_rotation = Rotation.from_euler("xyz", s_thetas, degrees=False).as_matrix()
    e_rotation = Rotation.from_euler("xyz", e_thetas, degrees=False).as_matrix()
    rotation_diff = e_rotation @ s_rotation.T
    theta_diff = Rotation.from_matrix(rotation_diff).as_euler("xyz", degrees=False)

    # --

    s_gripper = start[-1:]
    e_gripper = end[-1:]
    gripper_diff = e_gripper - s_gripper

    action = np.concatenate([xyz_diff, theta_diff, gripper_diff], axis=0)
    return torch.from_numpy(action)