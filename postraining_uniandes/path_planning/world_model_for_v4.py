# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F

from notebooks.utils.mpc_utils import cem, compute_new_pose


class WorldModel(object):

    def __init__(
        self,
        encoder,
        predictor,
        tokens_per_frame,
        transform,
        mpc_args={
            "rollout": 2,
            "samples": 400,
            "topk": 10,
            "cem_steps": 10,
            "momentum_mean": 0.15,
            "momentum_std": 0.15,
            "maxnorm": 0.05,
            "verbose": True,
        },
        normalize_reps=True,
        device="cuda:0",
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalize_reps = normalize_reps
        self.transform = transform
        self.tokens_per_frame = tokens_per_frame
        self.device = device
        self.mpc_args = mpc_args

    def infer_next_action(self, rep, pose, goal_rep, close_gripper=None):
        """
        rep: [1, 1, Tokens, D] (Initial Context)
        pose: [1, 1, 7] (Initial State)
        """

        def step_predictor_wrapper(frame_traj, action_traj, pose_traj):
            """
            Wraps predictor with MINI-BATCHING to prevent CUDA OOM.
            """
            
            # 1. Compute Physics (Next Pose) - fast, can be done in one go
            last_pose = pose_traj[:, -1:]    
            last_action = action_traj[:, -1:] 
            next_pose = compute_new_pose(last_pose, last_action) 
            
            # 2. Prepare Inputs
            states_in_full = torch.cat([pose_traj, next_pose], dim=1)
            actions_in_full = action_traj 
            
            B, T_curr, Tokens, D = frame_traj.shape
            dummy_z = torch.zeros((B, 1, Tokens, D), device=self.device, dtype=frame_traj.dtype)
            z_in_full = torch.cat([frame_traj, dummy_z], dim=1) 
            
            # --- MINI-BATCHING START ---
            # Process the samples in smaller chunks (e.g., 10 samples at a time)
            # This drastically reduces peak memory usage.
            chunk_size = 5  # Conservative chunk size (Adjust up to 10 or 20 if memory allows)
            num_samples = z_in_full.shape[0]
            next_reps_list = []
            
            for i in range(0, num_samples, chunk_size):
                # Slice the batch
                z_chunk = z_in_full[i : i + chunk_size]
                a_chunk = actions_in_full[i : i + chunk_size]
                s_chunk = states_in_full[i : i + chunk_size]
                
                # Flatten
                z_chunk_flat = z_chunk.flatten(1, 2)
                
                # Forward Pass (Only for this chunk)
                # This is where the OOM happened previously
                with torch.no_grad():
                    z_out_chunk = self.predictor(z_chunk_flat, a_chunk, s_chunk)
                
                # Extract and Normalize
                next_rep_chunk = z_out_chunk[:, -self.tokens_per_frame:] 
                if self.normalize_reps:
                    next_rep_chunk = F.layer_norm(next_rep_chunk, (next_rep_chunk.size(-1),))
                
                next_reps_list.append(next_rep_chunk)
                
            # Concatenate chunks back together
            next_rep = torch.cat(next_reps_list, dim=0)
            # --- MINI-BATCHING END ---
            
            # Reshape back to [S, 1, Tokens, D]
            next_rep = next_rep.view(B, 1, Tokens, D)
            
            return next_rep, next_pose

        mpc_action = cem(
            context_frame=rep,
            context_pose=pose,
            goal_frame=goal_rep,
            world_model=step_predictor_wrapper,
            close_gripper=close_gripper,
            **self.mpc_args,
        )[0]

        return mpc_action