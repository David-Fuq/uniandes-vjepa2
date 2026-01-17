import numpy as np
import torch
import torch.nn.functional as F
from postraining_uniandes.path_planning.mpc_utils_v2 import cem, compute_new_pose

class WorldModel(object):
    def __init__(self, encoder, predictor, tokens_per_frame, transform, mpc_args, normalize_reps=True, device="cuda:0"):
        self.encoder = encoder
        self.predictor = predictor
        self.tokens_per_frame = tokens_per_frame
        self.transform = transform
        self.mpc_args = mpc_args
        self.normalize_reps = normalize_reps
        self.device = device

    def encode(self, image):
        # Image input is [H, W, 3] numpy
        clip = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).unsqueeze(2) # [B, C, T=1, H, W]
        clip = self.transform(clip).to(self.device)
        
        with torch.no_grad():
            h = self.encoder(clip)
            # Flatten to [B, T*Tokens, D]
            if h.dim() == 4: # [B, T, Tokens, D]
                 h = h.flatten(1, 2)
            
            if self.normalize_reps:
                h = F.layer_norm(h, (h.size(-1),))
        return h

    def infer_next_action(self, rep, pose, goal_rep, close_gripper=None):
        """
        rep: [B, Tokens, D] (Current Visual State)
        pose: [B, 1, 7] (Current Physical State)
        goal_rep: [B, Tokens, D] (Target Visual State)
        """

        def step_predictor(curr_reps, action_seq, pose_seq):
            # curr_reps: [N, Tokens, D]
            # action_seq: [N, 1, 7] (CEM generates this)
            # pose_seq: [N, 1, 7] (Current pose)
            
            # 1. Calculate Next Pose based on Action (Physics)
            next_pose = compute_new_pose(pose_seq, action_seq) # [N, 1, 7]
            
            # 2. Prepare States for Predictor (Needs [s_t, s_{t+1}])
            states_in = torch.cat([pose_seq, next_pose], dim=1) # [N, 2, 7]
            
            # 3. Predict Next Frame
            # predictor(z, actions, states)
            pred_reps = self.predictor(curr_reps, action_seq, states_in) 
            
            # 4. Normalize
            if self.normalize_reps:
                pred_reps = F.layer_norm(pred_reps, (pred_reps.size(-1),))
                
            # Return next_rep and next_pose
            return pred_reps, next_pose

        # Run Cross-Entropy Method
        mpc_action = cem(
            context_frame=rep,
            context_pose=pose,
            goal_frame=goal_rep,
            world_model=step_predictor,
            close_gripper=close_gripper,
            **self.mpc_args,
        )[0] # Return the best action sequence found

        return mpc_action