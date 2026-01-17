import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

sys.path.insert(0, "..")

from postraining_uniandes.transforms import make_transforms
from notebooks.utils.mpc_utils import cem, compute_new_pose, poses_to_diff
from postraining_uniandes.path_planning.world_model_for_v4 import WorldModel
from postraining_uniandes.encoder_decoder_init import init_video_model

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
main_dir  = os.path.dirname(parent_dir)

# 1. Load Config & Model
checkpoint_dir = os.path.join(main_dir, "final_dataset_40_epoch_frozen_encoder")
config_path = os.path.join(checkpoint_dir, "params-pretrain.yaml")

with open(config_path, "r") as f:
    args = yaml.safe_load(f)

cfgs_model = args.get("model")
cfgs_data = args.get("data")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

patch_size = cfgs_data.get("patch_size", 16)
crop_size = cfgs_data.get("crop_size", 256)
tubelet_size = cfgs_data.get("tubelet_size", 2) 

print(f"[Init] Initializing model with Tubelet Size: {tubelet_size}")

encoder, predictor = init_video_model(
    uniform_power=cfgs_model.get("uniform_power", False),
    device=device,
    patch_size=patch_size,
    max_num_frames=512,
    tubelet_size=tubelet_size,
    model_name=cfgs_model.get("model_name"),
    crop_size=crop_size,
    pred_depth=cfgs_model.get("pred_depth"),
    pred_num_heads=cfgs_model.get("pred_num_heads"),
    pred_embed_dim=cfgs_model.get("pred_embed_dim"),
    action_embed_dim=7,
    pred_is_frame_causal=True,
    use_extrinsics=cfgs_model.get("use_extrinsics", False),
    use_rope=cfgs_model.get("use_rope", False)
)

# Load Checkpoint
checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

def remove_module_prefix(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

encoder.load_state_dict(remove_module_prefix(checkpoint["encoder"]))
predictor.load_state_dict(remove_module_prefix(checkpoint["predictor"]))
encoder.eval()
predictor.eval()

# 2. Data Preparation
tokens_per_frame = int((crop_size // patch_size) ** 2)

transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=[1., 1.],
    random_resize_scale=[1., 1.],
    crop_size=crop_size,
)

trajectory = np.load(os.path.join(script_dir, "test_trajectory_4.npz"))
np_clips = trajectory["observations"] 
np_states = trajectory["states"]

# ### CHANGE 1: Define Rollout Steps
# We want 3 actions.
# Since tubelet_size=2, this covers 3 * 2 = 6 video frames total.
rollout_steps = 3 
frames_needed = (rollout_steps + 1) * tubelet_size 

print(f"[Data] Planning {rollout_steps} actions (Model Steps).")
print(f"[Data] Each action covers {tubelet_size} frames.")
print(f"[Data] Need {frames_needed} video frames.")

if np_clips.shape[1] < frames_needed:
    raise ValueError(f"Trajectory too short. Need {frames_needed} frames, have {np_clips.shape[1]}.")

# Load Clips
video_tensor = torch.from_numpy(np_clips[0, 0:frames_needed]) # [T, H, W, C]
clips = transform(video_tensor).unsqueeze(0).to(device) # [1, C, T, H, W]

# Load States
# We need states at intervals of tubelet_size: 0, 2, 4, 6...
state_indices = np.arange(0, frames_needed + 1, tubelet_size)
# Ensure we don't go out of bounds if frames_needed aligns perfectly
state_indices = state_indices[state_indices < np_states.shape[1]]

states = torch.tensor(np_states[0, state_indices]).unsqueeze(0).float().to(device)

print(f"Clips shape: {clips.shape}")
print(f"States shape (Latent Steps): {states.shape}") 

# 3. Helper Functions
def get_latent_reps(c):
    with torch.no_grad():
        h = encoder(c) 
        B, Total_Tokens, D = h.shape
        T_latent = Total_Tokens // tokens_per_frame
        h = h.view(B, T_latent, tokens_per_frame, D)
        h = F.layer_norm(h, (D,))
    return h

# 4. Run CEM Planning
print("\n--- Running CEM Planning (Multi-Step) ---")

# Get Representations
h_all = get_latent_reps(clips) # [1, T_latent, Tokens, D]

# ### CHANGE 2: Define Start and Goal for MPC
# Start is the first latent frame
z_start_cem = h_all[:, 0].unsqueeze(1) # [1, 1, Tokens, D]
s_start_cem = states[:, 0:1]           # [1, 1, 7]

# Goal is the latent frame at the end of the rollout
# If we want 3 actions, we aim for latent frame index 3 (0->1, 1->2, 2->3)
z_goal_cem = h_all[:, rollout_steps].unsqueeze(1) # [1, 1, Tokens, D]

print(f"Start State (t=0): {s_start_cem[0,0,:3].cpu().numpy()}")
print(f"Goal Latent Index: {rollout_steps}")

wm = WorldModel(
    encoder=encoder,
    predictor=predictor,
    tokens_per_frame=tokens_per_frame,
    transform=transform,
    mpc_args={
        "rollout": rollout_steps,  # ### CHANGE 3: Set to 3
        "samples": 100,            # Increased samples for multi-step
        "topk": 20,
        "cem_steps": 10,
        "verbose": True
    },
    device=device
)

# This will now return a sequence of actions [1, rollout, 7]
planned_actions = wm.infer_next_action(z_start_cem, s_start_cem, z_goal_cem)

# 5. Compare Results
print(f"\n{'='*80}")
print(f"RESULTS (Tubelet Size = {tubelet_size})")
print(f"{'='*80}")
print(f"Planned action sequence shape: {planned_actions.shape}")

print(f"\nAction sequence comparison:")
print(f"{'Step':<6} {'Planned (x,y,z)':<30} {'Ground Truth (x,y,z)':<30} {'Error':<10}")
print(f"{'-'*80}")

planned_actions_np = planned_actions.cpu().numpy()
total_error = 0.0

for i in range(rollout_steps):
    # --- FIX IS HERE ---
    # The shape is [3, 7], so we just index [i] to get the ith action vector
    pred = planned_actions_np[i] 
    
    # If tubelet=2, action i moves from frame (i*2) to frame (i*2 + 2)
    t_start = i * tubelet_size
    t_end = (i + 1) * tubelet_size
    
    # Compute Ground Truth
    gt_action = poses_to_diff(
        torch.tensor(np_states[0, t_start]), 
        torch.tensor(np_states[0, t_end])
    ).numpy()
    
    # Ensure gt_action is 1D for subtraction
    gt_action = gt_action.flatten()

    error = np.linalg.norm(pred[:3] - gt_action[:3])
    total_error += error
    
    print(f"{i+1:<6} ({pred[0]:6.4f},{pred[1]:6.4f},{pred[2]:6.4f})      "
          f"({gt_action[0]:6.4f},{gt_action[1]:6.4f},{gt_action[2]:6.4f})      "
          f"{error:6.4f}")

avg_error = total_error / rollout_steps
print(f"{'-'*80}")
print(f"Average xyz error: {avg_error:.4f}")
print(f"{'='*80}")