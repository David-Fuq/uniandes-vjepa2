import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

sys.path.insert(0, "..")

# --- UPDATE THESE TO MATCH YOUR FOLDER STRUCTURE ---
from postraining_uniandes.transforms import make_transforms
from notebooks.utils.mpc_utils import cem, compute_new_pose, poses_to_diff
from postraining_uniandes.path_planning.world_model_for_v4 import WorldModel
from postraining_uniandes.encoder_decoder_init import init_video_model
# ---------------------------------------------------

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

# Device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Available memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
# Init Model
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

print("Models loaded successfully.")

# 2. Data Preparation
# Calculate tokens
tokens_per_frame = int((crop_size // patch_size) ** 2)

transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=[1., 1.],
    random_resize_scale=[1., 1.],
    crop_size=crop_size,
)

trajectory = np.load(os.path.join(script_dir, "test_trajectory_2.npz"))
np_clips = trajectory["observations"] 
np_states = trajectory["states"]

# --- CRITICAL FIX FOR TUBELETS ---
# If tubelet_size=2, one latent frame represents 2 video frames.
# To do a prediction step (Context -> Target), we need 2 LATENT frames.
# 2 Latent Frames = 2 * tubelet_size Video Frames.
frames_needed = 2 * tubelet_size
print(f"[Data] Loading {frames_needed} frames to generate 2 Latent Steps.")

if np_clips.shape[1] < frames_needed:
    raise ValueError(f"Trajectory too short. Need {frames_needed} frames.")

# Load Batch 0, Frames 0 to N
video_tensor = torch.from_numpy(np_clips[0, 0:frames_needed]) # [T, H, W, C]
clips = transform(video_tensor).unsqueeze(0).to(device) # [1, C, T, H, W]

# Load States corresponding to the START of each latent block
# Latent 0 starts at t=0. Latent 1 starts at t=tubelet_size.
state_indices = [0, tubelet_size]
states = torch.tensor(np_states[0, state_indices]).unsqueeze(0).float().to(device) # [1, 2, 7]

# Ground Truth Action (from State 0 to State 1)
gt_action_vec = poses_to_diff(torch.tensor(np_states[0, 0]), torch.tensor(np_states[0, tubelet_size]))
actions_gt = gt_action_vec.unsqueeze(0).unsqueeze(0).float().to(device) # [1, 1, 7]

print(f"Clips: {clips.shape}")   # [1, 3, 4, 256, 256]
print(f"States: {states.shape}") # [1, 2, 7]

# 3. Helper Functions

def get_latent_reps(c):
    """Encodes video clips into latent representations without hacking."""
    with torch.no_grad():
        h = encoder(c) # [B, T_latent*Tokens, D]
        
        # Reshape to split time
        B, Total_Tokens, D = h.shape
        T_latent = Total_Tokens // tokens_per_frame
        
        h = h.view(B, T_latent, tokens_per_frame, D)
        h = F.layer_norm(h, (D,))
    return h

def forward_actions_landscape(z_start, s_start, nsamples, grid_size=0.075):
    """
    Sweeps actions to generate energy landscape.
    z_start: [1, Tokens, D] (Context)
    s_start: [1, 1, 7] (Context State)
    """
    # 1. Create Action Grid
    action_samples = []
    for da in np.linspace(-grid_size, grid_size, nsamples):
        for db in np.linspace(-grid_size, grid_size, nsamples):
            for dc in np.linspace(-grid_size, grid_size, nsamples):
                action_samples.append([da, db, dc, 0, 0, 0, 0])
    
    # [N, 1, 7]
    grid = torch.tensor(action_samples, device=device).unsqueeze(1) 
    num_samples = grid.shape[0]

    # 2. Prepare Batch Inputs
    # Expand Context z -> [N, Tokens, D]
    z_batch = z_start.repeat(num_samples, 1, 1)
    
    # Create Dummy Target -> [N, Tokens, D]
    z_dummy = torch.zeros_like(z_batch)
    
    # Combine for Predictor Input -> [N, 2*Tokens, D]
    # (Flattening happens here because we concat spatial blocks then flatten)
    z_in = torch.cat([z_batch.unsqueeze(1), z_dummy.unsqueeze(1)], dim=1) # [N, 2, Tokens, D]
    z_in_flat = z_in.flatten(1, 2)
    
    # Expand Start State -> [N, 1, 7]
    s_batch = s_start.repeat(num_samples, 1, 1)
    
    # Calculate Next State for every action in grid -> [N, 1, 7]
    s_next = compute_new_pose(s_batch, grid)
    
    # Combine States -> [N, 2, 7]
    s_in = torch.cat([s_batch, s_next], dim=1)
    
    # Actions -> [N, 1, 7]
    a_in = grid.float()

    print(f"Sweeping {num_samples} actions...")
    
    # 3. Predict
    with torch.no_grad():
        # Input shapes match predictor requirements:
        # z: [N, 2*Tokens, D]
        # a: [N, 1, 7] (Length T-1)
        # s: [N, 2, 7] (Length T)
        z_out_flat = predictor(z_in_flat, a_in, s_in)
        
        # Extract last frame (Target)
        z_pred = z_out_flat[:, -tokens_per_frame:]
        z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))
        
    return z_pred, grid

# 4. Run Energy Landscape
print("\n--- Running Energy Landscape ---")

# Encode
h_all = get_latent_reps(clips) # [1, 2, Tokens, D]
z_context = h_all[:, 0] # Frame 1
z_target = h_all[:, 1]  # Frame 2

# Sweep
nsamples = 5
z_preds, grid = forward_actions_landscape(z_context, states[:, 0:1], nsamples)

# Loss
target_batch = z_target.repeat(grid.shape[0], 1, 1)
loss = torch.mean(torch.abs(z_preds - target_batch), dim=[1, 2]).cpu().numpy()

# Plot
grid_cpu = grid.cpu().numpy()
heatmap_data = []
for i in range(len(loss)):
    heatmap_data.append((grid_cpu[i, 0, 0], grid_cpu[i, 0, 2], loss[i]))

dx = [x[0] for x in heatmap_data]
dz = [x[1] for x in heatmap_data]
err = [x[2] for x in heatmap_data]

heatmap, xedges, yedges = np.histogram2d(dx, dz, weights=err, bins=nsamples)

plt.figure(figsize=(8,6))
plt.imshow(heatmap.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="viridis_r")
plt.colorbar(label="Prediction Error")
plt.title(f"Energy Landscape (Tubelet={tubelet_size})")
plt.xlabel("Delta X")
plt.ylabel("Delta Z")
gt_x = actions_gt[0,0,0].item()
gt_z = actions_gt[0,0,2].item()
plt.scatter([gt_x], [gt_z], c='red', marker='*', s=200, label='GT Action')
plt.legend()
plt.savefig(os.path.join(script_dir, "energy_landscape_adapted_final.png"))
print("Saved energy_landscape_adapted_final.png")

print("Clearing CUDA cache before planning...")
torch.cuda.empty_cache()

# 5. Run CEM Planning
print("\n--- Running CEM Planning ---")

# Define World Model
wm = WorldModel(
    encoder=encoder,
    predictor=predictor,
    tokens_per_frame=tokens_per_frame,
    transform=transform,
    mpc_args={
        "rollout": 1, 
        "samples": 30,
        "topk": 5,
        "cem_steps": 5,
        "verbose": True
    },
    device=device
)

# Prepare inputs for CEM
# CEM needs [1, 1, Tokens, D] for context
z_start_cem = z_context.unsqueeze(1)
z_goal_cem = z_target.unsqueeze(1)
s_start_cem = states[:, 0:1]

action_cem = wm.infer_next_action(z_start_cem, s_start_cem, z_goal_cem)

print(f"GT Action:  {gt_x:.3f}, {actions_gt[0,0,1].item():.3f}, {gt_z:.3f}")
print(f"CEM Action: {action_cem[0,0]:.3f}, {action_cem[0,1]:.3f}, {action_cem[0,2]:.3f}")