import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

import yaml

sys.path.insert(0, "..")

# --- UPDATE THESE IMPORTS TO MATCH YOUR FILE STRUCTURE ---
from postraining_uniandes.transforms import make_transforms
from postraining_uniandes.path_planning.mpc_utils_v2 import compute_new_pose, poses_to_diff
from postraining_uniandes.path_planning.world_model_wrapper_v2 import WorldModel
from postraining_uniandes.encoder_decoder_init import init_video_model
# ---------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load your specific models
device = "cuda:3" if torch.cuda.is_available() else "cpu"
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
main_dir  = os.path.dirname(parent_dir)
print(f"Script dir: {script_dir}; Parent dir: {parent_dir}")
# Load config file

checkpoint_dir = os.path.join(main_dir, "final_dataset_40_epoch_frozen_encoder")

print(f"Checkpoint dir: {checkpoint_dir}")

config_path = os.path.join(checkpoint_dir, "params-pretrain.yaml")

print(f"Loading config from: {config_path}")

with open(config_path, "r") as f:
    args = yaml.safe_load(f)


# Extract model configuration
cfgs_model = args.get("model")
cfgs_data = args.get("data")
model_name = cfgs_model.get("model_name")
pred_depth = cfgs_model.get("pred_depth")
pred_num_heads = cfgs_model.get("pred_num_heads")
pred_embed_dim = cfgs_model.get("pred_embed_dim")
pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
uniform_power = cfgs_model.get("uniform_power", False)
use_rope = cfgs_model.get("use_rope", False)
use_extrinsics = cfgs_model.get("use_extrinsics", False)
use_sdpa = args.get("meta", {}).get("use_sdpa", False)
use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)

crop_size = cfgs_data.get("crop_size", 256)
patch_size = cfgs_data.get("patch_size")
tubelet_size = cfgs_data.get("tubelet_size")

# Initialize models with same architecture as training
encoder, predictor = init_video_model(
    uniform_power=uniform_power,
    device="cpu",  # Load to CPU first
    patch_size=patch_size,
    max_num_frames=512,
    tubelet_size=tubelet_size,
    model_name=model_name,
    crop_size=crop_size,
    pred_depth=pred_depth,
    pred_num_heads=pred_num_heads,
    pred_embed_dim=pred_embed_dim,
    action_embed_dim=7,
    pred_is_frame_causal=pred_is_frame_causal,
    use_extrinsics=use_extrinsics,
    use_sdpa=use_sdpa,
    use_rope=use_rope,
    use_activation_checkpointing=use_activation_checkpointing,
)

# Load checkpoint (use latest.pt or specific epoch like e25.pt)
checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
# OR: checkpoint_path = os.path.join(checkpoint_dir, "e25.pt")

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Load state dictionaries
encoder.load_state_dict(checkpoint["encoder"])
predictor.load_state_dict(checkpoint["predictor"])

# Move to GPU
device = "cpu"
encoder = encoder.to(device)
predictor = predictor.to(device)


print("\n[DEBUG] Predictor architecture:")
print(predictor)

# After loading the checkpoint, add this check:
print("\n[DEBUG] Checking action encoder weights:")
action_encoder_weight = predictor.action_encoder.weight
print(f"Action encoder weight stats:")
print(f"  mean: {action_encoder_weight.mean():.6f}")
print(f"  std: {action_encoder_weight.std():.6f}")
print(f"  min: {action_encoder_weight.min():.6f}")
print(f"  max: {action_encoder_weight.max():.6f}")

# Check if it's close to initialization (Xavier uniform for Linear layers)
# Expected std for initialized weights: sqrt(2 / (in_features + out_features))
expected_std = np.sqrt(2 / (7 + 1024))
print(f"  expected init std: {expected_std:.6f}")
print(f"  ratio (actual/expected): {action_encoder_weight.std().item() / expected_std:.2f}")

print(f"Loaded checkpoint from {checkpoint_path}")
print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")

# Initialize transform
crop_size = 256
# Calculate tokens based on your encoder patch size
tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)

transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=(1., 1.),
    random_resize_scale=(1., 1.),
    reprob=0.,
    auto_augment=False,
    motion_shift=False,
    crop_size=crop_size,
)

# Load Trajectory
trajectory = np.load(os.path.join(script_dir, "franka_example_traj.npz"))
np_clips = trajectory["observations"] # Shape: [T, H, W, C] (uint8 0-255)
np_states = trajectory["states"]

# Compute Ground Truth Actions from states
gt_action_vec = poses_to_diff(torch.tensor(np_states[0, 0]), torch.tensor(np_states[0, 1]))
np_actions = gt_action_vec.numpy()[None, None, :] # [1, 1, 7]

# --- CORRECTED TENSOR PREPARATION ---
# 1. Convert numpy to Torch
# np_clips is likely [Batch, Time, H, W, C]. 
# We select Batch 0, and Time 0:2 (Context + Target)
video_tensor = torch.from_numpy(np_clips[0, 0:4]) # Shape: [2, H, W, C]

# 2. Apply transform 
# Expects [T, H, W, C] -> Returns [C, T, H, W]
clips = transform(video_tensor) 

# 3. Add batch dimension -> [B, C, T, H, W]
clips = clips.unsqueeze(0)
# ------------------------------------

# Ensure we select only the states for the first trajectory (Batch 0)
# np_states is [Batch, Time, Dim]. We want [1, 2, Dim]
states = torch.tensor(np_states[0:1, :2]).float() 

# Actions is already [1, 1, 7] from your previous calculation
actions = torch.tensor(np_actions).float() 

clips = clips.to(device)
states = states.to(device)
actions = actions.to(device)

print(f"Clips: {clips.shape}, States: {states.shape}, Actions: {actions.shape}")

# --- ADAPTED ENCODER CALL ---
def forward_target(c, normalize_reps=True):
    with torch.no_grad():
        # Pass the full video clip correctly to the encoder
        # c is [B, C, T, H, W]
        h = encoder(c) # Expected output: [B, T, Tokens, D] or [B, T*Tokens, D]
        
        # Reshape to [B, T, Tokens, D] if necessary
        if h.dim() == 3:
            B, Seq, D = h.shape
            T = c.shape[2]
            h = h.view(B, T, -1, D)
            
        # Flatten to [B, T*Tokens, D] for the predictor
        h = h.flatten(1, 2)
        
        if normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
    return h

# --- ADAPTED PREDICTOR CALL ---
def forward_actions(z_full, nsamples, grid_size=0.075, normalize_reps=True):
    """
    z_full: Encoded representations of the video [B, T*Tokens, D]
    We want to predict z_{t+1} given z_t and various actions.
    """
    
    # 1. Extract z_t (Context)
    # Assuming z_full is [B, T*Tokens, D], we take the first frame's tokens
    z_t = z_full[:, :tokens_per_frame] # [B, Tokens, D]
    
    # 2. Prepare Action Grid
    def make_action_grid(grid_size):
        action_samples = []
        # Create a grid around 0,0,0
        for da in np.linspace(-grid_size, grid_size, nsamples):
            for db in np.linspace(-grid_size, grid_size, nsamples):
                for dc in np.linspace(-grid_size, grid_size, nsamples):
                    # XYZ delta, 0 rotation, 0 gripper
                    action_samples.append([da, db, dc, 0, 0, 0, 0])
        return torch.tensor(action_samples, device=z_full.device, dtype=z_full.dtype)

    grid = make_action_grid(grid_size) # [N_samples, 7]
    num_samples = grid.shape[0]
    
    # 3. Batched Prediction setup
    
    # Expand Context z_t -> [N, Tokens, D]
    z_context = z_t.repeat(num_samples, 1, 1) 
    
    # --- FIX START: CREATE DUMMY TOKENS FOR TARGET FRAME ---
    # We need to feed the predictor T=2 frames.
    # Frame 1: z_context (Real data)
    # Frame 2: Dummy data (The placeholder for the prediction)
    z_dummy = torch.zeros_like(z_context)
    z_in = torch.cat([z_context, z_dummy], dim=1) # [N, 2*Tokens, D]
    # -------------------------------------------------------
    
    # Expand Actions -> [N, 1, 7] (Sequence length 1)
    # Actions map T=0 -> T=1, so we only need 1 action step for 2 frames
    a_in = grid.unsqueeze(1) 
    
    # Expand States
    # We need s_t and s_{t+1}.
    s_t = states[:, 0:1].repeat(num_samples, 1, 1) # [N, 1, 7]
    
    # Calculate next state for every action sample (Physics/Kinematics)
    s_next = compute_new_pose(s_t, a_in) # [N, 1, 7]
    
    # Combine states for predictor: [N, 2, 7]
    s_in = torch.cat([s_t, s_next], dim=1)
    
    print(f"Running batch prediction on {num_samples} samples...")
    print(f"Shapes - z: {z_in.shape}, a: {a_in.shape}, s: {s_in.shape}")

    # 4. Run Predictor
    with torch.no_grad():
        # z_in is now [N, 2*Tokens, D] to match s_in which is [N, 2, 7]
        z_out_full = predictor(z_in, a_in, s_in) 
        
        # z_out_full is [N, 2*Tokens, D]
        # We only care about the prediction for the second frame (the last tokens)
        z_out = z_out_full[:, -tokens_per_frame:] # [N, Tokens, D]
        
        if normalize_reps:
            z_out = F.layer_norm(z_out, (z_out.size(-1),))
            
    return z_out, grid

# --- LOSS CALCULATION ---
def get_loss(z_pred, z_target):
    # z_pred: [N, Tokens, D]
    # z_target: [1, Tokens, D] -> Needs to be broadcasted or repeated
    
    target_rep = z_target.repeat(z_pred.shape[0], 1, 1)
    
    # L1 Loss average over tokens and dimensions
    print("Calculating loss...")
    print(f"z_pred shape: {z_pred.shape}, target_rep shape: {target_rep.shape}")
    print(f"z_pred sample data: {z_pred.size()}, target_rep sample data: {target_rep.size()}")
    loss = torch.abs(z_pred - target_rep)
    loss = torch.mean(loss, dim=[1, 2]) # [N]
    return loss.tolist()

# --- MAIN EXECUTION ---

# 1. Get Representations
h = forward_target(clips) # [1, T*Tokens, D]
print(f"[DEBUG] Encoder output shape h: {h.shape}")

# 2. Define Goal (z_{t+1})
z_target = h[:, tokens_per_frame : 2*tokens_per_frame] # The second frame

# 3. Run Sweep
nsamples = 5
grid_size = 0.075
z_preds, action_grid = forward_actions(h, nsamples, grid_size)

# 4. Compute Energy
energy = get_loss(z_preds, z_target)

# 5. Plotting
# (This logic remains mostly the same, just accessing tensors cleanly)
print("Plotting results...")
plot_data = []
action_grid_cpu = action_grid.cpu().numpy()

for i, err in enumerate(energy):
    plot_data.append((
        action_grid_cpu[i, 0], # dx
        action_grid_cpu[i, 1], # dy
        action_grid_cpu[i, 2], # dz
        err
    ))

delta_x = [d[0] for d in plot_data]
delta_z = [d[2] for d in plot_data] # plotting X vs Z
energies = [d[3] for d in plot_data]

# Create Heatmap
heatmap, xedges, yedges = np.histogram2d(delta_x, delta_z, weights=energies, bins=nsamples)

# Ground truth annotation
gt_x = gt_action_vec[0].item()
gt_y = gt_action_vec[1].item()
gt_z = gt_action_vec[2].item()

plt.figure()
plt.xlabel("Action Delta X")
plt.ylabel("Action Delta Z")
plt.title(f"Energy Landscape (Darker is Better)")
# Note: imshow typically needs transposition for correct x/y mapping with numpy histogram
plt.imshow(heatmap.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="viridis_r") # _r for reverse (dark=low energy)
plt.colorbar(label="Prediction Error")
plt.scatter([gt_x], [gt_z], c='red', marker='x', label='GT Action')
plt.legend()
plt.savefig(os.path.join(script_dir, "energy_landscape_adapted_uniandes_gemini.png"))
print(f"Saved to energy_landscape_adapted_uniandes_gemini.png")

# --- OPTIONAL: RUN PLANNER ---
print("\n--- Running World Model Planner ---")
world_model = WorldModel(
    encoder=encoder,
    predictor=predictor,
    tokens_per_frame=tokens_per_frame,
    transform=transform,
    mpc_args={
        "rollout": 1, # Keep short for testing
        "samples": 50,
        "topk": 5,
        "cem_steps": 5,
        "verbose": True
    },
    device=device
)

# Start: Frame 0 tokens, State 0
z_start = h[:, :tokens_per_frame]
s_start = states[:, 0:1]
# Goal: Frame 1 tokens
z_goal = z_target

# Infer
best_action = world_model.infer_next_action(z_start, s_start, z_goal)
print(f"GT Action: {gt_x:.3f}, {gt_y:.3f}, {gt_z:.3f}")
print(f"CEM Action: {best_action[0,0,0]:.3f}, {best_action[0,0,1]:.3f}, {best_action[0,0,2]:.3f}")