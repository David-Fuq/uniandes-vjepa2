import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import yaml

# --- IMPORTS ---
sys.path.insert(0, "..")
from postraining_uniandes.transforms import make_transforms
from postraining_uniandes.path_planning.mpc_utils_v2 import compute_new_pose, poses_to_diff
# We will define a local WorldModel to ensure it's clean
from postraining_uniandes.encoder_decoder_init import init_video_model

# --- SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
main_dir  = os.path.dirname(parent_dir)

checkpoint_dir = os.path.join(main_dir, "final_dataset_40_epoch_frozen_encoder")
config_path = os.path.join(checkpoint_dir, "params-pretrain.yaml")

with open(config_path, "r") as f:
    args = yaml.safe_load(f)

# --- MODEL INIT ---
cfgs_model = args.get("model")
cfgs_data = args.get("data")

# Force defaults if missing
crop_size = cfgs_data.get("crop_size", 256)
patch_size = cfgs_data.get("patch_size", 16)
tubelet_size = cfgs_data.get("tubelet_size", 2) # Likely 2

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Available memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

encoder, predictor = init_video_model(
    device="cpu", # Init on CPU first
    patch_size=patch_size,
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
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
encoder.load_state_dict(checkpoint["encoder"])
predictor.load_state_dict(checkpoint["predictor"])

encoder.to(device).eval()
predictor.to(device).eval()
print(f"Models loaded from {checkpoint_path}")

# --- DATA LOADING (THE FIX) ---
# We need exactly 2 frames to make 1 Tubelet (if tubelet_size=2)
# We need to create a context of at least 1 tubelet.

transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=[1., 1.],
    random_resize_scale=[1., 1.],
    crop_size=crop_size,
)

trajectory = np.load(os.path.join(script_dir, "franka_example_traj.npz"))
np_clips = trajectory["observations"] # [Batch, Time, H, W, C]
np_states = trajectory["states"]

print(f"Loaded trajectory with {np_clips.shape[0]} clips, {np_clips.shape[1]} frames each.")
print(f"Clip shape: {np_clips.shape}") 

# 1. Load actual temporal data
# We take Batch 0.
# We take Frame 0 and Frame 1. Together they make the first latent representation (z_0)
# We take Frame 2 and Frame 3. Together they make the target representation (z_1)
frames_needed = 4 
if np_clips.shape[1] < frames_needed:
    raise ValueError(f"Not enough frames in trajectory. Need {frames_needed}")

video_tensor = torch.from_numpy(np_clips[0, 0:frames_needed]) # [4, H, W, C]
clips = transform(video_tensor).unsqueeze(0).to(device) # [1, C, 4, H, W]

# Prepare States (t=0)
states = torch.tensor(np_states[0:1, :2]).float().to(device) 

# Prepare Action (The Ground Truth action between state 0 and 1)
gt_action_vec = poses_to_diff(torch.tensor(np_states[0, 0]), torch.tensor(np_states[0, 1]))
actions = gt_action_vec.unsqueeze(0).unsqueeze(0).float().to(device) # [1, 1, 7]

print(f"Input Clips: {clips.shape}") # Should be [1, 3, 4, 256, 256]

# --- HELPER FUNCTIONS ---

def get_latent_reps(c):
    """
    Encodes video clips into latent representations.
    c: [B, C, T, H, W]
    Returns: [B, T_latent, Tokens, D]
    """
    with torch.no_grad():
        h = encoder(c) # [B, T_latent*Tokens, D] (Output is flattened time)
        
        # Reshape back to separate Time and Space
        # Calculate tokens per frame (spatial)
        tokens_spatial = int((crop_size // encoder.patch_size) ** 2)
        B, Total_Tokens, D = h.shape
        T_latent = Total_Tokens // tokens_spatial
        
        h = h.view(B, T_latent, tokens_spatial, D)
        h = F.layer_norm(h, (D,))
    return h

# --- MAIN EXECUTION ---

# 1. Encode
# Clips is [1, 3, 4, 256, 256]. Encoder (tubelet=2) should output T_latent=2
h_all = get_latent_reps(clips) 
print(f"Encoded Shape: {h_all.shape}") # Should be [1, 2, 256, 1408]

z_current = h_all[:, 0] # Context (Frames 0-1)
z_target = h_all[:, 1]  # Goal (Frames 2-3)

# 2. Action Sweep
nsamples = 5
grid_size = 0.075

# Create Action Grid
action_samples = []
for da in np.linspace(-grid_size, grid_size, nsamples):
    for db in np.linspace(-grid_size, grid_size, nsamples):
        for dc in np.linspace(-grid_size, grid_size, nsamples):
            action_samples.append([da, db, dc, 0, 0, 0, 0])
action_grid = torch.tensor(action_samples, device=device).unsqueeze(1) # [N, 1, 7]
num_samples = action_grid.shape[0]

# 3. Batch Prediction
# Repeat context for all samples
z_batch = z_current.repeat(num_samples, 1, 1) # [N, Tokens, D]
s_batch = states[:, 0:1].repeat(num_samples, 1, 1) # [N, 1, 7]

# Prepare input sequence for Predictor
# We want: [z_current, dummy]
z_seq = torch.cat([z_batch.unsqueeze(1), torch.zeros_like(z_batch).unsqueeze(1)], dim=1) # [N, 2, Tokens, D]
z_seq = z_seq.flatten(1, 2) # [N, 2*Tokens, D]

# Prepare state sequence
# We calculate next state for every action
s_next = compute_new_pose(s_batch, action_grid)
s_seq = torch.cat([s_batch, s_next], dim=1) # [N, 2, 7]

print("Running Predictor on Grid...")
with torch.no_grad():
    # z_seq: [N, 2*Tokens, D]
    # action_grid: [N, 1, 7]
    # s_seq: [N, 2, 7]
    z_out_seq = predictor(z_seq, action_grid, s_seq)
    
    # Extract prediction for the 2nd step
    # Predictor output matches z_seq shape. 
    # The last 'tokens_spatial' tokens correspond to the predicted Frame 2
    tokens_spatial = z_current.shape[1]
    z_pred = z_out_seq[:, -tokens_spatial:]
    z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))

# 4. Loss
# Compare z_pred (Prediction of Frames 2-3) vs z_target (Actual Frames 2-3)
target_batch = z_target.repeat(num_samples, 1, 1)
loss = torch.mean(torch.abs(z_pred - target_batch), dim=[1, 2]).cpu().numpy()

# 5. Plot
print("Plotting...")
# ... (Standard plotting code remains similar)
heatmap_data = []
action_grid_cpu = action_grid.cpu().numpy()

for i in range(num_samples):
    heatmap_data.append((
        action_grid_cpu[i, 0, 0], # dx
        action_grid_cpu[i, 0, 2], # dz (Plotting X vs Z)
        loss[i]
    ))

dx = [x[0] for x in heatmap_data]
dz = [x[1] for x in heatmap_data]
err = [x[2] for x in heatmap_data]

heatmap, xedges, yedges = np.histogram2d(dx, dz, weights=err, bins=nsamples)

plt.figure()
plt.imshow(heatmap.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="viridis_r")
plt.colorbar(label="Prediction Error")
plt.title("Energy Landscape (Fixed)")
plt.xlabel("Delta X")
plt.ylabel("Delta Z")
gt_x = gt_action_vec[0].item()
gt_z = gt_action_vec[2].item()
plt.scatter([gt_x], [gt_z], c='red', marker='x', label='GT')
plt.legend()
plt.savefig("energy_landscape_fixed.png")
print("Saved energy_landscape_fixed.png")

# --- CLEAN WORLD MODEL CLASS FOR CEM ---
# Redefining logic cleanly for CEM usage
class CleanWorldModel:
    def __init__(self, predictor, tokens_spatial, normalize=True):
        self.predictor = predictor
        self.tokens_spatial = tokens_spatial
        self.normalize = normalize
        
    def step(self, reps, actions, poses):
        # reps: [B, T, Tokens, D]
        # actions: [B, T, 7]
        # poses: [B, T+1, 7]
        
        B = reps.shape[0]
        reps_flat = reps.flatten(1, 2)
        
        # Predict
        next_rep_flat = self.predictor(reps_flat, actions, poses)
        
        # Extract last frame
        next_rep = next_rep_flat[:, -self.tokens_spatial:]
        if self.normalize:
            next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
            
        next_rep = next_rep.view(B, 1, self.tokens_spatial, -1)
        next_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])
        
        return next_rep, next_pose

# Initialize clean wrapper
wm = CleanWorldModel(predictor, tokens_spatial)

# Import CEM (Assuming mpc_utils logic is available)
from postraining_uniandes.path_planning.mpc_utils_v2 import cem

print("\nRunning CEM...")
# z_current: [1, Tokens, D] -> [1, 1, Tokens, D] for CEM
z_init = z_current.unsqueeze(1)
z_goal_input = z_target.unsqueeze(1)
pose_init = states[:, 0:1] # [1, 1, 7]

best_action = cem(
    context_frame=z_init,
    context_pose=pose_init,
    goal_frame=z_goal_input,
    world_model=wm.step,
    rollout=1, # Single step prediction
    samples=50,
    cem_steps=5,
    verbose=True
)

print(f"CEM Best Action: {best_action[0, 0, :3].cpu().numpy()}")
print(f"GT Action: {gt_action_vec[:3].cpu().numpy()}")