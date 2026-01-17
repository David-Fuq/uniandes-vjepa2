import sys
sys.path.insert(0, "..")
import os
import yaml

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

from app.vjepa_droid.transforms import make_transforms
from notebooks.utils.mpc_utils import (
    compute_new_pose,
    poses_to_diff
)

from notebooks.utils.world_model_wrapper import WorldModel
from app.vjepa_droid.utils import init_video_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Available memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

script_dir = os.path.dirname(os.path.abspath(__file__))


encoder = torch.load(os.path.join(script_dir,"encoder_full_vjepa2_original.pt"), weights_only=False)
predictor = torch.load(os.path.join(script_dir,"predictor_full_vjepa2_original.pt"), weights_only=False)

# Move to GPU
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

# Initialize transform
crop_size = 256
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

play_in_reverse = False  # Use this FLAG to try loading the trajectory backwards, and see how the energy landscape changes

trajectory = np.load(os.path.join(script_dir,"test_trajectory_5_known.npz"))
np_clips_full = trajectory["observations"][:, :7]
np_states_full = trajectory["states"][:, :7]
if play_in_reverse:
    np_clips_full = trajectory["observations"][:, ::-1].copy()
    np_states_full = trajectory["states"][:, ::-1].copy()

# Extract information about the full trajectory
T_full = len(np_clips_full[0])
print(f"\n{'='*80}")
print(f"LOADED FULL TRAJECTORY")
print(f"{'='*80}")
print(f"Total frames: {T_full}")
print(f"Full clip shape: {np_clips_full.shape}")
print(f"Full states shape: {np_states_full.shape}")

# Extract first 2 frames for energy landscape analysis
np_clips_2frame = np_clips_full[:, :2]
np_states_2frame = np_states_full[:, :2]

# Convert to torch tensors
clips_full = transform(np_clips_full[0]).unsqueeze(0).to(device)  # [1, C, T, H, W]
clips_2frame = transform(np_clips_2frame[0]).unsqueeze(0).to(device)  # [1, C, 2, H, W]

states_full = torch.tensor(np_states_full, dtype=torch.float32).to(device)
states_2frame = torch.tensor(np_states_2frame, dtype=torch.float32).to(device)

# Compute ground truth action for first step
np_actions_first = np.expand_dims(poses_to_diff(np_states_full[0, 0], np_states_full[0, 1]), axis=(0, 1))
actions_first = torch.tensor(np_actions_first, dtype=torch.float32).to(device)


print(f"\n2-frame subset for energy landscape:")
print(f"  clips_2frame: {clips_2frame.shape}")
print(f"  states_2frame: {states_2frame.shape}")
print(f"  Ground truth first action: {actions_first.shape}")
print(f"\nFull trajectory for CEM planning:")
print(f"  clips_full: {clips_full.shape}")
print(f"  states_full: {states_full.shape}")

# Visualize loaded video frames from full trajectory
T = T_full
clips_vis = clips_full[0].permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]

# Denormalize
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
clips_vis = clips_vis * std + mean
clips_vis = np.clip(clips_vis, 0, 1)

# Reshape to horizontal strip
H, W = clips_vis.shape[1:3]
strip = clips_vis.transpose(1, 0, 2, 3).reshape(H, W * T, 3)

plt.figure(figsize=(20, 3))
plt.imshow(strip)
plt.title(f"Full Trajectory Visualization ({T} frames)")
plt.savefig(os.path.join(script_dir, "trajectory_frames_original_trajectory_5_known.png"), bbox_inches='tight', dpi=100)
print(f"\nSaved visualization to {os.path.join(script_dir, 'trajectory_frames_original_trajectory_5_known.png')}")
plt.close()

def forward_target(c, normalize_reps=True):
    B, C, T, H, W = c.size()
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def forward_actions(z, current_states, nsamples, grid_size=0.075, normalize_reps=True, action_repeat=1):
    """
    Predict next state for a grid of actions.
    
    Args:
        z: Current frame representation [1, tokens, D]
        current_states: Current robot states [1, T_current, 7]
        nsamples: Grid resolution
        grid_size: Action magnitude range
    """
    def make_action_grid(grid_size=grid_size):
        action_samples = []
        for da in np.linspace(-grid_size, grid_size, nsamples):
            for db in np.linspace(-grid_size, grid_size, nsamples):
                for dc in np.linspace(-grid_size, grid_size, nsamples):
                    action_samples += [torch.tensor([da, db, dc, 0, 0, 0, 0], device=z.device, dtype=z.dtype)]
        return torch.stack(action_samples, dim=0).unsqueeze(1)

    # Sample grid of actions
    action_samples = make_action_grid()
    print(f"Sampled grid of actions; num actions = {len(action_samples)}")

    def step_predictor(_z, _a, _s):
        _z = predictor(_z, _a, _s)[:, -tokens_per_frame:]
        if normalize_reps:
            _z = F.layer_norm(_z, (_z.size(-1),))
        _s = compute_new_pose(_s[:, -1:], _a[:, -1:])
        return _z, _s

    # Context frame rep and context pose
    z_hat = z[:, :tokens_per_frame].repeat(int(nsamples**3), 1, 1)  # [S, N, D]
    s_hat = current_states[:, :1].repeat((int(nsamples**3), 1, 1))  # [S, 1, 7]
    a_hat = action_samples  # [S, 1, 7]

    for _ in range(action_repeat):
        _z, _s = step_predictor(z_hat, a_hat, s_hat)
        z_hat = torch.cat([z_hat, _z], dim=1)
        s_hat = torch.cat([s_hat, _s], dim=1)
        a_hat = torch.cat([a_hat, action_samples], dim=1)

    return z_hat, s_hat, a_hat

def loss_fn(z, h):
    z, h = z[:, -tokens_per_frame:], h[:, -tokens_per_frame:]
    loss = torch.abs(z - h)  # [B, N, D]
    loss = torch.mean(loss, dim=[1, 2])
    return loss.tolist()


# ============================================================================
# PART 1: ENERGY LANDSCAPE (using first 2 frames only)
# ============================================================================
print(f"\n{'='*80}")
print(f"COMPUTING ENERGY LANDSCAPE (frames 1-2)")
print(f"{'='*80}")

nsamples = 7  # Increased for better resolution
grid_size = 0.075
with torch.no_grad():
    h_2frame = forward_target(clips_2frame)
    z_hat, s_hat, a_hat = forward_actions(h_2frame, states_2frame, nsamples=nsamples, grid_size=grid_size)
    loss = loss_fn(z_hat, h_2frame)  # jepa prediction loss

# Extract action coordinates and energies
delta_x = a_hat[:, 0, 0].cpu().numpy()  # x component of actions
delta_z = a_hat[:, 0, 2].cpu().numpy()  # z component of actions
energy = np.array(loss)

gt_x = actions_first[0, 0, 0].item()
gt_y = actions_first[0, 0, 1].item()
gt_z = actions_first[0, 0, 2].item()

# Reshape energy into proper grid (nsamples x nsamples x nsamples)
# We need to average over y-dimension since we're plotting x-z plane
energy_grid_3d = energy.reshape(nsamples, nsamples, nsamples)
energy_grid_2d = np.mean(energy_grid_3d, axis=1)  # Average over y-axis

# Get unique x and z values
x_vals = np.linspace(-grid_size, grid_size, nsamples)
z_vals = np.linspace(-grid_size, grid_size, nsamples)

# Create the plot
plt.figure(figsize=(10, 8))
plt.xlabel("Action Delta x", fontsize=12)
plt.ylabel("Action Delta z", fontsize=12)
plt.title(f"Energy Landscape (Frame 1 → Frame 2)", fontsize=14)

# Display the heatmap with proper extent
im = plt.imshow(
    energy_grid_2d.T,  # Transpose for correct orientation
    origin="lower",
    extent=[-grid_size, grid_size, -grid_size, grid_size],
    cmap="viridis",
    aspect='auto'
)
plt.colorbar(im, label="Prediction Error (L1)")

# Mark ground truth action
plt.plot(gt_x, gt_z, 'r*', markersize=20, markeredgewidth=2, markeredgecolor='white', label='Ground Truth', zorder=10)

# Add grid lines for reference
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add contour lines
contours = plt.contour(x_vals, z_vals, energy_grid_2d.T, levels=5, colors='white', alpha=0.3, linewidths=1)
plt.clabel(contours, inline=True, fontsize=8)

plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig(os.path.join(script_dir, "energy_landscape_original_v3_clear_cup.png"), bbox_inches='tight', dpi=150)
print(f"\nGround truth action frame 1→2: (x={gt_x:.4f}, y={gt_y:.4f}, z={gt_z:.4f})")
print(f"Saved energy landscape to {os.path.join(script_dir, 'energy_landscape_original_v3_clear_cup.png')}")
plt.close()


# ============================================================================
# PART 2: CEM PLANNING (full trajectory from frame 1 to frame T)
# ============================================================================
print(f"\n{'='*80}")
print(f"CEM PLANNING (frame 1 → frame {T_full})")
print(f"{'='*80}")

rollout_steps = T_full - 1  # Number of actions needed

world_model = WorldModel(
    encoder=encoder,
    predictor=predictor,
    tokens_per_frame=tokens_per_frame,
    transform=transform,
    mpc_args={
        "rollout": rollout_steps,
        "samples": 100,  # Increased for better optimization
        "topk": 20,  # Increased
        "cem_steps": 10,  # Increased for better convergence
        "momentum_mean": 0.15,
        "momentum_mean_gripper": 0.15,
        "momentum_std": 0.75,
        "momentum_std_gripper": 0.15,
        "maxnorm": 0.075,
        "verbose": True
    },
    normalize_reps=True,
    device=device  # Use GPU
)

with torch.no_grad():
    h_full = forward_target(clips_full)
    z_start = h_full[:, :tokens_per_frame]
    z_goal = h_full[:, -tokens_per_frame:]
    s_start = states_full[:, :1]
    
    print(f"Planning trajectory: {T_full} frames = {rollout_steps} actions")
    print(f"Start state: {s_start[0, 0, :3].cpu().numpy()}")
    print(f"Goal state: {states_full[0, -1, :3].cpu().numpy()}")
    
    planned_actions = world_model.infer_next_action(z_start, s_start, z_goal).cpu().numpy()

print(f"\n{'='*80}")
print(f"RESULTS (Meta Original Model)")
print(f"{'='*80}")
print(f"Inference mode: Frame-by-frame (Δt = 1)")
print(f"Planned action sequence shape: {planned_actions.shape}")

print(f"\nAction sequence comparison (frame-by-frame):")
print(f"{'Step':<6} {'Planned (x,y,z)':<30} {'Ground Truth (x,y,z)':<30} {'Error':<10}")
print(f"{'-'*80}")

total_error = 0.0

for i in range(rollout_steps):
    pred = planned_actions[i]
    
    # Frame-by-frame: action i moves from frame i to frame i+1
    t_start = i
    t_end = i + 1
    
    # Compute Ground Truth (frame-to-frame difference)
    gt_action = poses_to_diff(
        np_states_full[0, t_start],
        np_states_full[0, t_end]
    ).numpy()
    
    error = np.linalg.norm(pred[:3] - gt_action[:3])
    total_error += error
    
    print(f"{i+1:<6} ({pred[0]:6.4f},{pred[1]:6.4f},{pred[2]:6.4f})      "
          f"({gt_action[0]:6.4f},{gt_action[1]:6.4f},{gt_action[2]:6.4f})      "
          f"{error:6.4f}")

avg_error = total_error / rollout_steps
print(f"{'-'*80}")
print(f"Average xyz error (per frame): {avg_error:.4f}")

print(f"\n{'='*80}")
print(f"DONE")
print(f"{'='*80}")