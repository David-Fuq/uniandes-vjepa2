import sys
sys.path.insert(0, "..")
import os
import yaml

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

from postraining_uniandes.transforms import make_transforms
from notebooks.utils.mpc_utils import (
    compute_new_pose,
    poses_to_diff
)

from notebooks.utils.world_model_wrapper import WorldModel
from postraining_uniandes.encoder_decoder_init import init_video_model


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

play_in_reverse = True  # Use this FLAG to try loading the trajectory backwards, and see how the energy landscape changes

trajectory = np.load(os.path.join(script_dir,"franka_example_traj.npz"))
np_clips = trajectory["observations"]
np_states = trajectory["states"]
if play_in_reverse:
    np_clips = trajectory["observations"][:, ::-1].copy()
    np_states = trajectory["states"][:, ::-1].copy()
np_actions = np.expand_dims(poses_to_diff(np_states[0, 0], np_states[0, 1]), axis=(0, 1))

# Convert trajectory to torch tensors
clips = transform(np_clips[0]).unsqueeze(0)
states = torch.tensor(np_states).to(device)
actions = torch.tensor(np_actions).to(device)
print(f"clips: {clips.shape}; states: {states.shape}; actions: {actions.shape}")


# Visualize loaded video frames from traj

T = len(np_clips[0])
plt.figure(figsize=(20, 3))
plt.imshow(np.transpose(np_clips[0], (1, 0, 2, 3)).reshape(256, 256 * T, 3))
plt.savefig(os.path.join(script_dir, "trajectory_frames_uniandes_2frames_reverse.png"), bbox_inches='tight', dpi=100)
print(f"Saved visualization to {os.path.join(script_dir, 'trajectory_frames_uniandes_2frames_reverse.png')}")
plt.close()

def forward_target(c, normalize_reps=True):
    B, C, T, H, W = c.size()
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def forward_actions(z, nsamples, grid_size=0.075, normalize_reps=True, action_repeat=1):

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
        # _z: [Batch, Tokens, Dim] (Current Frame Representation)
        # _a: [Batch, 1, 7]        (Proposed Action)
        # _s: [Batch, 1, 7]        (Current State)

        # --- THE FIX ---
        # We construct a sequence of length T=2.
        # t=0: "Dummy" past. The model uses internal zero-action.
        # t=1: "Current" step. The model uses _a (actions[0]) to predict next state.
        
        # 1. Create a sequence by duplicating the current frame and state
        # This tells the model: "We were at state _s, saw _z, and now we are here again."
        _z_seq = torch.cat([_z, _z], dim=1)  # Shape: [B, 2*Tokens, D]
        _s_seq = torch.cat([_s, _s], dim=1)  # Shape: [B, 2, 7]
        
        # 2. Pass the sequence to the predictor.
        # Note: We do NOT duplicate _a. The predictor expects actions to be T-1 relative to states.
        # Internal logic: 
        #   t=0 uses dummy_action
        #   t=1 uses _a[:, 0] <--- This is where your action finally gets used!
        pred_sequence = predictor(_z_seq, _a, _s_seq)
        
        # 3. Extract only the prediction for the last frame
        _z_next = pred_sequence[:, -tokens_per_frame:] 
        
        if normalize_reps:
            _z_next = F.layer_norm(_z_next, (_z_next.size(-1),))
            
        # Update physical state (simple physics integration)
        _s_next = compute_new_pose(_s[:, -1:], _a[:, -1:])
        
        return _z_next, _s_next

    # Context frame rep and context pose
    # (Rest of the function remains exactly the same)
    z_hat = z[:, :tokens_per_frame].repeat(int(nsamples**3), 1, 1)
    s_hat = states[:, :1].repeat((int(nsamples**3), 1, 1))
    a_hat = action_samples

    for _ in range(action_repeat):
        _z, _s = step_predictor(z_hat, a_hat, s_hat)
        z_hat = torch.cat([z_hat, _z], dim=1)
        s_hat = torch.cat([s_hat, _s], dim=1)
        a_hat = torch.cat([a_hat, action_samples], dim=1)

    return z_hat, s_hat, a_hat

def loss_fn(z, h):
    z, h = z[:, -tokens_per_frame:], h[:, -tokens_per_frame:]
    loss = torch.abs(z - h)  # [B, N, D]
    print(f"[DEBUG] z stats - min: {z.min():.4f}, max: {z.max():.4f}, mean: {z.mean():.4f}, std: {z.std():.4f}")
    print(f"[DEBUG] h stats - min: {h.min():.4f}, max: {h.max():.4f}, mean: {h.mean():.4f}, std: {h.std():.4f}")
    print(f"[DEBUG] diff stats - min: {loss.min():.4f}, max: {loss.max():.4f}, mean: {loss.mean():.4f}, std: {loss.std():.4f}")
    
    loss = torch.mean(loss, dim=[1, 2])
    print(f"[DEBUG] final loss range: {loss.min():.6f} to {loss.max():.6f}")
    return loss.tolist()




# Compute energy for cartesian action grid of size (nsample x nsamples x nsamples)
nsamples = 5
grid_size = 0.075
with torch.no_grad():
    h = forward_target(clips)
    print(f"[DEBUG] h stats before action prediction:")
    print(h)
    print("---"*50)
    z_hat, s_hat, a_hat = forward_actions(h, nsamples=nsamples, grid_size=grid_size)
    loss = loss_fn(z_hat, h)  # jepa prediction loss
    print(loss)

# Plot the energy

plot_data = []
for b, v in enumerate(loss):
    plot_data.append((
        a_hat[b, :-1, 0].sum().item(),
        a_hat[b, :-1, 1].sum().item(),
        a_hat[b, :-1, 2].sum().item(),
        v,
    ))

delta_x = [d[0] for d in plot_data]
delta_y = [d[1] for d in plot_data]
delta_z = [d[2] for d in plot_data]
energy = [d[3] for d in plot_data]

print("delta x: ", delta_x)
print("delta y: ", delta_y)
print("delta z: ", delta_z)
print("energy: ", energy)

gt_x = actions[0, 0, 0].item()
gt_y = actions[0, 0, 1].item()
gt_z = actions[0, 0, 2].item()

# Create the 2D histogram
heatmap, xedges, yedges = np.histogram2d(delta_x, delta_z, weights=energy, bins=nsamples)

# Set axis labels
plt.xlabel("Action Delta x")
plt.ylabel("Action Delta z")
plt.title(f"Energy Landscape")

# Display the heatmap
print(f"Ground truth action (x,y,z) = ({gt_x:.2f},{gt_y:.2f},{gt_z:.2f})")
plt.imshow(heatmap.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="viridis")
plt.colorbar()
plt.savefig(os.path.join(script_dir, "energy_landscape_uniandes_2frames_reverse.png"), bbox_inches='tight', dpi=100)
print(f"Saved energy landscape to {os.path.join(script_dir, 'energy_landscape_uniandes_2frames_reverse.png')}")
plt.close()



# Action Prediction using CEM
world_model = WorldModel(
    encoder=encoder,
    predictor=predictor,
    tokens_per_frame=tokens_per_frame,
    transform=transform,
    # Doing very few CEM iterations with very few samples just to run efficiently on CPU...
    # ... increase cem_steps and samples for more accurate optimization of energy landscape
    mpc_args={
        "rollout": 2,
        "samples": 25,
        "topk": 10,
        "cem_steps": 2,
        "momentum_mean": 0.15,
        "momentum_mean_gripper": 0.15,
        "momentum_std": 0.75,
        "momentum_std_gripper": 0.15,
        "maxnorm": 0.075,
        "verbose": True
    },
    normalize_reps=True,
    device="cpu"
)

with torch.no_grad():
    h = forward_target(clips)
    z_n, z_goal = h[:, :tokens_per_frame], h[:, -tokens_per_frame:]
    s_n = states[:, :1]
    print(f"Starting planning using Cross-Entropy Method...")
    actions = world_model.infer_next_action(z_n, s_n, z_goal).numpy()

print(f"Actions returned by planning with CEM (x,y,z) = ({actions[0, 0]:.2f},{actions[0, 1]:.2f} {actions[0, 2]:.2f})")

