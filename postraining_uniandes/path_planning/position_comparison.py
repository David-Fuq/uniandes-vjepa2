"""
Evaluate trajectory prediction by applying predicted actions to initial state.

This script takes predicted actions and applies them sequentially from the initial
robot state to compute where the robot would end up. It then compares these
predicted positions against the ground truth positions from the trajectory.

Usage:
    python evaluate_trajectory_from_actions.py \
        --npz path/to/test_trajectory.npz \
        --actions "(-0.0170,0.0442,0.0432);(-0.0097,-0.0136,0.0218);..." \
        --model_name "My Model"

This allows you to:
1. Evaluate accumulated positional error (not just per-step action error)
2. Reuse existing test results without re-running inference
3. Compare how different models' actions translate to actual robot movement
"""

import argparse
import numpy as np
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_actions(actions_str):
    """
    Parse action string into numpy array.
    
    Input format: "(-0.0170,0.0442,0.0432);(-0.0097,-0.0136,0.0218);..."
    
    Returns:
        np.array of shape [num_actions, 3] (xyz only) or [num_actions, 7] (full)
    """
    # Remove any whitespace
    actions_str = actions_str.strip()
    
    # Split by semicolon
    action_strs = actions_str.split(";")
    
    actions = []
    for action_str in action_strs:
        action_str = action_str.strip()
        if not action_str:
            continue
            
        # Extract numbers from tuple format (x,y,z) or (x,y,z,rx,ry,rz,g)
        # Remove parentheses and split by comma
        action_str = action_str.strip("()")
        values = [float(v.strip()) for v in action_str.split(",")]
        actions.append(values)
    
    return np.array(actions)


def apply_actions_to_state(initial_state, actions):
    """
    Apply a sequence of actions to an initial state to get predicted trajectory.
    
    For simplicity, we assume actions are additive for position (xyz).
    For rotations, this is an approximation but sufficient for small movements.
    
    Args:
        initial_state: [7] array (x, y, z, rx, ry, rz, gripper)
        actions: [T, 3] or [T, 7] array of actions
        
    Returns:
        predicted_states: [T+1, 7] array of states (including initial)
    """
    num_actions = len(actions)
    action_dim = actions.shape[1] if actions.ndim > 1 else len(actions[0])
    
    # Initialize trajectory with initial state
    predicted_states = np.zeros((num_actions + 1, 7))
    predicted_states[0] = initial_state
    
    for t in range(num_actions):
        # Get current state
        current_state = predicted_states[t].copy()
        
        # Apply action (additive for position and rotation)
        if action_dim >= 3:
            # XYZ position
            current_state[:3] += actions[t, :3] if actions.ndim > 1 else actions[t][:3]
        
        if action_dim >= 6:
            # Rotation (Euler angles - additive approximation)
            current_state[3:6] += actions[t, 3:6]
        
        if action_dim >= 7:
            # Gripper
            current_state[6] += actions[t, 6]
        
        predicted_states[t + 1] = current_state
    
    return predicted_states


def compute_position_errors(predicted_states, ground_truth_states):
    """
    Compute position errors between predicted and ground truth trajectories.
    
    Args:
        predicted_states: [T, 7] predicted states
        ground_truth_states: [T, 7] ground truth states
        
    Returns:
        errors: dict with per-step and accumulated errors
    """
    # Ensure same length
    T = min(len(predicted_states), len(ground_truth_states))
    predicted_states = predicted_states[:T]
    ground_truth_states = ground_truth_states[:T]
    
    # Per-step XYZ error
    xyz_errors = np.linalg.norm(
        predicted_states[:, :3] - ground_truth_states[:, :3], 
        axis=1
    )
    
    # Final position error
    final_error = xyz_errors[-1]
    
    # Average position error
    avg_error = np.mean(xyz_errors[1:])  # Exclude initial state (always 0)
    
    # Accumulated drift (how much the error grows over time)
    drift_per_step = np.diff(xyz_errors)
    
    return {
        "per_step_errors": xyz_errors,
        "final_error": final_error,
        "avg_error": avg_error,
        "drift_per_step": drift_per_step,
        "predicted_positions": predicted_states[:, :3],
        "ground_truth_positions": ground_truth_states[:, :3],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trajectory by applying predicted actions'
    )
    parser.add_argument(
        '--npz', type=str, required=True,
        help='Path to .npz file containing observations and states'
    )
    parser.add_argument(
        '--actions', type=str, required=True,
        help='Predicted actions as semicolon-separated tuples: "(x,y,z);(x,y,z);..."'
    )
    parser.add_argument(
        '--model_name', type=str, default='Model',
        help='Name of the model for display purposes'
    )
    parser.add_argument(
        '--start_frame', type=int, default=0,
        help='Starting frame index in the trajectory (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Load trajectory data
    print(f"Loading trajectory from: {args.npz}")
    data = np.load(args.npz)
    
    observations = data["observations"]  # [batch, T, H, W, C]
    states = data["states"]  # [batch, T, 7]
    
    # Use first batch
    states = states[0]  # [T, 7]
    
    print(f"Trajectory: {len(states)} states")
    print(f"State format: [x, y, z, rx, ry, rz, gripper]")
    
    # Parse actions
    print(f"\nParsing actions...")
    actions = parse_actions(args.actions)
    num_actions = len(actions)
    
    print(f"Parsed {num_actions} actions")
    print(f"Action dimension: {actions.shape[1] if actions.ndim > 1 else 'variable'}")
    
    # Ensure actions have at least 3 dimensions (xyz)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 3)
    
    # Pad actions to 7 dimensions if needed
    if actions.shape[1] < 7:
        padded_actions = np.zeros((num_actions, 7))
        padded_actions[:, :actions.shape[1]] = actions
        actions = padded_actions
    
    # Get initial state and ground truth trajectory
    start_idx = args.start_frame
    end_idx = start_idx + num_actions + 1
    
    if end_idx > len(states):
        print(f"Warning: Requested {num_actions} actions from frame {start_idx}, "
              f"but only {len(states) - start_idx - 1} available. Truncating.")
        end_idx = len(states)
        num_actions = end_idx - start_idx - 1
        actions = actions[:num_actions]
    
    initial_state = states[start_idx]
    ground_truth_states = states[start_idx:end_idx]
    
    print(f"\nInitial state (frame {start_idx}): "
          f"({initial_state[0]:.4f}, {initial_state[1]:.4f}, {initial_state[2]:.4f})")
    print(f"Final GT state (frame {end_idx-1}): "
          f"({ground_truth_states[-1, 0]:.4f}, {ground_truth_states[-1, 1]:.4f}, {ground_truth_states[-1, 2]:.4f})")
    
    # Apply actions to get predicted trajectory
    print(f"\nApplying {num_actions} actions to initial state...")
    predicted_states = apply_actions_to_state(initial_state, actions)
    
    # Compute errors
    errors = compute_position_errors(predicted_states, ground_truth_states)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS ({args.model_name})")
    print(f"{'='*80}")
    print(f"Evaluation: Predicted Position vs Ground Truth Position")
    print(f"Starting from frame {start_idx}, applying {num_actions} actions")
    
    print(f"\nPosition trajectory comparison:")
    print(f"{'Frame':<8} {'Predicted (x,y,z)':<32} {'Ground Truth (x,y,z)':<32} {'Error':<10}")
    print(f"{'-'*82}")
    
    for t in range(len(predicted_states)):
        pred = errors["predicted_positions"][t]
        gt = errors["ground_truth_positions"][t]
        err = errors["per_step_errors"][t]
        
        frame_idx = start_idx + t
        print(f"{frame_idx:<8} "
              f"({pred[0]:8.4f},{pred[1]:8.4f},{pred[2]:8.4f})      "
              f"({gt[0]:8.4f},{gt[1]:8.4f},{gt[2]:8.4f})      "
              f"{err:8.4f}")
    
    print(f"{'-'*82}")
    print(f"\nSummary Statistics:")
    print(f"  Average position error: {errors['avg_error']:.4f}")
    print(f"  Final position error:   {errors['final_error']:.4f}")
    print(f"  Max position error:     {np.max(errors['per_step_errors']):.4f}")
    
    # Show drift analysis
    print(f"\nDrift Analysis (error growth per step):")
    print(f"  Mean drift: {np.mean(errors['drift_per_step']):.4f}")
    print(f"  Max drift:  {np.max(errors['drift_per_step']):.4f}")
    
    print(f"\n{'='*80}")
    print(f"DONE")
    print(f"{'='*80}")
    
    return 0


#"(-0.0170,0.0442,0.0432);(-0.0097,-0.0136,0.0218);(0.0058,-0.0237,0.0068);(-0.0088,-0.0087,0.0197);(-0.0025,-0.0100,0.0207);(-0.0143,-0.0024,0.0104)"

if __name__ == '__main__':
    sys.exit(main())
