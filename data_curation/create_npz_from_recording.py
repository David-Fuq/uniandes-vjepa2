"""
Convert DROID recording (video + trajectory.h5 + metadata.json) to .npz format
for use with energy landscape testing scripts.

Usage:
    python create_npz_from_recording.py --input /path/to/recording/folder --output test_trajectory.npz

The recording folder should contain:
    - trajectory.h5
    - metadata*.json 
    - recordings/MP4/*.mp4 (video files)
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
from decord import VideoReader, cpu
from math import ceil
from scipy.spatial.transform import Rotation

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_json(directory):
    """Load metadata JSON file from recording directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")
    return None


def poses_to_diffs(poses):
    """Convert absolute poses to relative differences (actions)."""
    xyz = poses[:, :3]  # shape [T, 3]
    thetas = poses[:, 3:6]  # euler angles, shape [T, 3]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    
    xyz_diff = xyz[1:] - xyz[:-1]
    angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    angle_diff = np.stack([d for d in angle_diff], axis=0)
    
    closedness = poses[:, -1:]
    closedness_delta = closedness[1:] - closedness[:-1]
    
    return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)


def load_recording(
    recording_path,
    camera_view="left_mp4_path",
    frames_per_clip=None,
    fps=5,
    frameskip=2,
    start_frame=0,
):
    """
    Load video frames and trajectory from a DROID recording.
    
    Args:
        recording_path: Path to recording folder
        camera_view: Which camera view to use (default: "left_mp4_path")
        frames_per_clip: Number of frames to extract (None = all available)
        fps: Target FPS for sampling
        frameskip: Frame skip for temporal subsampling
        start_frame: Starting frame index (for deterministic sampling)
        
    Returns:
        observations: Video frames [T, H, W, C] - raw uint8 format
        states: Robot states [T, 7]
    """
    print(f"Loading recording from: {recording_path}")
    
    # Load metadata
    metadata = get_json(recording_path)
    if metadata is None:
        raise Exception(f"No metadata found in {recording_path}")
    
    # Load trajectory
    tpath = os.path.join(recording_path, "trajectory.h5")
    if not os.path.exists(tpath):
        raise Exception(f"trajectory.h5 not found in {recording_path}")
    
    trajectory = h5py.File(tpath, 'r')
    
    # Get video path from metadata
    if camera_view not in metadata:
        raise Exception(f"Camera view '{camera_view}' not found in metadata. Available: {list(metadata.keys())}")
    
    mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
    vpath = os.path.join(recording_path, "recordings/MP4", mp4_name)
    
    if not os.path.exists(vpath):
        raise Exception(f"Video file not found: {vpath}")
    
    print(f"Loading video: {vpath}")
    
    # Load video
    vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
    vfps = vr.get_avg_fps()
    vlen = len(vr)
    
    print(f"Video: {vlen} frames at {vfps:.2f} FPS")
    
    # Load states
    states = np.concatenate(
        [
            np.array(trajectory["observation"]["robot_state"]["cartesian_position"]),
            np.array(trajectory["observation"]["robot_state"]["gripper_position"])[:, None],
        ],
        axis=1,
    )  # [T, 7]
    
    print(f"Trajectory: {len(states)} states")
    
    # Determine sampling parameters
    if fps is None:
        fps = vfps
    fstp = ceil(vfps / fps)  # Frame step
    
    if frames_per_clip is None:
        # Use all available frames
        nframes = vlen
        indices = np.arange(0, vlen, fstp).astype(np.int64)
    else:
        nframes = int(frames_per_clip * fstp)
        if vlen < nframes:
            raise Exception(f"Video too short: {vlen} frames, need {nframes}")
        
        # Sample from start_frame
        sf = start_frame
        ef = sf + nframes
        if ef > vlen:
            raise Exception(f"Requested frames exceed video length: {ef} > {vlen}")
        
        indices = np.arange(sf, ef, fstp).astype(np.int64)
    
    print(f"Sampling {len(indices)} frames with step {fstp} (target FPS: {fps})")
    
    # Sample states at the same indices
    states = states[indices, :][::frameskip]
    
    # Load video frames
    vr.seek(0)
    buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C] in uint8
    
    # Apply frameskip to video as well
    buffer = buffer[::frameskip]
    
    print(f"Final shapes: observations={buffer.shape} (dtype={buffer.dtype}), states={states.shape}")
    
    # Ensure correct format: [T, H, W, C] with uint8
    assert buffer.ndim == 4, f"Expected 4D array, got {buffer.ndim}D"
    assert buffer.shape[-1] == 3, f"Expected 3 channels, got {buffer.shape[-1]}"
    
    return buffer, states


def main():
    parser = argparse.ArgumentParser(description='Convert DROID recording to .npz format')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to recording folder')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .npz filename')
    parser.add_argument('--camera', type=str, default='left_mp4_path',
                        help='Camera view to use (default: left_mp4_path)')
    parser.add_argument('--frames', type=int, default=None,
                        help='Number of frames to extract (default: all)')
    parser.add_argument('--fps', type=int, default=5,
                        help='Target FPS for sampling (default: 5)')
    parser.add_argument('--frameskip', type=int, default=2,
                        help='Frame skip for temporal subsampling (default: 2)')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting frame index (default: 0)')
    
    args = parser.parse_args()
    
    # Load recording
    try:
        observations, states = load_recording(
            recording_path=args.input,
            camera_view=args.camera,
            frames_per_clip=args.frames,
            fps=args.fps,
            frameskip=args.frameskip,
            start_frame=args.start,
        )
    except Exception as e:
        print(f"Error loading recording: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Add batch dimension to match expected format
    # Expected: [batch=1, time, height, width, channels]
    observations = np.expand_dims(observations, axis=0)  # [1, T, H, W, C]
    states = np.expand_dims(states, axis=0)  # [1, T, 7]
    
    # Save to .npz
    print(f"\nSaving to: {args.output}")
    np.savez(
        args.output,
        observations=observations,
        states=states,
    )
    
    print("Done!")
    print(f"\nSaved arrays:")
    print(f"  observations: {observations.shape} (dtype={observations.dtype})")
    print(f"  states: {states.shape} (dtype={states.dtype})")
    print(f"\nFormat: [batch, time, height, width, channels]")
    print(f"You can now use this .npz file in energy_landscape_example_uniandes.py")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())