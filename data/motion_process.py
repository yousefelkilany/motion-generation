"""
Motion Processing Pipeline: Raw BVH to Learnable Motion Features

This module implements a feature extraction pipeline that converts raw BVH motion files
into learnable motion features suitable for machine learning models.

Feature Representation:
- Root rotation velocity (1D): Angular velocity around Y-axis
- Root linear velocity (2D): X, Z velocity (horizontal plane)
- Root height (1D): Y position
- Joint rotations (joint_num * 6): 6D continuous rotation representation
- Root-invariant positions (RIC) (joint_num * 3): Local positions relative to root
- Joint velocities (joint_num * 3): Velocity of each joint
- Foot contacts (4D): Binary contact labels for left/right foot joints

Preprocessing Steps:
1. Resampling to target FPS (downsampling or upsampling with interpolation)
2. Forward kinematics to get global positions
3. Floor alignment (put character on ground)
4. Forward direction extraction and smoothing
5. Root rotation normalization (all poses face Z+)
6. Foot contact detection based on velocity and height

Uses custom Skeleton and quaternion utilities from utils/ directory.
"""

import logging
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from utils.math import qbetween_np, qinv_np, qmul_np, qrot_np, quaternion_to_cont6d_np
from utils.skeleton import Skeleton

try:
    from pymotion.io.bvh import BVH

    PYMOTION_AVAILABLE = True
except ImportError:
    PYMOTION_AVAILABLE = False
    print("Warning: pymotion not available. Install with: pip install pymotion")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default joint names for contact and face direction detection
# Updated to match motion-s BVH naming convention (lowercase with underscores)
CONTACT_JOINT_NAMES = ["left_ankle", "left_foot", "right_ankle", "right_foot"]

FACE_JOINT_NAMES = ["left_hip", "right_hip", "left_collar", "right_collar"]

# Alternative naming patterns for more flexible matching
# Prioritizes motion-s convention (left_ankle, left_foot, etc.) but includes other common variations
CONTACT_JOINT_PATTERNS = {
    "left": [
        "left_ankle",
        "left_foot",
        "leftankle",
        "leftfoot",
        "lefttoe",
        "l_ankle",
        "l_foot",
        "foot_l",
        "lfoot",
        "L_foot",
        "LFoot",
        "L_ankle",
        "LAnkle",
        "LeftAnkle",
        "LeftToe",
        "LeftFoot",
    ],
    "right": [
        "right_ankle",
        "right_foot",
        "rightankle",
        "rightfoot",
        "righttoe",
        "r_ankle",
        "r_foot",
        "foot_r",
        "rfoot",
        "R_foot",
        "RFoot",
        "R_ankle",
        "RAnkle",
        "RightAnkle",
        "RightToe",
        "RightFoot",
    ],
}

FACE_JOINT_PATTERNS = {
    "r_hip": [
        "right_hip",
        "righthip",
        "R_hip",
        "RHip",
        "r_hip",
        "r_legupper",
        "R_legUpper",
        "RightHip",
    ],
    "l_hip": [
        "left_hip",
        "lefthip",
        "L_hip",
        "LHip",
        "l_hip",
        "l_legupper",
        "L_legUpper",
        "LeftHip",
    ],
    "r_sdr": [
        "right_collar",
        "rightcollar",
        "r_clavicle",
        "rightclavicle",
        "R_clavicle",
        "RClavicle",
        "r_shoulder",
        "rightshoulder",
        "RightCollar",
        "RightShoulder",
    ],
    "l_sdr": [
        "left_collar",
        "leftcollar",
        "l_clavicle",
        "leftclavicle",
        "L_clavicle",
        "LClavicle",
        "l_shoulder",
        "leftshoulder",
        "LeftCollar",
        "LeftShoulder",
    ],
}


def find_joint_by_pattern(joint_names, patterns):
    """
    Find joint index by matching patterns (case-insensitive).

    Args:
        joint_names: List of joint names
        patterns: List of pattern strings to match

    Returns:
        Joint index or None if not found
    """
    for i, name in enumerate(joint_names):
        name_lower = name.lower()
        for pattern in patterns:
            if pattern.lower() in name_lower:
                return i
    return None


def find_joints_dict(joint_names, contact_names=None, face_names=None):
    """
    Find joint indices for contact and face direction joints.

    Args:
        joint_names: List of all joint names
        contact_names: List of contact joint names (optional)
        face_names: List of face joint names (optional)

    Returns:
        Dictionary with joint indices
    """
    joints_dict = {name: i for i, name in enumerate(joint_names)}

    # Find contact joints
    contact_ids = []
    if contact_names:
        for name in contact_names:
            if name in joints_dict:
                contact_ids.append(joints_dict[name])
            else:
                # Try pattern matching
                idx = find_joint_by_pattern(joint_names, [name])
                if idx is not None:
                    contact_ids.append(idx)

    # If not found with exact names, try patterns
    if len(contact_ids) < 4:
        contact_ids = []
        for side in ["left", "right"]:
            for pattern in CONTACT_JOINT_PATTERNS[side]:
                idx = find_joint_by_pattern(joint_names, [pattern])
                if idx is not None:
                    contact_ids.append(idx)
                    break

    # Find face direction joints
    face_joint_ids = []
    if face_names:
        for name in face_names:
            if name in joints_dict:
                face_joint_ids.append(joints_dict[name])
            else:
                idx = find_joint_by_pattern(joint_names, [name])
                if idx is not None:
                    face_joint_ids.append(idx)

    # If not found, try patterns
    if len(face_joint_ids) < 4:
        face_joint_ids = []
        for key in ["r_hip", "l_hip", "r_sdr", "l_sdr"]:
            for pattern in FACE_JOINT_PATTERNS[key]:
                idx = find_joint_by_pattern(joint_names, [pattern])
                if idx is not None:
                    face_joint_ids.append(idx)
                    break

    return {
        "joints_dict": joints_dict,
        "contact_ids": contact_ids[:4] if len(contact_ids) >= 4 else contact_ids,
        "face_joint_ids": face_joint_ids[:4]
        if len(face_joint_ids) >= 4
        else face_joint_ids,
    }


def process_bvh_motion(
    filepath,
    now_fps=30,
    target_fps=30,
    feet_thre=0.11,
    shift_one_frame=False,
    contact_joint_names=None,
    face_joint_names=None,
):
    """
    Process BVH motion file and extract learnable motion features.

    Feature representation:
    - Root rotation velocity (1D)
    - Root linear velocity (2D: X, Z)
    - Root height (1D: Y)
    - Joint rotations in 6D continuous representation (joint_num * 6)
    - Root-invariant positions (RIC) (joint_num * 3)
    - Joint velocities (joint_num * 3)
    - Foot contacts (4D: left foot 1, left foot 2, right foot 1, right foot 2)

    Args:
        filepath: Path to BVH file
        now_fps: Original frame rate
        target_fps: Target frame rate
            - If now_fps > target_fps: Downsampling (now_fps must be divisible by target_fps)
            - If now_fps < target_fps: Upsampling (interpolation will be applied)
            - If now_fps == target_fps: No resampling
        feet_thre: Threshold for foot contact detection
        shift_one_frame: Whether to skip first frame
        contact_joint_names: Custom contact joint names (optional)
        face_joint_names: Custom face direction joint names (optional)

    Returns:
        data: (N-1, D) feature array where D = 1 + 2 + 1 + joint_num*6 + joint_num*3 + joint_num*3 + 4
    """
    if not PYMOTION_AVAILABLE:
        raise ImportError("pymotion is required for BVH loading")

    # Load BVH file
    bvh = BVH()
    bvh.load(filepath)

    # Extract data
    local_quaternions, local_positions, parents, offsets, _, _ = bvh.get_data()
    joint_names = bvh.data["names"]

    # Shift frame if needed
    if shift_one_frame:
        local_quaternions = local_quaternions[1:]
        local_positions = local_positions[1:]

    # Find joint indices
    joints_info = find_joints_dict(
        joint_names,
        contact_joint_names or CONTACT_JOINT_NAMES,
        face_joint_names or FACE_JOINT_NAMES,
    )
    contact_ids = joints_info["contact_ids"]
    face_joint_ids = joints_info["face_joint_ids"]

    if len(face_joint_ids) < 4:
        logger.warning(
            f"Could not find all face direction joints. Found: {face_joint_ids}"
        )
        logger.warning("Using alternative method for forward direction")
        # Use hips and shoulders if available
        if len(face_joint_ids) >= 2:
            # Duplicate if needed
            while len(face_joint_ids) < 4:
                face_joint_ids.append(face_joint_ids[-1])
        else:
            # Fallback: use first 4 joints
            face_joint_ids = list(range(min(4, len(joint_names))))

    r_hip, l_hip, r_sdr, l_sdr = face_joint_ids[:4]

    if len(contact_ids) >= 4:
        fid_l, fid_r = contact_ids[:2], contact_ids[2:4]
    elif len(contact_ids) >= 2:
        fid_l, fid_r = [contact_ids[0]], [contact_ids[1]]
    else:
        logger.warning(f"Could not find contact joints. Found: {contact_ids}")
        fid_l, fid_r = [], []

    # Resampling (downsampling or upsampling)
    if now_fps == target_fps:
        # No resampling needed
        rotations = local_quaternions
        positions = local_positions
    elif now_fps > target_fps:
        # Downsampling: require divisibility for clean downsampling
        assert now_fps % target_fps == 0, (
            f"Invalid target fps! {now_fps} must be divisible by {target_fps} for downsampling"
        )
        ds_rate = now_fps // target_fps
        rotations = local_quaternions[::ds_rate]
        positions = local_positions[::ds_rate]
    else:
        # Upsampling: interpolate frames
        # Calculate upsampling factor
        upsample_factor = target_fps / now_fps

        # Create time arrays
        original_times = np.arange(len(local_quaternions))
        # Create new time array with upsampled rate
        # We want to go from 0 to len-1 with step of 1/upsample_factor
        new_length = int((len(local_quaternions) - 1) * upsample_factor) + 1
        new_times = np.linspace(0, len(local_quaternions) - 1, new_length)

        # Interpolate positions (linear interpolation is fine for positions)
        positions = np.zeros(
            (new_length, local_positions.shape[1], local_positions.shape[2])
        )
        for joint_idx in range(local_positions.shape[1]):
            for coord_idx in range(local_positions.shape[2]):
                interp_func = interp1d(
                    original_times,
                    local_positions[:, joint_idx, coord_idx],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                positions[:, joint_idx, coord_idx] = interp_func(new_times)

        # Interpolate quaternions (use spherical linear interpolation)
        # For quaternions, we'll use linear interpolation of components followed by normalization
        # This is a reasonable approximation for small time steps
        rotations = np.zeros(
            (new_length, local_quaternions.shape[1], local_quaternions.shape[2])
        )
        for joint_idx in range(local_quaternions.shape[1]):
            # Linear interpolation of quaternion components
            interp_quat = np.zeros((new_length, 4))
            for comp_idx in range(4):
                interp_func = interp1d(
                    original_times,
                    local_quaternions[:, joint_idx, comp_idx],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                interp_quat[:, comp_idx] = interp_func(new_times)

            # Normalize quaternions to ensure they remain valid
            norms = np.linalg.norm(interp_quat, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)  # Avoid division by zero
            rotations[:, joint_idx, :] = interp_quat / norms

        logger.info(
            f"Upsampled motion from {now_fps} fps to {target_fps} fps: {len(local_quaternions)} -> {new_length} frames"
        )

    # Forward Kinematics
    skeleton = Skeleton(offsets, parents, device="cpu")
    global_quat, global_pos = skeleton.fk_local_quat_np(rotations, positions[:, 0])

    # Put on Floor
    # Y+ is up axis
    if len(contact_ids) > 0:
        seq = np.sort(global_pos[:, contact_ids, 1].flatten())
        nmin_seq = int(len(seq) * 0.1)
        floor_height = seq[:nmin_seq].mean()
    else:
        # Fallback: use minimum height
        floor_height = global_pos[:, :, 1].min()

    global_pos[:, :, 1] -= floor_height

    # Extract forward direction and smooth
    across = (global_pos[:, l_hip] - global_pos[:, r_hip]) + (
        global_pos[:, l_sdr] - global_pos[:, r_sdr]
    )
    across = across / (np.sqrt((across**2).sum(axis=-1))[..., np.newaxis] + 1e-8)

    direction_filterwidth = 5
    forward = gaussian_filter1d(
        np.cross(across, np.array([[0, 1, 0]])),
        direction_filterwidth,
        axis=0,
        mode="nearest",
    )
    forward = forward / (np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis] + 1e-8)

    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    root_rotations = qbetween_np(forward, target)[:, np.newaxis]
    root_rotations = np.repeat(root_rotations, global_pos.shape[1], axis=1)

    # All initially face z+
    root_rotation_init = root_rotations[0:1].repeat(len(root_rotations), axis=0)
    root_rotations = qmul_np(qinv_np(root_rotation_init), root_rotations)
    global_quat = qmul_np(root_rotation_init, global_quat)
    global_pos = qrot_np(root_rotation_init, global_pos)

    # Re-gain global positions
    global_pos = skeleton.fk_global_quat_np(global_quat, global_pos[:, 0])

    # Get foot contact
    def detect_contact(positions, thres):
        """Detect foot contact based on velocity and height."""
        velfactor, heightfactor = np.array([thres, thres]), np.array([9.0, 4.0])

        if len(fid_l) == 0 or len(fid_r) == 0:
            # Return zeros if no foot joints found
            feet_l = np.zeros((len(positions) - 1, 1))
            feet_r = np.zeros((len(positions) - 1, 1))
            return feet_l, feet_r

        # def get_f(fid):
        #     if len(fid) >= 2:
        #         v1 = np.sum(
        #             (positions[1:, fid[0]] - positions[:-1, fid[0]]) ** 2, axis=-1
        #         )
        #         v2 = np.sum(
        #             (positions[1:, fid[1]] - positions[:-1, fid[1]]) ** 2, axis=-1
        #         )
        #         return np.stack(
        #             [
        #                 (v1 < velfactor[0])
        #                 & (positions[:-1, fid[0], 1] < heightfactor[0]),
        #                 (v2 < velfactor[1])
        #                 & (positions[:-1, fid[1], 1] < heightfactor[1]),
        #             ],
        #             -1,
        #         ).astype(float)
        #     v = np.sum((positions[1:, fid[0]] - positions[:-1, fid[0]]) ** 2, axis=-1)
        #     f = (
        #         (v < velfactor[0]) & (positions[:-1, fid[0], 1] < heightfactor[0])
        #     ).astype(float)
        #     return np.stack([f, f], -1)

        # return get_f(fid_l), get_f(fid_r)

        # Left foot
        if len(fid_l) >= 2:
            feet_l_x = (positions[1:, fid_l[0], 0] - positions[:-1, fid_l[0], 0]) ** 2
            feet_l_y = (positions[1:, fid_l[0], 1] - positions[:-1, fid_l[0], 1]) ** 2
            feet_l_z = (positions[1:, fid_l[0], 2] - positions[:-1, fid_l[0], 2]) ** 2
            feet_l_h = positions[:-1, fid_l[0], 1]
            feet_l_1 = (
                ((feet_l_x + feet_l_y + feet_l_z) < velfactor[0])
                & (feet_l_h < heightfactor[0])
            ).astype(float)

            feet_l_x2 = (positions[1:, fid_l[1], 0] - positions[:-1, fid_l[1], 0]) ** 2
            feet_l_y2 = (positions[1:, fid_l[1], 1] - positions[:-1, fid_l[1], 1]) ** 2
            feet_l_z2 = (positions[1:, fid_l[1], 2] - positions[:-1, fid_l[1], 2]) ** 2
            feet_l_h2 = positions[:-1, fid_l[1], 1]
            feet_l_2 = (
                ((feet_l_x2 + feet_l_y2 + feet_l_z2) < velfactor[1])
                & (feet_l_h2 < heightfactor[1])
            ).astype(float)
            feet_l = np.stack([feet_l_1, feet_l_2], axis=-1)
        else:
            feet_l_x = (positions[1:, fid_l[0], 0] - positions[:-1, fid_l[0], 0]) ** 2
            feet_l_y = (positions[1:, fid_l[0], 1] - positions[:-1, fid_l[0], 1]) ** 2
            feet_l_z = (positions[1:, fid_l[0], 2] - positions[:-1, fid_l[0], 2]) ** 2
            feet_l_h = positions[:-1, fid_l[0], 1]
            feet_l_1 = (
                ((feet_l_x + feet_l_y + feet_l_z) < velfactor[0])
                & (feet_l_h < heightfactor[0])
            ).astype(float)
            feet_l = np.stack(
                [feet_l_1, feet_l_1], axis=-1
            )  # Duplicate if only one joint

        # Right foot
        if len(fid_r) >= 2:
            feet_r_x = (positions[1:, fid_r[0], 0] - positions[:-1, fid_r[0], 0]) ** 2
            feet_r_y = (positions[1:, fid_r[0], 1] - positions[:-1, fid_r[0], 1]) ** 2
            feet_r_z = (positions[1:, fid_r[0], 2] - positions[:-1, fid_r[0], 2]) ** 2
            feet_r_h = positions[:-1, fid_r[0], 1]
            feet_r_1 = (
                ((feet_r_x + feet_r_y + feet_r_z) < velfactor[0])
                & (feet_r_h < heightfactor[0])
            ).astype(float)

            feet_r_x2 = (positions[1:, fid_r[1], 0] - positions[:-1, fid_r[1], 0]) ** 2
            feet_r_y2 = (positions[1:, fid_r[1], 1] - positions[:-1, fid_r[1], 1]) ** 2
            feet_r_z2 = (positions[1:, fid_r[1], 2] - positions[:-1, fid_r[1], 2]) ** 2
            feet_r_h2 = positions[:-1, fid_r[1], 1]
            feet_r_2 = (
                ((feet_r_x2 + feet_r_y2 + feet_r_z2) < velfactor[1])
                & (feet_r_h2 < heightfactor[1])
            ).astype(float)
            feet_r = np.stack([feet_r_1, feet_r_2], axis=-1)
        else:
            feet_r_x = (positions[1:, fid_r[0], 0] - positions[:-1, fid_r[0], 0]) ** 2
            feet_r_y = (positions[1:, fid_r[0], 1] - positions[:-1, fid_r[0], 1]) ** 2
            feet_r_z = (positions[1:, fid_r[0], 2] - positions[:-1, fid_r[0], 2]) ** 2
            feet_r_h = positions[:-1, fid_r[0], 1]
            feet_r_1 = (
                ((feet_r_x + feet_r_y + feet_r_z) < velfactor[0])
                & (feet_r_h < heightfactor[0])
            ).astype(float)
            feet_r = np.stack(
                [feet_r_1, feet_r_1], axis=-1
            )  # Duplicate if only one joint

        return feet_l, feet_r

    def get_con6d_params(r_rot, r_pos, quat_params):
        """Remove root rotations and convert to 6D representation."""
        # Remove root rotations from joint rotations
        quat_params = qmul_np(r_rot, quat_params)

        # Quaternion to continuous 6D representation
        cont6d_params = quaternion_to_cont6d_np(quat_params)

        # Root Linear Velocity
        velocity = (r_pos[1:] - r_pos[:-1]).copy()
        velocity = qrot_np(r_rot[:-1, 0], velocity)

        # Root angular velocity
        r_velocity = qmul_np(r_rot[1:, 0], qinv_np(r_rot[:-1, 0]))
        r_velocity = r_velocity / (np.linalg.norm(r_velocity, axis=-1)[:, None] + 1e-8)
        r_velocity = np.arctan2(r_velocity[:, 2:3], r_velocity[:, 0:1]) * 2

        return cont6d_params[:-1], velocity, r_velocity

    def get_local_positions(r_rot, positions):
        """Get root-invariant local positions."""
        positions = positions.copy()

        # Local pose (remove root XZ translation)
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]

        # All pose face Z+
        positions = qrot_np(r_rot, positions)

        # Get Joint Velocity
        local_vel = positions[1:] - positions[:-1]

        return positions[:-1], local_vel

    # Process features
    feet_l, feet_r = detect_contact(global_pos, thres=feet_thre)
    cont6d_param, l_velocity, r_velocity = get_con6d_params(
        root_rotations, global_pos[:, 0], global_quat
    )
    local_positions, local_velocity = get_local_positions(root_rotations, global_pos)

    # Root height
    root_y = local_positions[:, 0, 1:2]

    # Linear root velocity (only X, Z)
    l_velocity = l_velocity[:, [0, 2]]

    # Root data: [r_velocity, l_velocity, root_y]
    root_data = np.concatenate([r_velocity, l_velocity, root_y], axis=-1)

    # Get joint rotation representation
    rot_data = cont6d_param.reshape(len(cont6d_param), -1)

    # Get root-rotation-invariant position representation
    ric_data = local_positions.reshape(len(local_positions), -1)

    # Get Joint Velocity Representation
    vel_data = local_velocity.reshape(len(local_velocity), -1)

    # Concatenate foot contacts
    if feet_l.shape[-1] == 2 and feet_r.shape[-1] == 2:
        foot_contacts = np.concatenate([feet_l, feet_r], axis=-1)
    elif feet_l.shape[-1] == 1 and feet_r.shape[-1] == 1:
        # Duplicate to match expected 4D
        foot_contacts = np.concatenate([feet_l, feet_l, feet_r, feet_r], axis=-1)
    else:
        # Fallback: create zeros
        foot_contacts = np.zeros((len(rot_data), 4))

    # Concatenate all features
    data = np.concatenate(
        [root_data, rot_data, ric_data, vel_data, foot_contacts], axis=-1
    )

    return data


def batch_process_bvh(
    src_root,
    tgt_root,
    now_fps=24,
    target_fps=30,
    feet_thre=0.11,
    shift_one_frame=False,
    overwrite=False,
):
    """
    Process stitched BVH files from dataset folder structure.

    Expected structure:
        src_root/
            1630/
                1630.bvh  <- stitched file (named after folder)
                dream.bvh, have.bvh, me.bvh  <- source files (ignored)
            1885/
                1885.bvh
                ...

    Args:
        src_root: Root directory containing sentence folders with stitched BVH files
        tgt_root: Root directory for output numpy files (flat structure)
        now_fps: Original frame rate of stitched BVH files (default: 24)
        target_fps: Target frame rate for features (default: 30)
        feet_thre: Foot contact threshold
        shift_one_frame: Whether to skip first frame
        overwrite: Whether to overwrite existing files
    """
    src_root = Path(src_root)
    tgt_root = Path(tgt_root)
    tgt_root.mkdir(parents=True, exist_ok=True)

    # Find all sentence folders with stitched BVH files
    stitched_files = []
    for folder in os.listdir(src_root):
        folder_path = src_root / folder
        if not folder_path.is_dir():
            continue

        # Look for stitched BVH file (named after the folder)
        stitched_bvh = folder_path / f"{folder}.bvh"
        if stitched_bvh.exists():
            stitched_files.append((folder, stitched_bvh))

    logger.info(f"Found {len(stitched_files)} stitched BVH files to process")

    if len(stitched_files) == 0:
        logger.warning(f"No stitched BVH files found in {src_root}")
        logger.warning("Expected structure: src_root/ID/ID.bvh")
        return

    num_frame = 0
    processed = 0
    skipped = 0
    failed = 0

    for sentence_id, bvh_path in tqdm(stitched_files, desc="Processing stitched BVHs"):
        try:
            output_path = tgt_root / f"{sentence_id}.npy"

            # Skip if already exists and not overwriting
            if output_path.exists() and not overwrite:
                skipped += 1
                continue

            data = process_bvh_motion(
                str(bvh_path),
                now_fps=now_fps,
                target_fps=target_fps,
                feet_thre=feet_thre,
                shift_one_frame=shift_one_frame,
            )

            np.save(output_path, data)
            num_frame += len(data)
            processed += 1

        except Exception as e:
            logger.error(f"Failed to process {sentence_id}: {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed: {processed}, Skipped: {skipped}, Failed: {failed}")
    if processed > 0:
        logger.info(f"Total duration: {num_frame / target_fps / 3600:.4f}h")
        logger.info(f"Average duration: {num_frame / target_fps / processed:.4f}s")
        logger.info(f"Total frames: {num_frame}")
