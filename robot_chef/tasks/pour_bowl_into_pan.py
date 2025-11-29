"""Perception-in-the-loop pipeline for pouring with a fixed external camera."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import time # For pauses

import numpy as np
import pybullet as p
import struct
import zlib

from ..camera import Camera, CameraNoiseModel
from ..config import Pose6D, PourTaskConfig
from ..controller_vision import VisionRefineController # Keep using this for IK and moves
from ..perception.bowl_rim import detect_bowl_rim
from ..simulation import RobotChefSimulation

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Small geometry helpers (Unchanged)
# ... (Assume _matrix_to_quaternion, _write_png, _pose_to_quaternion, _apply_world_yaw are correct) ...
def _matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    m = R
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0; w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s; y = 0.25 * s
        x = (m[0, 1] + m[1, 0]) / s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s; z = 0.25 * s
        x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s
    quat = np.array([x, y, z, w], dtype=float); quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat)

def _write_png(path: Path, image: np.ndarray) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    img = image;
    if img.dtype != np.uint8: img = np.clip(img, 0, 255).astype(np.uint8)
    h, w, c = img.shape;
    if c != 3: raise ValueError("Overlay writer expects RGB images")
    raw = bytearray();
    for row in img: raw.append(0); raw.extend(row.tobytes())
    compressed = zlib.compress(bytes(raw), level=6)
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
    with path.open("wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        fh.write(chunk(b"IHDR", ihdr)); fh.write(chunk(b"IDAT", compressed)); fh.write(chunk(b"IEND", b""))

def _pose_to_quaternion(pose: Pose6D) -> Tuple[float, float, float, float]:
    return p.getQuaternionFromEuler([pose.roll, pose.pitch, pose.yaw])

def _apply_world_yaw(pose: Pose6D, yaw_deg: float) -> Pose6D:
    yaw = math.radians(yaw_deg); cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    x = pose.x * cos_y - pose.y * sin_y; y = pose.x * sin_y + pose.y * cos_y
    return Pose6D(x=x, y=y, z=pose.z, roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw + yaw)
# --------------------------------------------------------------------------- #

class PourBowlIntoPanAndReturn:
    """Reworked pouring task focusing on pour location and smoothness."""

    def __init__(self) -> None:
        self.sim: Optional[RobotChefSimulation] = None
        self.cfg: Optional[PourTaskConfig] = None
        self.camera: Optional[Camera] = None
        self.controller: Optional[VisionRefineController] = None
        self.active_arm_name: str = "left"
        self._bowl_uid: Optional[int] = None
        self._pan_uid: Optional[int] = None
        self._metrics: Dict[str, float] = {"transfer_ratio": 0.0, "in_pan": 0, "total": 0}

    # --- Lifecycle (Unchanged) ---
    def setup(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> None:
        self.sim = sim; self.cfg = cfg
        view_cfg = cfg.camera.get_active_view(); noise_cfg = cfg.camera.noise
        LOGGER.info("Setting up fixed camera view: '%s'", cfg.camera.active_view)
        self.camera = Camera( client_id=sim.client_id, view_xyz=view_cfg.xyz, view_rpy_deg=view_cfg.rpy_deg, fov_deg=view_cfg.fov_deg, near=cfg.camera.near, far=cfg.camera.far, resolution=view_cfg.resolution, noise=CameraNoiseModel(depth_std=noise_cfg.depth_std, drop_prob=noise_cfg.drop_prob))
        active = sim.left_arm; self.active_arm_name = "left"
        self.controller = VisionRefineController( client_id=sim.client_id, arm_id=active.body_id, ee_link=active.ee_link, arm_joints=active.joint_indices, dt=sim.dt, camera=self.camera, handeye_T_cam_in_eef=np.eye(4), gripper_open=lambda width=0.08: sim.gripper_open(width=width, arm=self.active_arm_name), gripper_close=lambda force=200.0: sim.gripper_close(force=force, arm=self.active_arm_name))
        sim.gripper_open(arm=self.active_arm_name)
        if cfg.particles > 0: sim.spawn_rice_particles(cfg.particles, seed=cfg.seed)
        bowl_entry = sim.objects.get("bowl") or sim.objects.get("rice_bowl")
        pan_entry = sim.objects.get("pan")
        if bowl_entry is None or pan_entry is None: raise RuntimeError("Missing bowl or pan objects.")
        self._bowl_uid = int(bowl_entry["body_id"]); self._pan_uid = int(pan_entry["body_id"])

    def plan(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        LOGGER.info("Planning completed (fixed camera, no IBVS)."); return True

    def execute(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        if not self.camera or not self.controller: raise RuntimeError("Task not set up.")
        bowl_entry = sim.objects.get("bowl") or sim.objects.get("rice_bowl")
        if bowl_entry is None: LOGGER.error("Bowl not found."); return False
        detection = self._run_perception(sim, cfg, bowl_entry)
        if detection is None: LOGGER.error("Perception failed."); self._metrics = {"transfer_ratio":0.0,"in_pan":0,"total":0}; return False
        success = self._perform_pour_sequence(sim, cfg, detection)
        self._metrics = sim.count_particles_in_pan(); return success

    def metrics(self, sim: RobotChefSimulation) -> Dict[str, float]:
        return dict(self._metrics)

    # --- Perception (Unchanged) ---
    def _run_perception( self, sim: RobotChefSimulation, cfg: PourTaskConfig, bowl_entry: Dict[str, object],) -> Optional[Dict[str, object]]:
        assert self.camera is not None and self.controller is not None
        bowl_pose: Pose6D = bowl_entry["pose"]
        if bowl_pose is None: LOGGER.error("Bowl pose missing."); return None
        self.camera.aim_at_world(bowl_pose.position)
        LOGGER.info("Aimed camera at final bowl pose: (%.3f, %.3f, %.3f)", bowl_pose.x, bowl_pose.y, bowl_pose.z)
        LOGGER.info("Running perception from fixed camera.")
        detection = self._capture_detection(sim, cfg, bowl_entry, bowl_pose, attempt=1)
        if detection is not None: return detection
        LOGGER.error("Failed to detect bowl rim."); return None

    def _capture_detection( self, sim: RobotChefSimulation, cfg: PourTaskConfig, bowl_entry: Dict[str, object], bowl_pose: Pose6D, attempt: int,) -> Optional[Dict[str, object]]:
        assert self.camera is not None
        rgb, depth, K, seg_mask = self.camera.get_rgbd(with_segmentation=True)
        seg = { "client_id": sim.client_id, "camera_from_world": self.camera.camera_from_world, "world_from_camera": self.camera.world_from_camera, "perception_cfg": cfg.perception, "bowl_properties": bowl_entry.get("properties", {}), "bowl_pose": bowl_pose, "segmentation": seg_mask,}
        try: detection = detect_bowl_rim(rgb=rgb, depth=depth, K=K, seg=seg, bowl_uid=int(bowl_entry["body_id"]))
        except Exception as exc: LOGGER.warning("Rim detection failed: %s", exc); return None
        candidates = detection.get("grasp_candidates", [])
        if not candidates: LOGGER.warning("No grasp candidates found."); return None
        out_path = Path(f"attempt{attempt:02d}_overlay.png"); _write_png(out_path, rgb);
        LOGGER.info("Saved detection overlay to %s", out_path.absolute()); LOGGER.info("Detected rim.")
        return detection

    # --- Execution ---
    def _perform_pour_sequence(
        self, sim: RobotChefSimulation, cfg: PourTaskConfig, detection: Dict[str, object],
    ) -> bool:
        assert self.controller is not None
        assert self.camera is not None

        candidates = sorted(detection["grasp_candidates"], key=lambda c: c.get("quality", 0.0), reverse=True)
        if not candidates: LOGGER.error("No grasp candidates."); return False

        MIN_GRASP_WIDTH = 0.005
        LONG_TIMEOUT = 12.0 # Increased slightly again
        NORMAL_TIMEOUT = 9.0 # Increased slightly again

        POUR_STEPS = 150 # Even more steps
        TOTAL_POUR_TIME_S = 8.0 # Pour over 10 seconds
        POUR_STEP_SIM_TIME = max(1, int( (TOTAL_POUR_TIME_S / POUR_STEPS) / sim.dt ))
        POUR_STEP_GAIN = 0.015 # Very low gain


        for idx, candidate in enumerate(candidates[:3]):
            LOGGER.info("Attempting grasp candidate %d", idx + 1)
            # === Step 1: Grip the rim === (Unchanged logic)
            position = np.array(candidate["pose_world"]["position"], dtype=float)
            quat_grasp = np.array(candidate["pose_world"]["quaternion"], dtype=float)
            clearance = cfg.perception.grasp_clearance_m
            pregrasp_pose = position.copy(); pregrasp_pose[2] += clearance + 0.15
            self.controller.open_gripper()
            if not self.controller.move_waypoint(pregrasp_pose, quat_grasp, timeout_s=NORMAL_TIMEOUT): LOGGER.warning("Pre-grasp failed."); continue
            approach_pose = position.copy(); approach_pose[2] += clearance + 0.01
            if not self.controller.move_waypoint(approach_pose, quat_grasp, timeout_s=NORMAL_TIMEOUT): LOGGER.warning("Approach failed."); continue
            grasp_pose = position.copy(); grasp_pose[2] += max(0.0, cfg.perception.rim_thickness_m * 0.5) - 0.003
            if not self.controller.move_waypoint(grasp_pose, quat_grasp, timeout_s=NORMAL_TIMEOUT): LOGGER.warning("Grasp pose failed."); continue
            self.controller.close_gripper(); LOGGER.info("Pausing after grasp command..."); sim.step_simulation(steps=240)
            current_width = sim.get_gripper_width(arm=self.active_arm_name)
            if current_width < MIN_GRASP_WIDTH: LOGGER.warning("Grasp FAILED (%.4f < %.4f). Retrying.", current_width, MIN_GRASP_WIDTH); self.controller.open_gripper(); self.controller.move_waypoint(pregrasp_pose, quat_grasp, timeout_s=NORMAL_TIMEOUT); continue
            LOGGER.info("Grasp SUCCEEDED (%.4f).", current_width)

            # === Step 2: Lift the bowl === (Unchanged logic)
            pan_pose_sim: Pose6D = sim.objects["pan"]["pose"]
            if pan_pose_sim is None: LOGGER.error("Pan pose missing."); continue
            safe_lift_z = pan_pose_sim.z + 0.35
            intermediate_lift_pos = grasp_pose.copy(); intermediate_lift_pos[2] += 0.20
            LOGGER.info("Lifting straight up to %.3f m", intermediate_lift_pos[2])
            if not self.controller.move_waypoint(intermediate_lift_pos, quat_grasp, timeout_s=NORMAL_TIMEOUT): LOGGER.warning("Initial lift failed."); continue
            LOGGER.info("Pausing after initial lift..."); sim.step_simulation(steps=120)
            safe_lift_pos = intermediate_lift_pos.copy(); safe_lift_pos[2] = safe_lift_z
            LOGGER.info("Lifting to safe height: %.3f m", safe_lift_z)
            self.controller.move_waypoint(safe_lift_pos, quat_grasp, timeout_s=LONG_TIMEOUT)

            # === Step 3: Move above the pan (No rotation, Corrected XY) ===
            pan_center_xy = np.array([pan_pose_sim.x, pan_pose_sim.y], dtype=float)
            pan_quat_xyzw = _pose_to_quaternion(pan_pose_sim)
            pan_R = np.array(p.getMatrixFromQuaternion(pan_quat_xyzw)).reshape(3, 3)
            # --- CORRECTED OFFSET CALCULATION ---
            # Pan local +X points along the handle. We want to move in the -X direction.
            offset_dist = 0.20 # Offset 15cm from center, AWAY from handle
            offset_local = np.array([-offset_dist, 0.0, 0.0], dtype=float) # NEGATIVE X offset
            offset_world = pan_R @ offset_local
            above_pan_target_xy = pan_center_xy + offset_world[:2]
            above_pan_target_pos = np.array([above_pan_target_xy[0], above_pan_target_xy[1], safe_lift_z], dtype=float)
            level_quat = quat_grasp
            # --- END CORRECTED OFFSET ---

            # 3a. Smooth horizontal move
            HORIZONTAL_MOVE_STEPS = 25
            current_ls = p.getLinkState(self.controller.arm_id, self.controller.ee_link, physicsClientId=sim.client_id)
            start_pos_horizontal = np.array(current_ls[0], dtype=float)
            LOGGER.info("Moving smoothly above pan cooking area (offset %.2f m) over %d steps...", offset_dist, HORIZONTAL_MOVE_STEPS)
            for i in range(HORIZONTAL_MOVE_STEPS + 1):
                alpha = i / HORIZONTAL_MOVE_STEPS
                inter_pos = (1.0 - alpha) * start_pos_horizontal + alpha * above_pan_target_pos
                step_timeout = max(sim.dt * 12, LONG_TIMEOUT / HORIZONTAL_MOVE_STEPS)
                if not self.controller.move_waypoint(inter_pos, level_quat, timeout_s=step_timeout): LOGGER.warning("Intermediate horizontal move step %d failed.", i); break
            LOGGER.info("Ensuring final horizontal position above pan cooking area.")
            self.controller.move_waypoint(above_pan_target_pos, level_quat, timeout_s=NORMAL_TIMEOUT)

            # 3b. Define Pour Start Pose
            pan_props = sim.objects["pan"]["properties"]
            pan_rim_height = pan_pose_sim.z + pan_props.get("depth", 0.045)
            pour_start_z = pan_rim_height + 0.25 # Start pour high
            pour_start_pos = np.array([above_pan_target_xy[0], above_pan_target_xy[1], pour_start_z], dtype=float)
            pour_start_quat = level_quat

            LOGGER.info("Moving to pour start pose Z=%.3f m", pour_start_z)
            if not self.controller.move_waypoint(pour_start_pos, pour_start_quat, timeout_s=LONG_TIMEOUT):
                 LOGGER.warning("Move to pour start pose failed."); continue


            # === Step 4: Perform the pour slowly (using joint interpolation) ===
            tilt_rad = math.radians(cfg.tilt_angle_deg)
            current_orn_matrix = np.array(p.getMatrixFromQuaternion(pour_start_quat)).reshape(3, 3)
            cy, sy = math.cos(tilt_rad), math.sin(tilt_rad)
            R_pitch_local = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
            pour_orn_matrix = current_orn_matrix @ R_pitch_local
            pour_quat_target = _matrix_to_quaternion(pour_orn_matrix)
            # Pour target position keeps the same XY, drops Z slightly but stays high
            pour_pos_target = pour_start_pos.copy()
            pour_pos_target[2] = max(pan_rim_height + 0.15, pour_start_pos[2] - 0.03) # End pour >= 15cm above rim

            # Get IK for start and end
            q_start_pour = self.controller.get_ik_for_pose(pour_start_pos, pour_start_quat)
            q_end_pour = self.controller.get_ik_for_pose(pour_pos_target, pour_quat_target)
            if q_start_pour is None or q_end_pour is None:
                LOGGER.error("Failed to get IK for pour motion. Aborting."); continue

            # --- GENTLE POUR LOOP ---
            LOGGER.info("Executing gentle pour rotation over %d steps...", POUR_STEPS)
            for i in range(POUR_STEPS + 1):
                alpha = i / POUR_STEPS
                q_current = (1.0 - alpha) * q_start_pour + alpha * q_end_pour
                self.controller.move_to_joint_target(q_current, gain=POUR_STEP_GAIN)
                sim.step_simulation(steps=POUR_STEP_SIM_TIME)
            # --- END GENTLE POUR LOOP ---

            LOGGER.info("Holding pour..."); hold_steps = max(1, int(cfg.hold_sec / sim.dt));
            for _ in range(hold_steps): sim.step_simulation(steps=1)

            # === Step 5: Put the bowl back ===
            # 5a. Rotate back slowly to level pose (reverse loop)
            LOGGER.info("Returning SLOWLY bowl to level pose...")
            for i in range(POUR_STEPS + 1):
                alpha = (POUR_STEPS - i) / POUR_STEPS # Reverse alpha
                q_current = (1.0 - alpha) * q_start_pour + alpha * q_end_pour
                self.controller.move_to_joint_target(q_current, gain=POUR_STEP_GAIN)
                sim.step_simulation(steps=POUR_STEP_SIM_TIME)

            # 5b. Lift back to safe height
            LOGGER.info("Lifting bowl back to safe height above pan.")
            above_pan_safe_pos = pour_start_pos.copy(); above_pan_safe_pos[2] = safe_lift_z
            self.controller.move_waypoint(above_pan_safe_pos, level_quat, timeout_s=LONG_TIMEOUT)

            # 5c. Move horizontally back
            LOGGER.info("Moving smoothly back above bowl start position...")
            back_target_pos = intermediate_lift_pos
            current_ls_return = p.getLinkState(self.controller.arm_id, self.controller.ee_link, physicsClientId=sim.client_id)
            start_pos_return = np.array(current_ls_return[0], dtype=float)
            for i in range(HORIZONTAL_MOVE_STEPS + 1):
                alpha = i / HORIZONTAL_MOVE_STEPS
                inter_pos = (1.0 - alpha) * start_pos_return + alpha * back_target_pos
                step_timeout = max(sim.dt * 12, LONG_TIMEOUT / HORIZONTAL_MOVE_STEPS)
                if not self.controller.move_waypoint(inter_pos, level_quat, timeout_s=step_timeout): LOGGER.warning("Intermediate return move step %d failed.", i)
            LOGGER.info("Ensuring final return position above bowl.")
            self.controller.move_waypoint(back_target_pos, quat_grasp, timeout_s=NORMAL_TIMEOUT)

            # 5d. Place bowl down
            place_pose = grasp_pose.copy(); place_pose[2] += clearance + 0.05
            LOGGER.info("Placing bowl back.")
            self.controller.move_waypoint(place_pose, quat_grasp, timeout_s=NORMAL_TIMEOUT)

            # 5e. Open gripper and retract
            self.controller.open_gripper(); sim.step_simulation(steps=60)
            LOGGER.info("Retracting arm.")
            retract_pos_1 = place_pose.copy(); retract_pos_1[2] = intermediate_lift_pos[2]
            self.controller.move_waypoint(retract_pos_1, quat_grasp, timeout_s=NORMAL_TIMEOUT)
            retract_pos_2 = retract_pos_1.copy(); retract_pos_2[2] += 0.15
            self.controller.move_waypoint(retract_pos_2, quat_grasp, timeout_s=NORMAL_TIMEOUT)

            LOGGER.info("Grasp and pour sequence successful."); return True # Success!

        # If loop finishes, all candidates failed
        LOGGER.error("All grasp candidates failed."); return False

__all__ = ["PourBowlIntoPanAndReturn"]