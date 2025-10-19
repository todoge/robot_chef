# robot_chef/perception/bowl_rim.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------- Helpers ----------------

def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    return vec if norm < 1e-9 else vec / norm

# ---------------- Context ----------------

@dataclass
class _BowlRimContext:
    bowl_min: np.ndarray
    bowl_max: np.ndarray
    world_from_cam: np.ndarray
    margin: float
    sample_count: int
    clearance: float
    reachability_fn: Optional[Callable[[np.ndarray, np.ndarray], Tuple[bool, float]]] = None


_CONTEXT: Optional[_BowlRimContext] = None


def configure_scene_context(
    bowl_aabb_min: Tuple[float, float, float],
    bowl_aabb_max: Tuple[float, float, float],
    world_from_cam: np.ndarray,
    roi_margin: float,
    sample_count: int,
    grasp_clearance: float,
    reachability_fn=None,
) -> None:
    """Set per-frame scene context used by detect_bowl_rim()."""
    global _CONTEXT
    _CONTEXT = _BowlRimContext(
        bowl_min=np.array(bowl_aabb_min, dtype=float),
        bowl_max=np.array(bowl_aabb_max, dtype=float),
        world_from_cam=np.array(world_from_cam, dtype=float),
        margin=float(roi_margin),
        sample_count=int(sample_count),
        clearance=float(grasp_clearance),
        reachability_fn=reachability_fn,
    )


# ---------------- Helpers ----------------

def _gaussian_blur_depth(depth: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    if ksize < 3 or ksize % 2 == 0:
        return depth
    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    ker = np.exp(-(ax**2) / (2.0 * sigma * sigma)).astype(np.float32)
    ker /= ker.sum()

    D = depth.astype(np.float32)
    mask = (D > 0.0).astype(np.float32)
    pad = ksize // 2

    # Horizontal
    Dp = np.pad(D, ((0, 0), (pad, pad)), mode="edge")
    Mp = np.pad(mask, ((0, 0), (pad, pad)), mode="edge")
    Dr = np.zeros_like(Dp)
    Mr = np.zeros_like(Mp)
    for i in range(ksize):
        sl = slice(i, i + D.shape[1])
        Dr[:, sl] += ker[i] * Dp[:, sl]
        Mr[:, sl] += ker[i] * Mp[:, sl]
    Dr = np.where(Mr > 1e-6, Dr / np.maximum(Mr, 1e-6), 0.0)
    Dr = Dr[:, pad:-pad]

    # Vertical
    Dp = np.pad(Dr, ((pad, pad), (0, 0)), mode="edge")
    Mp = np.pad((Mr[:, pad:-pad] > 0).astype(np.float32), ((pad, pad), (0, 0)), mode="edge")
    Dc = np.zeros_like(Dp)
    Mc = np.zeros_like(Mp)
    for i in range(ksize):
        sl = slice(i, i + Dr.shape[0])
        Dc[sl, :] += ker[i] * Dp[sl, :]
        Mc[sl, :] += ker[i] * Mp[sl, :]
    Dc = np.where(Mc > 1e-6, Dc / np.maximum(Mc, 1e-6), 0.0)
    Dc = Dc[pad:-pad, :]
    return Dc


def _depth_to_camera_points(depth: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    Z = depth.astype(np.float32)
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    pixels = np.stack([u, v], axis=-1).reshape(-1, 2)
    return points, pixels


def _transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    Pw = np.concatenate([pts, ones], axis=1) @ T.T
    return Pw[:, :3]


def _ransac_plane(points: np.ndarray, iterations: int = 200, tol: float = 0.004):
    N = points.shape[0]
    if N < 3:
        return None, None, np.zeros((N,), dtype=bool)
    rng = np.random.default_rng(123)
    best_inliers = np.zeros((N,), dtype=bool)
    best_n = None
    best_d = None
    for _ in range(iterations):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            continue
        n /= max(n_norm, 1e-12)
        d = -np.dot(n, p0)
        dist = np.abs(points @ n + d)
        inliers = dist < tol
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_n, best_d = n, d
    return best_n, best_d, best_inliers


def _adaptive_rim_band(z_vals: np.ndarray, default_bw: float = 0.015) -> Tuple[float, float]:
    if z_vals.size == 0:
        return 0.0, default_bw
    p97 = float(np.percentile(z_vals, 97.0))
    p90 = float(np.percentile(z_vals, 90.0))
    spread = max(1e-3, p97 - p90)
    bw = max(default_bw * 0.5, min(default_bw * 2.0, 2.5 * spread))
    return p97, bw


def _fit_circle_ransac(points_xy: np.ndarray, iters: int = 300, tol: float = 0.01):
    if points_xy.shape[0] < 3:
        return np.zeros(2), 0.0, np.zeros((points_xy.shape[0],), dtype=bool)
    rng = np.random.default_rng(7)
    best_c = np.zeros(2)
    best_r = 0.0
    best_in = np.zeros((points_xy.shape[0],), dtype=bool)
    for _ in range(iters):
        idx = rng.choice(points_xy.shape[0], 3, replace=False)
        A, B, C = points_xy[idx]
        a = 2 * (B[0] - A[0]); b = 2 * (B[1] - A[1]); c = B[0] ** 2 + B[1] ** 2 - A[0] ** 2 - A[1] ** 2
        d = 2 * (C[0] - A[0]); e = 2 * (C[1] - A[1]); f = C[0] ** 2 + C[1] ** 2 - A[0] ** 2 - A[1] ** 2
        denom = (a * e - b * d)
        if abs(denom) < 1e-9:
            continue
        cx = (c * e - b * f) / denom
        cy = (a * f - c * d) / denom
        center = np.array([cx, cy], dtype=float)
        r = float(np.mean(np.linalg.norm(points_xy - center, axis=1)))
        inliers = np.abs(np.linalg.norm(points_xy - center, axis=1) - r) < tol
        if inliers.sum() > best_in.sum():
            best_c = center; best_r = r; best_in = inliers
    return best_c, float(best_r), best_in


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> quaternion [x,y,z,w], numerically stable."""
    R = np.asarray(R, dtype=float)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    q /= (np.linalg.norm(q) + 1e-12)
    return q


def _sample_grasp_poses(
    center: np.ndarray,
    rim_points: np.ndarray,
    ctx: _BowlRimContext,
    *,
    gripper_width_max: float = 0.08,
    rim_thickness_hint: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Generate forced-closure rim grasp candidates with x=tangent, y=outward, z=up."""
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    rim_points = np.asarray(rim_points, dtype=float)
    sample_count = max(8, ctx.sample_count)
    results: List[Dict[str, object]] = []

    radii = np.linalg.norm(rim_points[:, :2] - center[:2], axis=1)
    radius_mean = float(np.mean(radii))
    thickness_est = float(
        rim_thickness_hint
        if rim_thickness_hint is not None
        else max(0.005, min(0.012, float(np.percentile(radii, 95) - np.percentile(radii, 5))))
    )
    span = thickness_est * 2.0
    if span >= gripper_width_max * 0.95:
        thickness_est = gripper_width_max * 0.45
        span = thickness_est * 2.0

    for idx in range(sample_count):
        theta = (2.0 * math.pi * idx) / sample_count
        rim_hint = center + np.array([radius_mean * math.cos(theta), radius_mean * math.sin(theta), 0.0])
        nearest = np.argmin(np.linalg.norm(rim_points[:, :2] - rim_hint[:2], axis=1))
        rim_point = rim_points[nearest]

        outward = rim_point - center
        outward /= np.linalg.norm(outward) + 1e-12
        tangent = np.cross(up, outward)
        if np.linalg.norm(tangent) < 1e-8:
            tangent = np.array([0.0, 1.0, 0.0], dtype=float)
        tangent = _normalize(tangent)

        # Forced-closure contacts: inner/outer along rim normal
        contact_outer = rim_point + outward * (span * 0.5)
        contact_inner = rim_point - outward * (span * 0.5)
        normal_outer = _normalize(contact_outer - center)
        normal_inner = _normalize(contact_inner - center)

        opp_angle = math.degrees(math.acos(np.clip(np.dot(normal_outer, -normal_inner), -1.0, 1.0)))
        clearance_ok = span <= gripper_width_max

        # Build grasp frame: z-axis aligned with -up (approach downward),
        # y-axis aligned with -outward (closing direction),
        # x-axis completes right-handed frame (approx tangent).
        z_axis = -up
        y_axis = -outward
        x_axis = np.cross(y_axis, z_axis)
        if np.linalg.norm(x_axis) < 1e-8:
            x_axis = tangent
        x_axis = _normalize(x_axis)
        y_axis = _normalize(np.cross(z_axis, x_axis))
        z_axis = _normalize(np.cross(x_axis, y_axis))
        R = np.column_stack([x_axis, y_axis, z_axis])
        quat = _rotation_matrix_to_quaternion(R)

        reach_score = 1.0
        reachable = True
        if ctx.reachability_fn is not None:
            ok, score = ctx.reachability_fn(rim_point, quat)
            reachable = bool(ok)
            reach_score = float(score if ok else 0.0)
        if not reachable:
            continue

        clearance_ratio = min(1.0, max(0.0, ctx.clearance / 0.05))
        closure_score = max(0.0, 1.0 - abs(opp_angle) / 40.0)
        width_penalty = 1.0 if clearance_ok else max(0.4, 1.0 - (span - gripper_width_max) * 12.5)
        quality = 0.5 * reach_score + 0.3 * closure_score + 0.2 * clearance_ratio
        quality *= max(0.1, width_penalty)
        quality = max(0.0, min(1.0, quality))

        results.append(
            {
                "pose_world": {
                    "position": tuple(float(v) for v in rim_point),
                    "quaternion": tuple(float(v) for v in quat),
                },
                "quality": float(max(0.0, min(1.0, quality))),
                "contacts": {
                    "outer": tuple(float(v) for v in contact_outer),
                    "inner": tuple(float(v) for v in contact_inner),
                },
                "span_m": float(span),
            }
        )

    results.sort(key=lambda c: c["quality"], reverse=True)
    return results


# ---------------- Fallback: model rim from AABB ----------------

def _model_rim_from_context(ctx: _BowlRimContext, N: int = 16) -> Dict[str, object]:
    """Synthesize a circular rim at the AABB top with relaxed radius."""
    c_xy = 0.5 * (ctx.bowl_min[:2] + ctx.bowl_max[:2])
    z_top = float(ctx.bowl_max[2])
    r_guess = 0.55 * float(np.max(ctx.bowl_max[:2] - ctx.bowl_min[:2]))
    N = max(8, int(N))
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    rim_pts = []
    cands = []
    for i in range(N):
        th = 2.0 * math.pi * i / N
        rp = np.array([c_xy[0] + r_guess * math.cos(th),
                       c_xy[1] + r_guess * math.sin(th),
                       z_top], dtype=float)
        outward = rp - np.array([c_xy[0], c_xy[1], z_top], dtype=float)
        outward /= (np.linalg.norm(outward) + 1e-12)
        tangent = np.cross(up, outward)
        tangent = _normalize(tangent)
        z_axis = -up
        y_axis = -outward
        x_axis = np.cross(y_axis, z_axis)
        if np.linalg.norm(x_axis) < 1e-8:
            x_axis = tangent
        x_axis = _normalize(x_axis)
        y_axis = _normalize(np.cross(z_axis, x_axis))
        z_axis = _normalize(np.cross(x_axis, y_axis))
        R = np.column_stack([x_axis, y_axis, z_axis])
        quat = _rotation_matrix_to_quaternion(R)
        rim_pts.append(rp)
        cands.append(
            {
                "pose_world": {
                    "position": tuple(float(v) for v in rp),
                    "quaternion": tuple(float(v) for v in quat),
                },
                "quality": 0.5,
                "contacts": {},
                "span_m": 0.02,
            }
        )

    center_3d = np.array([c_xy[0], c_xy[1], z_top], dtype=float)
    return {
        "rim_pts_3d": np.array(rim_pts, dtype=float),
        "center_3d": center_3d,
        "radius_m": float(r_guess),
        "grasp_candidates": cands,
    }


# ---------------- Segmentation decoding ----------------

def _decode_bowl_mask(seg_flat: np.ndarray, bowl_uid: int) -> np.ndarray:
    """Handle both plain-id and packed (uid<<24|linkIndex) encodings."""
    eq = (seg_flat == int(bowl_uid))
    if eq.any():
        return eq
    vals_u32 = seg_flat.astype(np.uint32)
    uid_from_mask = (vals_u32 >> np.uint32(24)).astype(np.int64)
    return uid_from_mask == int(bowl_uid)


# ---------------- Main detection ----------------

def detect_bowl_rim(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    seg: Optional[np.ndarray] = None,
    bowl_uid: Optional[int] = None,
) -> Dict[str, object]:
    """
    Extract rim points and grasp candidates from a single RGB-D frame.
    Always returns a dict with keys: rim_pts_3d, center_3d, radius_m, grasp_candidates.
    Falls back to a model-based rim if data are insufficient.
    """
    if _CONTEXT is None:
        raise RuntimeError("Rim detection context not configured. Call configure_scene_context(...) first.")
    if depth.shape[:2] != rgb.shape[:2]:
        raise ValueError("Depth and RGB resolutions must match.")

    # 1) Denoise
    depth = _gaussian_blur_depth(depth, ksize=5, sigma=1.0)

    # 2) Back-project all valid
    points_cam, pixels = _depth_to_camera_points(depth, K)
    valid = depth.reshape(-1) > 0.0
    points_cam = points_cam[valid]
    if points_cam.size == 0:
        return _model_rim_from_context(_CONTEXT, _CONTEXT.sample_count)

    # 3) Prefer segmentation for isolation, else keep all valid
    if seg is not None and bowl_uid is not None:
        seg_flat = seg.reshape(-1)[valid]
        seg_mask = _decode_bowl_mask(seg_flat, int(bowl_uid))
        if np.any(seg_mask):
            points_cam = points_cam[seg_mask]
            if points_cam.shape[0] < 30:
                points_cam = _depth_to_camera_points(depth, K)[0][valid]

    # 4) Camera → world
    Pw = _transform_points(points_cam, _CONTEXT.world_from_cam)

    # 5) Optional AABB ROI. If it over-prunes, revert.
    if Pw.shape[0] > 12000:
        bowl_min = _CONTEXT.bowl_min - _CONTEXT.margin
        bowl_max = _CONTEXT.bowl_max + _CONTEXT.margin
        roi = np.all((Pw >= bowl_min) & (Pw <= bowl_max), axis=1)
        Pw_roi = Pw[roi]
        if Pw_roi.shape[0] >= 50:
            Pw = Pw_roi

    if Pw.shape[0] < 30:
        return _model_rim_from_context(_CONTEXT, _CONTEXT.sample_count)

    # 6) Remove tabletop (RANSAC)
    n, d, inliers = _ransac_plane(Pw)
    if inliers is not None and inliers.sum() > 100 and (n is not None):
        dist = (Pw @ n + d)
        keep = dist > 0.004
        if keep.sum() >= 30:
            Pw = Pw[keep]
    if Pw.shape[0] < 30:
        return _model_rim_from_context(_CONTEXT, _CONTEXT.sample_count)

    # 7) Rim band near upper envelope
    rim_h, bw = _adaptive_rim_band(Pw[:, 2], default_bw=0.015)
    band = np.abs(Pw[:, 2] - rim_h) <= bw
    rim = Pw[band]
    if rim.shape[0] < 30:
        band = np.abs(Pw[:, 2] - rim_h) <= (bw * 1.8)
        rim = Pw[band]
    if rim.shape[0] < 30:
        return _model_rim_from_context(_CONTEXT, _CONTEXT.sample_count)

    # 8) Fit circle on XY; if it fails, use model
    cxy, radius, inl = _fit_circle_ransac(rim[:, :2])
    if not inl.any() or radius <= 0.0:
        return _model_rim_from_context(_CONTEXT, _CONTEXT.sample_count)

    rim_pts = rim[inl]
    rim_h = float(np.mean(rim_pts[:, 2]))
    center_3d = np.array([cxy[0], cxy[1], rim_h], dtype=float)

    # 9) Sample grasp poses
    grasp_candidates = _sample_grasp_poses(center_3d, rim_pts, _CONTEXT)

    return {
        "rim_pts_3d": rim_pts,
        "center_3d": center_3d,
        "radius_m": float(radius),
        "grasp_candidates": grasp_candidates,
    }
