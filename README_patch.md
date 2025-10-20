# Runtime refresh for Robot Chef Pour Task

## What changed
- Rebuilt the runtime around a streamlined `RobotChefSimulation` that spawns the plane, table, bowl, pan, dual Panda arms, and grain particles with deterministic seeds.
- Added typed configuration loading (`robot_chef/config.py`) to preserve the existing YAML schema while exposing structured dataclasses for poses, camera, perception, tolerances, and task timing.
- Implemented a reusable RGB-D `Camera` wrapper with intrinsic calibration, aiming control, and metric depth conversion for PyBullet captures.
- Mounted the active camera rigidly to the Panda wrist (hand–eye calibrated) so it tracks the arm during motion and reports consistent CV-frame extrinsics.
- Replaced the perception stack with `perception/bowl_rim.py`, which uses PyBullet segmentation + depth back-projection (with analytic fallback) to fit the rim robustly and emit grasp candidates plus IBVS-ready feature pairs.
- Added `VisionRefineController` to run closed-loop image-based visual servoing (IBVS) that maps camera error into Panda joint velocities via a damped SVD pseudo-inverse.
- Rewrote the pour task pipeline to use the wrist camera, sweep the table when the bowl is not immediately visible, and then sequence perception, grasp refinement, pouring, and placement with overlay diagnostics and structured logging.
- Added a `scripts/smoke_pour.sh` helper for the CLI contract and documented the refresh in this patch file.

## IBVS refresher
1. Capture RGB-D imagery and detect the target rim features (two opposing rim points).
2. Convert pixel residuals into normalized image coordinates and build the interaction matrix for each feature.
3. Solve for the camera twist with a damped pseudo-inverse, transform it into the world frame, and map it to arm joint velocities through the Panda Jacobian.
4. Integrate joint velocities for a short horizon, stop when RMS pixel error and mean depth error fall within tolerances, otherwise warn on timeout.

## Run the demo
```bash
bash scripts/smoke_pour.sh                  # headless smoke test
# or interactively
python main.py --task pour_bowl_into_pan --config config/recipes/pour_demo.yaml
```

## Tunables
- `perception.grasp_clearance_m`
- `perception.rim_sample_count`
- `controller pixel_tol`, `depth_tol`, `gain`, `max_joint_vel`
- `camera.noise.depth_std`, `camera.noise.drop_prob`
- `task.hold_sec`

## Known limitations
- IBVS assumes a fixed external camera; the hand-eye transform is identity, so aggressive motions may degrade accuracy.
- Grasp quality scoring is heuristic and does not inspect collision-free reachability.
- Pouring kinematics rely on pre-defined poses rather than torque control, so spill quality is sensitive to pose tuning.
- Overlay diagnostics are stored in the working directory and will accumulate over repeated runs.

## Revert instructions
Reset to the previous runtime with Git:
```bash
git reset --hard
```
