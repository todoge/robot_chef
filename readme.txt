Robot Chef Fried Rice Demonstration
==================================

This project provides a PyBullet-based simulation of a dual-arm robot executing a
preset fried rice recipe. The simplified demonstration focuses on robust action
sequencing, pouring, stirring, opening a sauce bottle, and plating the dish using
scripted motion primitives.

Getting Started
---------------

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run the simulation:

   ```bash
   python main.py           # Launch with GUI
   python main.py --headless  # Run without GUI for automated testing
   ```

Project Structure
-----------------

- `main.py`: Entry point that runs the recipe executor.
- `robot_chef/config.py`: Workspace and motion tuning parameters.
- `robot_chef/simulation.py`: Environment setup, robot helper functions, and utilities.
- `robot_chef/actions.py`: Definitions of the fried rice action primitives.
- `robot_chef/executor.py`: Finite state machine running the scripted recipe.

Extending the Demo
------------------

- Replace the placeholder Franka Panda arms with XArm URDFs or real hardware drivers.
- Integrate perception to verify object locations or adapt to perturbations.
- Calibrate pouring angles and stirring trajectories for physical execution.
- Instrument evaluation metrics such as completion time and spill estimation.
