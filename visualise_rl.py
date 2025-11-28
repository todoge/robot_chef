from robot_chef.rl_env import RobotChefSimulation
import argparse
from pathlib import Path
from stable_baselines3 import SAC
from robot_chef import config
import time


def parse_args():
    ap = argparse.ArgumentParser(description="Visualize Trained Robot Chef Agent")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--model-path", required=True, help="Path to trained model checkpoint (.zip)")
    ap.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return ap.parse_args()


args = parse_args()

# Load config and create environment with GUI enabled
cfg = config.load_pour_task_config(Path(args.config))
env = RobotChefSimulation(gui=True, recipe=cfg)  # Enable GUI for visualization

# Load the trained model
model = SAC.load(args.model_path)
print(f"Loaded model from {args.model_path}")

# Run visualization episodes
for episode in range(args.episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    step_count = 0
    
    print(f"\n=== Episode {episode + 1} ===")
    
    while not done:
        # Predict action (deterministic=True for evaluation)
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        # Optional: add delay for better visualization
        time.sleep(0.01)
        
        # Check if episode terminated
        if done or truncated:
            break
    
    print(f"Episode {episode + 1} finished after {step_count} steps")
    print(f"Total reward: {episode_reward:.2f}")

env.close()
print("\nVisualization complete!")
