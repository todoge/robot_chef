from robot_chef.rl_env import RobotChefSimulation
import argparse
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

def parse_args():
    ap = argparse.ArgumentParser(description="Robot Chef")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return ap.parse_args()


args = parse_args()
from robot_chef import config  # agent will create
cfg = config.load_pour_task_config(Path(args.config))
env = RobotChefSimulation(gui=not args.headless, recipe=cfg)
#obs, info = env.reset()
model = SAC(
        "MlpPolicy",   # Multilayer perceptron policy (fully connected)
        env,
        verbose=1,
        batch_size=256,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        seed=42
    )

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='franka_sac')
    
# 4. Start learning (training)
total_timesteps = 500_000  # set as needed
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# 5. Save the final model
model.save("pouring_final")

env.close()



