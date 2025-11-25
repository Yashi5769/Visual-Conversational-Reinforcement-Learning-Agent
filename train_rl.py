import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from modules.gym_env import ThorNavEnv

# 1. Setup Directories
models_dir = "models/ppo_nav"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 2. Initialize Environment
# We wrap it in DummyVecEnv because SB3 expects vectorized environments
env = DummyVecEnv([lambda: ThorNavEnv(scene="FloorPlan10")])

# 3. Initialize PPO Agent
# CnnPolicy is used because the input is an Image (Convolutional Neural Network)
model = PPO(
    "CnnPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir,
    learning_rate=0.0003,
    n_steps=2048,
)

print("🧠 Starting PPO Training... (Press Ctrl+C to stop manually)")

# 4. Training Loop
# We train for 20,000 steps for a quick demo. 
# Real obstacle avoidance usually takes 100k+ steps.
TIMESTEPS = 10000
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

# 5. Save Model
model_path = f"{models_dir}/ppo_navigator_final"
model.save(model_path)
print(f"✅ Model saved to {model_path}.zip")

env.close()