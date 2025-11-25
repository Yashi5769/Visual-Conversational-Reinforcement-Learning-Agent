import time
import cv2
import numpy as np
from stable_baselines3 import PPO
from modules.gym_env import ThorNavEnv

# 1. Load the Environment
print("🎮 Loading Environment...")
env = ThorNavEnv(scene="FloorPlan10")

# 2. Load the Trained Brain
model_path = "models/ppo_nav/ppo_navigator_final"
print(f"🧠 Loading Model from {model_path}...")
model = PPO.load(model_path)

# 3. The Testing Loop
obs, _ = env.reset()
print("🚀 Starting Autonomous Navigation Test...")

for i in range(500): # Run for 500 steps
    # Ask the AI what to do based on what it sees (obs)
    action, _states = model.predict(obs)
    
    # Execute the action
    obs, reward, done, truncated, info = env.step(action)
    
    # Show what the robot sees (The 84x84 Grayscale Input)
    # We resize it up to 300x300 so you can see it on your screen
    view = cv2.resize(obs, (300, 300))
    cv2.imshow("Robot Brain Input", view)
    cv2.waitKey(1)
    
    time.sleep(0.05) # Slow down so we can watch

print("🏁 Test Complete.")
env.close()