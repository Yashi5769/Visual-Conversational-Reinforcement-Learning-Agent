import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from ai2thor.controller import Controller

class ThorNavEnv(gym.Env):
    """
    Custom Gym Environment that wraps AI2-THOR for PPO Training.
    Goal: Learn to navigate without hitting obstacles.
    """
    def __init__(self, scene="FloorPlan10"):
        super(ThorNavEnv, self).__init__()
        
        print(f"🎮 Initializing RL Training Environment in {scene}...")
        self.controller = Controller(
            scene=scene,
            gridSize=0.25,
            width=300,  # Smaller resolution for faster training
            height=300,
            visibilityDistance=1.0, # Short view to force learning proximity
            renderDepthImage=False,
            renderObjectImage=False,
            renderClassImage=False
        )

        # ACTION SPACE: 0=MoveAhead, 1=RotateRight, 2=RotateLeft
        self.action_space = spaces.Discrete(3)

        # OBSERVATION SPACE: The robot sees a Grayscale Image (84x84)
        # We shrink it to 84x84 to mimic the standard Atari/DeepMind setup
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def step(self, action):
        """
        The Agent takes a step. We return (observation, reward, done, info)
        """
        event = self.controller.last_event
        reward = 0
        terminated = False
        truncated = False
        
        # --- 1. EXECUTE ACTION ---
        action_success = False
        if action == 0:   # Move Forward
            event = self.controller.step(action="MoveAhead")
            action_success = event.metadata["lastActionSuccess"]
        elif action == 1: # Turn Right
            event = self.controller.step(action="RotateRight", degrees=45)
            action_success = True 
        elif action == 2: # Turn Left
            event = self.controller.step(action="RotateLeft", degrees=45)
            action_success = True

        # --- 2. CALCULATE REWARD ---
        # CASE A: Collision (Move failed)
        if action == 0 and not action_success:
            reward = -10  # Punishment for hitting a wall
        
        # CASE B: Survival/Movement
        elif action == 0 and action_success:
            reward = +1   # Reward for moving forward safely
        
        else:
            reward = -0.1 # Small penalty for spinning in circles

        # --- 3. PROCESS OBSERVATION ---
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """
        Reset the robot to a random start location.
        """
        super().reset(seed=seed)
        
        # Teleport to a random safe spot
        positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        random_pos = positions[np.random.randint(0, len(positions))]
        
        self.controller.step(
            action="Teleport",
            position=random_pos,
            rotation=dict(x=0, y=np.random.choice([0, 90, 180, 270]), z=0)
        )
        
        return self._get_observation(), {}

    def _get_observation(self):
        """
        Helper: Convert Unity Frame -> 84x84 Grayscale for the Neural Network
        """
        frame = self.controller.last_event.cv2img
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2. Resize to 84x84 (Standard RL size)
        resized = cv2.resize(gray, (84, 84))
        # 3. Add channel dimension (84, 84, 1)
        return np.expand_dims(resized, axis=-1)

    def close(self):
        self.controller.stop()