import time
import cv2
import math
import numpy as np
from ai2thor.controller import Controller
from stable_baselines3 import PPO

class RobotAgent:
    def __init__(self, scene="FloorPlan10", grid_size=0.25):
        print(f"🎮 Launching AI2-THOR Simulator in {scene}...")
        
        self.controller = Controller(
            scene=scene,
            gridSize=grid_size,
            width=600,
            height=600,
            visibilityDistance=3.0, 
            renderDepthImage=False,
            renderObjectImage=False,
            renderClassImage=False,
        )
        
        self.scene = scene
        self.current_location = "Hallway"
        
        try:
            self.rl_model = PPO.load("models/ppo_nav/ppo_navigator_final")
            print("🧠 RL Navigation Model Loaded Successfully.")
        except:
            self.rl_model = None

        self.location_map = {
            "Kitchen":     {"x": 0.0,  "z": 0.0, "y": 0.90, "rotation": 180},
            "Living Room": {"x": -1.5, "z": 1.0, "y": 0.90, "rotation": 90},
            "Hallway":     {"x": 0.5,  "z": 2.0, "y": 0.90, "rotation": 0},
            "Bedroom":     {"x": 2.5,  "z": 1.0, "y": 0.90, "rotation": 270},
            "Fridge":      {"x": 0.0,  "z": 1.0, "y": 0.90, "rotation": 180},
            "Microwave":   {"x": 0.0,  "z": 1.0, "y": 0.90, "rotation": 180},
            "Sink":        {"x": -0.5, "z": 1.0, "y": 0.90, "rotation": 180},
            "Sofa":        {"x": -1.5, "z": 1.0, "y": 0.90, "rotation": 90},
            "Bed":         {"x": 2.5,  "z": 1.0, "y": 0.90, "rotation": 270}
        }
        print("✅ Robot Online.")

    def _preprocess_for_rl(self):
        frame = self.controller.last_event.cv2img
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return np.expand_dims(resized, axis=-1)

    def face_object(self, obj_id):
        obj = next((o for o in self.controller.last_event.metadata["objects"] if o["objectId"] == obj_id), None)
        if not obj: return

        meta = self.controller.last_event.metadata
        camera_pos = meta["cameraPosition"] 
        agent_pos = meta["agent"]["position"]
        
        dx = obj["position"]["x"] - agent_pos["x"]
        dz = obj["position"]["z"] - agent_pos["z"]
        yaw = math.degrees(math.atan2(dx, dz))
        
        flat_dist = math.sqrt(dx**2 + dz**2)
        dy = obj["position"]["y"] - camera_pos["y"]
        pitch = -math.degrees(math.atan2(dy, flat_dist))
        
        self.controller.step(
            action="Teleport",
            position=agent_pos,
            rotation={"x": 0, "y": yaw, "z": 0},
            horizon=pitch
        )
        time.sleep(0.1)

    def snap_to_cardinal(self):
        agent_rot = self.controller.last_event.metadata["agent"]["rotation"]["y"]
        cardinal_rot = round(agent_rot / 90) * 90
        agent_pos = self.controller.last_event.metadata["agent"]["position"]
        self.controller.step(
            action="Teleport",
            position=agent_pos,
            rotation={"x": 0, "y": cardinal_rot, "z": 0},
            horizon=0 
        )
        time.sleep(0.1)

    def approach_object(self, obj_id, callback=None):
        print(f"🏃 Approaching object...")
        self.face_object(obj_id)
        previous_dist = float('inf')
        
        for _ in range(15): 
            # Update Web Interface
            if callback: callback(self.controller.last_event.cv2img)

            obj = next((o for o in self.controller.last_event.metadata["objects"] if o["objectId"] == obj_id), None)
            if not obj: break
            
            dist = obj["distance"]
            print(f"   📏 Distance: {dist:.2f}m")
            
            if dist <= 1.1: return True
            if dist > previous_dist: return True
            previous_dist = dist
            
            event = self.controller.step(action="MoveAhead")
            if not event.metadata["lastActionSuccess"]: break
            
            self.face_object(obj_id)
            
        return False

    # --- MAIN EXECUTION ---
    def execute_action(self, action_type, target, vision_system, visualizer_callback=None):
        """
        Now accepts a 'visualizer_callback' function to update the Web UI live.
        """
        status = "FAILED"
        observations = []

        # Helper to update UI
        def update_view():
            if visualizer_callback:
                visualizer_callback(self.controller.last_event.cv2img)

        # Show initial view
        update_view()

        if action_type == "GOTO":
            target_coords = None
            for key in self.location_map:
                if key.lower() in target.lower():
                    target_coords = self.location_map[key]
                    break
            
            if target_coords:
                print(f"🚀 Teleporting to {target}...")
                self.controller.step(
                    action="Teleport",
                    position={"x": target_coords['x'], "y": 0.90, "z": target_coords['z']},
                    rotation={"x": 0, "y": target_coords['rotation'], "z": 0},
                    horizon=0,
                    standing=True
                )
                self.current_location = target
                status = "SUCCESS"
                update_view() # Update UI after move
                time.sleep(1)
            else:
                status = "FAILED"

        elif action_type == "WANDER":
            if self.rl_model:
                print(f"🧠 Autonomous Exploring (Wander)...")
                status = "SUCCESS"
                for _ in range(20):
                    obs = self._preprocess_for_rl()
                    action, _ = self.rl_model.predict(obs)
                    if action == 0: self.controller.step(action="MoveAhead")
                    elif action == 1: self.controller.step(action="RotateRight", degrees=45)
                    elif action == 2: self.controller.step(action="RotateLeft", degrees=45)
                    
                    update_view() # Live Feed
                    time.sleep(0.05)
            else: status = "FAILED"

        elif action_type == "SCAN":
            print(f"👀 Scanning for {target}...")
            status = "NOT_FOUND"
            categories = {
                "food": ["apple", "orange", "bread", "lettuce", "tomato", "egg"],
                "fruit": ["apple", "orange", "tomato"],
                "container": ["bowl", "cup", "mug", "bottle"],
                "electronics": ["laptop", "cell phone", "remote", "tv", "microwave"],
                "potato": ["apple", "orange", "rock", "ball"],
                "sofa": ["couch", "sofa"],
                "remote control": ["remote"]
            }
            acceptable = categories.get(target.lower(), [target.lower()])

            horizons = [30, 0]
            for h in horizons:
                if status == "FOUND": break
                
                agent = self.controller.last_event.metadata["agent"]
                self.controller.step(
                    action="Teleport", position=agent["position"], rotation=agent["rotation"], horizon=h
                )
                
                for i in range(8): 
                    update_view() # Live Feed
                    
                    frame = self.controller.last_event.cv2img 
                    detected_objects = vision_system.detect_objects(frame)
                    
                    found_item = None
                    for detected in detected_objects:
                        for acc in acceptable:
                            if acc in detected.lower():
                                found_item = detected
                                break
                        if found_item: break
                    
                    if found_item:
                        status = "FOUND"
                        observations.append(f"I found {found_item}.")
                        break 
                    else:
                        self.controller.step(action="RotateRight", degrees=45)
                        time.sleep(0.15)

        elif action_type == "GRAB":
            print(f"✋ Attempting to GRAB {target}...")
            obj_id = None
            
            for i in range(12):
                update_view()
                for obj in self.controller.last_event.metadata["objects"]:
                    if target.lower() in obj["objectType"].lower() and obj["visible"]:
                        obj_id = obj["objectId"]
                        break
                if obj_id: break
                else:
                    if i < 11: self.controller.step(action="RotateRight", degrees=30)

            if obj_id:
                self.approach_object(obj_id, callback=visualizer_callback)
                self.face_object(obj_id) 
                update_view()
                try:
                    self.controller.step(action="PickupObject", objectId=obj_id, forceAction=False)
                    if len(self.controller.last_event.metadata["inventoryObjects"]) > 0:
                        status = "SUCCESS"
                        observations.append(f"I picked up the {target}.")
                    else:
                        status = "FAILED"
                except: status = "FAILED"
            else: status = "FAILED"

        elif action_type == "OPEN":
            print(f"👐 Attempting to OPEN {target}...")
            obj_id = None
            
            for i in range(12):
                update_view()
                for obj in self.controller.last_event.metadata["objects"]:
                    if target.lower() in obj["objectType"].lower() and obj["visible"]:
                        obj_id = obj["objectId"]
                        break
                if obj_id: break
                else:
                    if i < 11: self.controller.step(action="RotateRight", degrees=30)
            
            if obj_id:
                self.approach_object(obj_id, callback=visualizer_callback) 
                self.snap_to_cardinal()
                update_view()
                try:
                    self.controller.step(action="OpenObject", objectId=obj_id, forceAction=False)
                    status = "SUCCESS"
                    observations.append(f"I opened the {target}.")
                    update_view() # Show opened state
                    self.controller.step(action="MoveBack") 
                    time.sleep(1)
                except Exception as e:
                    status = "FAILED"
            else: status = "FAILED"

        elif action_type == "CLOSE":
            print(f"👐 Attempting to CLOSE {target}...")
            obj_id = None
            for i in range(12):
                update_view()
                for obj in self.controller.last_event.metadata["objects"]:
                    if target.lower() in obj["objectType"].lower() and obj["visible"]:
                        obj_id = obj["objectId"]
                        break
                if obj_id: break
                else:
                    if i < 11: self.controller.step(action="RotateRight", degrees=30)
            
            if obj_id:
                self.approach_object(obj_id, callback=visualizer_callback)
                self.snap_to_cardinal()
                update_view()
                try:
                    self.controller.step(action="CloseObject", objectId=obj_id, forceAction=False)
                    status = "SUCCESS"
                    observations.append(f"I closed the {target}.")
                    update_view()
                    time.sleep(1)
                except: status = "FAILED"
            else: status = "FAILED"

        elif action_type == "GIVE":
            print(f"🎁 Giving item to {target}...")
            if len(self.controller.last_event.metadata["inventoryObjects"]) > 0:
                self.controller.step(action="DropHandObject", forceAction=True)
                status = "SUCCESS"
                observations.append("I delivered the item.")
                update_view()
            else: status = "FAILED"

        return status, observations