import time
from ai2thor.controller import Controller

print("⏳ Initializing AI2-THOR (First run will download 500MB)...")
controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene="FloorPlan10",
    # This forces the window to show up
    renderDepthImage=True,
    renderInstanceSegmentation=True,
    width=900,
    height=900
)

print("✅ Simulator Started! Look for the Unity Window.")
time.sleep(2)

print("🔄 Rotating Robot...")
for _ in range(4):
    controller.step(action="RotateRight", degrees=90)
    time.sleep(1)

print("🏁 Test Complete.")


