🧠 VCRL-DKR: Visual-Conversational Reinforcement Learning Agentwith Dynamic Knowledge Retrieval & Embodied AI📖 Executive SummaryVCRL-DKR is a state-of-the-art Neuro-Symbolic AI system designed to bridge the gap between high-level natural language understanding and low-level physical robotic interaction.Unlike traditional chatbots that exist only in text, this agent operates in a photorealistic 3D world (AI2-THOR). It utilizes a "Brain-Body" architecture where a Large Language Model (LLM) acts as the high-level Planner, Computer Vision (YOLO) acts as the Eyes, and Reinforcement Learning (RL) provides autonomous navigation capabilities.🌟 Key Capabilities1. 🧠 Cognitive Reasoning (The Brain)Contextual Awareness: Uses RAG (Retrieval-Augmented Generation) via ChromaDB to recall household facts (e.g., "Where is the remote?" → "It's usually on the sofa").Dynamic Planning: Converts abstract requests like "I am hungry" into actionable physical steps (GOTO Kitchen → OPEN Fridge → GRAB Food).2. 👁️ Visual Perception (The Eyes)Object Detection: Integrated YOLOv8 to identify objects in real-time from the robot's camera feed.Visual Servoing: Implements intelligent Approach Logic—the robot spots an object from a distance, rotates to face it, and walks until it is within arm's reach (1.0m) before interacting.3. 🦾 Physical Interaction (The Body)Manipulation: Capable of Opening doors, Closing containers, Grabbing objects, and Dropping items using the physics engine.Precision Alignment: Includes logic to "Square Up" to objects (0°, 90°, 180°) and tilt the head dynamically to see inside containers or on tables.4. 🧭 Hybrid Navigation (The Feet)Teleportation: For known semantic locations (Kitchen, Bedroom).RL Wandering: A trained PPO (Proximal Policy Optimization) agent that can autonomously explore unknown environments without colliding with walls.🏗️ System ArchitectureThe project follows a modular pipeline:graph LR
    A[User Input] --> B{RAG Memory};
    B -->|Context| C[LLM Planner];
    C -->|JSON Plan| D[Execution Engine];
    D --> E[Robot Agent];
    E -->|Action| F[AI2-THOR Simulator];
    F -->|RGB Frame| G[YOLOv8 Vision];
    G -->|Detections| E;
    E -->|Status Updates| H[Streamlit Dashboard];
📂 Directory StructureVCRL_DKR/
├── app.py                 # Main Web Application (Streamlit Interface)
├── config.py              # Configuration Settings & API Keys
├── requirements.txt       # Dependency List
├── .env                   # API Keys (Hidden)
├── models/
│   ├── yolov8n.pt         # Computer Vision Model
│   └── ppo_nav/           # Trained Reinforcement Learning Brain
└── modules/
    ├── agent.py           # Robot Control Logic (Movement, Physics, RL)
    ├── knowledge.py       # RAG System (ChromaDB Vector Store)
    ├── planner.py         # LLM Interface (Prompt Engineering)
    └── vision.py          # YOLO Wrapper for Object Detection
⚙️ Installation & Setup1. PrerequisitesOS: macOS (Silicon/Intel), Windows 10/11, or Linux.Python: Version 3.10 or 3.11 (Strict requirement for AI2-THOR).2. Setup Virtual Environment# Create environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
3. Install Dependenciespip install -r requirements.txt
4. Configure SecretsCreate a file named .env in the root directory:GROQ_API_KEY="gsk_your_actual_key_here"
# OPENAI_API_KEY="sk_..." (Optional fallback)
HEADLESS_MODE="False"
5. Run the Applicationstreamlit run app.py
🚀 Usage ScenariosScenario 1: The "Grand Finale" (Full Stack Demo)User: "I am hungry. Find an apple in the fridge and bring it to me."Behavior:Planner infers location (Kitchen/Fridge).Agent navigates to Kitchen.Agent Opens the Fridge door.Vision detects Apple inside.Agent Grabs Apple.Agent returns to User and Drops the Apple.Scenario 2: Object Permanence & PhysicsUser: "Open the microwave, check for food, and then close it."Behavior:Robot approaches microwave.Rotates to face it perfectly (Visual Servoing).Opens door -> Scans -> Closes door.Scenario 3: Autonomous ExplorationUser: "Go to the bedroom and wander around."Behavior:Teleports to Bedroom.Switches to RL Mode (Neural Network).Drives autonomously avoiding obstacles.🧠 Technical Modules BreakdownModuleDescriptionplanner.pyUses Few-Shot Prompting with Llama-3 to generate strict JSON plans. Includes a robust Regex parser to handle LLM verbosity.agent.pyThe core controller. Implements Visual Servoing (approach logic), Head Tracking (calculating pitch/yaw to face objects), and PPO Integration for wandering.vision.pyWraps YOLOv8. Includes logic to map synonyms (e.g., "Couch" = "Sofa") and adjustable confidence thresholds.gym_env.pyA custom OpenAI Gym wrapper used to train the PPO agent on 84x84 grayscale images from the simulator.🛠️ Future ScopeSLAM (Simultaneous Localization and Mapping): Currently, the robot uses a pre-defined coordinate map. Future work involves building this map dynamically using depth sensors.Voice Interaction: Integrating Whisper (STT) and ElevenLabs (TTS) for full verbal communication.Open Vocabulary Vision: Replacing YOLO with CLIP or DETIC to detect any object in the world without pre-training on specific classes.👨‍💻 Author[Your Name]*Final Year
