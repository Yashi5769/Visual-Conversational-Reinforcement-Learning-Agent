import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()

# 2. Project Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

# 3. API Key Logic (Supports both OpenAI and Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# If a Groq key exists, we use it as our main API key
if (GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_")) or \
   (OPENAI_API_KEY and OPENAI_API_KEY.startswith("gsk_")):
    
    print("✅ Detected Groq Key (gsk_...). Routing to Groq Llama-3.")
    actual_key = GROQ_API_KEY if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_") else OPENAI_API_KEY
    
    OPENAI_API_KEY = actual_key
    LLM_BASE_URL = "https://api.groq.com/openai/v1"
    LLM_MODEL = "llama-3.3-70b-versatile"
elif OPENAI_API_KEY and not OPENAI_API_KEY.startswith("gsk_"):
    print("✅ Detected OpenAI API Key. Using GPT-3.5 model.")
    LLM_BASE_URL = None # Default OpenAI URL
    LLM_MODEL = "gpt-3.5-turbo"
else:
    # If neither is found, we crash with a helpful error
    raise ValueError("❌ No API Key found! Please set GROQ_API_KEY or OPENAI_API_KEY in your .env file.")
# 4. Computer Vision Config (YOLO)
VISION_CONF = {
    "model_path": MODELS_DIR / "yolov8n.pt",
    "confidence_threshold": 0.5,
    "save_output": True,
    "output_path": OUTPUT_DIR / "vision_logs"
}

# 5. RAG / Knowledge Base Config
KNOWLEDGE_CONF = {
    "collection_name": "home_knowledge_v1",
    "persist_directory": str(DATA_DIR / "chroma_db"),
    "embedding_model": "all-MiniLM-L6-v2"
}

# 6. RL Environment Config
ENV_CONF = {
    "scene_name": "FloorPlan10",
    "grid_size": 0.25,
    "render_depth": True,
    "width": 640,
    "height": 640,
    "headless": os.getenv("HEADLESS_MODE", "False").lower() == "true"
}