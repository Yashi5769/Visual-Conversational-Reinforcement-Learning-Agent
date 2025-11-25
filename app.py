import streamlit as st
import cv2
import config
from modules.knowledge import KnowledgeBase
from modules.planner import LLMPlanner
from modules.vision import VisionSystem
from modules.agent import RobotAgent

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VCRL-DKR Robot Controller",
    page_icon="🤖",
    layout="wide"
)

# --- 2. ADVANCED CSS STYLING (NEURAL HUD THEME) ---
st.markdown("""
<style>
    /* 1. IMPORT FUTURISTIC FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;900&family=JetBrains+Mono:wght@300;400;700&display=swap');

    /* 2. GLOBAL APP THEME */
    .stApp {
        background-color: #050505;
        background-image: 
            linear-gradient(rgba(0, 255, 136, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 136, 0.03) 1px, transparent 1px);
        background-size: 30px 30px; /* Grid texture */
        color: #e0e0e0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* 3. HIDE STREAMLIT UI CLUTTER */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stToolbar"] {visibility: hidden;}

    /* 4. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 12, 15, 0.95);
        border-right: 1px solid rgba(0, 255, 136, 0.2);
        box-shadow: 5px 0 20px rgba(0, 0, 0, 0.5);
    }
    
    /* 5. TYPOGRAPHY HEADERS */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #00ff88, #00f2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }

    /* 6. CHAT CARDS (HUD STYLE) */
    .user-card {
        background: linear-gradient(135deg, rgba(0, 242, 255, 0.05) 0%, rgba(0, 0, 0, 0) 100%);
        border-left: 3px solid #00f2ff;
        border-top: 1px solid rgba(0, 242, 255, 0.1);
        padding: 15px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0, 242, 255, 0.05);
        position: relative;
    }
    
    .bot-card {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.05) 0%, rgba(0, 0, 0, 0) 100%);
        border-left: 3px solid #00ff88;
        border-top: 1px solid rgba(0, 255, 136, 0.1);
        padding: 15px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.05);
    }
    
    /* Add tiny tech labels to cards */
    .user-card::before { content: "OPERATOR_INPUT"; position: absolute; top: 2px; right: 5px; font-size: 0.6em; opacity: 0.5; color: #00f2ff; }
    .bot-card::before { content: "SYS_RESPONSE"; position: absolute; top: 2px; right: 5px; font-size: 0.6em; opacity: 0.5; color: #00ff88; }

    /* 7. STATUS GRID & ANIMATIONS */
    .status-grid {
        display: flex;
        gap: 8px;
        margin-bottom: 20px;
    }
    .status-item {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid #333;
        border-radius: 4px;
        padding: 10px;
        flex: 1;
        text-align: center;
        font-size: 0.75em;
        font-weight: 700;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    /* Pulsing Dot Animation */
    @keyframes pulse-green { 0% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); } 70% { box-shadow: 0 0 0 6px rgba(0, 255, 136, 0); } 100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); } }
    
    .dot {
        height: 8px;
        width: 8px;
        background-color: #333;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .dot.active {
        background-color: #00ff88;
        box-shadow: 0 0 5px #00ff88;
        animation: pulse-green 2s infinite;
    }

    /* 8. VISION FEED FRAME (VIEWFINDER LOOK) */
    .vision-frame {
        position: relative;
        border: 2px solid #222;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
    }
    /* Corner Accents for "Camera" look */
    .vision-frame::after {
        content: "";
        position: absolute;
        top: 10px; left: 10px; right: 10px; bottom: 10px;
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 4px;
        pointer-events: none;
        z-index: 10;
    }
    div[data-testid="stImage"] img {
        border-radius: 8px;
        filter: contrast(1.1) saturate(1.1); /* Slight enhancement to video */
    }

    /* 9. INPUT FIELDS (GLOWING) */
    .stTextInput input {
        background-color: rgba(0, 0, 0, 0.5) !important;
        color: #00f2ff !important;
        border: 1px solid #333 !important;
        border-radius: 5px;
        font-family: 'JetBrains Mono', monospace;
    }
    .stTextInput input:focus {
        border-color: #00f2ff !important;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.2) !important;
    }
    
    /* 10. EXPANDERS & INFO BOXES */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.05);
        border-radius: 5px;
        font-size: 0.9em;
    }
    div[data-testid="stAlert"] {
        background-color: rgba(0, 255, 136, 0.1);
        color: #00ff88;
        border: 1px solid rgba(0, 255, 136, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION (Cached) ---
# This prevents the simulator from reloading every time you click a button
@st.cache_resource
def initialize_system():
    print("🚀 BOOTING UP VCRL-DKR SYSTEM...")
    
    # 1. Vision
    vision = VisionSystem(model_path=config.VISION_CONF['model_path'])
    
    # 2. Knowledge Base
    kb = KnowledgeBase(collection_name=config.KNOWLEDGE_CONF['collection_name'])
    # Pre-seed knowledge
    kb.ingest_documents([
        "Books are usually stored on the coffee table in the Living Room.",
        "The kitchen is north of the hallway.",
        "Apples are in the fridge.",
        "The remote control is usually on the sofa.",
        "The microwave is in the kitchen."
    ])
    
    # 3. Planner
    planner = LLMPlanner()
    
    # 4. Robot Agent (Launches Unity)
    bot = RobotAgent(
        scene=config.ENV_CONF['scene_name'], 
        grid_size=config.ENV_CONF['grid_size']
    )
    
    return vision, kb, planner, bot

# Show a spinner while Unity loads
with st.spinner("Initializing Robot & Simulator... (This takes ~10 seconds)"):
    vision, kb, planner, bot = initialize_system()

# --- LAYOUT SETUP ---
st.title("🤖 VCRL-DKR: Neuro-Symbolic Agent")

# Create two columns: Left for Vision, Right for Chat
col_feed, col_chat = st.columns([1.2, 1])

with col_feed:
    st.subheader("👁️ Robot Vision Feed")
    # This is the placeholder where the video will stream
    camera_view = st.empty() 
    # Default placeholder image
    camera_view.info("Waiting for robot to move...")

with col_chat:
    st.subheader("💬 Mission Control")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "System Online. How can I help you?"}]

    # Display Chat History
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

# --- REAL-TIME VISUALIZER CALLBACK ---
def update_streamlit_view(frame):
    """
    This function is passed to the Robot Agent.
    The Agent calls this function every time it moves or looks.
    """
    if frame is not None:
        # Convert BGR (OpenCV/Unity standard) to RGB (Web standard)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Update the image placeholder immediately
        camera_view.image(rgb_frame, channels="RGB", caption="Live Agent View", use_container_width=True)

# --- MAIN LOGIC ---
if prompt := st.chat_input("Enter command (e.g., 'Find the apple in the fridge'):"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Robot Processing
        with st.chat_message("assistant"):
            status_box = st.empty()
            status_box.markdown("🤔 *Thinking...*")
            
            # A. RAG Retrieval
            context = kb.retrieve_context(prompt)
            
            # B. LLM Planning
            plan = planner.generate_plan(prompt, context)
            
            if not plan:
                response = "I'm sorry, I couldn't generate a valid plan for that command."
                status_box.error(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Show Plan
                plan_str = "\n".join([f"- **{step.get('action')}** {step.get('target', 'area')}" for step in plan])
                status_box.markdown(f"**📋 Plan Generated:**\n{plan_str}\n\n*Executing...*")
                
                # C. Execution Loop
                full_status_text = f"**📋 Plan Generated:**\n{plan_str}\n\n"
                
                for step in plan:
                    action = step.get('action')
                    target = step.get('target', 'area')
                    
                    # Show current step
                    status_box.markdown(f"{full_status_text}⏳ *Executing: {action} {target}...*")
                    
                    # --- CRITICAL: EXECUTE WITH CALLBACK ---
                    # We pass 'update_streamlit_view' so the agent can send images back here
                    status, _ = bot.execute_action(
                        action, 
                        target, 
                        vision_system=vision, 
                        visualizer_callback=update_streamlit_view
                    )
                    
                    # Update Log
                    icon = "✅" if status == "SUCCESS" or status == "FOUND" else "❌"
                    full_status_text += f"{icon} **{action} {target}**: {status}\n\n"
                    
                    if status == "FAILED":
                        full_status_text += "⚠️ *Plan stopped due to failure.*"
                        status_box.markdown(full_status_text)
                        break
                
                # D. Final Response
                final_reply = planner.generate_response(f"Plan executed. Status: {full_status_text}")
                final_output = f"{full_status_text}\n---\n**🤖 Agent:** {final_reply}"
                status_box.markdown(final_output)
                
                st.session_state.messages.append({"role": "assistant", "content": final_output})