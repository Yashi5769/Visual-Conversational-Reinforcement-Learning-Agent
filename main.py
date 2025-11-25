import time
import config
from modules.knowledge import KnowledgeBase
from modules.planner import LLMPlanner
from modules.vision import VisionSystem
from modules.agent import RobotAgent

def main():
    print("\n===========================================")
    print(" 🤖 VCRL-DKR: VISUAL-CONVERSATIONAL AGENT")
    print("===========================================\n")

    # 1. Initialize Components
    print(f"🔹 Loading Vision Model from: {config.VISION_CONF['model_path']}")
    vision = VisionSystem(model_path=config.VISION_CONF['model_path'])
    
    print(f"🔹 Initializing Knowledge Base...")
    kb = KnowledgeBase(collection_name=config.KNOWLEDGE_CONF['collection_name'])
    
    print(f"🔹 Connecting to Planner (LLM)...")
    planner = LLMPlanner()

    # 2. Initialize Robot
    print(f"🔹 Initializing Robot Agent in {config.ENV_CONF['scene_name']}...")
    bot = RobotAgent(
        scene=config.ENV_CONF['scene_name'], 
        grid_size=config.ENV_CONF['grid_size']
    )

    # 3. Seeding Knowledge
    print("🔹 Seeding Knowledge Base...")
    kb.ingest_documents([
        "Books are usually stored on the coffee table in the Living Room.",
        "The kitchen is north of the hallway.",
        "Apples are in the fridge.",
        "The remote control is usually on the sofa.",
        "The microwave is in the kitchen."
    ])

    print("\n✅ SYSTEM ONLINE. Ready for commands.")
    print("---------------------------------------")

    # 4. Interaction Loop
    while True:
        try:
            # Simple Text Input
            user_input = input("\n👤 COMMAND: ")
            
            if user_input.lower() in ['exit', 'quit']: 
                print("🛑 Shutting down system.")
                break

            # A. Retrieve Context (RAG)
            context = kb.retrieve_context(user_input)

            # B. Generate Plan (LLM)
            plan = planner.generate_plan(user_input, context)
            
            if not plan:
                print("🤖 Agent: I'm sorry, I couldn't create a plan for that.")
                continue
            
            print(f"📋 PLAN: {plan}")

            # C. Execute Plan (RL + Vision)
            execution_log = []
            
            for step in plan:
                action = step.get('action') # Use .get() for safety
                # CRITICAL FIX: Use .get() with a default string "area"
                # This prevents the crash if 'target' is missing
                target = step.get('target', "area") 
                
                # Execute the step
                status, obs = bot.execute_action(action, target, vision_system=vision)
                
                log_entry = f"Action: {action} {target} | Result: {status}"
                execution_log.append(log_entry)
                
                # Print status nicely
                icon = "✅" if status == "SUCCESS" or status == "FOUND" else "❌"
                print(f"   {icon} {log_entry}")
                
                if status == "FAILED":
                    print("   ⚠️ Stopping plan due to failure.")
                    break

            # D. Final Response
            final_reply = planner.generate_response(execution_log)
            print(f"🤖 AGENT: {final_reply}")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"⚠️ System Error: {e}")

if __name__ == "__main__":
    main()