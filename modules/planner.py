import json
from openai import OpenAI
import config

class LLMPlanner:
    def __init__(self, api_key=None):
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.LLM_BASE_URL
        )
        self.model = config.LLM_MODEL

    def generate_plan(self, user_command, rag_context):
        print("🤔 [LLM] Thinking...")
        
        # Using double braces {{ }} for JSON examples inside the f-string
        system_prompt = f"""
        You are the Planner for a household robot.
        
        AVAILABLE PRIMITIVES:
        - GOTO(location): Navigate to a room.
        - SCAN(object): Look for a specific object.
        - OPEN(object): Open a container.
        - CLOSE(object): Close a container.
        - GRAB(object): Pick up an object.
        - GIVE(person/location): Drop the item at the user's location or furniture.
        - WANDER: Explore the area randomly.
        CONTEXT:
        {rag_context}
        
        RULES:
        1. Output ONLY a valid JSON list.
        2. If the user implies "inside" a container, include an "OPEN" action first. <--- NEW
        3. If user says "Bring to me", use {{"action": "GIVE", "target": "User"}}.
        4. If user says "Put on [X]", use {{"action": "GIVE", "target": "[X]"}}.
        
        EXAMPLE:
        User: "Check the fridge and close it."
        Output: [{{"action": "GOTO", "target": "Kitchen"}}, {{"action": "OPEN", "target": "Fridge"}}, {{"action": "SCAN", "target": "Food"}}, {{"action": "CLOSE", "target": "Fridge"}}]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_command}
                ],
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            
            # --- ROBUST PARSING LOGIC (Backtracking) ---
            # 1. Find the FIRST opening bracket
            start_idx = content.find('[')
            if start_idx == -1:
                print(f"❌ No JSON list found in: {content}")
                return []
            
            # 2. Find the LAST closing bracket
            end_idx = content.rfind(']')
            
            # 3. Try parsing. If it fails (Extra Data), peel back to the previous ']'
            while end_idx > start_idx:
                candidate = content[start_idx : end_idx + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # We likely captured extra text with brackets. 
                    # Move end_idx back to the next ']'
                    end_idx = content.rfind(']', 0, end_idx)
            
            print("❌ Could not parse JSON plan.")
            return []

        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return []

    def generate_response(self, execution_log):
        prompt = f"""
        You are a helpful robot. You just executed:
        {execution_log}
        Generate a brief response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60
            )
            return response.choices[0].message.content
        except:
            return "Task complete."