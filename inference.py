import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# 1. MANDATORY HACKATHON VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-70b-chat-hf") # Example fallback
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. LOCAL ENV URL
ENV_URL = "https://chrry07-cloudops-rl.hf.space"

# Initialize compliant client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_llm_action(state):
    """Asks the LLM to choose an action based on the state."""
    prompt = f"""
    You are an AI DevOps Agent managing a cloud environment.
    Current System State:
    {state}

    You must reply with EXACTLY ONE of these action strings, and nothing else:
    scale_up
    scale_down
    restart_service
    restart_database
    rebalance_traffic
    clear_cache
    remove_idle_resource
    noop
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=20
        )
        # Get the raw text and make it lowercase
        raw_output = response.choices[0].message.content.strip().lower()
        print(f"[DEBUG] Raw AI Response: {raw_output}")
        
        valid_actions = ["scale_up", "scale_down", "restart_service", "restart_database", "rebalance_traffic", "clear_cache", "remove_idle_resource", "noop"]
        
        # Smarter parsing: check if any valid action is INSIDE the AI's sentence
        for action in valid_actions:
            if action in raw_output:
                return action
                
        return "noop" # Fallback if it hallucinated completely
        
    except Exception as e:
        print(f"LLM Error: {e}")
        return "noop"

def main():
    print("Starting CloudOps-RL Baseline Inference...")
    
    # Reset Environment
    res = requests.post(f"{ENV_URL}/reset")
    state = res.json().get("state", {})
    
    done = False
    step = 1
    total_reward = 0.0

    # Run loop
    while not done and step <= 10:
        print(f"\n--- Step {step} ---")
        action = get_llm_action(state)
        print(f"Agent chose: {action}")
        
        # Step Environment
        res = requests.post(f"{ENV_URL}/step", json={"action": action})
        result = res.json()
        
        state = result.get("state")
        reward = result.get("reward", 0)
        done = result.get("done", False)
        
        total_reward += reward
        print(f"Reward: {reward} | Total Reward: {total_reward} | Done: {done}")
        
        step += 1

    print("\nInference Complete.")

if __name__ == "__main__":
    main()