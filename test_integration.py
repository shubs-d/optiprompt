import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from app.api.optimize import optimize_prompt, OptimizeRequest
import json
import asyncio

def run():
    req = OptimizeRequest(
        prompt="Could you write a Python script to deploy this app to Kubernetes?",
        constraints={"max_tokens": 15, "semantic_threshold": 0.8},
        compute_mode="balanced"
    )
    try:
        res = optimize_prompt(req)
        print("Response:", json.dumps(res, indent=2))
        
        # Check log file
        log_path = Path("/data/optimization_logs.json")
        if not log_path.exists():
            log_path = Path("data/optimization_logs.json")
            
        print(f"\nLog file {log_path} exists: {log_path.exists()}")
        if log_path.exists():
            with open(log_path, "r") as f:
                logs = json.load(f)
                print(f"Num logs: {len(logs)}")
                print("Last log:", json.dumps(logs[-1], indent=2))
    except Exception as e:
        print("Error:", e)
        raise

if __name__ == "__main__":
    run()
