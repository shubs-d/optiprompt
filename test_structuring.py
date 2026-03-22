import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent))

from app.api.optimize import optimize_prompt, OptimizeRequest

def run():
    long_prompt = (
        "Act as an expert Cloud Architect. I need you to write a comprehensive design document "
        "and a Python script that will deploy my FastAPI application to Kubernetes on AWS EKS. "
        "Ensure that the script handles VPC creation, IAM roles, and sets up a Load Balancer. "
        "Must be under 500 words and use strict security practices. Do not use plain text secrets."
    )
    req = OptimizeRequest(prompt=long_prompt, constraints={}, compute_mode="balanced")
    
    try:
        res = optimize_prompt(req)
        print("Response:", json.dumps(res, indent=2))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    run()
