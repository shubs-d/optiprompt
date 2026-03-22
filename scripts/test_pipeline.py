"""Test script to verify the new multi-model pipeline."""

import json
from app.core.pipeline import OptiPromptPipeline
import warnings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def test():
    print("Initializing Multi-Model Pipeline...")
    pipeline = OptiPromptPipeline()
    
    prompt = "Can you write a cool python script that fetches the HTML content of a webpage and parses it to get all the links please?"
    print(f"Original Prompt: {prompt}")
    
    print("Running optimization...")
    res = pipeline.optimize(prompt)
    
    print("\n--- RESULTS ---")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    test()
