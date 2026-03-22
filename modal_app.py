"""Modal deployment entrypoint for OptiPrompt FastAPI service."""

import modal
import os

def download_models():
    # This runs at build time to cache weights in the image
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer

    print("Downloading all-MiniLM-L6-v2...")
    SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Downloading distilgpt2...")
    AutoTokenizer.from_pretrained("distilgpt2")
    AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    print("Downloading TinyLlama...")
    AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.41.2",
        "sentence-transformers==3.0.1",
        "accelerate==0.31.0",
        "numpy"
    )
    .run_function(download_models)
    .add_local_dir("app", remote_path="/root/app")
)

app = modal.App("optiprompt")

# Persist data (like offline failed_optimizations.jsonl logs) across runs
volume = modal.Volume.from_name("optiprompt-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/optiprompt-logs": volume},
    timeout=600
)
@modal.asgi_app()
def fastapi_app_entry():
    """Expose the FastAPI app through Modal ASGI integration."""
    from app.main import app as fastapi_app
    import os
    # Divert dynamic logs into the persistent Modal volume instead of read-only Mount
    os.environ["FAILED_OPT_LOG_PATH"] = "/root/optiprompt-logs/failed_optimizations.jsonl"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return fastapi_app
