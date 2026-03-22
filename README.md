# OptiPrompt

OptiPrompt is a highly optimized, feedback-driven Small Language Model (SLM) engine designed to compress, structure, and optimize raw user prompts before sending them to large frontier models (e.g., GPT-4, Claude).

The system prioritizes compute efficiency, avoiding massive models in favor of composing smaller, specialized models that perform intent classification, entropy pruning, semantic evaluation, and heuristic tuning on edge GPUs.

---

## 🏗️ 1. High-Level System Layers

The project spans three main deployment environments:

1. **Frontend Interface (`chrome-extension/`)**: A Chromium browser extension that injects "⚡ Optimize" buttons directly into native DOM text areas. It reads raw text and communicates with the backend via REST.
2. **API Gateway (`app/api/`)**: An async FastAPI backend that acts as the ingress controller. It enforces strict JSON schemas and manages long-running inference requests dynamically tracking token compression savings over time.
3. **Optimization Engine (`app/core/`, `app/` modules)**: The specialized multi-model stack running prompt extraction, heuristical structuring, binary-search aggression tuning, and semantic generation.
4. **Cloud Deployment (`modal_app.py`)**: A Modal serverless GPU integration script configured to spin up ephemeral Debian containers running NVIDIA T4 GPUs. It uses a custom entrypoint to cache HuggingFace weights at image build time to prevent cold-start latencies.

---

## 🧠 2. The Multi-Model ML Stack

OptiPrompt shifted from static rule-based systems to an orchestrated multi-model methodology that achieves low memory overhead while preserving nuanced conversational semantics.

| Architecture | Model Loaded | Project Role (Module) |
| :--- | :--- | :--- |
| **Generative SLM** | `TinyLlama-1.1B` | Reconstructs and generates context-aware variants (`app/generation/generator.py`). |
| **Masked/Causal LM** | `distilgpt2` | Extract Negative Log Probabilities for Surprise-based pruning known as GEPA (`app/gepa/entropy.py`). |
| **Embedding Model** | `all-MiniLM-L6-v2` | Computes dense vector representations mapped to Cosine Similarity metrics (`app/semantic/embedding.py`). |

---

## ⚡ 3. The Core Optimization Pipeline

The core lifecycle of a prompt optimization request triggers when passing through the `AggressionController` (Binary Search Orchestrator). 

Instead of arbitrarily cutting prompts, the controller iterates over the following steps using **Binary Search Feedback** ($O(\log N)$ steps):

1. **Cleaner (`app/core/cleaner.py`)**: Fast and deterministic static pruning. Eliminates filler phrases using regex mappings.
2. **GEPA Entropy (`app/gepa/entropy.py`)**: Computes the Surprisal value ($-\log(P(token_{i} | context)$). High aggression thresholds drop highly contextual tokens that an LLM would effortlessly guess anyway, preserving the core informative tokens without losing context density.
3. **Variant Generator (`app/generation/generator.py`)**: Intercepts the pruned token soup and interpolates it into explicit template instructions using `TinyLlama`, forcefully restricting its generation properties dynamically.
4. **Intent Refinement (`app/intent/classifier.py`)**: Maps inputs against base keyword mappings but integrates `MiniLM` to measure semantic proximity to prototype queries if rule-based classification fails.

Through the Binary Search logic, the model generates candidate prompts, validates them against constraints containing both **token limits** and **semantic similarity**, and self-corrects the aggression bounds until the optimal threshold is dialed in.

---

## 🛠️ 4. Local Quick Start

### 1. Requirements Setup
OptiPrompt requires Python 3.11+. The model sizes are lightweight enough that they can be run on general CPU backends.

```bash
# Set up isolated VENV
python -m venv .venv
source .venv/bin/activate

# Install strictly defined PyTorch and Transformers
pip install -r requirements.txt
```

### 2. Startup Server Manually
Boot up the core multi-model backend server:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --reload
```

### 3. Deploy to Cloud GPUs
Instead of running heavy inference locally permanently, push directly to Modal where optimized builds cache the Model Checkpoints seamlessly:

```bash
modal serve modal_app.py
```
