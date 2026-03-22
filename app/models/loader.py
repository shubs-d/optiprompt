"""Singleton model loader for HuggingFace models."""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.distilgpt2_model = None
        self.distilgpt2_tokenizer = None
        self.minilm_model = None
        self.generator_pipeline = None
        
        self._initialized = True

    def get_distilgpt2(self):
        """Lazy load distilgpt2 for GEPA / entropy calculation."""
        if self.distilgpt2_model is None or self.distilgpt2_tokenizer is None:
            logger.info(f"Loading distilgpt2 on {self.device}")
            model_name = "distilgpt2"
            self.distilgpt2_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.distilgpt2_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.distilgpt2_model.eval()
        return self.distilgpt2_model, self.distilgpt2_tokenizer

    def get_minilm(self):
        """Lazy load MiniLM for semantic embeddings."""
        if self.minilm_model is None:
            logger.info(f"Loading all-MiniLM-L6-v2 on {self.device}")
            self.minilm_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        return self.minilm_model

    def get_generator(self):
        """Lazy load TinyLlama for variant generation to save memory and ensure speed."""
        if self.generator_pipeline is None:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            try:
                logger.info(f"Loading {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                device_map = "auto" if self.device == "cuda" else None
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    device_map=device_map,
                    torch_dtype=dtype
                )
                self.generator_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                raise e
        return self.generator_pipeline

# Global singleton instance
model_loader = ModelLoader()
