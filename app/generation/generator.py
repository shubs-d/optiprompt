"""Variant generation using SLM."""

from app.models.loader import model_loader
import logging

logger = logging.getLogger(__name__)

PROMPT_TEMPLATES = {
    "aggressive": (
        "Rewrite the prompt with compression level: {aggression}.\n"
        "0 = minimal compression\n"
        "1 = maximum compression\n"
        "Keep meaning intact.\n"
        "Original: {prompt}\n\nOptimized prompt:"
    ),
    "balanced": (
        "Rewrite the prompt with compression level: {aggression}.\n"
        "0 = minimal, 1 = maximum. "
        "Remove unnecessary fillers but keep it structured.\n"
        "Original: {prompt}\n\nOptimized prompt:"
    ),
    "structure-enhanced": (
        "Rewrite the prompt with compression level: {aggression}.\n"
        "Use structured formats (bullets) while compressing where possible.\n"
        "Original: {prompt}\n\nOptimized prompt:"
    )
}

def generate_variants(prompt: str, aggression: float = 0.5) -> dict[str, str]:
    """Generate 3 variants of the prompt using dynamic aggression."""
    generator = model_loader.get_generator()
    variants = {}
    
    # Format the aggression to 2 decimal places for clarity to the model
    aggression_str = f"{max(0.0, min(1.0, aggression)):.2f}"
    
    for mode, template in PROMPT_TEMPLATES.items():
        formatted_prompt = template.format(prompt=prompt, aggression=aggression_str)
        try:
            outputs = generator(
                formatted_prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                return_full_text=False,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            generated_text = outputs[0]["generated_text"].strip()
            if not generated_text:
                generated_text = prompt
            variants[mode] = generated_text
        except Exception as e:
            logger.error(f"Failed to generate {mode} variant: {e}")
            variants[mode] = prompt
            
    return variants
