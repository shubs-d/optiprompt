"""GEPA Entropy calculation using distilgpt2."""

import torch
from app.models.loader import model_loader

def calculate_surprisal(prompt: str):
    """Calculate token surprisal (negative log probability) for each token.
    Returns a list of (token_id, token_string, surprisal_score).
    """
    model, tokenizer = model_loader.get_distilgpt2()
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    log_probs = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    surprisals = [-log_p.item() for log_p in log_probs]
    surprisals = [0.0] + surprisals # First token context padding
    
    # Extract IDs and strings
    ids_list = input_ids.view(-1).tolist()
    tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in ids_list]
    
    return list(zip(ids_list, tokens, surprisals))

def prune_prompt(prompt: str, aggression: float = 0.5) -> str:
    """Prune low-information tokens using entropy and aggression factor."""
    if not prompt.strip():
        return prompt
        
    model, tokenizer = model_loader.get_distilgpt2()
    scored_tokens = calculate_surprisal(prompt)
    
    base_threshold = 0.5
    scaling_factor = 2.5
    threshold = base_threshold + (aggression * scaling_factor)
    
    PROTECTED_KEYWORDS = set([
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must",
        "write", "create", "make", "build", "generate", "code", "explain", "analyze", "deploy", "run",
        "test", "debug", "fix", "update", "delete", "remove", "add", "insert", "select", "query",
        "kubernetes", "aws", "gcp", "azure", "docker", "python", "javascript", "react", "node", "sql", "api",
        "not", "no", "never", "always"
    ])
    
    kept_ids = []
    for t_id, t_str, surprisal in scored_tokens:
        # Check for protected tokens (strip typical subword markers like 'Ġ')
        clean_str = t_str.lower().strip(' \t\n\r.,!?#*Ġ')
        is_protected = (
            clean_str in PROTECTED_KEYWORDS or
            any(char.isdigit() for char in t_str) or
            not t_str.isalnum()
        )
        
        # Always keep first token (surprisal 0.0) or protected or high entropy tokens
        if surprisal >= threshold or surprisal == 0.0 or is_protected:
            kept_ids.append(t_id)
            
    if not kept_ids:
        return prompt
        
    return tokenizer.decode(kept_ids, clean_up_tokenization_spaces=True)
