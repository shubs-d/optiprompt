"""Semantic Structuring Layer for high-density prompt transformation."""

def extract_role(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for prefix in ["act as a ", "act as an ", "act as ", "you are a ", "you are an ", "you are "]:
        if prefix in prompt_lower:
            idx = prompt_lower.find(prefix) + len(prefix)
            end = prompt_lower.find(".", idx)
            end2 = prompt_lower.find(",", idx)
            end3 = prompt_lower.find("\n", idx)
            candidates = [e for e in [end, end2, end3] if e != -1]
            end_idx = min(candidates) if candidates else len(prompt)
            return prompt[idx:end_idx].strip().title()
    return "Expert AI Assistant"

def extract_objective(prompt: str) -> str:
    prompt_lower = prompt.lower()
    verbs = ["write", "create", "explain", "analyze", "generate", "build", "describe", "summarize", "help", "provide", "solve"]
    for v in verbs:
        if v in prompt_lower:
            idx = prompt_lower.find(v)
            end = prompt_lower.find(".", idx)
            end2 = prompt_lower.find("\n", idx)
            candidates = [e for e in [end, end2] if e != -1]
            end_idx = min(candidates) if candidates else len(prompt)
            return prompt[idx:end_idx].strip().capitalize()
    
    sentences = prompt.split('.')
    return sentences[0].strip() if sentences else prompt.strip()

def extract_constraints(prompt: str) -> list:
    constraints = []
    prompt_lower = prompt.lower()
    
    if "under" in prompt_lower and "words" in prompt_lower:
        constraints.append("Follow strict length limits")
    elif "max" in prompt_lower and "tokens" in prompt_lower:
         constraints.append("Follow token limits")
         
    markers = ["must", "do not", "ensure", "only", "never", "always", "in python", "using"]
    for m in markers:
        if m in prompt_lower:
            idx = prompt_lower.find(m)
            end = prompt_lower.find(".", idx)
            end2 = prompt_lower.find(",", idx)
            end3 = prompt_lower.find("\n", idx)
            candidates = [e for e in [end, end2, end3] if e != -1]
            end_idx = min(candidates) if candidates else len(prompt)
            c = prompt[idx:end_idx].strip()
            if len(c) > 3:
                constraints.append(c)
                
    return list(set(constraints)) if constraints else ["No explicit constraints provided"]

def extract_input(prompt: str) -> str:
    """Extract context/input excluding obvious instructions."""
    lines = [line.strip() for line in prompt.split('\n') if line.strip()]
    if len(lines) > 1:
        # assume first line might be instruction
        return "\n".join(lines[1:])
    return prompt.strip()

def format_constraints(constraints: list) -> str:
    if not constraints:
        return "- None"
    return "\n".join(f"- {c.capitalize()}" for c in constraints)

def build_structured_prompt(role: str, objective: str, constraints: list, input_text: str) -> str:
    return f"Role: {role}\nObjective: {objective}\nConstraints:\n{format_constraints(constraints)}\n\nInput:\n{input_text}"
