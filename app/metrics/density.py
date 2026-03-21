"""Information density metrics."""

from typing import Iterable

from app.utils.text_utils import is_meaningful_token


def information_density(tokens: Iterable[str]) -> float:
    """Return meaningful_token_count / total_token_count."""
    token_list = list(tokens)
    if not token_list:
        return 0.0

    meaningful = sum(1 for token in token_list if is_meaningful_token(token))
    return round(meaningful / len(token_list), 4)
